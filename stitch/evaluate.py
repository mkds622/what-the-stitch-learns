"""Evaluation.

Two questions, answered separately.

**Does the coder describe the activation?** Reconstruction error, sparsity and
how much of the dictionary is in use, measured on activations the coder was not
trained on. These are cheap and run during training.

**Does the model still work with the reconstruction in place of the real
activation?** The upper half of the model runs normally, the coder replaces the
activation at the cut point, and the lower half carries on. If the prediction
survives, the coder kept what the rest of the model was using. If it does not,
a low reconstruction error was measuring something the model does not care
about. This is expensive and runs at the end.

The second is the one that cannot be argued with. A coder can reach a good
mean squared error by reproducing the large, uninformative directions of an
activation and discarding the small ones the classifier actually reads.

Validity gates sit on top of both. They do not say a coder is good; they say it
is not obviously broken, and a coder that fails one should not be compared with
anything.

Explained variance
------------------
The one definition used by every measurement here and by the linear baseline in
the sweep, written down once because the reconstruction gate compares the two:

    1 - (sum over dimensions of the mean squared residual)
        / (sum over dimensions of the variance of the target)

Summed over dimensions rather than averaged over them, so a dimension carrying
more of the activation's variance counts for more. Both sums are accumulated
over the whole evaluation set rather than averaged per batch, which is a
different number. One is an exact reconstruction, zero is no better than
predicting the target's mean, and a map fitted on tokens other than the ones it
is scored on can land below zero.

It is invariant to a single scalar per layer, because that scalar squares out of
the residual and the total alike. The activation scaling is one scalar per
layer, so a coder working in scaled units and a linear fit measured in whatever
units the sweep fed it are comparable. Per-dimension scaling would not cancel
and the two would not be comparable.
"""

from __future__ import annotations

import logging
import math
from typing import Iterator

import torch
from torch import Tensor, nn

from stitch.coder import SparseCoder

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# how the firing is spread across the dictionary
# ----------------------------------------------------------------------

def frequency_concentration(frequencies: Tensor) -> dict[str, float]:
    """Whether the live dictionary shares the work or a few features do it all.

    Sparsity says how many features fire per token. It says nothing about
    whether the same few fire every time. A dictionary where a handful of
    features carry almost all the firing is closer to a dense code in a small
    basis than to a sparse code in a large one, and the two are not
    distinguishable by L0.

    Measured over the features that fired at least once. Dead features have
    their own metric, and counting them here would make this a second reading
    of that one rather than a statement about the dictionary in use.

    Two readings of the same distribution, comparable across different things.
    ``gini`` is a share, so it is comparable between runs holding the same
    number of live features and not between a dictionary with a thousand live
    features and one with five thousand: the same unevenness reads differently
    at the two sizes. ``effective`` is a count of features, so it is comparable
    between any two runs reconstructing the same activation, whatever their
    live counts or configured expansions.

    Args:
        frequencies: share of tokens each feature fired on, from
            :func:`feature_frequencies` or accumulated alongside the metrics.

    Returns:
        ``gini``, zero when every live feature fires equally often and
        approaching one when a single feature does everything; ``median``, the
        typical live feature's share of tokens; and ``effective``, the number
        of features the firing is spread over, as the exponential of the
        entropy of the normalised firing frequencies. That is exactly ``m``
        when ``m`` features fire equally often and the rest are dead, one when
        a single feature does everything, and between the two otherwise, so a
        dictionary with five thousand live features of which a few hundred do
        the work reports a few hundred. All three are zero when nothing fired,
        which the collapse gate catches separately.
    """
    live = frequencies[frequencies > 0].to(torch.float64)
    n = live.numel()
    if n == 0:
        return {"gini": 0.0, "median": 0.0, "effective": 0.0}

    ordered = torch.sort(live).values
    total = float(ordered.sum())
    if total <= 0:
        return {"gini": 0.0, "median": 0.0, "effective": 0.0}

    rank = torch.arange(1, n + 1, dtype=torch.float64, device=ordered.device)
    gini = 2.0 * float((rank * ordered).sum()) / (n * total) - (n + 1) / n

    # Normalised into a distribution over the live features, so this is the
    # shape of the firing and not its overall rate: doubling every frequency
    # leaves it unchanged.
    share = ordered / total
    entropy = float(-(share * share.log()).sum())

    return {
        "gini": max(0.0, gini),
        "median": float(ordered.median()),
        "effective": math.exp(entropy),
    }


# ----------------------------------------------------------------------
# reconstruction
# ----------------------------------------------------------------------

@torch.no_grad()
def reconstruction_metrics(coder: SparseCoder, batches: Iterator[dict[str, Tensor]],
                           source_key: str, target_key: str, count: int,
                           device: torch.device | str = "cpu") -> dict[str, float]:
    """How well the coder describes activations it was not trained on.

    Args:
        coder: the coder, in whatever state it is in.
        batches: an activation stream, usually over the evaluation split.
        source_key: layer the coder reads.
        target_key: layer it reconstructs. The same as ``source_key`` when it
            reproduces its own input.
        count: batches to average over. More is tighter and slower.
        device: where to run.

    Returns:
        Averages over those batches, plus two numbers about the dictionary:
        the share of features silent here, and the largest share of tokens
        any single feature fired on. A feature active on almost every token is
        not a feature, it is a bias with extra steps.

        ``eval_explained_variance`` is the quantity defined at the top of this
        module, accumulated over every token read rather than averaged per
        batch.

        ``eval_l0_frac`` is the active features per token over that
        width, which is the unit sparsity is compared in: at one, the code has
        as many degrees of freedom as the thing it describes.

        ``eval_effective_expansion`` is the features that fired at least once
        over the same width, which is how overcomplete the dictionary is in
        practice rather than in configuration. Below one the live dictionary
        cannot span the activation, so nothing that was superposed in it can
        be separated out. The denominator is the width being reconstructed,
        while the configured expansion is built from the width being read, so
        the two are the same base only when a coder reproduces its own input.

        ``eval_frequency_gini``, ``eval_median_feature_frequency`` and
        ``eval_effective_features`` describe how the firing is spread across
        the live dictionary, which L0 does not say. The last is a count of
        features rather than a share, which is what makes it readable between
        runs whose live counts differ; the Gini coefficient is not. See
        :func:`frequency_concentration`.
    """
    device = torch.device(device)
    coder = coder.to(device).eval()

    totals = {"mse": 0.0, "l0": 0.0}
    residual_sum = torch.zeros(coder.d_out, device=device)
    target_sum = torch.zeros(coder.d_out, device=device)
    target_square = torch.zeros(coder.d_out, device=device)
    fired = torch.zeros(coder.d_latent, device=device)
    tokens = 0

    for index, batch in enumerate(batches):
        if index >= count:
            break

        x = batch[source_key].to(device)
        target = batch[target_key].to(device)
        reconstruction, latents = coder(x)

        n = x.shape[0]
        tokens += n

        totals["mse"] += float(((reconstruction - target) ** 2).mean()) * n
        totals["l0"] += float((latents > 0).float().sum(dim=-1).sum())

        # Accumulated rather than averaged per batch, so the variance is over
        # the whole evaluation set rather than the mean of per batch variances.
        residual_sum += ((target - reconstruction) ** 2).sum(dim=0)
        target_sum += target.sum(dim=0)
        target_square += (target ** 2).sum(dim=0)
        fired += (latents > 0).float().sum(dim=0)

    if tokens == 0:
        raise ValueError("the activation stream produced no batches")

    target_variance = (target_square / tokens) - (target_sum / tokens) ** 2
    unexplained = (residual_sum / tokens).sum()
    total_variance = target_variance.sum()
    explained = float(1.0 - unexplained / total_variance) if total_variance > 0 else 0.0

    frequencies = fired / tokens

    l0 = totals["l0"] / tokens
    alive = float((frequencies > 0).float().sum())
    concentration = frequency_concentration(frequencies)

    return {
        "eval_mse": totals["mse"] / tokens,
        "eval_l0": l0,
        "eval_l0_frac": l0 / coder.d_out,
        "eval_explained_variance": explained,
        "eval_dead_or_dormant": float((frequencies == 0).float().mean()),
        "eval_effective_expansion": alive / coder.d_out,
        "eval_max_feature_frequency": float(frequencies.max()),
        "eval_median_feature_frequency": concentration["median"],
        "eval_frequency_gini": concentration["gini"],
        "eval_effective_features": concentration["effective"],
        "eval_tokens": float(tokens),
    }


@torch.no_grad()
def feature_frequencies(coder: SparseCoder, batches: Iterator[dict[str, Tensor]],
                        source_key: str, count: int,
                        device: torch.device | str = "cpu") -> Tensor:
    """Share of tokens each feature fires on.

    Kept separate from the metrics because it is a distribution rather than a
    number, and is written out as an artifact rather than logged.
    """
    device = torch.device(device)
    coder = coder.to(device).eval()

    fired = torch.zeros(coder.d_latent, device=device)
    tokens = 0
    for index, batch in enumerate(batches):
        if index >= count:
            break
        latents = coder.encode(batch[source_key].to(device))
        fired += (latents > 0).float().sum(dim=0)
        tokens += latents.shape[0]

    if tokens == 0:
        raise ValueError("the activation stream produced no batches")
    return fired / tokens


# ----------------------------------------------------------------------
# substitution
# ----------------------------------------------------------------------

@torch.no_grad()
def substitution_accuracy(coder: SparseCoder, upper: nn.Module, lower: nn.Module,
                          loader, scale_in: float, scale_out: float,
                          max_images: int, device: torch.device | str = "cpu",
                          label_map: Tensor | None = None,
                          reference_upper: nn.Module | None = None,
                          baseline: nn.Module | None = None,
                          ) -> dict[str, float]:
    """What the model predicts when the coder's output replaces the activation.

    The upper half runs normally, its activation is scaled the way the coder was
    trained to expect, the coder reconstructs it, the reconstruction is scaled
    back, and the lower half carries on from there.

    Args:
        coder: the trained coder.
        upper: the part of the model before the cut, producing the activation.
        lower: the part after the cut, consuming it.
        loader: evaluation images.
        scale_in: the scalar the coder's input layer was divided by.
        scale_out: the scalar its target layer was divided by. The same number
            as ``scale_in`` when the coder reproduces its own input.
        max_images: stop after this many.
        device: where to run.
        label_map: translation from the loader's class numbering into the
            model's, from ``stitch.labels.label_map_for``. None when the two
            already agree, which is the case when the loader sees every class
            the model was trained on. Only the two accuracies depend on it;
            agreement compares two predictions and never reads a label.
        reference_upper: the upper half of the model ``lower`` belongs to, used
            to produce the unmodified prediction. Required whenever ``upper``
            and ``lower`` come from different models: there, feeding ``upper``'s
            activation straight into ``lower`` is not a model at all, it is one
            network's activation handed to another's layers, and it predicts at
            chance. Left as None when both halves come from one model, where
            the activation already in hand is the right one and a second
            forward pass would buy nothing.
        baseline: the reduced-rank map as a module, from
            ``stitch.sweep.LinearStitch``, substituted at the same cut point so
            its effect on the prediction can be read beside the coder's.
            Without it, agreement is a number with no scale: there is no way to
            tell whether a coder preserving 0.95 of the model's predictions is
            the dictionary doing work or what any map of that cost reaches at
            that depth. Run inside the same loop over the same images, so it
            costs one more pass through the lower half and no extra pass over
            the data.

            It reads the activation directly, in the units the activation
            arrives in, because the sweep that fitted it never scaled anything.
            The coder is the one working in scaled units, and the scaling is
            undone on its output alone.

    Returns:
        Top-1 accuracy of the unmodified model, top-1 with the reconstruction
        substituted, the gap between them, and the share of images where both
        predict the same class. Agreement is the stricter of the two: a coder
        can preserve accuracy while changing which images are right.

        With a baseline given, the same three readings for it under
        ``baseline_`` names, matching the keys its explained variance is
        already recorded under. Absent without one, rather than present and
        zero, so a run that had no baseline cannot be mistaken for one whose
        baseline destroyed the prediction.

        The unmodified model is the one the coder reconstructs into, cut at the
        same depth and rejoined with nothing in between, so the only difference
        between the two is the coder itself.
    """
    device = torch.device(device)
    coder = coder.to(device).eval()
    upper = upper.to(device).eval()
    lower = lower.to(device).eval()
    if reference_upper is not None:
        reference_upper = reference_upper.to(device).eval()
    if baseline is not None:
        baseline = baseline.to(device).eval()

    if label_map is not None:
        label_map = label_map.to(device)

    correct_plain = correct_substituted = agreed = seen = 0
    correct_baseline = agreed_baseline = 0

    for images, labels in loader:
        if seen >= max_images:
            break
        images = images.to(device)
        labels = labels.to(device)

        if label_map is not None:
            labels = label_map[labels]

        activation = upper(images)
        plain = lower(activation if reference_upper is None else reference_upper(images))

        raw = activation.reshape(-1, activation.shape[-1])
        reconstruction, _ = coder(raw * scale_in)
        restored = (reconstruction / scale_out).reshape(activation.shape)
        substituted = lower(restored)

        plain_class = plain.argmax(dim=-1)
        substituted_class = substituted.argmax(dim=-1)

        correct_plain += int((plain_class == labels).sum())
        correct_substituted += int((substituted_class == labels).sum())
        agreed += int((plain_class == substituted_class).sum())
        seen += images.shape[0]

        if baseline is not None:
            # Already fitted in the activation's own units, so no scale enters
            # on either side. The second argument it returns is an empty latent
            # tensor it carries only to match a coder's signature.
            predicted, _ = baseline(raw)
            baseline_class = lower(predicted.reshape(activation.shape)).argmax(dim=-1)
            correct_baseline += int((baseline_class == labels).sum())
            agreed_baseline += int((baseline_class == plain_class).sum())

    if seen == 0:
        raise ValueError("the loader produced no images")

    plain_top1 = correct_plain / seen
    substituted_top1 = correct_substituted / seen

    result = {
        "plain_top1": plain_top1,
        "substituted_top1": substituted_top1,
        "top1_drop": plain_top1 - substituted_top1,
        "agreement": agreed / seen,
        "images": float(seen),
    }

    if baseline is not None:
        baseline_top1 = correct_baseline / seen
        result.update({
            "baseline_substituted_top1": baseline_top1,
            "baseline_top1_drop": plain_top1 - baseline_top1,
            "baseline_agreement": agreed_baseline / seen,
        })

    return result


# ----------------------------------------------------------------------
# validity gates
# ----------------------------------------------------------------------

def check_gates(metrics: dict[str, float], cfg, width: int) -> dict[str, bool]:
    """Whether a coder is in a state worth comparing with anything.

    These are not a measure of quality. Each one catches a way a run can finish
    looking finished while meaning nothing:

    ``finite``
        a loss that went to infinity leaves plausible looking weights.
    ``reconstructs``
        a coder that reconstructs nothing still has an error to report. Judged
        against what a closed-form linear stitch reaches at the same depth,
        because that is what is reachable there at all, and not against a fixed
        number, which would ask more of a depth where little transfers than of
        one where everything does. Both sides are the explained variance
        defined at the top of this module, which is what makes the comparison
        mean anything: the coder's measured here, the linear fit's measured by
        the sweep and carried in on the configuration.
    ``not_collapsed``
        with every feature off, the reconstruction is a constant and the
        sparsity is perfect.
    ``not_dense``
        with enough features on at once, any target is reachable, the
        reconstruction is good and nothing has been disentangled. Judged as a
        fraction of the width of the activation being reconstructed, which is
        what the code has to describe, and not of the dictionary, which would
        move the threshold with the expansion factor and say nothing about how
        compactly the token was described.
    ``features_alive``
        a coder with most of its dictionary silent is a smaller coder than its
        configuration says, so comparing it against one with none silent
        compares two dictionary sizes. Silent and not dead: what this reads is
        the share that fired on none of the evaluation tokens, which holds
        together the features that never fired in training and the ones too
        rare for the window to catch. Separating those needs the training
        tracker, and for this gate's purpose they count the same, since a
        feature the evaluation never sees contributes nothing to what is
        measured there either way.
    ``no_always_on_feature``
        a feature active on nearly every token carries no information about any
        particular token.

    Args:
        metrics: what ``reconstruction_metrics`` produced.
        cfg: the run configuration, holding the thresholds.
        width: the width of the activation being reconstructed, ``coder.d_out``.

    Returns:
        One entry per gate, plus ``passed``, true only when all of them are.
    """
    gates = {
        "finite": all(math.isfinite(value) for value in metrics.values()),
        "reconstructs": (metrics["eval_explained_variance"]
                         >= cfg.eval.fit_fraction * cfg.eval.linear_fit),
        "not_collapsed": metrics["eval_l0"] >= cfg.eval.min_l0,
        "not_dense": metrics["eval_l0"] <= cfg.eval.max_l0_frac * width,
        "features_alive": (metrics["eval_dead_or_dormant"]
                           <= cfg.eval.max_allowed_silent),
        "no_always_on_feature":
            metrics["eval_max_feature_frequency"] <= cfg.eval.max_feature_frequency,
    }
    gates["passed"] = all(gates.values())

    failed = [name for name, ok in gates.items() if not ok and name != "passed"]
    if failed:
        log.warning("validity gates failed: %s", ", ".join(failed))
    return gates


def evaluate(coder: SparseCoder, batches: Iterator[dict[str, Tensor]],
             source_key: str, target_key: str, cfg,
             device: torch.device | str = "cpu") -> dict[str, float]:
    """Reconstruction metrics and the gates, in the form the logger wants.

    Shaped to be passed to the training loop as its evaluation hook, so the
    gates are visible during a run rather than only at the end of one. Booleans
    become one and zero, since a metric store holds numbers.

    The linear fit the coder was judged against is recorded alongside the
    metrics, with the ratio between the two, so a result carries the comparison
    rather than leaving it in a configuration file read separately. A ratio
    above one is a coder that beat the linear map.
    """
    metrics = reconstruction_metrics(
        coder, batches, source_key, target_key, cfg.eval.batches, device)
    metrics["eval_linear_fit"] = cfg.eval.linear_fit
    metrics["eval_fit_ratio"] = metrics["eval_explained_variance"] / cfg.eval.linear_fit
    gates = check_gates(metrics, cfg, coder.d_out)
    metrics.update({f"gate_{name}": float(value) for name, value in gates.items()})
    return metrics