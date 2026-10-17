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
        the share of features that never fired, and the largest share of tokens
        any single feature fired on. A feature active on almost every token is
        not a feature, it is a bias with extra steps.
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

    return {
        "eval_mse": totals["mse"] / tokens,
        "eval_l0": totals["l0"] / tokens,
        "eval_explained_variance": explained,
        "eval_never_fired": float((frequencies == 0).float().mean()),
        "eval_max_feature_frequency": float(frequencies.max()),
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

    Returns:
        Top-1 accuracy of the unmodified model, top-1 with the reconstruction
        substituted, the gap between them, and the share of images where both
        predict the same class. Agreement is the stricter of the two: a coder
        can preserve accuracy while changing which images are right.
    """
    device = torch.device(device)
    coder = coder.to(device).eval()
    upper = upper.to(device).eval()
    lower = lower.to(device).eval()

    if label_map is not None:
        label_map = label_map.to(device)

    correct_plain = correct_substituted = agreed = seen = 0

    for images, labels in loader:
        if seen >= max_images:
            break
        images = images.to(device)
        labels = labels.to(device)

        if label_map is not None:
            labels = label_map[labels]

        activation = upper(images)
        plain = lower(activation)

        flat = activation.reshape(-1, activation.shape[-1]) * scale_in
        reconstruction, _ = coder(flat)
        restored = (reconstruction / scale_out).reshape(activation.shape)
        substituted = lower(restored)

        plain_class = plain.argmax(dim=-1)
        substituted_class = substituted.argmax(dim=-1)

        correct_plain += int((plain_class == labels).sum())
        correct_substituted += int((substituted_class == labels).sum())
        agreed += int((plain_class == substituted_class).sum())
        seen += images.shape[0]

    if seen == 0:
        raise ValueError("the loader produced no images")

    plain_top1 = correct_plain / seen
    substituted_top1 = correct_substituted / seen

    return {
        "plain_top1": plain_top1,
        "substituted_top1": substituted_top1,
        "top1_drop": plain_top1 - substituted_top1,
        "agreement": agreed / seen,
        "images": float(seen),
    }


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
        a coder that reconstructs nothing still has an error to report.
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
        a coder with most of its dictionary dead is a smaller coder than its
        configuration says, so comparing it against one with none dead compares
        two dictionary sizes.
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
        "reconstructs": metrics["eval_explained_variance"] >= cfg.eval.min_explained_variance,
        "not_collapsed": metrics["eval_l0"] >= cfg.eval.min_l0,
        "not_dense": metrics["eval_l0"] <= cfg.eval.max_l0_frac * width,
        "features_alive": metrics["eval_never_fired"] <= cfg.eval.max_never_fired,
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
    """
    metrics = reconstruction_metrics(
        coder, batches, source_key, target_key, cfg.eval.batches, device)
    gates = check_gates(metrics, cfg, coder.d_out)
    metrics.update({f"gate_{name}": float(value) for name, value in gates.items()})
    return metrics