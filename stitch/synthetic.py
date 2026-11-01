"""Activations with the right second-order statistics and nothing else.

The control this exists for. A trained dictionary that reaches a good sparsity
and a good reconstruction has not thereby shown it found features: a Gaussian
with the same mean and covariance has no features in it at all, being a
correlated blob and nothing more, and if a coder does equally well on that then
whatever it found on the real activations was already described by their second
moments. What the dictionary is worth is the gap between the two, not the
number it reaches on real data.

The generated stream satisfies the same interface the real one does, so the
training loop, the evaluation and the gates are unchanged and the only
difference between a control run and the run it controls is where the tokens
came from.

    real activations      ->  coder  ->  L0, explained variance, dictionary
    matched Gaussian      ->  coder  ->  the same numbers, with nothing to find

Matched on the joint distribution, not on each layer separately. A coder
reading one layer and reconstructing another is being asked to exploit the
relationship between them, so a control whose two halves were independent would
be a control for the wrong thing.

The statistics come from the sums the layer sweep already accumulated, so no
pass through the backbones is needed and the control is matched to exactly the
tokens the sweep saw.
"""

from __future__ import annotations

import logging
import math
from typing import Iterator

import torch
from torch import Tensor

from stitch.activations import ActivationSource, TokenBatch
from stitch.sweep import CrossMoments

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# the statistics being matched
# ----------------------------------------------------------------------

def joint_statistics(moments: CrossMoments, keys: list[str]) -> tuple[Tensor, Tensor]:
    """Mean and covariance of one or two layers, stacked into one vector.

    One key describes a coder reproducing its own input, where there is a
    single activation and the covariance is its own. Two describe a coder
    reading one layer and reconstructing another, where the off-diagonal blocks
    are the relationship the coder is being asked to learn, and dropping them
    would control for the wrong thing.

    Args:
        moments: accumulated sums, from the layer sweep.
        keys: one or two layer keys, in the order the stream presents them.

    Returns:
        ``(mean, covariance)`` over the concatenation of the layers, in double
        precision.

    Raises:
        ValueError: on no tokens, on the wrong number of keys, or on a layer
            the moments do not cover.
    """
    if moments.tokens == 0:
        raise ValueError("no tokens were accumulated")
    if not 1 <= len(keys) <= 2:
        raise ValueError(f"expected one or two layers, got {keys}")

    n = moments.tokens

    def first_moment(key: str) -> Tensor:
        if key in moments.sum_x:
            return moments.sum_x[key] / n
        if key in moments.sum_y:
            return moments.sum_y[key] / n
        raise ValueError(f"the moments do not cover layer {key!r}")

    def second_moment(key: str) -> Tensor:
        if key in moments.sum_xx:
            return moments.sum_xx[key]
        if key in moments.sum_yy:
            return moments.sum_yy[key]
        raise ValueError(f"the moments hold no covariance for layer {key!r}")

    means = [first_moment(key) for key in keys]
    mean = torch.cat(means)

    if len(keys) == 1:
        centred = second_moment(keys[0]) / n - torch.outer(means[0], means[0])
        return mean, 0.5 * (centred + centred.T)

    source, target = keys
    cross = moments.sum_xy.get((source, target))
    if cross is None:
        raise ValueError(
            f"the moments hold no cross term for {source} -> {target}, so the "
            f"relationship between the two layers cannot be matched")

    sxx = second_moment(source) / n - torch.outer(means[0], means[0])
    syy = second_moment(target) / n - torch.outer(means[1], means[1])
    sxy = cross / n - torch.outer(means[0], means[1])

    covariance = torch.cat([torch.cat([sxx, sxy], dim=1),
                            torch.cat([sxy.T, syy], dim=1)], dim=0)
    return mean, 0.5 * (covariance + covariance.T)


def cholesky_factor(covariance: Tensor, jitter: float = 1e-8) -> Tensor:
    """A factor ``L`` with ``L @ L.T`` equal to the covariance.

    An activation covariance is close to singular, since a few directions carry
    most of the variance and the rest almost none, and a decomposition of it
    fails rather than returning something slightly wrong. The diagonal is
    therefore lifted by a fraction of its own mean until it succeeds, which
    adds isotropic variance far below the directions that carry anything.

    Raises:
        ValueError: if it still fails after the jitter has grown by six orders
            of magnitude, which means the matrix is not a covariance.
    """
    scale = float(torch.diagonal(covariance).mean()).__abs__()
    eye = torch.eye(covariance.shape[0], dtype=covariance.dtype,
                    device=covariance.device)

    for power in range(7):
        damped = covariance + (jitter * 10 ** power) * scale * eye
        try:
            return torch.linalg.cholesky(damped)
        except RuntimeError:
            continue

    raise ValueError("the covariance could not be factorised even with jitter")


# ----------------------------------------------------------------------
# the stream
# ----------------------------------------------------------------------

class GaussianActivations(ActivationSource):
    """Tokens drawn from a Gaussian matched to real activations.

    Endless, like the real stream, and in the same units: scales are measured
    from the generated tokens by the rule the real source uses, so a coefficient
    means the same amount of pressure here as it does on real data. They are
    measured rather than copied because the two streams are matched in
    distribution, so the rule returns the same number either way, and measuring
    keeps a control run self-contained.

    Tokens are independent by construction. The real stream shuffles because
    tokens from one image are correlated; here there is nothing to decorrelate.
    """

    def __init__(self, cfg, moments: CrossMoments, keys: list[str]):
        """
        Args:
            cfg: the run configuration, for the batch size, the seed, the
                device and the scaling rule.
            moments: accumulated sums covering the layers in ``keys``.
            keys: the layer keys the stream yields, in the order
                ``resolve_layers`` returns them.

        Raises:
            ValueError: on an unknown scaling mode, or on moments that do not
                cover the layers asked for.
        """
        self.cfg = cfg
        self.keys = list(keys)
        self.device = torch.device(cfg.train.device)
        self.token_batch = cfg.train.token_batch

        self.scaling = cfg.activations.scaling
        if self.scaling not in ("unit_norm", "none"):
            raise ValueError(
                f"unknown activations.scaling {self.scaling!r}, "
                f"expected 'unit_norm' or 'none'")
        self.scale_tokens = cfg.activations.scale_tokens

        mean, covariance = joint_statistics(moments, self.keys)
        self.mean = mean.to(self.device)
        self.factor = cholesky_factor(covariance).to(self.device)

        self._widths = {key: int(moments.widths[key]) for key in self.keys}
        self._offsets: list[tuple[str, int, int]] = []
        start = 0
        for key in self.keys:
            width = self._widths[key]
            self._offsets.append((key, start, start + width))
            start += width

        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(cfg.train.seed)
        self._scales: dict[str, float] | None = None

        log.info("gaussian activations for %s, matched over %d tokens",
                 ", ".join(self.keys), moments.tokens)

    # ------------------------------------------------------------------

    def _draw(self, count: int) -> TokenBatch:
        """One block of tokens, unscaled."""
        noise = torch.randn(count, self.mean.shape[0], generator=self.generator,
                            dtype=self.mean.dtype, device=self.device)
        tokens = self.mean + noise @ self.factor.T
        return {key: tokens[:, start:stop].to(torch.float32)
                for key, start, stop in self._offsets}

    def _measure_scales(self) -> None:
        """The same rule the real stream uses, on generated tokens."""
        if self._scales is not None:
            return

        if self.scaling == "none":
            self._scales = {key: 1.0 for key in self.keys}
            return

        totals: dict[str, float] = {key: 0.0 for key in self.keys}
        seen = 0
        while seen < self.scale_tokens:
            count = min(self.token_batch, self.scale_tokens - seen)
            block = self._draw(count)
            for key, value in block.items():
                totals[key] += float(value.norm(dim=-1).sum())
            seen += count

        self._scales = {}
        for key, total in totals.items():
            mean_norm = total / seen
            self._scales[key] = (math.sqrt(self._widths[key]) / mean_norm
                                 if mean_norm > 0 else 1.0)

        log.info("gaussian activation scales: %s",
                 {key: round(value, 4) for key, value in self._scales.items()})

    def batches(self) -> Iterator[TokenBatch]:
        """Scaled token batches, forever."""
        self._measure_scales()
        while True:
            block = self._draw(self.token_batch)
            for key, value in block.items():
                value.mul_(self._scales[key])
            yield block

    def widths(self) -> dict[str, int]:
        """Taken from the moments, so a coder is built at the real width."""
        return dict(self._widths)

    def scales(self) -> dict[str, float]:
        self._measure_scales()
        return dict(self._scales)

    def centres(self) -> dict[str, torch.Tensor]:
        """Where this stream sits, sliced per layer and scaled.

        The real stream reports a geometric median, because a residual stream
        carries outlier tokens that pull an arithmetic mean off centre. A
        Gaussian carries none. It is symmetric about its location parameter, so
        its geometric median and its arithmetic mean are the same point, and
        that point is already held here as the vector the stream generates
        from. Running Weiszfeld over generated tokens would hand it back with
        sampling noise added.

        That location is the real activations' arithmetic mean, since that is
        what this stream was matched to. So a control run is centred on a
        slightly different vector from the run it controls. That is correct
        rather than an oversight: each stream is centred on its own
        distribution, and a control centred where its own tokens are not would
        be measuring the centring instead of the features.
        """
        self._measure_scales()
        return {key: (self.mean[start:stop].to(torch.float32).cpu()
                      * self._scales[key])
                for key, start, stop in self._offsets}

    def set_scales(self, scales: dict[str, float]) -> None:
        self._scales = dict(scales)