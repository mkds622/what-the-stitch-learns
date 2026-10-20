"""Reduced-rank regression, as the baseline a trained coder has to beat.

A sparse coder spends some number of features per token. The question this
answers is whether spending the same number on a fixed set of directions does
as well. If it does, the dictionary is not buying anything: a rank-k linear
map would have described the activation just as closely at the same cost per
token, without any of the machinery.

The estimator is reduced-rank regression: the best rank-k linear map from the
layer the coder reads to the layer it reconstructs. It covers both objectives
with one piece of arithmetic, because a coder reproducing its own input is the
case where the two layers are the same, and reduced-rank regression from an
activation to itself is principal component analysis of that activation. One
estimator means the coder-over-baseline ratio means the same thing in both
cases, which is the only way the two can be compared.

    rank-k map from x to y   ==   PCA of x, when y is x

Rank is matched to the coder's measured L0, so both spend the same count of
numbers per token. The comparison is deliberately generous to the baseline: the
coder also has to record which features are active, and the rank-k map does
not, so a coder that merely ties has not justified its own index cost.

The fit never holds tokens. It works from the same accumulated sums the layer
sweep collects, so a map at any rank costs a solve rather than a pass over the
data, and one accumulation serves every rank and every run at that depth.

Fitted on one split and scored on another, for the reason set out in
``stitch.sweep``: a map with this many coefficients flatters itself on its own
fitting tokens, and the coder it is compared against is always scored on
held-out activations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import Tensor

from stitch.sweep import CrossMoments, score_linear

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# the fit
# ----------------------------------------------------------------------

@dataclass
class ReducedRankFit:
    """A full-rank solution plus the ordering that truncates it.

    Holding these rather than one map means every rank comes from one solve.
    ``basis`` is ordered by how much of the fitted variance each direction
    carries, so taking its first k columns gives the best rank-k map rather
    than an arbitrary one.

    Attributes:
        weight: the unconstrained least-squares map, ``(d_source, d_target)``.
        basis: directions in the target space, ``(d_target, d_target)``,
            best first.
        spectrum: variance carried by each direction, descending, in the units
            the moments were accumulated in.
        mean_x: mean of the source layer.
        mean_y: mean of the target layer.
        total: total variance of the target, for turning the spectrum into a
            share.
    """

    weight: Tensor
    basis: Tensor
    spectrum: Tensor
    mean_x: Tensor
    mean_y: Tensor
    total: float

    @property
    def max_rank(self) -> int:
        """The largest rank that is not the same map as the one below it."""
        return int(self.basis.shape[1])


def fit_reduced_rank(moments: CrossMoments, source: str, target: str,
                 ridge: float = 1e-3) -> ReducedRankFit:
    """Solve once, at every rank.

    Reduced-rank regression factors into two steps that can be done in either
    order. The unconstrained map is solved first, and its fitted values are
    then projected onto the directions carrying the most variance. Truncating
    that projection at k gives the best rank-k map, which is why one solve
    answers every rank.

    Args:
        moments: accumulated sums over the tokens to fit on.
        source: the layer read.
        target: the layer reconstructed. The same as ``source`` for a coder
            reproducing its own input, which makes this PCA.
        ridge: added to the diagonal as a fraction of its mean. An activation
            covariance is close to singular, and solving without damping fits
            noise in the directions that carry almost nothing.

    Returns:
        A :class:`ReducedRankFit`, from which any rank is a truncation.

    Raises:
        ValueError: if no tokens were accumulated.
    """
    if moments.tokens == 0:
        raise ValueError("no tokens were accumulated")

    n = moments.tokens
    mean_x = moments.sum_x[source] / n
    mean_y = moments.sum_y[target] / n

    sxx = moments.sum_xx[source] - n * torch.outer(mean_x, mean_x)
    sxy = moments.sum_xy[(source, target)] - n * torch.outer(mean_x, mean_y)
    total = float((moments.sum_y2[target] - n * mean_y ** 2).sum())

    damping = ridge * float(torch.diagonal(sxx).mean())
    sxx = sxx + damping * torch.eye(sxx.shape[0], dtype=sxx.dtype, device=sxx.device)

    weight = torch.linalg.solve(sxx, sxy)

    # Covariance of the fitted values. Its leading directions are the ones a
    # rank-limited map should spend its budget on, which is what makes the
    # truncation optimal rather than merely a projection.
    fitted = weight.T @ sxx @ weight
    fitted = 0.5 * (fitted + fitted.T)          # symmetric up to rounding

    spectrum, basis = torch.linalg.eigh(fitted)
    order = torch.argsort(spectrum, descending=True)

    return ReducedRankFit(
        weight=weight,
        basis=basis[:, order],
        spectrum=spectrum[order].clamp_min(0.0),
        mean_x=mean_x,
        mean_y=mean_y,
        total=total,
    )


def rank_k_map(fit: ReducedRankFit, k: int) -> tuple[Tensor, Tensor]:
    """The best rank-k map, as a weight and a bias.

    Shaped so it can be scored by :func:`stitch.sweep.score_linear` and run by
    :class:`stitch.sweep.LinearStitch`, which is what lets the baseline be put
    through the same substitution measurement as a trained coder.

    Args:
        fit: from :func:`fit_reduced_rank`.
        k: how many directions to keep. Matched to the coder's measured L0.

    Returns:
        ``(weight, bias)`` in single precision, so the map is applied in the
        same precision the coder is.

    Raises:
        ValueError: if k is outside 1 to the target width.
    """
    if not 1 <= k <= fit.max_rank:
        raise ValueError(f"rank must be between 1 and {fit.max_rank}, got {k}")

    kept = fit.basis[:, :k]
    weight = fit.weight @ kept @ kept.T
    bias = fit.mean_y - fit.mean_x @ weight

    return weight.to(torch.float32), bias.to(torch.float32)


def explained_at_rank(fit: ReducedRankFit, k: int,
                      held_out: CrossMoments | None = None,
                      source: str | None = None,
                      target: str | None = None) -> float:
    """How much of the target a rank-k map accounts for.

    Args:
        fit: from :func:`fit_reduced_rank`.
        k: the rank.
        held_out: sums over tokens the map was not fitted to. Given, the score
            is measured on those, which is what makes it comparable with a
            coder's. Omitted, the score is in-sample and read straight off the
            spectrum, which is exact and costs nothing.
        source: the layer read. Required with ``held_out``.
        target: the layer reconstructed. Required with ``held_out``.

    Returns:
        Explained variance, as defined at the top of ``stitch.evaluate``. The
        held-out value can be negative where a fit does not generalise.
    """
    if held_out is None:
        if fit.total <= 0:
            return 0.0
        if not 1 <= k <= fit.max_rank:
            raise ValueError(f"rank must be between 1 and {fit.max_rank}, got {k}")
        return float(fit.spectrum[:k].sum()) / fit.total

    if source is None or target is None:
        raise ValueError("scoring on held-out tokens needs the source and target layers")

    weight, bias = rank_k_map(fit, k)
    return score_linear(held_out, source, target, weight, bias)


def baseline_for(fit: ReducedRankFit, l0: float, held_out: CrossMoments | None = None,
                 source: str | None = None, target: str | None = None) -> dict[str, float]:
    """The baseline a coder with this L0 has to beat, and the rank it was taken at.

    Args:
        fit: from :func:`fit_reduced_rank`.
        l0: the coder's measured active features per token. Rounded, and held
            inside the available range, since a coder can be sparser than one
            direction or denser than the target is wide and neither is a rank.
        held_out: sums over other tokens, as in :func:`explained_at_rank`.
        source: the layer read.
        target: the layer reconstructed.

    Returns:
        The rank used, the explained variance there, and whether the rank had
        to be clamped, which says the comparison is no longer at matched cost.
    """
    wanted = int(round(l0))
    rank = max(1, min(wanted, fit.max_rank))

    return {
        "baseline_rank": float(rank),
        "baseline_rank_wanted": float(wanted),
        "baseline_rank_clamped": float(rank != wanted),
        "baseline_explained_variance": explained_at_rank(fit, rank, held_out, source, target),
    }