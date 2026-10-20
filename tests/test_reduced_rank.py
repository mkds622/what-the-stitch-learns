"""Tests for the matched-rank linear baseline.

Reduced-rank regression has a closed form, so these check it against answers
known by construction: data that is exactly rank r must be reproduced at rank r
and not below it, and a map from an activation to itself must come out as that
activation's principal components. Small widths, CPU, seconds.

    python -m pytest tests/test_reduced_rank.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")

from stitch.reduced_rank import (                                           # noqa: E402
    baseline_for, baseline_from_moments, explained_at_rank, fit_reduced_rank,
    rank_k_map, self_pair_moments,
)
from stitch.sweep import CrossMoments, LinearStitch, fit_linear        # noqa: E402

WIDTH = 6
TOKENS = 4000
A = "A@add_14"
B = "B@add_14"


def moments_from(x: torch.Tensor, y: torch.Tensor, source=A, target=B) -> CrossMoments:
    moments = CrossMoments(pairs=[(source, target)],
                           widths={source: x.shape[-1], target: y.shape[-1]})
    moments.update({source: x, target: y})
    return moments


def self_moments(x: torch.Tensor, key=A) -> CrossMoments:
    """A layer against itself, which is what makes the fit PCA."""
    moments = CrossMoments(pairs=[(key, key)], widths={key: x.shape[-1]})
    moments.update({key: x})
    return moments


def reduced_rank_data(tokens: int, rank: int, width: int = WIDTH, seed: int = 0):
    """Source tokens, and a target that is exactly a rank-r function of them."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(tokens, width, generator=generator)
    left = torch.randn(width, rank, generator=generator)
    right = torch.randn(rank, width, generator=generator)
    return x, x @ left @ right


# ----------------------------------------------------------------------
# the rank is the budget
# ----------------------------------------------------------------------

def test_an_exactly_rank_r_target_is_recovered_at_rank_r():
    """The premise: a rank limit that matches the structure costs nothing."""
    x, y = reduced_rank_data(TOKENS, rank=2)
    fit = fit_reduced_rank(moments_from(x, y), A, B, ridge=1e-9)

    assert explained_at_rank(fit, 2) == pytest.approx(1.0, abs=1e-4)


def test_one_rank_short_cannot_recover_it():
    """Otherwise the rank is not doing anything and the baseline is not a budget.

    How much the missing direction is worth depends on the data, so the test is
    that it is worth something and that the full rank is exact, not that the
    shortfall reaches any particular size.
    """
    x, y = reduced_rank_data(TOKENS, rank=3)
    fit = fit_reduced_rank(moments_from(x, y), A, B, ridge=1e-9)

    short, exact = explained_at_rank(fit, 2), explained_at_rank(fit, 3)
    assert exact == pytest.approx(1.0, abs=1e-4)
    assert short < 0.999
    assert exact - short > 1e-3


def test_more_rank_never_explains_less():
    x, y = reduced_rank_data(TOKENS, rank=4)
    fit = fit_reduced_rank(moments_from(x, y), A, B, ridge=1e-6)

    scores = [explained_at_rank(fit, k) for k in range(1, WIDTH + 1)]
    assert scores == sorted(scores)


def test_full_rank_matches_the_unconstrained_fit():
    """At the target's width the rank constraint binds on nothing."""
    torch.manual_seed(1)
    x = torch.randn(TOKENS, WIDTH)
    y = x @ torch.randn(WIDTH, WIDTH) + torch.randn(TOKENS, WIDTH)
    moments = moments_from(x, y)

    _, _, unconstrained = fit_linear(moments, A, B, ridge=1e-6)
    fit = fit_reduced_rank(moments, A, B, ridge=1e-6)

    assert explained_at_rank(fit, WIDTH) == pytest.approx(unconstrained, abs=1e-4)


def test_a_rank_outside_the_range_raises():
    x, y = reduced_rank_data(TOKENS, rank=2)
    fit = fit_reduced_rank(moments_from(x, y), A, B, ridge=1e-6)

    for bad in (0, -1, WIDTH + 1):
        with pytest.raises(ValueError, match="rank must be between"):
            rank_k_map(fit, bad)


def test_fitting_nothing_raises():
    moments = CrossMoments(pairs=[(A, B)], widths={A: WIDTH, B: WIDTH})
    with pytest.raises(ValueError, match="no tokens"):
        fit_reduced_rank(moments, A, B, ridge=1e-6)


# ----------------------------------------------------------------------
# a layer against itself is PCA
#
# This is what lets one estimator serve both objectives, so the ratio a coder
# is reported against means the same thing in each.
# ----------------------------------------------------------------------

def test_a_layer_against_itself_gives_its_principal_components():
    torch.manual_seed(2)
    x = torch.randn(TOKENS, 3) @ torch.randn(3, WIDTH) + 0.1 * torch.randn(TOKENS, WIDTH)

    fit = fit_reduced_rank(self_moments(x), A, A, ridge=1e-9)
    weight, _ = rank_k_map(fit, 3)

    centred = (x - x.mean(dim=0)).to(torch.float64)
    covariance = centred.T @ centred
    values, vectors = torch.linalg.eigh(covariance)
    top = vectors[:, torch.argsort(values, descending=True)[:3]]
    projector = (top @ top.T).to(torch.float32)

    assert torch.allclose(weight, projector, atol=1e-3)


def test_pca_at_the_data_rank_loses_nothing():
    """Three directions of signal, so three components reconstruct it."""
    torch.manual_seed(3)
    x = torch.randn(TOKENS, 3) @ torch.randn(3, WIDTH)

    fit = fit_reduced_rank(self_moments(x), A, A, ridge=1e-9)
    assert explained_at_rank(fit, 3) == pytest.approx(1.0, abs=1e-4)


def test_the_projector_is_idempotent():
    """A projection applied twice is the same projection, which a rank-k map
    from a layer to itself has to be."""
    torch.manual_seed(4)
    x = torch.randn(TOKENS, WIDTH)

    fit = fit_reduced_rank(self_moments(x), A, A, ridge=1e-9)
    weight, _ = rank_k_map(fit, 2)

    assert torch.allclose(weight @ weight, weight, atol=1e-3)


# ----------------------------------------------------------------------
# scored on tokens it was not fitted to
# ----------------------------------------------------------------------

def test_a_real_relation_survives_unseen_tokens():
    generator = torch.Generator().manual_seed(5)
    left = torch.randn(WIDTH, 2, generator=generator)
    right = torch.randn(2, WIDTH, generator=generator)

    fit_x = torch.randn(TOKENS, WIDTH, generator=generator)
    fit = fit_reduced_rank(moments_from(fit_x, fit_x @ left @ right), A, B, ridge=1e-9)

    unseen = torch.randn(TOKENS, WIDTH, generator=generator)
    held_out = moments_from(unseen, unseen @ left @ right)

    assert explained_at_rank(fit, 2, held_out, A, B) == pytest.approx(1.0, abs=1e-3)


def test_a_map_that_does_not_generalise_scores_below_zero():
    generator = torch.Generator().manual_seed(6)
    fit_x = torch.randn(200, WIDTH, generator=generator)
    fit_y = torch.randn(200, WIDTH, generator=generator)
    fit = fit_reduced_rank(moments_from(fit_x, fit_y), A, B, ridge=1e-9)

    unrelated = moments_from(torch.randn(TOKENS, WIDTH, generator=generator),
                             torch.randn(TOKENS, WIDTH, generator=generator))

    assert explained_at_rank(fit, WIDTH, unrelated, A, B) < 0.0


def test_held_out_scoring_needs_to_be_told_the_layers():
    x, y = reduced_rank_data(TOKENS, rank=2)
    fit = fit_reduced_rank(moments_from(x, y), A, B, ridge=1e-6)
    with pytest.raises(ValueError, match="source and target"):
        explained_at_rank(fit, 2, moments_from(x, y))


# ----------------------------------------------------------------------
# the map as something a model can be run through
# ----------------------------------------------------------------------

def test_the_map_runs_like_a_coder():
    """So the baseline goes through the same substitution measurement."""
    x, y = reduced_rank_data(TOKENS, rank=2)
    fit = fit_reduced_rank(moments_from(x, y), A, B, ridge=1e-9)
    weight, bias = rank_k_map(fit, 2)

    output, latents = LinearStitch(weight, bias)(x[:16])
    assert output.shape == (16, WIDTH)
    assert latents.shape == (16, 0)
    assert torch.allclose(output, y[:16], atol=1e-2)


# ----------------------------------------------------------------------
# matching the rank to a coder
# ----------------------------------------------------------------------

def test_the_rank_is_the_coders_active_count():
    x, y = reduced_rank_data(TOKENS, rank=3)
    fit = fit_reduced_rank(moments_from(x, y), A, B, ridge=1e-6)

    assert baseline_for(fit, l0=3.4)["baseline_rank"] == 3.0
    assert baseline_for(fit, l0=2.5)["baseline_rank"] == 2.0


def test_a_coder_sparser_than_one_feature_still_gets_a_baseline():
    """L0 is an average, so it can fall below one without the rank doing so."""
    x, y = reduced_rank_data(TOKENS, rank=2)
    fit = fit_reduced_rank(moments_from(x, y), A, B, ridge=1e-6)

    result = baseline_for(fit, l0=0.2)
    assert result["baseline_rank"] == 1.0
    assert result["baseline_rank_clamped"] == 1.0


def test_a_coder_denser_than_the_target_is_flagged():
    """Past the target's width there is no rank left to match, so the
    comparison is no longer at equal cost and has to say so."""
    x, y = reduced_rank_data(TOKENS, rank=2)
    fit = fit_reduced_rank(moments_from(x, y), A, B, ridge=1e-6)

    result = baseline_for(fit, l0=900.0)
    assert result["baseline_rank"] == float(WIDTH)
    assert result["baseline_rank_wanted"] == 900.0
    assert result["baseline_rank_clamped"] == 1.0


def test_a_matched_rank_is_not_flagged():
    x, y = reduced_rank_data(TOKENS, rank=2)
    fit = fit_reduced_rank(moments_from(x, y), A, B, ridge=1e-6)
    assert baseline_for(fit, l0=3.0)["baseline_rank_clamped"] == 0.0


def test_the_baseline_is_scored_held_out_when_given_other_tokens():
    generator = torch.Generator().manual_seed(7)
    fit_x = torch.randn(400, WIDTH, generator=generator)
    fit_y = fit_x @ torch.randn(WIDTH, WIDTH, generator=generator) \
        + 2.0 * torch.randn(400, WIDTH, generator=generator)
    fit = fit_reduced_rank(moments_from(fit_x, fit_y), A, B, ridge=1e-9)

    unseen = torch.randn(TOKENS, WIDTH, generator=generator)
    held_out = moments_from(unseen, unseen @ torch.randn(WIDTH, WIDTH, generator=generator))

    in_sample = baseline_for(fit, l0=WIDTH)["baseline_explained_variance"]
    out_of_sample = baseline_for(fit, l0=WIDTH, held_out=held_out,
                                 source=A, target=B)["baseline_explained_variance"]

    assert out_of_sample < in_sample


# ----------------------------------------------------------------------
# a coder reproducing its own input
#
# The sweep accumulates A against B and never A against A, so the sums a case 1
# baseline needs are not in the file. They are all derivable from what is.
# ----------------------------------------------------------------------

def test_a_layer_is_arranged_as_a_pair_with_itself():
    torch.manual_seed(20)
    x = torch.randn(TOKENS, WIDTH)
    paired = self_pair_moments(moments_from(x, torch.randn(TOKENS, WIDTH)), A)

    assert paired.pairs == [(A, A)]
    assert paired.tokens == TOKENS
    assert torch.allclose(paired.sum_xy[(A, A)], (x.T @ x).to(torch.float64), atol=1e-2)
    assert torch.allclose(paired.sum_y[A], paired.sum_x[A])
    assert torch.allclose(paired.sum_y2[A], torch.diagonal(paired.sum_xx[A]))


def test_a_derived_self_pair_is_the_layer_s_own_components():
    """Which is what makes the case 1 baseline PCA, with no extra accumulation."""
    torch.manual_seed(21)
    x = torch.randn(TOKENS, 2) @ torch.randn(2, WIDTH)
    paired = self_pair_moments(moments_from(x, torch.randn(TOKENS, WIDTH)), A)

    assert explained_at_rank(fit_reduced_rank(paired, A, A, ridge=1e-9), 2) \
        == pytest.approx(1.0, abs=1e-4)


def test_a_layer_the_sums_do_not_cover_raises():
    x, y = reduced_rank_data(TOKENS, rank=2)
    with pytest.raises(ValueError, match="do not cover"):
        self_pair_moments(moments_from(x, y), "A@nowhere")


def test_the_target_side_of_a_pair_can_be_arranged_too():
    """B is only ever a target in the sweep, so its sums live elsewhere."""
    torch.manual_seed(22)
    y = torch.randn(TOKENS, WIDTH)
    paired = self_pair_moments(moments_from(torch.randn(TOKENS, WIDTH), y), B)

    assert paired.pairs == [(B, B)]
    assert torch.allclose(paired.sum_xx[B], (y.T @ y).to(torch.float64), atol=1e-2)


# ----------------------------------------------------------------------
# the one call a run makes
# ----------------------------------------------------------------------

def test_a_run_gets_its_baseline_from_the_sums():
    torch.manual_seed(23)
    x = torch.randn(TOKENS, WIDTH)
    y = x @ torch.randn(WIDTH, WIDTH)
    result = baseline_from_moments(moments_from(x, y), A, B, l0=WIDTH, ridge=1e-9)

    assert result["baseline_rank"] == float(WIDTH)
    assert result["baseline_explained_variance"] == pytest.approx(1.0, abs=1e-3)


def test_the_same_layer_on_both_sides_needs_no_cross_term():
    """The case the sweep never accumulates, which must work anyway."""
    torch.manual_seed(24)
    x = torch.randn(TOKENS, 3) @ torch.randn(3, WIDTH)
    result = baseline_from_moments(moments_from(x, torch.randn(TOKENS, WIDTH)),
                                   A, A, l0=3, ridge=1e-9)

    assert result["baseline_rank"] == 3.0
    assert result["baseline_explained_variance"] == pytest.approx(1.0, abs=1e-3)


def test_the_run_level_call_scores_held_out_too():
    generator = torch.Generator().manual_seed(25)
    fit_x = torch.randn(300, WIDTH, generator=generator)
    fit_y = torch.randn(300, WIDTH, generator=generator)
    unseen = torch.randn(TOKENS, WIDTH, generator=generator)

    result = baseline_from_moments(
        moments_from(fit_x, fit_y), A, B, l0=WIDTH, ridge=1e-9,
        held_out=moments_from(unseen, torch.randn(TOKENS, WIDTH, generator=generator)))

    assert result["baseline_explained_variance"] < 0.0


def test_every_value_is_a_number():
    """It goes into a run summary beside the coder's own metrics."""
    x, y = reduced_rank_data(TOKENS, rank=2)
    fit = fit_reduced_rank(moments_from(x, y), A, B, ridge=1e-6)
    assert all(isinstance(value, float) for value in baseline_for(fit, l0=2).values())