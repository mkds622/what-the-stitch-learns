"""Tests for the layer pair sweep.

The fit has a closed form, so most of these check it against an answer known in
advance: a target built as an exact linear function of the source must come back
with that function and a perfect score, and an unrelated target must come back
with a score near zero. Small widths, CPU, no GPU anywhere.

The end to end test needs the dataset, the graph library and two models, and is
skipped without them.

    python -m pytest tests/test_sweep.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")

from stitch.sweep import (                                              # noqa: E402
    CrossMoments, LinearStitch, accumulate, candidate_pairs, fit_linear,
    rank, read_ranking, top_pairs, write_ranking,
)

WIDTH = 6
TOKENS = 512
A = "A@add_14"
B = "B@add_14"


def moments_from(x: torch.Tensor, y: torch.Tensor, source=A, target=B) -> CrossMoments:
    """Fold two tensors in as one batch, which is all the sums need."""
    moments = CrossMoments(pairs=[(source, target)],
                           widths={source: x.shape[-1], target: y.shape[-1]})
    moments.update({source: x, target: y})
    return moments


# ----------------------------------------------------------------------
# which pairs
# ----------------------------------------------------------------------

def test_matched_depths_pair_block_with_block():
    pairs = candidate_pairs(["a0", "a1", "a2"], ["b0", "b1", "b2"], True)
    assert pairs == [("a0", "b0"), ("a1", "b1"), ("a2", "b2")]


def test_every_pair_is_the_square():
    pairs = candidate_pairs(["a0", "a1"], ["b0", "b1", "b2"], False)
    assert len(pairs) == 6


def test_matched_depths_need_equal_counts():
    """Otherwise 'matched' has no meaning and a silent misalignment follows."""
    with pytest.raises(ValueError, match="same number"):
        candidate_pairs(["a0", "a1"], ["b0"], True)


def test_unmatched_counts_are_fine_for_every_pair():
    assert len(candidate_pairs(["a0"], ["b0", "b1"], False)) == 2


# ----------------------------------------------------------------------
# the fit
# ----------------------------------------------------------------------

def test_an_exactly_linear_target_is_recovered():
    """The whole premise: if a linear map exists, the closed form finds it."""
    torch.manual_seed(0)
    x = torch.randn(TOKENS, WIDTH)
    true_weight = torch.randn(WIDTH, WIDTH)
    true_bias = torch.randn(WIDTH)
    y = x @ true_weight + true_bias

    weight, bias, score = fit_linear(moments_from(x, y), A, B, ridge=1e-9)

    assert torch.allclose(weight, true_weight, atol=1e-3)
    assert torch.allclose(bias, true_bias, atol=1e-3)
    assert score == pytest.approx(1.0, abs=1e-4)


def test_an_unrelated_target_scores_near_zero():
    """Which is the signal that no coder could succeed at that depth either."""
    torch.manual_seed(1)
    x = torch.randn(20_000, 4)
    y = torch.randn(20_000, 4)

    _, _, score = fit_linear(moments_from(x, y), A, B, ridge=1e-6)
    assert score < 0.05


def test_a_partly_linear_target_scores_in_between():
    torch.manual_seed(2)
    x = torch.randn(20_000, WIDTH)
    weight = torch.randn(WIDTH, WIDTH)
    y = x @ weight + torch.randn(20_000, WIDTH) * 3.0

    _, _, score = fit_linear(moments_from(x, y), A, B, ridge=1e-6)
    assert 0.05 < score < 0.95


def test_the_bias_is_found_when_the_target_is_offset():
    """Without centring, a constant offset would show up as unexplained variance."""
    torch.manual_seed(3)
    x = torch.randn(TOKENS, WIDTH)
    y = x + 17.0

    weight, bias, score = fit_linear(moments_from(x, y), A, B, ridge=1e-9)
    assert torch.allclose(bias, torch.full((WIDTH,), 17.0), atol=1e-3)
    assert score == pytest.approx(1.0, abs=1e-4)


def test_the_score_ignores_scale():
    """It is a ratio of variances, which is why activations are not scaled here."""
    torch.manual_seed(4)
    x = torch.randn(TOKENS, WIDTH)
    y = x @ torch.randn(WIDTH, WIDTH) + torch.randn(TOKENS, WIDTH)

    _, _, plain = fit_linear(moments_from(x, y), A, B, ridge=1e-6)
    _, _, scaled = fit_linear(moments_from(x * 50.0, y * 0.02), A, B, ridge=1e-6)
    assert scaled == pytest.approx(plain, abs=1e-3)


def test_widths_may_differ():
    torch.manual_seed(5)
    x = torch.randn(TOKENS, 6)
    y = torch.randn(TOKENS, 10)

    weight, bias, _ = fit_linear(moments_from(x, y), A, B, ridge=1e-6)
    assert weight.shape == (6, 10)
    assert bias.shape == (10,)


def test_a_constant_target_scores_zero_rather_than_dividing_by_zero():
    x = torch.randn(TOKENS, WIDTH)
    y = torch.ones(TOKENS, WIDTH)
    _, _, score = fit_linear(moments_from(x, y), A, B, ridge=1e-6)
    assert score == 0.0


def test_fitting_nothing_raises():
    moments = CrossMoments(pairs=[(A, B)], widths={A: WIDTH, B: WIDTH})
    with pytest.raises(ValueError, match="no tokens"):
        fit_linear(moments, A, B, ridge=1e-6)


# ----------------------------------------------------------------------
# accumulating in pieces
# ----------------------------------------------------------------------

def test_many_batches_give_the_same_answer_as_one():
    """The sums are the whole point: the fit must not depend on batching."""
    torch.manual_seed(6)
    x = torch.randn(600, WIDTH)
    y = x @ torch.randn(WIDTH, WIDTH) + torch.randn(600, WIDTH)

    whole = fit_linear(moments_from(x, y), A, B, ridge=1e-6)

    pieces = CrossMoments(pairs=[(A, B)], widths={A: WIDTH, B: WIDTH})
    for start in range(0, 600, 37):
        pieces.update({A: x[start:start + 37], B: y[start:start + 37]})
    split = fit_linear(pieces, A, B, ridge=1e-6)

    assert pieces.tokens == 600
    assert torch.allclose(whole[0], split[0], atol=1e-4)
    assert whole[2] == pytest.approx(split[2], abs=1e-6)


def test_accumulate_stops_at_the_token_budget():
    torch.manual_seed(7)
    loader = [(torch.randn(4, 3), None) for _ in range(100)]

    def extract(images):
        return {A: images.unsqueeze(1).repeat(1, 5, 1), B: images.unsqueeze(1).repeat(1, 5, 1)}

    moments = accumulate(extract, loader, [(A, B)], {A: 3, B: 3}, fit_tokens=50)
    assert 50 <= moments.tokens < 50 + 20


def test_accumulate_on_an_empty_loader_raises():
    with pytest.raises(ValueError, match="no images"):
        accumulate(lambda images: {}, [], [(A, B)], {A: 3, B: 3}, fit_tokens=10)


# ----------------------------------------------------------------------
# ranking
# ----------------------------------------------------------------------

def three_pairs():
    """Three sources, the first unrelated to its target, the last identical."""
    torch.manual_seed(8)
    x = torch.randn(4000, WIDTH)
    pairs = [("A@one", "B@one"), ("A@two", "B@two"), ("A@three", "B@three")]
    widths = {name: WIDTH for pair in pairs for name in pair}

    moments = CrossMoments(pairs=pairs, widths=widths)
    moments.update({
        "A@one": x, "B@one": torch.randn(4000, WIDTH),
        "A@two": x, "B@two": x @ torch.randn(WIDTH, WIDTH) + torch.randn(4000, WIDTH) * 2,
        "A@three": x, "B@three": x.clone(),
    })
    return moments


def test_the_ranking_is_best_first():
    records = rank(three_pairs(), ridge=1e-6)
    scores = [record["explained_variance"] for record in records]
    assert scores == sorted(scores, reverse=True)
    assert records[0]["source"] == "A@three"
    assert records[-1]["source"] == "A@one"


def test_every_pair_appears_once():
    records = rank(three_pairs(), ridge=1e-6)
    assert len(records) == 3
    assert len({record["source"] for record in records}) == 3


def test_rank_positions_start_at_zero():
    records = rank(three_pairs(), ridge=1e-6)
    assert [record["rank"] for record in records] == [0, 1, 2]


# ----------------------------------------------------------------------
# the file the training grid reads
# ----------------------------------------------------------------------

def test_the_ranking_survives_a_round_trip(tmp_path):
    records = rank(three_pairs(), ridge=1e-6)
    path = tmp_path / "sweeps" / "layer_sweep.json"

    write_ranking(records, path, context={"tokens": 4000})
    assert read_ranking(path) == records


def test_the_context_is_kept(tmp_path):
    """A ranking is only meaningful alongside what produced it."""
    path = tmp_path / "ranking.json"
    write_ranking(rank(three_pairs(), ridge=1e-6), path,
                  context={"models": {"A": "vit_base_patch16_224"}, "tokens": 4000})

    stored = json.loads(path.read_text())
    assert stored["context"]["tokens"] == 4000
    assert stored["context"]["models"]["A"] == "vit_base_patch16_224"


def test_the_directory_is_created(tmp_path):
    path = tmp_path / "deep" / "deeper" / "ranking.json"
    write_ranking(rank(three_pairs(), ridge=1e-6), path)
    assert path.exists()


def test_top_pairs_takes_the_best_few(tmp_path):
    path = tmp_path / "ranking.json"
    write_ranking(rank(three_pairs(), ridge=1e-6), path)

    assert top_pairs(path, 2) == [("A@three", "B@three"), ("A@two", "B@two")]


def test_top_pairs_asking_for_more_than_exist_is_fine(tmp_path):
    path = tmp_path / "ranking.json"
    write_ranking(rank(three_pairs(), ridge=1e-6), path)
    assert len(top_pairs(path, 99)) == 3


# ----------------------------------------------------------------------
# the fitted map as something evaluable
# ----------------------------------------------------------------------

def test_the_fitted_map_can_be_run_like_a_coder():
    """So the linear reference and a trained coder are measured the same way."""
    torch.manual_seed(9)
    x = torch.randn(TOKENS, WIDTH)
    y = x @ torch.randn(WIDTH, WIDTH) + torch.randn(WIDTH)

    weight, bias, _ = fit_linear(moments_from(x, y), A, B, ridge=1e-9)
    output, latents = LinearStitch(weight, bias)(x)

    assert torch.allclose(output, y, atol=1e-3)
    assert latents.shape == (TOKENS, 0)


def test_the_fitted_map_holds_no_gradients():
    weight, bias, _ = fit_linear(
        moments_from(torch.randn(TOKENS, WIDTH), torch.randn(TOKENS, WIDTH)), A, B, 1e-6)
    stitch = LinearStitch(weight, bias)
    assert not any(parameter.requires_grad for parameter in stitch.parameters())
    