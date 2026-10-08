"""Tests for the sparse coder.

Small widths and a handful of steps, so the whole file runs on CPU in a couple
of seconds. No GPU is used anywhere here, and none is required: every property
under test is about shapes, constraints and arithmetic, none of which depends
on the device.

    python -m pytest tests/test_coder.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")

from stitch.coder import (                                            # noqa: E402
    DeadFeatureTracker, SparseCoder, build_coder, compute_loss,
)
from stitch.config import Config                                      # noqa: E402

D_IN = 8
D_OUT = 8
EXPANSION = 4
TOKENS = 32


@pytest.fixture
def coder():
    torch.manual_seed(0)
    return SparseCoder(D_IN, D_OUT, EXPANSION)


@pytest.fixture
def x():
    torch.manual_seed(1)
    return torch.randn(TOKENS, D_IN)


# ----------------------------------------------------------------------
# construction
# ----------------------------------------------------------------------

def test_dictionary_is_expansion_times_the_input(coder):
    assert coder.d_latent == D_IN * EXPANSION


def test_forward_returns_a_reconstruction_and_its_features(coder, x):
    reconstruction, latents = coder(x)
    assert reconstruction.shape == (TOKENS, D_OUT)
    assert latents.shape == (TOKENS, D_IN * EXPANSION)


def test_input_and_output_widths_can_differ():
    """The only structural difference between the objectives being compared."""
    coder = SparseCoder(d_in=8, d_out=12, expansion=2)
    reconstruction, latents = coder(torch.randn(TOKENS, 8))
    assert reconstruction.shape == (TOKENS, 12)
    assert latents.shape == (TOKENS, 16)


def test_features_are_never_negative(coder, x):
    """A feature is present or absent, so a negative amount of one is meaningless."""
    _, latents = coder(x)
    assert (latents >= 0).all()


@pytest.mark.parametrize("kwargs, message", [
    ({"d_in": 0, "d_out": 8, "expansion": 2}, "positive"),
    ({"d_in": 8, "d_out": 8, "expansion": 0}, "positive"),
    ({"d_in": 8, "d_out": 8, "expansion": 2, "sparsity": "lasso"}, "sparsity mode"),
    ({"d_in": 8, "d_out": 8, "expansion": 2, "sparsity": "topk"}, "needs k"),
    ({"d_in": 8, "d_out": 8, "expansion": 2, "sparsity": "topk", "k": 0}, "between"),
    ({"d_in": 8, "d_out": 8, "expansion": 2, "sparsity": "topk", "k": 99}, "between"),
])
def test_bad_arguments_raise(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SparseCoder(**kwargs)


def test_the_same_seed_builds_the_same_coder():
    torch.manual_seed(7)
    one = SparseCoder(D_IN, D_OUT, EXPANSION)
    torch.manual_seed(7)
    two = SparseCoder(D_IN, D_OUT, EXPANSION)
    assert torch.equal(one.decoder.weight, two.decoder.weight)


# ----------------------------------------------------------------------
# sparsity
# ----------------------------------------------------------------------

def test_topk_keeps_exactly_k_features(x):
    """Under topk the sparsity is exact, not something a penalty aims at."""
    coder = SparseCoder(D_IN, D_OUT, EXPANSION, sparsity="topk", k=3)
    _, latents = coder(x)
    assert ((latents > 0).sum(dim=-1) <= 3).all()


def test_topk_keeps_the_largest_features(x):
    coder = SparseCoder(D_IN, D_OUT, EXPANSION, sparsity="topk", k=3)
    before = torch.relu(coder.encoder(x - coder.b_pre))
    _, after = coder(x)

    for row_before, row_after in zip(before, after):
        kept = row_after > 0
        if kept.any():
            assert row_before[kept].min() >= row_before[~kept].max()


def test_topk_has_no_penalty_so_the_loss_is_the_reconstruction(x):
    coder = SparseCoder(D_IN, D_OUT, EXPANSION, sparsity="topk", k=3)
    loss, metrics = compute_loss(coder, x, x, coeff=1.0)
    assert metrics["sparsity_penalty"] == 0.0
    assert pytest.approx(metrics["loss"]) == metrics["mse"]


def test_l1_penalty_grows_with_activity(coder, x):
    """A larger penalty must mean a larger loss, or the coefficient does nothing."""
    _, small = compute_loss(coder, x, x, coeff=0.0)
    _, large = compute_loss(coder, x, x, coeff=1.0)
    assert large["loss"] > small["loss"]
    assert large["sparsity_penalty"] == pytest.approx(small["sparsity_penalty"])


# ----------------------------------------------------------------------
# dictionary normalisation
#
# Without this an L1 penalty is trivially satisfiable: shrink the features,
# grow the dictionary vectors, leave the reconstruction unchanged and report a
# smaller penalty. Nothing has become sparser.
# ----------------------------------------------------------------------

def test_dictionary_starts_at_unit_length(coder):
    assert torch.allclose(coder.decoder_norms(), torch.ones(coder.d_latent), atol=1e-5)


def test_normalising_restores_unit_length(coder):
    with torch.no_grad():
        coder.decoder.weight.mul_(3.7)
    coder.normalize_decoder_columns()
    assert torch.allclose(coder.decoder_norms(), torch.ones(coder.d_latent), atol=1e-5)


def test_the_gradient_loses_its_lengthwise_part(coder):
    """What remains must be perpendicular to each vector, so the step stays on the sphere."""
    coder.decoder.weight.grad = torch.randn_like(coder.decoder.weight)
    coder.project_decoder_gradient()

    along = (coder.decoder.weight.grad * coder.decoder.weight).sum(dim=0)
    assert torch.allclose(along, torch.zeros_like(along), atol=1e-5)


def test_the_gradient_keeps_its_other_part(coder):
    """Projection must remove one component, not the whole gradient."""
    coder.decoder.weight.grad = torch.randn_like(coder.decoder.weight)
    coder.project_decoder_gradient()
    assert coder.decoder.weight.grad.abs().sum() > 0


def test_normalisation_can_be_turned_off():
    coder = SparseCoder(D_IN, D_OUT, EXPANSION, normalize_decoder=False)
    with torch.no_grad():
        coder.decoder.weight.mul_(3.0)
    coder.normalize_decoder_columns()
    assert coder.decoder_norms().mean() > 2.0


def test_the_penalty_weighs_features_by_their_dictionary_vector(x):
    """Otherwise the decoder can rescale its way out of the penalty."""
    coder = SparseCoder(D_IN, D_OUT, EXPANSION, normalize_decoder=False)
    _, before = compute_loss(coder, x, x, coeff=1.0)

    with torch.no_grad():
        coder.decoder.weight.mul_(2.0)
        coder.encoder.weight.div_(2.0)
        coder.encoder.bias.div_(2.0)

    _, after = compute_loss(coder, x, x, coeff=1.0)
    assert after["sparsity_penalty"] == pytest.approx(before["sparsity_penalty"], rel=1e-4)


# ----------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------

def test_a_perfect_reconstruction_explains_everything(coder, x):
    reconstruction, _ = coder(x)
    _, metrics = compute_loss(coder, x, reconstruction.detach(), coeff=0.0)
    assert metrics["mse"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["explained_variance"] == pytest.approx(1.0, abs=1e-5)


def test_l0_counts_active_features(x):
    coder = SparseCoder(D_IN, D_OUT, EXPANSION, sparsity="topk", k=5)
    _, metrics = compute_loss(coder, x, x, coeff=0.0)
    assert metrics["l0"] <= 5.0


def test_metrics_are_plain_numbers(coder, x):
    """They go straight to the logger, so a tensor here would keep the graph alive."""
    _, metrics = compute_loss(coder, x, x, coeff=1e-3)
    assert all(isinstance(value, float) for value in metrics.values())


def test_the_loss_carries_gradients(coder, x):
    loss, _ = compute_loss(coder, x, x, coeff=1e-3)
    loss.backward()
    assert coder.encoder.weight.grad is not None
    assert coder.decoder.weight.grad is not None


# ----------------------------------------------------------------------
# dead features
# ----------------------------------------------------------------------

def test_a_silent_feature_dies_after_the_window():
    tracker = DeadFeatureTracker(d_latent=4, window=3)
    latents = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    for _ in range(4):
        tracker.update(latents)

    assert tracker.dead_mask.tolist() == [False, True, True, True]
    assert tracker.dead_fraction == pytest.approx(0.75)


def test_firing_resets_the_count():
    tracker = DeadFeatureTracker(d_latent=2, window=2)
    for _ in range(5):
        tracker.update(torch.tensor([[1.0, 0.0]]))
    assert tracker.dead_mask.tolist() == [False, True]

    tracker.update(torch.tensor([[0.0, 1.0]]))
    assert tracker.dead_mask.tolist() == [False, False]


def test_firing_for_any_token_counts():
    tracker = DeadFeatureTracker(d_latent=2, window=1)
    tracker.update(torch.tensor([[0.0, 0.0], [0.0, 2.0]]))
    tracker.update(torch.tensor([[0.0, 0.0], [0.0, 0.0]]))
    assert tracker.dead_mask.tolist() == [True, False]


def test_nothing_is_dead_at_the_start():
    assert DeadFeatureTracker(d_latent=16, window=10).dead_fraction == 0.0


def test_the_count_survives_a_checkpoint():
    """A resumed run must not begin by calling every feature alive."""
    tracker = DeadFeatureTracker(d_latent=4, window=2)
    for _ in range(3):
        tracker.update(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))

    restored = DeadFeatureTracker(d_latent=4, window=2)
    restored.load_state_dict(tracker.state_dict())
    assert restored.dead_mask.tolist() == tracker.dead_mask.tolist()


def test_a_bad_window_raises():
    with pytest.raises(ValueError, match="positive"):
        DeadFeatureTracker(d_latent=4, window=0)


# ----------------------------------------------------------------------
# construction from configuration
# ----------------------------------------------------------------------

def test_build_coder_sizes_from_the_widths_given():
    cfg = Config()
    coder, tracker = build_coder(cfg, d_in=16, d_out=24)

    assert coder.d_in == 16 and coder.d_out == 24
    assert coder.d_latent == 16 * cfg.coder.expansion
    assert tracker.steps_since_fired.shape == (coder.d_latent,)
    assert tracker.window == cfg.coder.dead_features.track_window


def test_build_coder_follows_the_sparsity_setting():
    cfg = Config()
    cfg.coder.sparsity.mode = "l1"
    coder, _ = build_coder(cfg, d_in=8, d_out=8)
    assert coder.sparsity == "l1"


# ----------------------------------------------------------------------
# it actually learns
#
# Not a test of quality, which needs a real run. A test that the pieces are
# wired the right way round: a coder that cannot reduce its own loss on a fixed
# batch has something connected backwards, and every later number would be
# measured against that.
# ----------------------------------------------------------------------

def test_the_loss_falls_on_a_fixed_batch():
    torch.manual_seed(0)

    # A low rank batch, so a small dictionary can represent it well.
    basis = torch.randn(3, D_IN)
    weights = torch.rand(256, 3)
    batch = weights @ basis

    coder = SparseCoder(D_IN, D_OUT, EXPANSION)
    optimiser = torch.optim.Adam(coder.parameters(), lr=1e-2)

    _, first = compute_loss(coder, batch, batch, coeff=1e-4)
    for _ in range(300):
        optimiser.zero_grad()
        loss, last = compute_loss(coder, batch, batch, coeff=1e-4)
        loss.backward()
        coder.project_decoder_gradient()
        optimiser.step()
        coder.normalize_decoder_columns()

    assert last["mse"] < first["mse"] * 0.5
    assert last["explained_variance"] > first["explained_variance"]


def test_it_learns_a_target_that_is_not_its_input():
    """The cross case, which is what the later objectives need."""
    torch.manual_seed(0)

    source = torch.randn(256, D_IN)
    target = source @ torch.randn(D_IN, 12)

    coder = SparseCoder(d_in=D_IN, d_out=12, expansion=EXPANSION)
    optimiser = torch.optim.Adam(coder.parameters(), lr=1e-2)

    _, first = compute_loss(coder, source, target, coeff=1e-4)
    for _ in range(300):
        optimiser.zero_grad()
        loss, last = compute_loss(coder, source, target, coeff=1e-4)
        loss.backward()
        coder.project_decoder_gradient()
        optimiser.step()
        coder.normalize_decoder_columns()

    assert last["mse"] < first["mse"] * 0.5

def test_build_coder_resolves_k_from_the_fraction():
    cfg = Config()
    cfg.coder.sparsity.mode = "topk"
    cfg.coder.sparsity.k = None
    coder, _ = build_coder(cfg, d_in=768, d_out=768)
    assert coder.k == round(cfg.coder.sparsity.k_frac * 768)


def test_an_explicit_k_wins_over_the_fraction():
    cfg = Config()
    cfg.coder.sparsity.mode = "topk"
    cfg.coder.sparsity.k = 32
    coder, _ = build_coder(cfg, d_in=768, d_out=768)
    assert coder.k == 32

def test_the_penalty_pushes_back_on_dictionary_growth():
    """Without a gradient path to the decoder, growing a vector is free."""
    coder = SparseCoder(D_IN, D_OUT, EXPANSION, normalize_decoder=False)
    batch = torch.randn(TOKENS, D_IN)

    coder.zero_grad()
    compute_loss(coder, batch, batch, coeff=0.0)[0].backward()
    without = coder.decoder.weight.grad.clone()

    coder.zero_grad()
    compute_loss(coder, batch, batch, coeff=1.0)[0].backward()
    with_penalty = coder.decoder.weight.grad.clone()

    assert not torch.allclose(without, with_penalty)