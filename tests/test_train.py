"""Tests for the training loop.

A fake activation source over a fixed low rank signal, so the whole file runs
on CPU in a few seconds with no backbone, no dataset and no GPU. Nothing here
uses a GPU and nothing here can: the loop's behaviour is about step counts,
ordering, checkpoints and logging, none of which depends on the device. The one
device-dependent path, half precision, is only enabled on CUDA and is therefore
inactive and untested here.

    python -m pytest tests/test_train.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")

from stitch.coder import DeadFeatureTracker, SparseCoder                # noqa: E402
from stitch.config import Config                                        # noqa: E402
from stitch.train import (                                              # noqa: E402
    build_optimizer, load_checkpoint, prune_checkpoints, resolve_layers,
    save_checkpoint, set_seed, train,
)

WIDTH = 8
TOKEN_BATCH = 16
KEY = "A@add_14"
TARGET_KEY = "A@add_16"


class FakeSource:
    """An endless stream of a fixed low rank signal.

    Low rank so a small dictionary can represent it, fixed so the loss falling
    means the loop works rather than that the data got easier.
    """

    def __init__(self, keys=(KEY,), width=WIDTH, seed=0):
        generator = torch.Generator().manual_seed(seed)
        basis = torch.randn(3, width, generator=generator)
        weights = torch.rand(512, 3, generator=generator)
        self.pool = weights @ basis
        self.keys = keys
        self.width = width
        self.position = 0

    def batches(self):
        while True:
            start = self.position % (self.pool.shape[0] - TOKEN_BATCH)
            self.position += TOKEN_BATCH
            chunk = self.pool[start:start + TOKEN_BATCH]
            yield {key: chunk.clone() for key in self.keys}

    def widths(self):
        return {key: self.width for key in self.keys}

    def scales(self):
        return {key: 1.0 for key in self.keys}


class FakeLogger:
    """Records what it was asked to log."""

    def __init__(self):
        self.calls: list[tuple[int, dict]] = []

    def log_metrics(self, metrics, step):
        self.calls.append((step, dict(metrics)))

    @property
    def steps(self):
        return [step for step, _ in self.calls]


@pytest.fixture
def cfg():
    c = Config()
    c.train.device = "cpu"
    c.train.dtype = "float32"
    c.train.token_batch = TOKEN_BATCH
    c.train.tokens = TOKEN_BATCH * 20
    c.train.lr = 1e-2
    c.train.log_every = 5
    c.train.eval_every = 10
    c.train.checkpoint_every = 10
    c.train.keep_checkpoints = 2
    c.coder.expansion = 4
    c.coder.sparsity.coeff = 1e-4
    c.coder.dead_features.track_window = 3
    c.activations.layers = [KEY]
    return c


@pytest.fixture
def pieces(cfg):
    set_seed(0)
    coder = SparseCoder(WIDTH, WIDTH, cfg.coder.expansion)
    tracker = DeadFeatureTracker(coder.d_latent, cfg.coder.dead_features.track_window)
    return coder, tracker


# ----------------------------------------------------------------------
# which layer is which
# ----------------------------------------------------------------------

def test_one_layer_is_read_and_reconstructed(cfg):
    assert resolve_layers(cfg) == (KEY, KEY)


def test_two_layers_are_read_then_reconstructed(cfg):
    """Order carries the meaning, so it is the first that is read."""
    cfg.activations.layers = [KEY, TARGET_KEY]
    assert resolve_layers(cfg) == (KEY, TARGET_KEY)


def test_three_layers_raise(cfg):
    cfg.activations.layers = [KEY, TARGET_KEY, "B@add_2"]
    with pytest.raises(ValueError, match="one or two"):
        resolve_layers(cfg)


# ----------------------------------------------------------------------
# the optimiser
# ----------------------------------------------------------------------

@pytest.mark.parametrize("name, expected", [("adam", torch.optim.Adam),
                                            ("AdamW", torch.optim.AdamW)])
def test_the_named_optimiser_is_built(cfg, pieces, name, expected):
    cfg.train.optimizer = name
    assert isinstance(build_optimizer(cfg, pieces[0]), expected)


def test_an_unknown_optimiser_raises(cfg, pieces):
    cfg.train.optimizer = "lbfgs"
    with pytest.raises(ValueError, match="train.optimizer"):
        build_optimizer(cfg, pieces[0])


# ----------------------------------------------------------------------
# the loop
# ----------------------------------------------------------------------

def test_it_runs_the_budget_and_no_more(cfg, pieces, tmp_path):
    """The stream is endless, so the budget is the only thing that stops it."""
    logger = FakeLogger()
    train(cfg, FakeSource(), *pieces, logger, tmp_path)

    assert max(logger.steps) == cfg.train.tokens // cfg.train.token_batch


def test_the_loss_falls(cfg, pieces, tmp_path):
    cfg.train.tokens = TOKEN_BATCH * 200
    logger = FakeLogger()
    final = train(cfg, FakeSource(), *pieces, logger, tmp_path)

    first = logger.calls[0][1]
    assert final["mse"] < first["mse"] * 0.5


def test_the_dictionary_stays_at_unit_length(cfg, pieces, tmp_path):
    """Renormalisation has to happen after every step, not once at the end."""
    coder, _ = pieces
    train(cfg, FakeSource(), *pieces, FakeLogger(), tmp_path)

    norms = coder.decoder_norms()
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_a_single_layer_reconstructs_itself(cfg, pieces, tmp_path):
    """Case one reads and reconstructs the same tensor, so the error can reach zero."""
    cfg.train.tokens = TOKEN_BATCH * 200
    final = train(cfg, FakeSource(), *pieces, FakeLogger(), tmp_path)
    assert final["explained_variance"] > 0.5


def test_two_layers_are_handled(cfg, pieces, tmp_path):
    cfg.activations.layers = [KEY, TARGET_KEY]
    final = train(cfg, FakeSource(keys=(KEY, TARGET_KEY)), *pieces,
                  FakeLogger(), tmp_path)
    assert "mse" in final


# ----------------------------------------------------------------------
# logging
# ----------------------------------------------------------------------

def test_metrics_are_logged_on_the_interval(cfg, pieces, tmp_path):
    logger = FakeLogger()
    train(cfg, FakeSource(), *pieces, logger, tmp_path)
    assert logger.steps == [5, 10, 15, 20]


def test_the_final_step_is_always_logged(cfg, pieces, tmp_path):
    """Otherwise a budget that is not a multiple of the interval loses its end."""
    cfg.train.tokens = TOKEN_BATCH * 7
    logger = FakeLogger()
    train(cfg, FakeSource(), *pieces, logger, tmp_path)
    assert logger.steps[-1] == 7


def test_tokens_and_dead_fraction_are_reported(cfg, pieces, tmp_path):
    logger = FakeLogger()
    train(cfg, FakeSource(), *pieces, logger, tmp_path)

    _, metrics = logger.calls[-1]
    assert metrics["tokens"] == cfg.train.tokens
    assert 0.0 <= metrics["dead_fraction"] <= 1.0


def test_the_gradient_norm_is_reported_when_clipping(cfg, pieces, tmp_path):
    cfg.train.grad_clip = 1.0
    logger = FakeLogger()
    train(cfg, FakeSource(), *pieces, logger, tmp_path)
    assert "grad_norm" in logger.calls[0][1]


def test_no_gradient_norm_when_clipping_is_off(cfg, pieces, tmp_path):
    cfg.train.grad_clip = None
    logger = FakeLogger()
    train(cfg, FakeSource(), *pieces, logger, tmp_path)
    assert "grad_norm" not in logger.calls[0][1]


def test_the_evaluation_hook_runs_on_its_own_interval(cfg, pieces, tmp_path):
    seen = []

    def evaluate(coder, step):
        seen.append(step)
        return {"gate": 1.0}

    train(cfg, FakeSource(), *pieces, FakeLogger(), tmp_path, evaluate=evaluate)
    assert seen == [10, 20]


# ----------------------------------------------------------------------
# checkpoints
# ----------------------------------------------------------------------

def test_a_checkpoint_is_written_on_the_interval(cfg, pieces, tmp_path):
    train(cfg, FakeSource(), *pieces, FakeLogger(), tmp_path)
    assert (tmp_path / "checkpoints" / "step_000000020.pt").exists()


def test_only_the_newest_are_kept(cfg, pieces, tmp_path):
    """A coder at the real width is a few hundred megabytes."""
    cfg.train.tokens = TOKEN_BATCH * 50
    cfg.train.checkpoint_every = 10
    cfg.train.keep_checkpoints = 2

    train(cfg, FakeSource(), *pieces, FakeLogger(), tmp_path)
    assert len(list((tmp_path / "checkpoints").glob("step_*.pt"))) == 2


def test_keeping_everything_keeps_everything(cfg, pieces, tmp_path):
    cfg.train.keep_checkpoints = 0
    train(cfg, FakeSource(), *pieces, FakeLogger(), tmp_path)
    assert len(list((tmp_path / "checkpoints").glob("step_*.pt"))) >= 2


def test_pruning_returns_what_survived(tmp_path):
    for step in range(5):
        (tmp_path / f"step_{step:09d}.pt").write_text("x")
    survivors = prune_checkpoints(tmp_path, keep=2)
    assert [path.name for path in survivors] == ["step_000000003.pt", "step_000000004.pt"]


def test_a_checkpoint_carries_everything_needed_to_continue(cfg, pieces, tmp_path):
    coder, tracker = pieces
    optimiser = build_optimizer(cfg, coder)
    path = tmp_path / "one.pt"

    save_checkpoint(path, 42, 4200, coder, tracker, optimiser, {KEY: 2.5})
    state = torch.load(path, map_location="cpu", weights_only=False)

    assert set(state) == {"step", "tokens", "coder", "tracker", "optimizer", "scales"}
    assert state["step"] == 42 and state["tokens"] == 4200
    assert state["scales"] == {KEY: 2.5}


def test_a_resumed_run_continues_rather_than_restarting(cfg, pieces, tmp_path):
    """A resumed run is the same run, so the step count carries over."""
    logger = FakeLogger()
    train(cfg, FakeSource(), *pieces, logger, tmp_path)

    checkpoint = sorted((tmp_path / "checkpoints").glob("step_*.pt"))[-1]

    cfg.train.tokens = TOKEN_BATCH * 30
    second = FakeLogger()
    train(cfg, FakeSource(), *pieces, second, tmp_path, resume=checkpoint)

    assert min(second.steps) > 20
    assert max(second.steps) == 30


def test_resuming_restores_the_weights(cfg, pieces, tmp_path):
    coder, tracker = pieces
    train(cfg, FakeSource(), coder, tracker, FakeLogger(), tmp_path)
    trained = coder.decoder.weight.clone()

    fresh = SparseCoder(WIDTH, WIDTH, cfg.coder.expansion)
    fresh_tracker = DeadFeatureTracker(fresh.d_latent, 3)
    optimiser = build_optimizer(cfg, fresh)
    assert not torch.allclose(fresh.decoder.weight, trained)

    checkpoint = sorted((tmp_path / "checkpoints").glob("step_*.pt"))[-1]
    step, tokens = load_checkpoint(checkpoint, fresh, fresh_tracker, optimiser)

    assert torch.allclose(fresh.decoder.weight, trained)
    assert step == 20 and tokens == cfg.train.tokens


def test_the_dead_feature_count_survives_the_round_trip(cfg, pieces, tmp_path):
    coder, tracker = pieces
    train(cfg, FakeSource(), coder, tracker, FakeLogger(), tmp_path)

    fresh = SparseCoder(WIDTH, WIDTH, cfg.coder.expansion)
    fresh_tracker = DeadFeatureTracker(fresh.d_latent, 3)
    checkpoint = sorted((tmp_path / "checkpoints").glob("step_*.pt"))[-1]
    load_checkpoint(checkpoint, fresh, fresh_tracker, build_optimizer(cfg, fresh))

    assert torch.equal(fresh_tracker.steps_since_fired, tracker.steps_since_fired)


# ----------------------------------------------------------------------
# shutdown
# ----------------------------------------------------------------------

def test_the_stream_is_closed_when_a_step_fails(cfg, pieces, tmp_path):
    """A loader with workers holds processes open, crash or no crash."""
    closed = []

    class ClosingSource(FakeSource):
        def batches(self):
            try:
                yield from super().batches()
            finally:
                closed.append(True)

    def explode(coder, step):
        raise RuntimeError("deliberate failure")

    with pytest.raises(RuntimeError, match="deliberate"):
        train(cfg, ClosingSource(), *pieces, FakeLogger(), tmp_path, evaluate=explode)

    assert closed == [True]


# ----------------------------------------------------------------------
# seeding
# ----------------------------------------------------------------------

def test_the_same_seed_trains_the_same_coder(cfg, tmp_path):
    """The comparison is read against the spread between seeds, so seeds must bite."""
    results = []
    for _ in range(2):
        set_seed(3)
        coder = SparseCoder(WIDTH, WIDTH, cfg.coder.expansion)
        tracker = DeadFeatureTracker(coder.d_latent, 3)
        train(cfg, FakeSource(), coder, tracker, FakeLogger(), tmp_path)
        results.append(coder.decoder.weight.clone())

    assert torch.equal(results[0], results[1])


def test_a_different_seed_trains_a_different_coder(cfg, tmp_path):
    results = []
    for seed in (3, 4):
        set_seed(seed)
        coder = SparseCoder(WIDTH, WIDTH, cfg.coder.expansion)
        tracker = DeadFeatureTracker(coder.d_latent, 3)
        train(cfg, FakeSource(), coder, tracker, FakeLogger(), tmp_path)
        results.append(coder.decoder.weight.clone())

    assert not torch.equal(results[0], results[1])