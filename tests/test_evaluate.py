"""Tests for evaluation.

Coders with known behaviour rather than trained ones, so each metric can be
checked against a number worked out by hand. A coder that returns its input
unchanged must explain all the variance; one that returns zero must explain
none. Runs on CPU in a couple of seconds; nothing here uses a GPU.

    python -m pytest tests/test_evaluate.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")

from torch import nn                                                    # noqa: E402

from stitch.coder import SparseCoder                                    # noqa: E402
from stitch.config import Config                                        # noqa: E402
from stitch.evaluate import (                                           # noqa: E402
    check_gates, evaluate, feature_frequencies, reconstruction_metrics,
    substitution_accuracy,
)

WIDTH = 8
LATENT = 32
TOKENS = 64
KEY = "A@add_14"
CLASSES = 5


class PerfectCoder(SparseCoder):
    """Returns its input unchanged, with one feature active per token.

    Not something training could produce. It exists so the metrics can be
    checked against values that are known rather than plausible.
    """

    def __init__(self, width=WIDTH, latent=LATENT, active=1):
        super().__init__(width, width, latent // width)
        self.active = active

    def encode(self, x):
        latents = torch.zeros(x.shape[0], self.d_latent, device=x.device)
        latents[:, :self.active] = 1.0
        return latents

    def forward(self, x):
        return x.clone(), self.encode(x)


class ZeroCoder(PerfectCoder):
    """Returns zero and fires nothing."""

    def encode(self, x):
        return torch.zeros(x.shape[0], self.d_latent, device=x.device)

    def forward(self, x):
        return torch.zeros_like(x), self.encode(x)


class DenseCoder(PerfectCoder):
    """Fires every feature on every token."""

    def encode(self, x):
        return torch.ones(x.shape[0], self.d_latent, device=x.device)


def stream(keys=(KEY,), width=WIDTH, batches=8, seed=0):
    """A finite, repeatable activation stream."""
    generator = torch.Generator().manual_seed(seed)
    for _ in range(batches):
        chunk = torch.randn(TOKENS, width, generator=generator)
        yield {key: chunk.clone() for key in keys}


@pytest.fixture
def cfg():
    c = Config()
    c.train.device = "cpu"
    c.eval.batches = 4
    c.eval.images = 32
    c.eval.min_explained_variance = 0.5
    c.eval.min_l0 = 1.0
    c.eval.max_l0_frac = 0.25
    c.eval.max_never_fired = 0.5
    c.eval.max_feature_frequency = 0.9
    return c


# ----------------------------------------------------------------------
# reconstruction metrics
# ----------------------------------------------------------------------

def test_a_perfect_reconstruction_explains_everything():
    metrics = reconstruction_metrics(PerfectCoder(), stream(), KEY, KEY, count=4)
    assert metrics["eval_mse"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["eval_explained_variance"] == pytest.approx(1.0, abs=1e-5)


def test_reconstructing_zero_explains_nothing():
    """Predicting zero for a zero mean target leaves all the variance unexplained."""
    metrics = reconstruction_metrics(ZeroCoder(), stream(), KEY, KEY, count=4)
    assert metrics["eval_explained_variance"] == pytest.approx(0.0, abs=0.1)


def test_l0_counts_the_active_features():
    metrics = reconstruction_metrics(PerfectCoder(active=3), stream(), KEY, KEY, count=4)
    assert metrics["eval_l0"] == pytest.approx(3.0)


def test_only_the_requested_number_of_batches_is_read():
    """The stream may be endless, so the count is what stops it."""
    metrics = reconstruction_metrics(PerfectCoder(), stream(batches=99), KEY, KEY, count=3)
    assert metrics["eval_tokens"] == 3 * TOKENS


def test_a_short_stream_is_used_in_full():
    metrics = reconstruction_metrics(PerfectCoder(), stream(batches=2), KEY, KEY, count=50)
    assert metrics["eval_tokens"] == 2 * TOKENS


def test_an_empty_stream_raises():
    with pytest.raises(ValueError, match="no batches"):
        reconstruction_metrics(PerfectCoder(), iter([]), KEY, KEY, count=4)


def test_silent_features_are_reported():
    """One feature in use out of thirty-two."""
    metrics = reconstruction_metrics(PerfectCoder(active=1), stream(), KEY, KEY, count=4)
    assert metrics["eval_never_fired"] == pytest.approx(31 / 32)


def test_an_always_on_feature_is_reported():
    """A feature active on every token is a bias, not a feature."""
    metrics = reconstruction_metrics(DenseCoder(), stream(), KEY, KEY, count=4)
    assert metrics["eval_max_feature_frequency"] == pytest.approx(1.0)


def test_a_different_target_is_used_as_the_target():
    """The cross case: what it reads and what it is judged against differ."""
    def crossed():
        for batch in stream(batches=4):
            yield {KEY: batch[KEY], "B@add_2": batch[KEY] * 3.0}

    metrics = reconstruction_metrics(PerfectCoder(), crossed(), KEY, "B@add_2", count=4)
    assert metrics["eval_mse"] > 0.1


# ----------------------------------------------------------------------
# feature frequencies
# ----------------------------------------------------------------------

def test_frequencies_are_one_per_feature():
    frequencies = feature_frequencies(PerfectCoder(), stream(), KEY, count=4)
    assert frequencies.shape == (LATENT,)


def test_frequencies_are_shares_of_tokens():
    frequencies = feature_frequencies(PerfectCoder(active=2), stream(), KEY, count=4)
    assert frequencies[:2].tolist() == [1.0, 1.0]
    assert frequencies[2:].sum() == 0.0


# ----------------------------------------------------------------------
# substitution
# ----------------------------------------------------------------------

class Upper(nn.Module):
    """Stands in for the part of a model before the cut."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(WIDTH, WIDTH)

    def forward(self, x):
        return self.fc(x)


class Lower(nn.Module):
    """Stands in for the part after it."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(WIDTH, CLASSES)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def loader():
    torch.manual_seed(0)
    images = torch.randn(32, WIDTH)
    labels = torch.randint(0, CLASSES, (32,))
    return [(images[:16], labels[:16]), (images[16:], labels[16:])]


def test_a_perfect_coder_changes_nothing(loader):
    """If the reconstruction is exact, the model must predict exactly as before."""
    result = substitution_accuracy(PerfectCoder(), Upper(), Lower(), loader,
                                   scale_in=1.0, scale_out=1.0, max_images=32)
    assert result["agreement"] == 1.0
    assert result["top1_drop"] == 0.0
    assert result["plain_top1"] == result["substituted_top1"]


def test_a_useless_coder_is_visible(loader):
    """Substituting zero should move the prediction, usually to one class."""
    result = substitution_accuracy(ZeroCoder(), Upper(), Lower(), loader,
                                   scale_in=1.0, scale_out=1.0, max_images=32)
    assert result["agreement"] < 1.0


def test_the_scale_is_undone_before_the_lower_half(loader):
    """The coder works in scaled units; the model below it does not."""
    result = substitution_accuracy(PerfectCoder(), Upper(), Lower(), loader,
                                   scale_in=7.5, scale_out=7.5, max_images=32)
    assert result["agreement"] == 1.0


def test_it_stops_at_the_image_limit(loader):
    result = substitution_accuracy(PerfectCoder(), Upper(), Lower(), loader,
                                   scale_in=1.0, scale_out=1.0, max_images=16)
    assert result["images"] == 16


def test_an_empty_loader_raises():
    with pytest.raises(ValueError, match="no images"):
        substitution_accuracy(PerfectCoder(), Upper(), Lower(), [],
                              scale_in=1.0, scale_out=1.0, max_images=32)


def test_accuracy_is_a_proportion(loader):
    result = substitution_accuracy(PerfectCoder(), Upper(), Lower(), loader,
                                   scale_in=1.0, scale_out=1.0, max_images=32)
    assert 0.0 <= result["plain_top1"] <= 1.0
    assert 0.0 <= result["agreement"] <= 1.0


# ----------------------------------------------------------------------
# validity gates
# ----------------------------------------------------------------------

def passing(**overrides):
    metrics = {
        "eval_mse": 0.1,
        "eval_l0": 8.0,
        "eval_explained_variance": 0.9,
        "eval_never_fired": 0.05,
        "eval_max_feature_frequency": 0.3,
        "eval_tokens": 1000.0,
    }
    metrics.update(overrides)
    return metrics


def test_a_healthy_coder_passes(cfg):
    assert check_gates(passing(), cfg, 64)["passed"]


@pytest.mark.parametrize("gate, metrics", [
    ("finite", {"eval_mse": float("nan")}),
    ("finite", {"eval_explained_variance": float("inf")}),
    ("reconstructs", {"eval_explained_variance": 0.1}),
    ("not_collapsed", {"eval_l0": 0.0}),
    ("not_dense", {"eval_l0": 64.0}),
    ("features_alive", {"eval_never_fired": 0.9}),
    ("no_always_on_feature", {"eval_max_feature_frequency": 0.99}),
])
def test_each_gate_catches_its_own_failure(cfg, gate, metrics):
    result = check_gates(passing(**metrics), cfg, 64)
    assert result[gate] is False
    assert result["passed"] is False


def test_one_failure_is_enough_to_fail(cfg):
    result = check_gates(passing(eval_l0=0.0), cfg, 64)
    assert result["reconstructs"] is True
    assert result["passed"] is False


def test_a_collapsed_coder_is_caught_despite_perfect_sparsity(cfg):
    """Every feature off gives a perfect sparsity score and no information."""
    result = check_gates(passing(eval_l0=0.0, eval_explained_variance=0.0), cfg, 64)
    assert result["not_collapsed"] is False


# ----------------------------------------------------------------------
# the hook
# ----------------------------------------------------------------------

def test_evaluate_returns_numbers_only(cfg):
    """It goes straight to a metric store, which holds numbers."""
    metrics = evaluate(PerfectCoder(), stream(), KEY, KEY, cfg)
    assert all(isinstance(value, float) for value in metrics.values())


def test_evaluate_reports_every_gate(cfg):
    metrics = evaluate(PerfectCoder(), stream(), KEY, KEY, cfg)
    assert "gate_passed" in metrics
    assert "gate_not_collapsed" in metrics


def test_evaluate_fails_a_coder_that_fires_nothing(cfg):
    metrics = evaluate(ZeroCoder(), stream(), KEY, KEY, cfg)
    assert metrics["gate_passed"] == 0.0
    assert metrics["gate_not_collapsed"] == 0.0