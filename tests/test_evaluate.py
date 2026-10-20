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


class HalfCoder(PerfectCoder):
    """Returns half its input, leaving three quarters of the variance.

    The residual is a quarter of the target in squared terms, so the explained
    variance is a known 0.75 whatever units the activation arrives in.
    """

    def forward(self, x):
        return x * 0.5, self.encode(x)


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
    c.eval.linear_fit = 1.0
    c.eval.fit_fraction = 0.5
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


def test_explained_variance_is_a_share_of_the_targets_variance():
    """Half the activation leaves a quarter of its squared magnitude as error."""
    metrics = reconstruction_metrics(HalfCoder(), stream(), KEY, KEY, count=4)
    assert metrics["eval_explained_variance"] == pytest.approx(0.75, abs=0.02)


def test_explained_variance_does_not_depend_on_the_activation_scale():
    """One scalar per layer squares out of the residual and the total alike.

    This is what lets the reconstruction gate compare a coder working in scaled
    units against a linear fit measured in whatever units the sweep fed it. The
    error itself moves with the scale; the share of the variance does not.
    """
    def scaled(factor):
        for batch in stream(batches=4):
            yield {KEY: batch[KEY] * factor}

    plain = reconstruction_metrics(HalfCoder(), scaled(1.0), KEY, KEY, count=4)
    big = reconstruction_metrics(HalfCoder(), scaled(37.0), KEY, KEY, count=4)

    assert big["eval_explained_variance"] == pytest.approx(
        plain["eval_explained_variance"], abs=1e-5)
    assert big["eval_mse"] == pytest.approx(plain["eval_mse"] * 37.0 ** 2, rel=1e-4)


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
# substitution against the right class numbering
#
# The bug these exist to prevent: the loader numbers a subset's classes from
# zero, the model predicts over the full set, and comparing the two gives
# almost exactly zero rather than an error. The model below is right on every
# image by construction, so anything other than a perfect score is the
# numbering and not the model.
# ----------------------------------------------------------------------

FULL_CLASSES = 10
#: Where each of the five subset classes sits in the model's numbering. None of
#: them coincides with its own position, so an untranslated comparison scores
#: zero rather than something that could be mistaken for a weak model.
SUBSET_TO_FULL = [3, 7, 1, 9, 5]


class ScatterLower(nn.Module):
    """Predicts ``SUBSET_TO_FULL[j]`` for an image that is one-hot at ``j``."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(WIDTH, FULL_CLASSES, bias=False)
        with torch.no_grad():
            self.fc.weight.zero_()
            for subset_index, full_index in enumerate(SUBSET_TO_FULL):
                self.fc.weight[full_index, subset_index] = 1.0

    def forward(self, x):
        return self.fc(x)


class Identity(nn.Module):
    """An upper half that passes the activation through untouched."""

    def forward(self, x):
        return x


@pytest.fixture
def scattered_loader():
    """One-hot images labelled in the loader's own numbering, zero to four."""
    subset_labels = torch.arange(CLASSES).repeat(4)          # 20 images
    images = torch.zeros(subset_labels.shape[0], WIDTH)
    images[torch.arange(subset_labels.shape[0]), subset_labels] = 1.0
    return [(images[:10], subset_labels[:10]), (images[10:], subset_labels[10:])]


def test_without_the_map_a_perfect_model_scores_zero(scattered_loader):
    """The symptom the defect produced in every run before this."""
    result = substitution_accuracy(PerfectCoder(), Identity(), ScatterLower(),
                                   scattered_loader, scale_in=1.0, scale_out=1.0,
                                   max_images=20)
    assert result["plain_top1"] == 0.0
    assert result["substituted_top1"] == 0.0


def test_with_the_map_a_perfect_model_scores_one(scattered_loader):
    result = substitution_accuracy(PerfectCoder(), Identity(), ScatterLower(),
                                   scattered_loader, scale_in=1.0, scale_out=1.0,
                                   max_images=20,
                                   label_map=torch.tensor(SUBSET_TO_FULL))
    assert result["plain_top1"] == 1.0
    assert result["substituted_top1"] == 1.0


# ----------------------------------------------------------------------
# the reference the stitch is compared against
#
# When the two halves come from different models, feeding the first model's
# activation straight into the second model's layers is not a model. It
# predicts at chance, and comparing the stitch against it says nothing. These
# fix the reference as the target model cut and rejoined.
# ----------------------------------------------------------------------

class ZeroUpper(nn.Module):
    """An upper half whose activation the lower half cannot use."""

    def forward(self, x):
        return torch.zeros_like(x)


def test_without_a_reference_the_baseline_is_the_broken_hybrid(scattered_loader):
    """The defect this exists to prevent, reproduced deliberately."""
    result = substitution_accuracy(PerfectCoder(), ZeroUpper(), ScatterLower(),
                                   scattered_loader, scale_in=1.0, scale_out=1.0,
                                   max_images=20,
                                   label_map=torch.tensor(SUBSET_TO_FULL))
    # Both sides are the same useless thing, so they agree perfectly and the
    # accuracy says nothing about the coder.
    assert result["plain_top1"] == result["substituted_top1"]
    assert result["agreement"] == 1.0


def test_a_reference_upper_supplies_the_unmodified_prediction(scattered_loader):
    """`plain` comes from the target model's own upper half, not the source's."""
    result = substitution_accuracy(PerfectCoder(), ZeroUpper(), ScatterLower(),
                                   scattered_loader, scale_in=1.0, scale_out=1.0,
                                   max_images=20,
                                   label_map=torch.tensor(SUBSET_TO_FULL),
                                   reference_upper=Identity())
    assert result["plain_top1"] == 1.0
    assert result["substituted_top1"] < 1.0
    assert result["top1_drop"] > 0.0


def test_the_reference_is_ignored_when_both_halves_are_one_model(scattered_loader):
    """Passing the same upper half twice changes nothing."""
    without = substitution_accuracy(PerfectCoder(), Identity(), ScatterLower(),
                                    scattered_loader, scale_in=1.0, scale_out=1.0,
                                    max_images=20)
    with_same = substitution_accuracy(PerfectCoder(), Identity(), ScatterLower(),
                                      scattered_loader, scale_in=1.0, scale_out=1.0,
                                      max_images=20, reference_upper=Identity())
    assert without == with_same


def test_agreement_does_not_depend_on_the_map(scattered_loader):
    """It compares two predictions, so no labels enter it either way."""
    without = substitution_accuracy(PerfectCoder(), Identity(), ScatterLower(),
                                    scattered_loader, scale_in=1.0, scale_out=1.0,
                                    max_images=20)
    with_map = substitution_accuracy(PerfectCoder(), Identity(), ScatterLower(),
                                     scattered_loader, scale_in=1.0, scale_out=1.0,
                                     max_images=20,
                                     label_map=torch.tensor(SUBSET_TO_FULL))
    assert without["agreement"] == with_map["agreement"] == 1.0


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


def test_density_is_judged_against_the_activation_not_the_dictionary(cfg):
    """A code using most of the activation's own width is dense, however wide
    the dictionary is.

    Six active features describing an eight-wide activation is dense: it is
    most of what it takes to reach any point in that space at all. Against a
    thirty-two entry dictionary the same six would read as comfortably sparse,
    which is the reading this gate used to take.
    """
    assert check_gates(passing(eval_l0=6.0), cfg, WIDTH)["not_dense"] is False
    assert check_gates(passing(eval_l0=6.0), cfg, LATENT)["not_dense"] is True


def test_the_threshold_scales_with_the_activation_width(cfg):
    """The same fraction at twice the width permits twice the active features."""
    assert check_gates(passing(eval_l0=3.0), cfg, 8)["not_dense"] is False
    assert check_gates(passing(eval_l0=3.0), cfg, 16)["not_dense"] is True


# ----------------------------------------------------------------------
# the reconstruction gate against the linear fit
#
# What is reachable at a depth is a property of the model pair and the depth.
# A fixed threshold asks more of a depth where little transfers than of one
# where everything does, and the same coder then passes or fails on where it
# was placed rather than on what it learned.
# ----------------------------------------------------------------------

def test_the_reconstruction_gate_moves_with_the_fit(cfg):
    """The same coder passes where little transfers and fails where much does."""
    cfg.eval.fit_fraction = 0.8
    cfg.eval.linear_fit = 0.5
    assert check_gates(passing(eval_explained_variance=0.45), cfg, 64)["reconstructs"] is True
    cfg.eval.linear_fit = 1.0
    assert check_gates(passing(eval_explained_variance=0.45), cfg, 64)["reconstructs"] is False


def test_the_threshold_is_the_fraction_of_the_fit(cfg):
    """Eight tenths of a fit of six tenths, and the line itself counts as passing.

    The threshold is computed rather than written out, because a coder landing
    exactly on it is the case the comparison has to get right and a literal
    0.48 is not the same double as 0.8 times 0.6.
    """
    cfg.eval.fit_fraction = 0.8
    cfg.eval.linear_fit = 0.6
    threshold = cfg.eval.fit_fraction * cfg.eval.linear_fit
    assert check_gates(passing(eval_explained_variance=threshold),
                       cfg, 64)["reconstructs"] is True
    assert check_gates(passing(eval_explained_variance=0.47), cfg, 64)["reconstructs"] is False
    assert check_gates(passing(eval_explained_variance=0.49), cfg, 64)["reconstructs"] is True


def test_beating_the_linear_fit_is_allowed(cfg):
    """The fit is a floor, not a ceiling. Nothing caps the coder at it."""
    cfg.eval.linear_fit = 0.4
    assert check_gates(passing(eval_explained_variance=0.9), cfg, 64)["passed"] is True


def test_the_whole_fit_can_be_demanded(cfg):
    """A fraction of one asks the coder to match the linear map exactly."""
    cfg.eval.fit_fraction = 1.0
    cfg.eval.linear_fit = 0.6
    assert check_gates(passing(eval_explained_variance=0.6), cfg, 64)["reconstructs"] is True
    assert check_gates(passing(eval_explained_variance=0.59), cfg, 64)["reconstructs"] is False


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


def test_evaluate_judges_density_against_the_output_width(cfg):
    """The hook passes the reconstruction's width, not the dictionary's.

    Eight features active, an eight-wide activation, a thirty-two entry
    dictionary. Dense against the activation, sparse against the dictionary.
    """
    metrics = evaluate(PerfectCoder(active=8), stream(), KEY, KEY, cfg)
    assert metrics["eval_l0"] == pytest.approx(8.0)
    assert metrics["gate_not_dense"] == 0.0


def test_evaluate_records_what_it_judged_against(cfg):
    """A result carries the comparison, not only the side of it it produced."""
    cfg.eval.linear_fit = 0.5
    metrics = evaluate(PerfectCoder(), stream(), KEY, KEY, cfg)
    assert metrics["eval_linear_fit"] == pytest.approx(0.5)
    assert metrics["eval_fit_ratio"] == pytest.approx(2.0, abs=1e-4)


def test_the_ratio_is_against_the_fit_not_against_one(cfg):
    """A coder reaching exactly the fit has a ratio of one, whatever the fit is."""
    cfg.eval.linear_fit = 1.0
    metrics = evaluate(PerfectCoder(), stream(), KEY, KEY, cfg)
    assert metrics["eval_fit_ratio"] == pytest.approx(1.0, abs=1e-4)