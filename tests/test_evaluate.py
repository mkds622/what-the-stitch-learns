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
    check_gates, evaluate, feature_frequencies, frequency_concentration,
    reconstruction_metrics, substitution_accuracy,
)
from stitch.sweep import LinearStitch                                   # noqa: E402

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


def test_sparsity_is_reported_against_the_activation_width():
    """Three active features describing an eight-wide activation."""
    metrics = reconstruction_metrics(PerfectCoder(active=3), stream(), KEY, KEY, count=4)
    assert metrics["eval_l0_frac"] == pytest.approx(3 / WIDTH)


def test_effective_expansion_counts_only_features_that_fired():
    """The configured dictionary is four times the width; one entry is in use."""
    metrics = reconstruction_metrics(PerfectCoder(active=1), stream(), KEY, KEY, count=4)
    assert metrics["eval_effective_expansion"] == pytest.approx(1 / WIDTH)


def test_a_fully_alive_dictionary_reports_its_configured_expansion():
    metrics = reconstruction_metrics(DenseCoder(), stream(), KEY, KEY, count=4)
    assert metrics["eval_effective_expansion"] == pytest.approx(LATENT / WIDTH)


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
# how the firing is spread
#
# L0 says how many features fire per token and nothing about whether the same
# few fire every time. These pin the measure that does, against distributions
# whose answer is known by construction.
# ----------------------------------------------------------------------

def test_a_dictionary_sharing_the_work_evenly_is_not_concentrated():
    result = frequency_concentration(torch.full((64,), 0.25))
    assert result["gini"] == pytest.approx(0.0, abs=1e-9)
    assert result["median"] == pytest.approx(0.25)


def test_one_feature_doing_everything_is_concentrated():
    frequencies = torch.zeros(64)
    frequencies[0] = 1.0
    frequencies[1:] = 1e-9
    assert frequency_concentration(frequencies)["gini"] > 0.95


def test_dead_features_are_not_counted():
    """They have their own metric, and counting them here would repeat it."""
    even = torch.full((32,), 0.25)
    padded = torch.cat([even, torch.zeros(96)])
    assert frequency_concentration(padded) == frequency_concentration(even)


def test_concentration_sits_between_the_two_extremes():
    """Half the live features at four times the rate of the other half."""
    frequencies = torch.cat([torch.full((32,), 0.4), torch.full((32,), 0.1)])
    gini = frequency_concentration(frequencies)["gini"]
    assert 0.0 < gini < 1.0


def test_the_median_is_the_typical_live_feature():
    frequencies = torch.tensor([0.0, 0.0, 0.1, 0.2, 0.3])
    assert frequency_concentration(frequencies)["median"] == pytest.approx(0.2)


def test_a_dictionary_that_never_fired_is_not_concentrated():
    """A collapsed coder is caught by its own gate, not by this reading as 1.0."""
    assert frequency_concentration(torch.zeros(16)) == {
        "gini": 0.0, "median": 0.0, "effective": 0.0}


def test_concentration_is_scale_free():
    """Doubling every rate changes how often features fire, not how unevenly."""
    frequencies = torch.tensor([0.05, 0.1, 0.2, 0.4])
    assert frequency_concentration(frequencies * 2)["gini"] == pytest.approx(
        frequency_concentration(frequencies)["gini"])
    assert frequency_concentration(frequencies * 2)["effective"] == pytest.approx(
        frequency_concentration(frequencies)["effective"])


# ----------------------------------------------------------------------
# how many features the firing is spread over
#
# The Gini coefficient is a share, so it reads differently at a thousand live
# features than at five thousand and cannot be compared between the two. These
# pin the count that can: it is in features, and its value is known exactly for
# a distribution whose shape is known.
# ----------------------------------------------------------------------

def test_an_even_dictionary_is_effectively_its_own_size():
    """Equal firing over m features is m features' worth of spread, exactly."""
    for m in (1, 8, 64, 1000):
        result = frequency_concentration(torch.full((m,), 0.25))
        assert result["effective"] == pytest.approx(float(m), rel=1e-9)


def test_dead_features_do_not_change_the_count():
    """A wider dictionary with the same live features spreads no further."""
    live = torch.full((32,), 0.25)
    padded = torch.cat([live, torch.zeros(4096)])
    assert frequency_concentration(padded)["effective"] == pytest.approx(32.0)


def test_one_feature_doing_everything_is_one_feature():
    frequencies = torch.zeros(4096)
    frequencies[0] = 1.0
    frequencies[1:] = 1e-12
    assert frequency_concentration(frequencies)["effective"] == pytest.approx(1.0, abs=0.01)


def test_a_large_dictionary_carried_by_a_few_reports_the_few():
    """The reading the live count cannot give and the Gini coefficient cannot compare.

    A thousand live features, ten of which do almost all the firing. The live
    count says a thousand, which is the number the covariance control is
    compared against and the number that would make a concentrated dictionary
    look like a spread one.
    """
    frequencies = torch.full((1000,), 1e-6)
    frequencies[:10] = 0.5
    effective = frequency_concentration(frequencies)["effective"]
    assert 10.0 <= effective <= 12.0


def test_the_count_never_exceeds_the_live_count():
    """Its ceiling is the number of features that fired, reached only when even."""
    frequencies = torch.cat([torch.full((16,), 0.4), torch.full((16,), 0.05)])
    result = frequency_concentration(frequencies)
    assert 1.0 < result["effective"] < 32.0


def test_the_count_and_the_gini_disagree_about_which_dictionary_is_concentrated():
    """Why the count was added: the same Gini coefficient at two live counts.

    Both dictionaries have half their live features firing at four times the
    rate of the other half, so they are equally uneven and their Gini
    coefficients match. One of them spreads its firing over eight times as
    many features, which only the count says.
    """
    small = torch.cat([torch.full((16,), 0.4), torch.full((16,), 0.1)])
    large = torch.cat([torch.full((128,), 0.4), torch.full((128,), 0.1)])

    assert frequency_concentration(large)["gini"] == pytest.approx(
        frequency_concentration(small)["gini"], abs=0.01)
    assert (frequency_concentration(large)["effective"]
            == pytest.approx(8.0 * frequency_concentration(small)["effective"], rel=1e-6))


def test_the_metrics_carry_the_concentration():
    """One feature in use out of thirty-two, so what is live is perfectly even."""
    metrics = reconstruction_metrics(PerfectCoder(active=1), stream(), KEY, KEY, count=4)
    assert metrics["eval_frequency_gini"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["eval_median_feature_frequency"] == pytest.approx(1.0)
    assert metrics["eval_effective_features"] == pytest.approx(1.0)


def test_the_metrics_carry_the_effective_count_in_features():
    """Three features firing on every token, so the firing spreads over three."""
    metrics = reconstruction_metrics(PerfectCoder(active=3), stream(), KEY, KEY, count=4)
    assert metrics["eval_effective_features"] == pytest.approx(3.0)
    assert metrics["eval_effective_expansion"] == pytest.approx(3 / WIDTH)


def test_concentration_is_reported_even_for_a_dead_coder():
    metrics = reconstruction_metrics(ZeroCoder(), stream(), KEY, KEY, count=4)
    assert metrics["eval_frequency_gini"] == 0.0
    assert metrics["eval_median_feature_frequency"] == 0.0
    assert metrics["eval_effective_features"] == 0.0


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
# the baseline put through the same substitution
#
# Agreement on its own is a number with no scale. These fix the reference it
# is read against: the same rank-limited map whose explained variance the run
# already records, substituted at the same cut point, in the same loop.
#
# The map is fitted in the activation's own units while the coder works in
# scaled ones, so the baseline must read the activation unscaled and have no
# scale undone on its output. That is the part a later change could silently
# break, and the scale test below is what catches it.
# ----------------------------------------------------------------------

def identity_map(width: int = WIDTH) -> LinearStitch:
    """A baseline that returns the activation untouched.

    The real thing is a rank-limited linear map. This is the degenerate case
    of one, and the only case whose effect on a prediction is known exactly.
    """
    return LinearStitch(torch.eye(width), torch.zeros(width))


def test_without_a_baseline_no_baseline_fields_are_reported(loader):
    """Absent rather than zero: a run with no baseline is not a run whose
    baseline destroyed the prediction."""
    result = substitution_accuracy(PerfectCoder(), Upper(), Lower(), loader,
                                   scale_in=1.0, scale_out=1.0, max_images=32)
    assert not any(key.startswith("baseline_") for key in result)


def test_an_exact_baseline_changes_nothing(loader):
    result = substitution_accuracy(PerfectCoder(), Upper(), Lower(), loader,
                                   scale_in=1.0, scale_out=1.0, max_images=32,
                                   baseline=identity_map())
    assert result["baseline_agreement"] == 1.0
    assert result["baseline_top1_drop"] == 0.0
    assert result["baseline_substituted_top1"] == result["plain_top1"]


def test_a_useless_baseline_is_visible(loader):
    """A map that returns zero should move the prediction, as the coder would."""
    dead = LinearStitch(torch.zeros(WIDTH, WIDTH), torch.zeros(WIDTH))
    result = substitution_accuracy(PerfectCoder(), Upper(), Lower(), loader,
                                   scale_in=1.0, scale_out=1.0, max_images=32,
                                   baseline=dead)
    assert result["baseline_agreement"] < 1.0
    assert result["agreement"] == 1.0


def test_the_baseline_reads_the_activation_in_its_own_units(loader):
    """The scale belongs to the coder and must not reach the baseline.

    The sweep that fitted the map never scaled anything, so the map is in the
    activation's units. Handing it the coder's scaled input would put it in a
    space it was never fitted for, and because the lower half has a bias that
    shows up as a changed prediction rather than as an error.
    """
    result = substitution_accuracy(PerfectCoder(), Upper(), Lower(), loader,
                                   scale_in=7.5, scale_out=7.5, max_images=32,
                                   baseline=identity_map())
    assert result["baseline_agreement"] == 1.0
    assert result["agreement"] == 1.0


def test_the_baseline_is_judged_against_the_same_reference_prediction(scattered_loader):
    """`plain` is the target model cut and rejoined, for the baseline too.

    With the halves taken from different models the reference comes from
    ``reference_upper``. The baseline is substituted at the cut like the coder,
    so it reads the source model's activation and cannot reproduce that
    reference, which is the honest reading rather than a flattering one.
    """
    result = substitution_accuracy(PerfectCoder(), ZeroUpper(), ScatterLower(),
                                   scattered_loader, scale_in=1.0, scale_out=1.0,
                                   max_images=20,
                                   label_map=torch.tensor(SUBSET_TO_FULL),
                                   reference_upper=Identity(),
                                   baseline=identity_map())
    assert result["plain_top1"] == 1.0
    assert result["baseline_agreement"] < 1.0


def test_the_baseline_does_not_disturb_the_coder_numbers(loader):
    """Adding a reference must not move the thing it is a reference for."""
    upper, lower = Upper(), Lower()                 # one model, used twice
    without = substitution_accuracy(PerfectCoder(), upper, lower, loader,
                                    scale_in=1.0, scale_out=1.0, max_images=32)
    with_baseline = substitution_accuracy(PerfectCoder(), upper, lower, loader,
                                          scale_in=1.0, scale_out=1.0, max_images=32,
                                          baseline=identity_map())
    assert all(with_baseline[key] == value for key, value in without.items())


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