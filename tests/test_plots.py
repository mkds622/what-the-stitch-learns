"""Tests for reading a grid as a curve.

The shaping is tested against curves whose answer is known by construction: one
that stops moving at a stated point, one that never does. Drawing is checked
only for producing a file, since asserting about pixels tests nothing useful.

    python -m pytest tests/test_plots.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stitch.plots import (                                            # noqa: E402
    Point, combine_results, flattening_point, noise_from, plot_curve,
    plot_feature_frequencies, plot_tradeoff, series, split_series,
)


def grid(values: dict[float, list[float]], metric="eval_l0", x="train.tokens"):
    """Records placing the given metric values at the given positions."""
    out = []
    for position, readings in values.items():
        for seed, reading in enumerate(readings):
            out.append({"point": {x: position, "train.seed": seed}, metric: reading})
    return out


# ----------------------------------------------------------------------
# shaping
# ----------------------------------------------------------------------

def test_one_point_per_position():
    points = series(grid({1e6: [10.0, 12.0], 2e6: [8.0]}), "train.tokens", "eval_l0")
    assert [point.x for point in points] == [1e6, 2e6]
    assert [point.n for point in points] == [2, 1]


def test_points_come_out_in_order():
    points = series(grid({4e6: [1.0], 1e6: [3.0], 2e6: [2.0]}), "train.tokens", "eval_l0")
    assert [point.x for point in points] == [1e6, 2e6, 4e6]


def test_a_position_is_averaged_over_its_runs():
    point = series(grid({1e6: [10.0, 20.0, 30.0]}), "train.tokens", "eval_l0")[0]
    assert point.mean == pytest.approx(20.0)
    assert (point.low, point.high) == (10.0, 30.0)
    assert point.std == pytest.approx(10.0)


def test_a_single_run_has_no_spread():
    assert series(grid({1e6: [10.0]}), "train.tokens", "eval_l0")[0].std is None


def test_failed_runs_are_left_out():
    records = grid({1e6: [10.0, 12.0]})
    records.append({"point": {"train.tokens": 1e6, "train.seed": 9}, "error": "boom"})
    assert series(records, "train.tokens", "eval_l0")[0].n == 2


def test_records_without_the_metric_are_left_out():
    records = grid({1e6: [10.0]})
    records.append({"point": {"train.tokens": 1e6, "train.seed": 9}, "eval_mse": 0.1})
    assert series(records, "train.tokens", "eval_l0")[0].n == 1


def test_records_off_the_axis_are_left_out():
    records = grid({1e6: [10.0]})
    records.append({"point": {"coder.expansion": 8}, "eval_l0": 99.0})
    assert len(series(records, "train.tokens", "eval_l0")) == 1


def test_an_empty_grid_gives_an_empty_series():
    assert series([], "train.tokens", "eval_l0") == []


# ----------------------------------------------------------------------
# several grids as one curve
#
# A sweep run in two sittings is one curve. The hazard is that a grid reuses a
# run it has already trained rather than training it again, so the same run can
# be recorded in two files, and counted twice it reads as two runs of one
# configuration agreeing exactly. That is a spread of zero where there is no
# spread, and the tolerance a budget curve is read against comes from exactly
# those spreads.
# ----------------------------------------------------------------------

def named(name: str, position: float, value: float, metric="eval_l0",
          x="train.tokens") -> dict:
    """A record carrying the run name a grid writes with it."""
    return {"run_name": name, "point": {x: position}, metric: value}


def test_two_grids_read_as_one_curve():
    first = [named("a", 1e6, 10.0), named("b", 2e6, 8.0)]
    second = [named("c", 4e6, 7.0)]
    points = series(combine_results([first, second]), "train.tokens", "eval_l0")
    assert [point.x for point in points] == [1e6, 2e6, 4e6]


def test_one_grid_is_unchanged():
    records = [named("a", 1e6, 10.0), named("b", 2e6, 8.0)]
    assert combine_results([records]) == records


def test_no_grids_give_nothing():
    assert combine_results([]) == []
    assert combine_results([[], []]) == []


def test_a_run_in_two_grids_is_counted_once():
    """The reuse case: the second grid records a run the first already had."""
    first = [named("a", 1e6, 10.0), named("b", 2e6, 8.0)]
    second = [named("b", 2e6, 8.0), named("c", 4e6, 7.0)]
    combined = combine_results([first, second])
    assert [record["run_name"] for record in combined] == ["a", "b", "c"]


def test_a_duplicate_does_not_invent_a_spread():
    """What the deduplication is for, stated as the number it protects.

    Counted twice, one run becomes two runs that agree exactly, and the
    tolerance a curve is read against is the largest spread any configuration
    shows. A zero where there should be no reading at all shrinks it.
    """
    twice = [[named("a", 1e6, 10.0)], [named("a", 1e6, 10.0)]]

    naive = [record for group in twice for record in group]
    assert series(naive, "train.tokens", "eval_l0")[0].n == 2
    assert series(naive, "train.tokens", "eval_l0")[0].std == pytest.approx(0.0)
    assert noise_from(naive, "eval_l0") == pytest.approx(0.0)

    combined = combine_results(twice)
    assert series(combined, "train.tokens", "eval_l0")[0].n == 1
    assert series(combined, "train.tokens", "eval_l0")[0].std is None
    assert noise_from(combined, "eval_l0") is None


def test_genuine_seed_repeats_still_average_together():
    """Different runs at one position are the spread, not a duplicate."""
    first = [named("s0", 1e6, 10.0)]
    second = [named("s1", 1e6, 20.0)]
    point = series(combine_results([first, second]), "train.tokens", "eval_l0")[0]
    assert (point.n, point.mean, point.std) == (2, 15.0, pytest.approx(7.071, abs=1e-3))


def test_the_first_record_of_a_run_is_the_one_kept():
    first = [named("a", 1e6, 10.0)]
    second = [named("a", 1e6, 99.0)]
    assert combine_results([first, second])[0]["eval_l0"] == 10.0


def test_records_without_a_name_are_all_kept():
    """Nothing identifies them as repeats, so dropping them would lose runs."""
    plain = {"point": {"train.tokens": 1e6}, "eval_l0": 10.0}
    assert len(combine_results([[dict(plain)], [dict(plain)]])) == 2


def test_a_failed_run_is_still_deduplicated():
    """It carries a name and no metric, and series drops it either way."""
    failed = {"run_name": "a", "point": {"train.tokens": 1e6}, "error": "boom"}
    assert len(combine_results([[failed], [dict(failed)]])) == 1


def test_combining_takes_any_iterable():
    """The script passes a generator over its --results paths."""
    groups = ([named("a", 1e6, 10.0)], [named("b", 2e6, 8.0)])
    assert len(combine_results(group for group in groups)) == 2


# ----------------------------------------------------------------------
# a grid that varied two things
#
# series averages over everything that is not the axis, which turns a second
# setting into part of the mean. These keep it as the comparison instead.
# ----------------------------------------------------------------------

def two_axis_grid():
    """Two expansions, two coefficients, a different metric value at each."""
    return [{"point": {"coder.sparsity.coeff": c, "coder.expansion": e}, "eval_l0": v}
            for e, c, v in ((8, 1.0, 40.0), (8, 2.0, 20.0),
                            (16, 1.0, 90.0), (16, 2.0, 50.0))]


def test_each_value_gets_its_own_curve():
    curves = split_series(two_axis_grid(), "coder.sparsity.coeff", "eval_l0",
                          "coder.expansion")
    assert list(curves) == ["8", "16"]
    assert [point.mean for point in curves["8"]] == [40.0, 20.0]
    assert [point.mean for point in curves["16"]] == [90.0, 50.0]


def test_the_curves_are_ordered_numerically():
    """Labels are strings, so plain sorting would put 16 before 8."""
    curves = split_series(two_axis_grid() + [
        {"point": {"coder.sparsity.coeff": 1.0, "coder.expansion": 32}, "eval_l0": 1.0}],
        "coder.sparsity.coeff", "eval_l0", "coder.expansion")
    assert list(curves) == ["8", "16", "32"]


def test_splitting_keeps_what_averaging_would_have_lost():
    """One curve through the mean of both expansions says nothing about either."""
    averaged = series(two_axis_grid(), "coder.sparsity.coeff", "eval_l0")
    assert [point.n for point in averaged] == [2, 2]
    assert averaged[0].mean == pytest.approx(65.0)

    curves = split_series(two_axis_grid(), "coder.sparsity.coeff", "eval_l0",
                          "coder.expansion")
    assert all(point.n == 1 for curve in curves.values() for point in curve)


def test_records_without_the_split_setting_are_left_out():
    records = two_axis_grid() + [{"point": {"coder.sparsity.coeff": 1.0}, "eval_l0": 7.0}]
    curves = split_series(records, "coder.sparsity.coeff", "eval_l0", "coder.expansion")
    assert sum(point.n for curve in curves.values() for point in curve) == 4


def test_failed_runs_are_left_out_of_a_split():
    records = two_axis_grid() + [
        {"point": {"coder.sparsity.coeff": 1.0, "coder.expansion": 8}, "error": "boom"}]
    curves = split_series(records, "coder.sparsity.coeff", "eval_l0", "coder.expansion")
    assert [point.n for point in curves["8"]] == [1, 1]


def test_splitting_an_empty_grid_gives_no_curves():
    assert split_series([], "coder.sparsity.coeff", "eval_l0", "coder.expansion") == {}


# ----------------------------------------------------------------------
# where it flattens
# ----------------------------------------------------------------------

def curve(*means):
    return [Point(x=float(i + 1), mean=value, std=None, low=value, high=value, n=1)
            for i, value in enumerate(means)]


def test_a_curve_that_stops_moving_is_found():
    """Steps of 10, 5, then 1: it flattens at the fourth point."""
    assert flattening_point(curve(100, 90, 85, 84), tolerance=2.0) == 4.0


def test_the_first_flat_step_is_the_answer_not_the_last():
    assert flattening_point(curve(100, 99, 98, 97), tolerance=2.0) == 2.0


def test_a_curve_still_moving_returns_nothing():
    """Which means the budget was too small to see the end of it."""
    assert flattening_point(curve(100, 80, 60, 40), tolerance=5.0) is None


def test_a_rising_curve_flattens_too():
    """Explained variance goes up where sparsity goes down."""
    assert flattening_point(curve(0.5, 0.7, 0.8, 0.805), tolerance=0.01) == 4.0


def test_one_point_cannot_flatten():
    assert flattening_point(curve(100), tolerance=1.0) is None


def test_no_points_cannot_flatten():
    assert flattening_point([], tolerance=1.0) is None


def test_a_zero_tolerance_needs_exact_agreement():
    assert flattening_point(curve(100, 100), tolerance=0.0) == 2.0
    assert flattening_point(curve(100, 99.999), tolerance=0.0) is None


def test_a_negative_tolerance_raises():
    with pytest.raises(ValueError, match="negative"):
        flattening_point(curve(1, 2), tolerance=-1.0)


# ----------------------------------------------------------------------
# the tolerance itself
# ----------------------------------------------------------------------

def test_the_tolerance_comes_from_the_seed_spread():
    noise = noise_from(grid({1e6: [10.0, 20.0, 30.0]}), "eval_l0")
    assert noise == pytest.approx(10.0)


def test_the_noisiest_configuration_sets_it():
    records = grid({1e6: [10.0, 10.5], 2e6: [5.0, 25.0]})
    assert noise_from(records, "eval_l0") == pytest.approx(14.142, abs=1e-2)


def test_without_repeats_there_is_no_tolerance():
    """Nothing is known about the spread from one run."""
    assert noise_from(grid({1e6: [10.0]}), "eval_l0") is None


# ----------------------------------------------------------------------
# drawing
# ----------------------------------------------------------------------

def test_a_curve_is_written(tmp_path):
    pytest.importorskip("matplotlib")
    path = plot_curve(series(grid({1e6: [100.0, 102.0], 2e6: [90.0], 4e6: [89.0]}),
                             "train.tokens", "eval_l0"),
                      "eval_l0", "tokens", tmp_path / "figures" / "curve.png",
                      tolerance=2.0)
    assert path.exists() and path.stat().st_size > 0


def test_an_empty_curve_raises(tmp_path):
    pytest.importorskip("matplotlib")
    with pytest.raises(ValueError, match="nothing to plot"):
        plot_curve([], "eval_l0", "tokens", tmp_path / "curve.png")


def test_several_curves_are_written(tmp_path):
    pytest.importorskip("matplotlib")
    curves = split_series(two_axis_grid(), "coder.sparsity.coeff", "eval_l0",
                          "coder.expansion")
    path = plot_curve(curves, "eval_l0", "coeff", tmp_path / "split.png",
                      legend_title="expansion")
    assert path.exists() and path.stat().st_size > 0


def test_curves_with_a_tolerance_are_written(tmp_path):
    """Each curve gets its own flattening mark rather than one for all of them."""
    pytest.importorskip("matplotlib")
    curves = split_series(two_axis_grid(), "coder.sparsity.coeff", "eval_l0",
                          "coder.expansion")
    assert plot_curve(curves, "eval_l0", "coeff", tmp_path / "split_tol.png",
                      tolerance=25.0).exists()


def test_an_empty_set_of_curves_raises(tmp_path):
    pytest.importorskip("matplotlib")
    with pytest.raises(ValueError, match="nothing to plot"):
        plot_curve({"8": [], "16": []}, "eval_l0", "tokens", tmp_path / "c.png")


def test_a_tradeoff_is_written(tmp_path):
    pytest.importorskip("matplotlib")
    records = [{"point": {"coder.expansion": e}, "eval_l0": l, "eval_explained_variance": v}
               for e, l, v in ((8, 40.0, 0.80), (16, 55.0, 0.86), (32, 70.0, 0.88))]
    path = plot_tradeoff(records, tmp_path / "tradeoff.png", label_by="coder.expansion")
    assert path.exists() and path.stat().st_size > 0


def test_a_tradeoff_without_labels_is_written(tmp_path):
    pytest.importorskip("matplotlib")
    records = [{"point": {}, "eval_l0": 40.0, "eval_explained_variance": 0.8}]
    assert plot_tradeoff(records, tmp_path / "t.png").exists()


def test_an_empty_tradeoff_raises(tmp_path):
    pytest.importorskip("matplotlib")
    with pytest.raises(ValueError, match="nothing to plot"):
        plot_tradeoff([], tmp_path / "t.png")


# ----------------------------------------------------------------------
# the frequency distribution
# ----------------------------------------------------------------------

def test_a_frequency_distribution_is_written(tmp_path):
    pytest.importorskip("matplotlib")
    path = plot_feature_frequencies([0.5, 0.2, 0.05, 0.0, 0.0],
                                    tmp_path / "figures" / "freq.png",
                                    title="block 6")
    assert path.exists() and path.stat().st_size > 0


def test_several_distributions_are_written(tmp_path):
    pytest.importorskip("matplotlib")
    path = plot_feature_frequencies({"coeff 12": [0.4, 0.3, 0.2, 0.1],
                                     "coeff 48": [0.9, 0.01, 0.0, 0.0]},
                                    tmp_path / "freqs.png",
                                    legend_title="setting")
    assert path.exists() and path.stat().st_size > 0


def test_a_dictionary_that_never_fired_raises(tmp_path):
    """A logarithmic axis has nothing to show, and an empty figure would read
    as a drawing failure rather than a dead coder."""
    pytest.importorskip("matplotlib")
    with pytest.raises(ValueError, match="no feature ever fired"):
        plot_feature_frequencies([0.0, 0.0, 0.0], tmp_path / "dead.png")


def test_one_dead_curve_among_live_ones_is_fine(tmp_path):
    pytest.importorskip("matplotlib")
    assert plot_feature_frequencies({"alive": [0.3, 0.1], "dead": [0.0, 0.0]},
                                    tmp_path / "mixed.png").exists()


# ----------------------------------------------------------------------
# the comparison figure
# ----------------------------------------------------------------------

def comparison_summary(**overrides):
    """A summary of the shape ``stitch.compare.summarise`` produces."""
    bins = 20
    summary = {
        "random": 0.19,
        "within": {"n": 10, "mean": 0.72, "std": 0.01, "min": 0.70, "max": 0.74},
        "across": {"n": 25, "mean": 0.45, "std": 0.02, "min": 0.42, "max": 0.48},
        "random_shape": [1.0 / bins] * bins,
        "within_shape": [1.0 / bins] * bins,
        "across_shape": [1.0 / bins] * bins,
    }
    summary.update(overrides)
    return summary


def test_the_comparison_figure_is_written(tmp_path):
    pytest.importorskip("matplotlib")
    from stitch.plots import plot_comparison
    path = plot_comparison(comparison_summary(), tmp_path / "figures" / "c.png",
                           title="block 6")
    assert path.exists() and path.stat().st_size > 0


def test_a_summary_without_distributions_raises(tmp_path):
    pytest.importorskip("matplotlib")
    from stitch.plots import plot_comparison
    with pytest.raises(ValueError, match="no distributions"):
        plot_comparison({"random": 0.1}, tmp_path / "c.png")


def test_the_figure_draws_without_a_measurement_to_place(tmp_path):
    """Seed repeats alone, before the grid: the upper panel still means something."""
    pytest.importorskip("matplotlib")
    from stitch.plots import plot_comparison
    summary = comparison_summary(across={"n": 0})
    summary.pop("across_shape")
    assert plot_comparison(summary, tmp_path / "c.png").exists()

# ----------------------------------------------------------------------
# the rest of the comparison figures
#
# Drawing is checked for producing a file and for refusing to draw an empty
# one, since asserting about pixels tests nothing. What is worth testing is
# that a figure says no rather than writing a blank image, because a blank
# image downstream reads as a drawing bug rather than as missing data.
# ----------------------------------------------------------------------

def per_feature(count=120, high=True):
    """Matched scores of roughly the shape one kind of comparison gives."""
    step = 1.0 / count
    base = 0.6 if high else 0.2
    return [base + step * index * 0.4 for index in range(count)]


def pair_rows(count=40, score=0.8):
    """Triples of where a feature sits, where its partner sits, and the score."""
    return [[index / count, ((index * 7) % count) / count, score]
            for index in range(count)]


def block(size=12, strength=0.9):
    """A similarity block with its pairing on the diagonal."""
    return [[strength if row == column else 0.2 for column in range(size)]
            for row in range(size)]


def test_the_squares_figure_is_written(tmp_path):
    pytest.importorskip("matplotlib")
    from stitch.plots import plot_feature_squares
    path = plot_feature_squares({"within": per_feature(),
                                 "across": per_feature(high=False)},
                                tmp_path / "figures" / "dictionary.png")
    assert path.exists() and path.stat().st_size > 0


def test_the_squares_figure_draws_one_kind(tmp_path):
    """Seed repeats alone, before there is anything to compare them to."""
    pytest.importorskip("matplotlib")
    from stitch.plots import plot_feature_squares
    assert plot_feature_squares({"within": per_feature()},
                                tmp_path / "one.png").exists()


def test_the_squares_figure_handles_fewer_features_than_a_row(tmp_path):
    pytest.importorskip("matplotlib")
    from stitch.plots import plot_feature_squares
    assert plot_feature_squares({"within": [0.5, 0.6, 0.7]},
                                tmp_path / "short.png").exists()


def test_squares_without_scores_raise(tmp_path):
    pytest.importorskip("matplotlib")
    from stitch.plots import plot_feature_squares
    with pytest.raises(ValueError, match="nothing to plot"):
        plot_feature_squares({"within": [], "across": []}, tmp_path / "s.png")


def test_the_pairing_figure_is_written(tmp_path):
    pytest.importorskip("matplotlib")
    from stitch.plots import plot_pairing
    path = plot_pairing({"within": pair_rows(), "across": pair_rows(score=0.4)},
                        tmp_path / "figures" / "pairing.png", title="block 12")
    assert path.exists() and path.stat().st_size > 0


def test_the_pairing_figure_takes_scores_outside_the_colour_range(tmp_path):
    """A cosine can be negative, and a colour map cannot be indexed below zero."""
    pytest.importorskip("matplotlib")
    from stitch.plots import plot_pairing
    assert plot_pairing({"across": [[0.1, 0.9, -0.4], [0.5, 0.2, 1.0]]},
                        tmp_path / "signed.png").exists()


def test_a_pairing_without_pairs_raises(tmp_path):
    pytest.importorskip("matplotlib")
    from stitch.plots import plot_pairing
    with pytest.raises(ValueError, match="nothing to plot"):
        plot_pairing({"within": []}, tmp_path / "p.png")


def test_the_matrix_figure_is_written(tmp_path):
    pytest.importorskip("matplotlib")
    from stitch.plots import plot_match_matrix
    path = plot_match_matrix({"within": block(), "across": block(strength=0.5)},
                             tmp_path / "figures" / "matrix.png")
    assert path.exists() and path.stat().st_size > 0


def test_a_matrix_figure_without_blocks_raises(tmp_path):
    pytest.importorskip("matplotlib")
    from stitch.plots import plot_match_matrix
    with pytest.raises(ValueError, match="nothing to plot"):
        plot_match_matrix({}, tmp_path / "m.png")


def test_the_distributions_figure_is_written(tmp_path):
    pytest.importorskip("matplotlib")
    from stitch.plots import plot_distributions
    path = plot_distributions(comparison_summary(),
                              tmp_path / "figures" / "distributions.png",
                              title="block 12")
    assert path.exists() and path.stat().st_size > 0


def test_the_distributions_figure_draws_what_it_has(tmp_path):
    pytest.importorskip("matplotlib")
    from stitch.plots import plot_distributions
    summary = comparison_summary()
    summary.pop("across_shape")
    assert plot_distributions(summary, tmp_path / "part.png").exists()


def test_a_flat_distribution_does_not_divide_by_zero(tmp_path):
    """A histogram of a dictionary with nothing in it is all zeros, and the
    rows are scaled by their own peak."""
    pytest.importorskip("matplotlib")
    from stitch.plots import plot_distributions
    assert plot_distributions(comparison_summary(within_shape=[0.0] * 20),
                              tmp_path / "flat.png").exists()


def test_distributions_without_shapes_raise(tmp_path):
    pytest.importorskip("matplotlib")
    from stitch.plots import plot_distributions
    with pytest.raises(ValueError, match="no distributions"):
        plot_distributions({"random": 0.1}, tmp_path / "d.png")


def test_the_shares_figure_is_written(tmp_path):
    pytest.importorskip("matplotlib")
    from stitch.plots import plot_match_shares
    path = plot_match_shares(
        {"within": {"strong": 0.78, "partial": 0.17, "absent": 0.05},
         "across": {"strong": 0.34, "partial": 0.29, "absent": 0.37}},
        tmp_path / "figures" / "shares.png")
    assert path.exists() and path.stat().st_size > 0


def test_the_shares_figure_omits_a_band_it_was_not_given(tmp_path):
    pytest.importorskip("matplotlib")
    from stitch.plots import plot_match_shares
    assert plot_match_shares({"across": {"strong": 0.4}},
                             tmp_path / "partial.png").exists()


def test_shares_without_kinds_raise(tmp_path):
    pytest.importorskip("matplotlib")
    from stitch.plots import plot_match_shares
    with pytest.raises(ValueError, match="nothing to plot"):
        plot_match_shares({}, tmp_path / "s.png")


def test_the_position_figure_is_written(tmp_path):
    pytest.importorskip("matplotlib")
    from stitch.plots import plot_position
    path = plot_position(comparison_summary(),
                         tmp_path / "figures" / "position.png", title="block 12")
    assert path.exists() and path.stat().st_size > 0


def test_a_position_without_a_ceiling_raises(tmp_path):
    """Without seed repeats there is no scale to place the measurement on, and
    a figure drawn anyway would be a number presented as a conclusion."""
    pytest.importorskip("matplotlib")
    from stitch.plots import plot_position
    summary = comparison_summary()
    summary.pop("within")
    with pytest.raises(ValueError, match="no measurement to place"):
        plot_position(summary, tmp_path / "p.png")


def test_a_floor_and_a_ceiling_that_agree_do_not_divide_by_zero(tmp_path):
    pytest.importorskip("matplotlib")
    from stitch.plots import plot_position
    summary = comparison_summary(random=0.5, within={"n": 2, "mean": 0.5},
                                 across={"n": 2, "mean": 0.5})
    assert plot_position(summary, tmp_path / "flat.png").exists()


def test_the_bands_figure_is_written(tmp_path):
    pytest.importorskip("matplotlib")
    from stitch.plots import plot_frequency_bands
    path = plot_frequency_bands({"within": [0.74, 0.82, 0.88, 0.93],
                                 "across": [0.31, 0.44, 0.56, 0.68]},
                                tmp_path / "figures" / "bands.png")
    assert path.exists() and path.stat().st_size > 0


def test_the_bands_figure_draws_one_kind(tmp_path):
    pytest.importorskip("matplotlib")
    from stitch.plots import plot_frequency_bands
    assert plot_frequency_bands({"within": [0.7, 0.8]},
                                tmp_path / "one.png").exists()


def test_bands_of_different_lengths_are_cut_to_the_shorter(tmp_path):
    """Two comparisons split into different numbers of bands would otherwise
    be drawn against each other on an axis neither of them is on."""
    pytest.importorskip("matplotlib")
    from stitch.plots import plot_frequency_bands
    assert plot_frequency_bands({"within": [0.7, 0.8, 0.9], "across": [0.3, 0.4]},
                                tmp_path / "ragged.png").exists()


def test_bands_without_curves_raise(tmp_path):
    pytest.importorskip("matplotlib")
    from stitch.plots import plot_frequency_bands
    with pytest.raises(ValueError, match="nothing to plot"):
        plot_frequency_bands({"within": []}, tmp_path / "b.png")

# ----------------------------------------------------------------------
# the depth as an axis
#
# Every other axis is a setting the grid swept, recorded in a run's point. The
# depth is not: a grid is told which depth to train at and varies the seed, so
# the depth never reaches the point and the four depths were run as four
# separate grids. It survives in the layer each run recorded, which is what
# makes a figure across the depths possible at all.
# ----------------------------------------------------------------------

def at_depth(layer: str, *values: float, metric="eval_l0") -> list[dict]:
    """Records from one depth's grid, one per seed, as a grid writes them."""
    return [{"run_name": f"{layer}-s{seed}", "point": {"train.seed": seed},
             "scales": {layer: 2.3}, metric: value}
            for seed, value in enumerate(values)]


def test_the_depth_is_read_from_the_layer():
    from stitch.plots import depth_of
    assert depth_of({"scales": {"A@add_12": 2.3}}) == 12.0
    assert depth_of({"scales": {"A@add_4": 1.0}}) == 4.0


def test_a_cross_model_run_is_read_from_the_layer_it_reads():
    """It records two and reads the first, and both are the same depth."""
    from stitch.plots import depth_of
    assert depth_of({"scales": {"A@add_20": 1.0, "B@add_20": 1.1}}) == 20.0


def test_a_record_with_no_layer_has_no_depth():
    from stitch.plots import depth_of
    assert depth_of({"point": {"train.seed": 0}}) is None
    assert depth_of({"scales": {}}) is None


def test_a_layer_with_no_number_has_no_depth():
    from stitch.plots import depth_of
    assert depth_of({"scales": {"A@head": 1.0}}) is None


def test_four_grids_become_one_curve_across_the_depths():
    """The figure the four-depth design exists for, from four grids that were
    each run on their own."""
    records = combine_results([at_depth("A@add_4", 10.0, 12.0),
                               at_depth("A@add_12", 20.0, 22.0),
                               at_depth("A@add_20", 30.0, 30.0),
                               at_depth("A@add_24", 40.0, 44.0)])
    points = series(records, "depth", "eval_l0")
    assert [point.x for point in points] == [4.0, 12.0, 20.0, 24.0]
    assert [point.n for point in points] == [2, 2, 2, 2]
    assert points[0].mean == pytest.approx(11.0)


def test_the_seeds_at_a_depth_are_its_spread():
    points = series(at_depth("A@add_12", 10.0, 20.0, 30.0), "depth", "eval_l0")
    assert points[0].std == pytest.approx(10.0)
    assert (points[0].low, points[0].high) == (10.0, 30.0)


def test_depths_come_out_in_order_whatever_order_the_grids_were_given():
    records = combine_results([at_depth("A@add_24", 1.0),
                               at_depth("A@add_4", 2.0),
                               at_depth("A@add_12", 3.0)])
    points = series(records, "depth", "eval_l0")
    assert [point.x for point in points] == [4.0, 12.0, 24.0]


def test_a_record_without_a_layer_is_left_off_the_depth_axis():
    """Rather than counted at zero, which would put a phantom point at the
    left-hand end of every figure."""
    records = at_depth("A@add_12", 10.0) + [
        {"run_name": "x", "point": {"train.seed": 9}, "eval_l0": 99.0}]
    points = series(records, "depth", "eval_l0")
    assert [point.x for point in points] == [12.0]
    assert points[0].n == 1


def test_a_failed_run_is_left_off_the_depth_axis_too():
    records = at_depth("A@add_12", 10.0, 12.0) + [
        {"run_name": "x", "scales": {"A@add_12": 2.3}, "error": "boom"}]
    assert series(records, "depth", "eval_l0")[0].n == 2


def test_the_ordinary_axes_are_unchanged():
    """The depth is an addition, and nothing that worked before moves."""
    points = series(grid({1e6: [10.0, 12.0], 2e6: [8.0]}), "train.tokens", "eval_l0")
    assert [point.x for point in points] == [1e6, 2e6]
    assert [point.n for point in points] == [2, 1]


def test_a_depth_curve_is_written(tmp_path):
    pytest.importorskip("matplotlib")
    records = combine_results([at_depth("A@add_4", 10.0, 12.0),
                               at_depth("A@add_12", 20.0, 22.0),
                               at_depth("A@add_20", 30.0, 31.0),
                               at_depth("A@add_24", 40.0, 44.0)])
    path = plot_curve(series(records, "depth", "eval_l0"),
                      "eval_l0", "depth", tmp_path / "by_depth.png", log_x=False)
    assert path.exists() and path.stat().st_size > 0