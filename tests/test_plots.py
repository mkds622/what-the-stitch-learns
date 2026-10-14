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
    Point, flattening_point, noise_from, plot_curve, plot_tradeoff, series,
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