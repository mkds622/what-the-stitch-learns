"""Tests for the grid driver.

The runner is replaced with a function that returns made-up numbers, so the
grid logic is exercised without a backbone, a dataset or a GPU. Nothing here
uses one.

    python -m pytest tests/test_grid.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stitch.grid import (                                              # noqa: E402
    as_overrides, describe_point, expand, group_spread, parse_axis,
    read_results, run_grid, separated, table, write_results,
)


# ----------------------------------------------------------------------
# describing a grid
# ----------------------------------------------------------------------

def test_an_axis_is_a_setting_and_its_values():
    assert parse_axis("train.seed=0,1,2") == ("train.seed", [0, 1, 2])


def test_axis_values_keep_their_types():
    """A seed must arrive as a number, not the string '0'."""
    key, values = parse_axis("coder.sparsity.coeff=1e-3,5e-4")
    assert key == "coder.sparsity.coeff"
    assert values == [1e-3, 5e-4]
    assert all(isinstance(value, float) for value in values)


def test_a_single_value_is_a_valid_axis():
    assert parse_axis("objective=case1") == ("objective", ["case1"])


@pytest.mark.parametrize("bad", ["train.seed", "train.seed=", "=1,2"])
def test_a_malformed_axis_raises(bad):
    with pytest.raises(ValueError):
        parse_axis(bad)


def test_the_grid_is_the_product_of_its_axes():
    points = expand({"a": [1, 2], "b": ["x", "y", "z"]})
    assert len(points) == 6
    assert {"a": 1, "b": "x"} in points
    assert {"a": 2, "b": "z"} in points


def test_the_last_axis_varies_fastest():
    """So a partial grid reads in a sensible order."""
    points = expand({"a": [1, 2], "b": [10, 20]})
    assert points == [{"a": 1, "b": 10}, {"a": 1, "b": 20},
                      {"a": 2, "b": 10}, {"a": 2, "b": 20}]


def test_no_axes_is_one_point():
    """A grid of nothing is still one run, which is the useful behaviour."""
    assert expand({}) == [{}]


def test_a_point_becomes_overrides():
    assert as_overrides({"train.seed": 3}) == ["train.seed=3"]


def test_a_point_describes_itself_briefly():
    assert describe_point({"train.seed": 3, "coder.expansion": 8}) == "seed=3, expansion=8"


# ----------------------------------------------------------------------
# running a grid
# ----------------------------------------------------------------------

class FakeConfig:
    """Just enough configuration for the grid to name and place a run."""

    def __init__(self, overrides, root):
        self.overrides = dict(pair.split("=", 1) for pair in overrides)
        self.logging = type("L", (), {"run_root": str(root)})()

    def run_name(self):
        return "run-" + "-".join(f"{k}{v}" for k, v in sorted(self.overrides.items()))


@pytest.fixture
def grid(monkeypatch, tmp_path):
    """A grid whose runs return a number derived from the seed."""
    calls = []

    def fake_load(path, overrides):
        return FakeConfig(overrides, tmp_path / "runs")

    def fake_run(cfg):
        calls.append(cfg.run_name())
        seed = int(cfg.overrides.get("train.seed", 0))
        return {"mse": 0.5 - 0.01 * seed, "l0": 20.0}

    import stitch.config
    monkeypatch.setattr(stitch.config, "load_config", fake_load)
    return calls, fake_run


def test_every_point_is_run(grid, tmp_path):
    calls, runner = grid
    results = run_grid(None, [], {"train.seed": [0, 1, 2]},
                       tmp_path / "grid.json", runner=runner)

    assert len(calls) == 3
    assert len(results) == 3
    assert all("mse" in record for record in results)


def test_each_record_carries_its_coordinates(grid, tmp_path):
    _, runner = grid
    results = run_grid(None, [], {"train.seed": [0, 1]},
                       tmp_path / "grid.json", runner=runner)
    assert [record["point"]["train.seed"] for record in results] == [0, 1]


def test_results_are_written_as_the_grid_goes(grid, tmp_path):
    """A grid of thirty runs must not lose everything to a crash at twenty-nine."""
    _, runner = grid
    out = tmp_path / "grid.json"

    def runner_then_fail(cfg):
        if "train.seed1" in cfg.run_name():
            raise RuntimeError("deliberate")
        return runner(cfg)

    run_grid(None, [], {"train.seed": [0, 1, 2]}, out, runner=runner_then_fail)
    stored = json.loads(out.read_text())
    assert len(stored) == 3


def test_a_failed_point_is_recorded_and_the_grid_continues(grid, tmp_path):
    _, runner = grid

    def sometimes(cfg):
        if "train.seed1" in cfg.run_name():
            raise RuntimeError("deliberate failure")
        return runner(cfg)

    results = run_grid(None, [], {"train.seed": [0, 1, 2]},
                       tmp_path / "grid.json", runner=sometimes)

    assert "deliberate failure" in results[1]["error"]
    assert "mse" in results[0] and "mse" in results[2]


def test_a_point_already_run_is_reused(grid, tmp_path):
    """So an interrupted grid continues rather than starting again."""
    calls, runner = grid

    done = tmp_path / "runs" / "run-train.seed1" / "summary.json"
    done.parent.mkdir(parents=True)
    done.write_text(json.dumps({"mse": 0.123}))

    results = run_grid(None, [], {"train.seed": [0, 1, 2]},
                       tmp_path / "grid.json", runner=runner)

    assert len(calls) == 2
    assert results[1]["reused"] is True
    assert results[1]["mse"] == 0.123


def test_reuse_can_be_turned_off(grid, tmp_path):
    calls, runner = grid

    done = tmp_path / "runs" / "run-train.seed1" / "summary.json"
    done.parent.mkdir(parents=True)
    done.write_text(json.dumps({"mse": 0.123}))

    run_grid(None, [], {"train.seed": [0, 1, 2]}, tmp_path / "grid.json",
             skip_existing=False, runner=runner)
    assert len(calls) == 3


def test_results_survive_a_round_trip(grid, tmp_path):
    _, runner = grid
    out = tmp_path / "grid.json"
    results = run_grid(None, [], {"train.seed": [0, 1]}, out, runner=runner)
    assert read_results(out) == results


# ----------------------------------------------------------------------
# the noise floor
# ----------------------------------------------------------------------

def records(**groups):
    """Build grid records from {expansion: [values across seeds]}."""
    out = []
    for expansion, values in groups.items():
        for seed, value in enumerate(values):
            out.append({"point": {"coder.expansion": int(expansion[1:]),
                                  "train.seed": seed},
                        "mse": value})
    return out


def test_seeds_collapse_into_one_group():
    spread = group_spread(records(x8=[0.5, 0.52, 0.48]), "mse")
    assert len(spread) == 1
    assert spread[0]["n"] == 3
    assert spread[0]["held"] == {"coder.expansion": 8}


def test_other_axes_stay_separate():
    spread = group_spread(records(x8=[0.5, 0.5], x16=[0.4, 0.4]), "mse")
    assert len(spread) == 2


def test_the_spread_is_the_sample_deviation():
    spread = group_spread(records(x8=[1.0, 2.0, 3.0]), "mse")
    assert spread[0]["mean"] == pytest.approx(2.0)
    assert spread[0]["std"] == pytest.approx(1.0)
    assert (spread[0]["min"], spread[0]["max"]) == (1.0, 3.0)


def test_one_run_has_no_spread_rather_than_zero_spread():
    """Zero would read as perfect agreement between runs that never happened."""
    spread = group_spread(records(x8=[0.5]), "mse")
    assert spread[0]["std"] is None


def test_failed_points_are_left_out():
    rows = records(x8=[0.5, 0.5])
    rows.append({"point": {"coder.expansion": 8, "train.seed": 9}, "error": "boom"})
    assert group_spread(rows, "mse")[0]["n"] == 2


def test_a_metric_that_is_missing_is_left_out():
    rows = records(x8=[0.5, 0.5])
    rows.append({"point": {"coder.expansion": 8, "train.seed": 9}, "l0": 3.0})
    assert group_spread(rows, "mse")[0]["n"] == 2


def test_groups_are_ordered_by_their_mean():
    spread = group_spread(records(x8=[0.2, 0.2], x16=[0.9, 0.9]), "mse")
    assert spread[0]["held"]["coder.expansion"] == 16


def test_any_axis_can_be_collapsed():
    spread = group_spread(records(x8=[0.5, 0.5], x16=[0.4, 0.4]), "mse",
                          across="coder.expansion")
    assert len(spread) == 2
    assert all(record["n"] == 2 for record in spread)


# ----------------------------------------------------------------------
# is a difference real
# ----------------------------------------------------------------------

def test_a_difference_above_the_noise_counts():
    spread = group_spread(records(x8=[0.50, 0.51, 0.49]), "mse")
    assert separated(spread, metric_difference=0.5)


def test_a_difference_below_the_noise_does_not():
    spread = group_spread(records(x8=[0.1, 0.9, 0.5]), "mse")
    assert not separated(spread, metric_difference=0.01)


def test_the_noisiest_group_is_what_has_to_be_beaten():
    """A claim has to survive the worst configuration it is made about."""
    spread = group_spread(records(x8=[0.50, 0.51], x16=[0.1, 0.9]), "mse")
    assert not separated(spread, metric_difference=0.1)


def test_nothing_is_separated_without_repeats():
    spread = group_spread(records(x8=[0.5]), "mse")
    assert not separated(spread, metric_difference=99.0)


def test_an_empty_spread_is_not_separated():
    assert not separated([], metric_difference=1.0)


# ----------------------------------------------------------------------
# printing
# ----------------------------------------------------------------------

def test_the_table_names_the_axes_that_remain():
    text = table(group_spread(records(x8=[0.5, 0.5], x16=[0.4, 0.4]), "mse"))
    assert "expansion" in text
    assert "seed" not in text


def test_the_table_marks_a_single_run():
    assert "-" in table(group_spread(records(x8=[0.5]), "mse"))


def test_an_empty_table_says_so():
    assert table([]) == "no results"


def test_write_results_creates_the_directory(tmp_path):
    path = tmp_path / "deep" / "grid.json"
    write_results([{"point": {}, "mse": 1.0}], path)
    assert path.exists()