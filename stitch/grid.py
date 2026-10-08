"""Running many configurations and reading the spread between them.

A single run produces numbers. It does not produce a result, because nothing
says whether a difference between two runs is the thing being studied or the
difference two runs would show anyway.

The same configuration trained at several seeds answers that. Everything is
held fixed except the initialisation and the data order, so whatever spread
appears is the spread that means nothing. A difference between objectives is
only a finding if it is larger than that.

One driver covers three jobs, because they are the same job:

- vary a hyperparameter, to tune on the control pair
- vary the seed, to measure the noise floor
- vary objective, layer pair and seed, for the comparison

A grid is a product of axes, each a setting and the values it takes. Every
point is an ordinary run with overrides applied, so a point can be reproduced
on its own by passing the same overrides to the run script.

Grids are long and machines are interrupted. A point whose summary already
exists is skipped, so a grid can be restarted and continues rather than
starting again.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
import statistics
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# describing a grid
# ----------------------------------------------------------------------

def _scalar(text: str) -> Any:
    """One value, typed.

    YAML 1.1 reads 1e-3 as a string, because it wants a decimal point and a
    signed exponent. Learning rates and sparsity coefficients are written the
    short way, so anything that survives as a string and parses as a number is
    taken as one.
    """
    import yaml

    value = yaml.safe_load(text)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def parse_axis(text: str) -> tuple[str, list[Any]]:
    """Turn ``train.seed=0,1,2`` into a setting and its values.

    Values are typed, so numbers arrive as numbers and ``true`` as a boolean,
    the same way a single override does. The types matter here as well as at
    load time, because the values become the coordinates results are grouped
    by, and a seed of ``0`` and a seed of ``"0"`` would be two groups.

    Raises:
        ValueError: if there is no ``=``, no setting before it, or no values
            after it.
    """
    if "=" not in text:
        raise ValueError(f"axis {text!r} is not of the form key=value,value")

    key, raw = text.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"axis {text!r} names no setting")

    values = [_scalar(piece) for piece in raw.split(",") if piece != ""]
    if not values:
        raise ValueError(f"axis {text!r} lists no values")
    return key, values


def expand(axes: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Every combination, in a stable order.

    The order follows the axes as given, with the last varying fastest, so a
    grid run twice visits its points in the same sequence and a partial grid is
    readable.
    """
    if not axes:
        return [{}]

    keys = list(axes)
    return [dict(zip(keys, values)) for values in itertools.product(*(axes[k] for k in keys))]


def as_overrides(point: dict[str, Any]) -> list[str]:
    """One grid point, in the form the configuration loader takes."""
    return [f"{key}={value}" for key, value in point.items()]


def describe_point(point: dict[str, Any]) -> str:
    """One grid point, short enough for a log line."""
    return ", ".join(f"{key.split('.')[-1]}={value}" for key, value in point.items()) or "default"


# ----------------------------------------------------------------------
# running one
# ----------------------------------------------------------------------

def already_done(cfg) -> Path | None:
    """The summary of a previous run of this exact configuration, if there is one.

    A run directory is named after the configuration hash, so this is an
    identity check rather than a guess.
    """
    path = Path(cfg.logging.run_root) / cfg.run_name() / "summary.json"
    return path if path.exists() else None


def run_grid(config_path: Path | None, overrides: list[str], axes: dict[str, list[Any]],
             out_file: Path, skip_existing: bool = True,
             runner: Callable | None = None) -> list[dict]:
    """Run every point and collect the summaries.

    Args:
        config_path: the base configuration.
        overrides: applied to every point, before the point's own.
        axes: setting to values.
        out_file: where the collected results go.
        skip_existing: reuse a point that has already been run.
        runner: what performs a run. Defaults to the real one; replaced in
            tests so the grid logic can be exercised without a backbone.

    Returns:
        One record per point: its coordinates and the run's summary. A point
        that raised is recorded with its error rather than stopping the grid,
        since losing eleven good points to the twelfth is worse than finishing
        with a hole in the table.
    """
    from stitch.config import load_config

    if runner is None:
        from stitch.run import run as runner

    points = expand(axes)
    log.info("grid of %d points", len(points))

    results = []
    for index, point in enumerate(points, start=1):
        cfg = load_config(config_path, list(overrides) + as_overrides(point))
        label = describe_point(point)

        existing = already_done(cfg) if skip_existing else None
        if existing is not None:
            log.info("[%d/%d] %s: already done", index, len(points), label)
            record = {"point": point, "reused": True, **json.loads(existing.read_text())}
        else:
            log.info("[%d/%d] %s: running", index, len(points), label)
            try:
                record = {"point": point, "reused": False, **runner(cfg)}
            except Exception as error:  # noqa: BLE001 - one bad point must not end the grid
                log.exception("[%d/%d] %s: failed", index, len(points), label)
                record = {"point": point, "reused": False,
                          "error": f"{type(error).__name__}: {error}"}

        results.append(record)
        write_results(results, out_file)

    failures = sum(1 for record in results if "error" in record)
    log.info("grid finished, %d points, %d failed", len(results), failures)
    return results


def write_results(results: list[dict], path: Path) -> None:
    """Write the collected results, after every point rather than at the end."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))


def read_results(path: Path) -> list[dict]:
    """Results collected by a previous grid."""
    return json.loads(Path(path).read_text())


# ----------------------------------------------------------------------
# reading the spread
# ----------------------------------------------------------------------

def group_spread(results: list[dict], metric: str,
                 across: str = "train.seed") -> list[dict]:
    """Spread of one metric across one axis, holding the others fixed.

    With ``across`` set to the seed, each group is one configuration trained
    several times, and the spread within it is the amount by which two runs of
    the same thing differ. That is the number any claimed difference has to
    exceed.

    Args:
        results: records from a grid.
        metric: which number to summarise.
        across: the axis to collapse.

    Returns:
        One record per group, with the remaining coordinates, the mean, the
        sample standard deviation, the range and the count. Groups of one have
        no standard deviation and report ``None`` rather than zero, which would
        read as agreement.
    """
    groups: dict[tuple, list[float]] = {}
    labels: dict[tuple, dict] = {}

    for record in results:
        if "error" in record or metric not in record:
            continue
        held = {key: value for key, value in record["point"].items() if key != across}
        key = tuple(sorted(held.items()))
        groups.setdefault(key, []).append(float(record[metric]))
        labels[key] = held

    out = []
    for key, values in groups.items():
        out.append({
            "held": labels[key],
            "metric": metric,
            "n": len(values),
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else None,
            "min": min(values),
            "max": max(values),
        })

    out.sort(key=lambda record: record["mean"], reverse=True)
    return out


def separated(spread: list[dict], metric_difference: float) -> bool:
    """Whether a difference of that size is larger than the noise.

    Compares against the largest standard deviation seen in any group, not the
    average, since a claim has to survive the noisiest configuration it is made
    about.

    Returns False when any group has fewer than two runs, since nothing is
    known about the spread then.
    """
    deviations = [record["std"] for record in spread]
    if not deviations or any(value is None for value in deviations):
        return False
    return metric_difference > max(deviations)


def table(spread: list[dict]) -> str:
    """The spread, printable."""
    if not spread:
        return "no results"

    keys = sorted({key for record in spread for key in record["held"]})
    header = "".join(f"{key.split('.')[-1]:<16}" for key in keys)
    lines = [f"{header}{'n':<4} {'mean':<10} {'std':<10} {'min':<10} max",
             "-" * (len(header) + 48)]

    for record in spread:
        cells = "".join(f"{str(record['held'].get(key, '')):<16}" for key in keys)
        std = "-" if record["std"] is None else f"{record['std']:.5f}"
        lines.append(f"{cells}{record['n']:<4} {record['mean']:<10.5f} {std:<10} "
                     f"{record['min']:<10.5f} {record['max']:.5f}")
    return "\n".join(lines)