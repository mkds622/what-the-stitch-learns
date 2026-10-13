"""Reading a grid as a curve.

Two questions, both answered from a grid's results file.

**Where does a curve flatten?** Reconstruction and sparsity against the token
budget. The answer is not a judgement about a line: a curve has flattened when
the change from one point to the next is smaller than the difference two runs
of the same configuration show, which is a number the seed grid provides.

**What does a coder trade?** Reconstruction against sparsity, as one point per
configuration. Neither alone says anything, because either can be made perfect
by ruining the other, so the pair is the thing that gets compared.

Plotting is kept apart from the shaping. The functions that turn results into
series are tested; the ones that draw are thin enough to read.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class Point:
    """One position on a curve, over however many runs sit at it."""

    x: float
    mean: float
    std: float | None
    low: float
    high: float
    n: int


def series(results: list[dict], x: str, metric: str) -> list[Point]:
    """One point per distinct value of ``x``, averaged over everything else.

    Args:
        results: records from a grid.
        x: the axis to plot along, such as ``train.tokens``.
        metric: the number to plot.

    Returns:
        Points in increasing order of ``x``. Records that failed or lack the
        metric are left out rather than counted as zero.
    """
    grouped: dict[float, list[float]] = {}
    for record in results:
        if "error" in record or metric not in record or x not in record.get("point", {}):
            continue
        grouped.setdefault(float(record["point"][x]), []).append(float(record[metric]))

    points = []
    for value in sorted(grouped):
        values = grouped[value]
        points.append(Point(
            x=value,
            mean=statistics.fmean(values),
            std=statistics.stdev(values) if len(values) > 1 else None,
            low=min(values),
            high=max(values),
            n=len(values),
        ))
    return points


def flattening_point(points: list[Point], tolerance: float) -> float | None:
    """The first position where the curve stopped moving.

    Args:
        points: a series, in increasing order.
        tolerance: how much change counts as none. The right value is the
            spread between seeds, so that "stopped moving" means "moved less
            than running the same thing twice would".

    Returns:
        The x at which the change from the previous point first fell within
        tolerance, or None if it never did, which means the curve was still
        moving when the data ran out.
    """
    if tolerance < 0:
        raise ValueError(f"tolerance must not be negative, got {tolerance}")

    for previous, current in zip(points, points[1:]):
        if abs(current.mean - previous.mean) <= tolerance:
            return current.x
    return None


def noise_from(results: list[dict], metric: str, across: str = "train.seed") -> float | None:
    """The largest spread any configuration shows, to use as a tolerance.

    The largest rather than the average, because a claim has to survive the
    noisiest configuration it is made about.
    """
    from stitch.grid import group_spread

    deviations = [record["std"] for record in group_spread(results, metric, across)]
    usable = [value for value in deviations if value is not None]
    return max(usable) if usable else None


# ----------------------------------------------------------------------
# drawing
# ----------------------------------------------------------------------

def plot_curve(points: list[Point], metric: str, x_label: str, path: Path,
               tolerance: float | None = None, log_x: bool = True) -> Path:
    """One metric against one axis, with the spread shown where there is any.

    Args:
        points: from :func:`series`.
        metric: for the vertical label.
        x_label: for the horizontal one.
        path: where the image goes.
        tolerance: if given, the flattening point is marked.
        log_x: budgets are usually chosen by doubling, which reads better on a
            logarithmic axis.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not points:
        raise ValueError("nothing to plot")

    figure, axes = plt.subplots(figsize=(7, 4.5))
    xs = [point.x for point in points]
    means = [point.mean for point in points]

    if any(point.n > 1 for point in points):
        axes.fill_between(xs, [p.low for p in points], [p.high for p in points], alpha=0.18)
    axes.plot(xs, means, marker="o")

    if tolerance is not None:
        flat = flattening_point(points, tolerance)
        if flat is not None:
            axes.axvline(flat, linestyle="--", linewidth=1)
            axes.annotate(f"flat from {flat:,.0f}", xy=(flat, means[xs.index(flat)]),
                          xytext=(6, 8), textcoords="offset points", fontsize=9)

    if log_x:
        axes.set_xscale("log")
    axes.set_xlabel(x_label)
    axes.set_ylabel(metric)
    axes.grid(alpha=0.25)
    figure.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    log.info("wrote %s", path)
    return path


def plot_tradeoff(results: list[dict], path: Path, label_by: str | None = None,
                  x_metric: str = "eval_l0",
                  y_metric: str = "eval_explained_variance") -> Path:
    """Reconstruction against sparsity, one point per run.

    The curve each objective traces out here is what the comparison is
    actually about. A point further up and further left is better; a point
    further up and further right has only bought reconstruction with density.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    usable = [record for record in results
              if "error" not in record and x_metric in record and y_metric in record]
    if not usable:
        raise ValueError("nothing to plot")

    figure, axes = plt.subplots(figsize=(6, 5))

    if label_by is None:
        axes.scatter([r[x_metric] for r in usable], [r[y_metric] for r in usable])
    else:
        groups: dict[str, list[dict]] = {}
        for record in usable:
            groups.setdefault(str(record["point"].get(label_by, "?")), []).append(record)
        for label, rows in sorted(groups.items()):
            axes.scatter([r[x_metric] for r in rows], [r[y_metric] for r in rows], label=label)
        axes.legend(title=label_by.split(".")[-1], fontsize=9)

    axes.set_xlabel(f"{x_metric}   (active features per token, lower is sparser)")
    axes.set_ylabel(f"{y_metric}   (higher reconstructs more)")
    axes.grid(alpha=0.25)
    figure.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    log.info("wrote %s", path)
    return path