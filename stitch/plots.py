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


# ----------------------------------------------------------------------
# what two dictionaries have in common
# ----------------------------------------------------------------------

#: Colours chosen to carry the argument: grey for the floor, blue for the
#: ceiling, orange for the measurement. Distinguishable in greyscale by fill
#: density as well as by hue.
_COMPARISON_COLOURS = {
    "random": "#9aa0a6",
    "within": "#3b78c3",
    "across": "#e8833a",
}


def plot_comparison(summary: dict, path: Path, centres: list[float] | None = None,
                    title: str | None = None) -> Path:
    """What two dictionaries have in common, drawn for someone who has not
    read the method.

    Two panels. The upper one shows, for every feature, how close the nearest
    feature in the other dictionary was: three distributions, one for random
    dictionaries, one for two runs of the same objective, one for two runs of
    different objectives. The lower one places the measurement on the line
    between the other two, which is the sentence the figure exists to make.

    Args:
        summary: output of ``stitch.compare.summarise``, which must have been
            built from records carrying distributions.
        path: where the image goes.
        centres: the cosine value each histogram bin sits at. Defaults to the
            bins ``stitch.compare`` records.
        title: a heading, usually which depth this is.

    Returns:
        The path written.

    Raises:
        ValueError: if the summary carries no distributions, which means the
            comparison was run without them.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shapes = {name: summary.get(f"{name}_shape")
              for name in ("random", "within", "across")}
    if not any(shapes.values()):
        raise ValueError("nothing to plot: the summary carries no distributions")

    if centres is None:
        from stitch.compare import bin_centres
        centres = bin_centres(len(next(shape for shape in shapes.values() if shape)))

    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(7.5, 6), height_ratios=[4, 1],
        gridspec_kw={"hspace": 0.45})

    labels = {
        "random": "two untrained dictionaries",
        "within": "same goal, different starting point",
        "across": "different goals",
    }
    for name in ("random", "within", "across"):
        shape = shapes.get(name)
        if not shape:
            continue
        colour = _COMPARISON_COLOURS[name]
        top.fill_between(centres, shape, alpha=0.35, color=colour)
        top.plot(centres, shape, color=colour, linewidth=2, label=labels[name])

    top.set_xlabel("how closely each feature is matched in the other dictionary\n"
                   "(1.0 is the same feature, 0.0 is nothing in common)")
    top.set_ylabel("share of features")
    top.set_xlim(-0.1, 1.0)
    top.set_yticks([])
    top.legend(frameon=False, fontsize=10)
    top.grid(alpha=0.2, axis="x")
    if title:
        top.set_title(title, fontsize=12)

    # The line from floor to ceiling, with the measurement on it.
    floor = summary.get("random")
    ceiling = (summary.get("within") or {}).get("mean")
    measured = (summary.get("across") or {}).get("mean")

    if floor is not None and ceiling is not None and measured is not None:
        bottom.plot([floor, ceiling], [0, 0], color="#c7ccd1", linewidth=8,
                    solid_capstyle="round", zorder=1)
        bottom.scatter([floor], [0], s=120, color=_COMPARISON_COLOURS["random"], zorder=2)
        bottom.scatter([ceiling], [0], s=120, color=_COMPARISON_COLOURS["within"], zorder=2)
        bottom.scatter([measured], [0], s=260, marker="D",
                       color=_COMPARISON_COLOURS["across"], zorder=3)

        span = max(ceiling - floor, 1e-9)
        bottom.annotate("nothing in common", (floor, 0), xytext=(0, -26),
                        textcoords="offset points", ha="center", fontsize=9,
                        color="#5f6368")
        bottom.annotate("as alike as two runs\nof the same goal", (ceiling, 0),
                        xytext=(0, -34), textcoords="offset points", ha="center",
                        fontsize=9, color="#5f6368")
        bottom.annotate(f"measured: {(measured - floor) / span:.0%} of the way",
                        (measured, 0), xytext=(0, 20), textcoords="offset points",
                        ha="center", fontsize=11, fontweight="bold",
                        color=_COMPARISON_COLOURS["across"])

        pad = span * 0.18
        bottom.set_xlim(floor - pad, ceiling + pad)
    else:
        bottom.text(0.5, 0.5, "not enough runs to place the measurement",
                    ha="center", va="center", fontsize=10, color="#5f6368")
        bottom.set_xlim(0, 1)

    bottom.set_ylim(-0.6, 0.6)
    bottom.set_yticks([])
    for side in ("left", "right", "top"):
        bottom.spines[side].set_visible(False)
    bottom.spines["bottom"].set_visible(False)
    bottom.set_xticks([])

    figure.subplots_adjust(left=0.1, right=0.97, top=0.9, bottom=0.14)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    log.info("wrote %s", path)
    return path