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


def format_x(value: float) -> str:
    """An axis value, readable whether it is a token count or a coefficient.

    Token budgets read best with thousands separators and no decimals, which is
    what every curve was originally plotted against. A sparsity coefficient
    below one prints as ``0`` under that format, so a perfectly good sweep reads
    as six points stacked at the origin.
    """
    return f"{value:,.0f}" if abs(value) >= 1000 else f"{value:g}"


#: The axis name that means the depth a run trained at. It is not an ordinary
#: setting, because a grid is told the depth and varies the seed, so the depth
#: never reaches a run's point.
DEPTH = "depth"

#: The objective a run was trained on. Like DEPTH, a name a caller can ask for
#: as though it were a setting, resolved from what the record carries.
OBJECTIVE = "objective"


def depth_of(record: dict) -> float | None:
    """The block a run trained at, as a number, from the layer it recorded.

    The depth is in every record, just not where a setting would be. A run
    writes the scalar it divided each layer by, and the key there is the layer
    it read: ``A@add_12``. The number in that node name is what orders the
    depths, and it is the only place the depth survives, since the grid is
    handed it rather than sweeping it.

    Returns:
        The number in the layer's node name, or None when the record carries
        no layer or the name holds no number.
    """
    scales = record.get("scales")
    if not isinstance(scales, dict) or not scales:
        return None

    node = next(iter(scales)).partition("@")[2]
    digits = "".join(character for character in node if character.isdigit())
    return float(digits) if digits else None


def objective_of(record: dict) -> str | None:
    """Which objective a run was trained on, from its name.

    Like the depth, this is in every record but not where a setting would be:
    the objective comes from the configuration file rather than from a swept
    axis, so two arms of the same programme are two grids rather than two
    values of one setting. The run name opens with it.

    Returns:
        The leading part of the run name, or None when the record carries no
        name to read it from.
    """
    name = record.get("run_name")
    if not isinstance(name, str) or not name:
        return None
    return name.split("-")[0]


def key_of(record: dict, name: str) -> str | None:
    """What separates one curve or one group of points from another.

    A grid records the axes it swept and nothing else, so the two things most
    worth separating a figure by, the depth and the objective, are both absent
    from ``point``. Both are recoverable from what the run did record, and
    resolving them here means a caller asks for them by name exactly as it
    asks for a setting.

    Args:
        record: one run's record.
        name: ``depth``, ``objective``, or a dotted setting in ``point``.

    Returns:
        The value as text, or None when this record does not carry it, which
        is how a record from another sweep is left out rather than grouped
        under a placeholder.
    """
    if name == DEPTH:
        depth = depth_of(record)
        return None if depth is None else format_x(depth)
    if name == OBJECTIVE:
        return objective_of(record)

    value = record.get("point", {}).get(name)
    return None if value is None else str(value)


def axis_of(record: dict, x: str) -> float | None:
    """Where a record sits on an axis, whether that is a setting or the depth.

    Returns:
        The position, or None when this record is not on this axis at all,
        which is how a record from another sweep is left out rather than
        counted at zero.
    """
    if x == DEPTH:
        return depth_of(record)

    value = record.get("point", {}).get(x)
    return None if value is None else float(value)


def combine_results(groups) -> list[dict]:
    """Several grids' records as one set of runs.

    A sweep extended by a later grid is one curve, and which file recorded a
    run says nothing about where that run sits on it.

    Runs are identified by ``run_name`` and kept once. A grid reuses a run
    whose configuration it has already trained rather than training it again,
    so the same run can be recorded in more than one file. Counted twice it
    reads as two runs of one configuration agreeing exactly, which reports a
    spread of zero where there is no spread to report, and :func:`noise_from`
    takes its tolerance from exactly those spreads. A tolerance that is too
    small makes a curve look like it never flattens.

    Args:
        groups: an iterable of record lists, one per grid file.

    Returns:
        One list, in the order the groups were given, each run once. The first
        record carrying a given name is the one kept. Records with no
        ``run_name`` are all kept, since nothing identifies them as repeats.
    """
    out: list[dict] = []
    seen: set[str] = set()
    dropped = 0

    for records in groups:
        for record in records:
            name = record.get("run_name")
            if name is not None:
                if name in seen:
                    dropped += 1
                    continue
                seen.add(name)
            out.append(record)

    if dropped:
        log.info("%d run(s) recorded in more than one grid, counted once", dropped)
    return out


def series(results: list[dict], x: str, metric: str) -> list[Point]:
    """One point per distinct value of ``x``, averaged over everything else.

    Args:
        results: records from a grid.
        x: the axis to plot along, such as ``train.tokens``, or ``depth`` for
            the block each run trained at, which is read from the layer it
            recorded because it is not a setting the grid swept.
        metric: the number to plot.

    Returns:
        Points in increasing order of ``x``. Records that failed, lack the
        metric, or do not sit on this axis are left out rather than counted as
        zero.
    """
    grouped: dict[float, list[float]] = {}
    for record in results:
        if "error" in record or metric not in record:
            continue
        position = axis_of(record, x)
        if position is None:
            continue
        grouped.setdefault(position, []).append(float(record[metric]))

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


def _sortable(label: str):
    """Order a split's labels numerically where they are numbers."""
    try:
        return (0, float(label), "")
    except ValueError:
        return (1, 0.0, label)


def split_series(results: list[dict], x: str, metric: str,
                 split_by: str) -> dict[str, list[Point]]:
    """One series per distinct value of another setting.

    :func:`series` averages over everything that is not the axis, which is
    right for a grid that varied one thing and wrong for one that varied two:
    the second setting disappears into the mean rather than being the
    comparison. This keeps it, as one curve per value.

    Args:
        results: records from a grid.
        x: the axis to plot along.
        metric: the number to plot.
        split_by: the setting that separates the curves, such as
            ``coder.expansion``.

    Returns:
        Label to series, numerically ordered where the labels are numbers.
        Records that failed or do not carry ``split_by`` are left out.
    """
    groups: dict[str, list[dict]] = {}
    for record in results:
        if "error" in record:
            continue
        label = key_of(record, split_by)
        if label is None:
            continue
        groups.setdefault(label, []).append(record)

    return {label: series(groups[label], x, metric)
            for label in sorted(groups, key=_sortable)}


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

def plot_curve(points: list[Point] | dict[str, list[Point]], metric: str,
               x_label: str, path: Path, tolerance: float | None = None,
               log_x: bool = True, legend_title: str | None = None) -> Path:
    """One metric against one axis, with the spread shown where there is any.

    Args:
        points: from :func:`series`, or from :func:`split_series` for several
            curves on one pair of axes.
        metric: for the vertical label.
        x_label: for the horizontal one.
        path: where the image goes.
        tolerance: if given, the flattening point is marked, once per curve.
        log_x: budgets are usually chosen by doubling, which reads better on a
            logarithmic axis.
        legend_title: what separates the curves, when there are several.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped = points if isinstance(points, dict) else {None: points}
    if not any(grouped.values()):
        raise ValueError("nothing to plot")

    figure, axes = plt.subplots(figsize=(7, 4.5))

    for label, curve in grouped.items():
        if not curve:
            continue
        xs = [point.x for point in curve]
        means = [point.mean for point in curve]

        if any(point.n > 1 for point in curve):
            axes.fill_between(xs, [p.low for p in curve], [p.high for p in curve],
                              alpha=0.18)
        line, = axes.plot(xs, means, marker="o", label=label)

        if tolerance is not None:
            flat = flattening_point(curve, tolerance)
            if flat is not None:
                axes.axvline(flat, linestyle="--", linewidth=1, color=line.get_color())
                axes.annotate(f"flat from {format_x(flat)}",
                              xy=(flat, means[xs.index(flat)]),
                              xytext=(6, 8), textcoords="offset points", fontsize=9,
                              color=line.get_color())

    if any(label is not None for label in grouped):
        axes.legend(title=legend_title, fontsize=9)

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
            groups.setdefault(key_of(record, label_by) or "?", []).append(record)
        for label in sorted(groups, key=_sortable):
            rows = groups[label]
            axes.scatter([r[x_metric] for r in rows], [r[y_metric] for r in rows],
                         label=label)
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


def plot_feature_frequencies(frequencies, path: Path, title: str | None = None,
                             legend_title: str | None = None) -> Path:
    """How often each feature fires, best first.

    The summary numbers say how concentrated the firing is; this says what the
    concentration looks like. A dictionary sharing the work evenly is a shallow
    line, one where a few features carry everything falls off a cliff, and the
    point where a line stops is where the live dictionary ends.

    Both axes are logarithmic because the interesting range spans several
    orders of magnitude: a feature on one token in ten thousand and a feature
    on one in three are both ordinary, and on a linear axis the first is
    indistinguishable from dead.

    Args:
        frequencies: a sequence of per-feature firing rates, or a mapping of
            label to sequence for several runs on one pair of axes.
        path: where the image goes.
        title: a heading, usually which run or setting this is.
        legend_title: what separates the curves, when there are several.

    Returns:
        The path written.

    Raises:
        ValueError: if no curve has a feature that ever fired, which means
            there is nothing a logarithmic axis can show.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped = frequencies if isinstance(frequencies, dict) else {None: frequencies}

    # Dead features sit at zero, which a logarithmic axis cannot place. They
    # are dropped from the line and reported as where it ends instead.
    live = {label: sorted((float(value) for value in values if float(value) > 0),
                          reverse=True)
            for label, values in grouped.items()}
    if not any(live.values()):
        raise ValueError("nothing to plot: no feature ever fired")

    figure, axes = plt.subplots(figsize=(7, 4.5))

    for label, values in live.items():
        if not values:
            continue
        total = len(grouped[label])
        shown = label if label is None else f"{label}  ({len(values)} of {total} alive)"
        axes.plot(range(1, len(values) + 1), values, label=shown)

    if any(label is not None for label in grouped):
        axes.legend(title=legend_title, fontsize=9)

    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlabel("feature, most active first")
    axes.set_ylabel("share of tokens it fires on")
    axes.grid(alpha=0.25, which="both")
    if title:
        axes.set_title(title, fontsize=11)
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


def ceilings_of(summary: dict) -> list[tuple[str, float, float | None]]:
    """Each objective's ceiling and where the measurement falls against it.

    There is one ceiling per objective rather than one overall, because how
    alike two dictionaries are when nothing about the task changed is a
    property of that objective. Two objectives need not agree on it, and here
    they do not.

    Returns:
        ``(name, mean, position)`` per objective, lowest ceiling first. The
        position is None when it could not be worked out. Empty when the
        summary carries no seed repeats.
    """
    within = summary.get("within")
    if not isinstance(within, dict):
        return []

    out = []
    for name, entry in within.items():
        if isinstance(entry, dict) and entry.get("n"):
            out.append((name, entry["mean"], entry.get("position")))
    return sorted(out, key=lambda row: row[1])


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

    shapes = shapes_of(summary)
    if not shapes:
        raise ValueError("nothing to plot: the summary carries no distributions")

    order = _panels(shapes)
    colours = _colours(order)

    if centres is None:
        from stitch.compare import bin_centres
        centres = bin_centres(len(shapes[order[0]]))

    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(7.5, 6), height_ratios=[4, 1],
        gridspec_kw={"hspace": 0.45})

    for name in order:
        shape = shapes[name]
        colour = colours[name]
        top.fill_between(centres, shape, alpha=0.35, color=colour)
        top.plot(centres, shape, color=colour, linewidth=2, label=_label(name))

    top.set_xlabel("how closely each feature is matched in the other dictionary\n"
                   "(1.0 is the same feature, 0.0 is nothing in common)")
    top.set_ylabel("share of features")
    top.set_xlim(-0.1, 1.0)
    top.set_yticks([])
    top.legend(frameon=False, fontsize=10)
    top.grid(alpha=0.2, axis="x")
    if title:
        top.set_title(title, fontsize=12)

    # The line from the floor to each objective's ceiling, with the
    # measurement on it. One ceiling per objective, since they differ.
    floor = summary.get("random")
    ceilings = ceilings_of(summary)
    measured = (summary.get("across") or {}).get("mean")

    if floor is not None and ceilings and measured is not None:
        highest = ceilings[-1][1]
        bottom.plot([floor, highest], [0, 0], color="#c7ccd1", linewidth=8,
                    solid_capstyle="round", zorder=1)
        bottom.scatter([floor], [0], s=120, color=_COMPARISON_COLOURS["random"], zorder=2)
        for _, ceiling, _ in ceilings:
            bottom.scatter([ceiling], [0], s=120,
                           color=_COMPARISON_COLOURS["within"], zorder=2)
        bottom.scatter([measured], [0], s=260, marker="D",
                       color=_COMPARISON_COLOURS["across"], zorder=3)

        span = max(highest - floor, 1e-9)
        bottom.annotate("nothing in common", (floor, 0), xytext=(0, -26),
                        textcoords="offset points", ha="center", fontsize=9,
                        color="#5f6368")
        bottom.annotate("as alike as two runs\nof the same goal", (highest, 0),
                        xytext=(0, -34), textcoords="offset points", ha="center",
                        fontsize=9, color="#5f6368")

        places = "  ".join(f"{position:.0%}" for _, _, position in ceilings
                           if position is not None)
        bottom.annotate(f"measured: {places} of the way",
                        (measured, 0), xytext=(0, 20), textcoords="offset points",
                        ha="center", fontsize=11, fontweight="bold",
                        color=_COMPARISON_COLOURS["across"])

        pad = span * 0.18
        bottom.set_xlim(floor - pad, highest + pad)
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

# ----------------------------------------------------------------------
# the same comparison, for different readers
#
# plot_comparison above is the whole argument in one frame, which is what a
# paper wants when it is allowed one figure. The rest of this section is the
# same small set of quantities taken apart: one figure that shows what a
# dictionary is before anything is compared, two that show the pairing itself,
# one that shows the distributions behind the numbers, one that reduces them to
# shares, one that states the conclusion alone, and one that asks whether the
# difference is about what was learned or about how much it was used.
#
# None of them measure anything. Everything they draw is produced by
# stitch.compare.
# ----------------------------------------------------------------------

#: The same wording plot_comparison uses, for the figures that label panels
#: rather than lines.
_PANEL_LABELS = {
    "random": "two untrained dictionaries",
    "within": "same goal, different starting point",
    "across": "different goals",
}

#: Order the side-by-side panels run in. The other way round, because a reader
#: meets a panel on the left before the one on the right, and the comparison
#: only means something once the ceilings beside it have been seen.
_PAIR_ORDER = ("within", "across", "random")


def _panels(given: dict) -> list[str]:
    """The panels to draw, left to right, from whatever was handed over.

    There is a seed pair per objective rather than one, so the panels are not
    a fixed set of three. The known kinds keep their order and anything else,
    which is an objective's own seed pair named after it, is drawn first and
    alphabetically, so every ceiling appears before the comparison it is the
    reference for.
    """
    present = [kind for kind in given if given.get(kind)]
    known = [kind for kind in _PAIR_ORDER if kind in present]
    other = sorted(kind for kind in present if kind not in _PAIR_ORDER)
    return other + known


def _label(kind: str) -> str:
    """What a panel is called. An objective's own name where it has one."""
    return _PANEL_LABELS.get(kind, f"two runs of {kind}")


#: Shades for the objectives' own curves. They are variations of the ceiling
#: colour, not new colours, because every one of them is a ceiling and should
#: read as one beside the floor and the comparison. Cycled, so any number of
#: objectives can be drawn.
_CEILING_SHADES = ("#3b78c3", "#7fb0e4", "#1d4e86", "#a9cbee")


def _stack(given: dict) -> list[str]:
    """The rows of a stacked figure, bottom upwards.

    The floor sits at the bottom and the comparison above it, which is the
    order a reader builds the claim in. Each objective's own ceiling goes on
    top, alphabetically, however many there are.
    """
    present = [kind for kind in given if given.get(kind)]
    low = [kind for kind in ("random", "across") if kind in present]
    return low + sorted(kind for kind in present if kind not in ("random", "across"))


def _colours(order: list[str]) -> dict[str, str]:
    """A colour per kind, with the fixed three keeping theirs.

    Anything else is an objective, and takes the next ceiling shade, so two
    objectives are told apart without either being mistaken for the floor or
    for the comparison.
    """
    out, shade = {}, 0
    for kind in order:
        if kind in _COMPARISON_COLOURS:
            out[kind] = _COMPARISON_COLOURS[kind]
        else:
            out[kind] = _CEILING_SHADES[shade % len(_CEILING_SHADES)]
            shade += 1
    return out


def shapes_of(summary: dict) -> dict[str, list[float]]:
    """Every match distribution a summary carries, keyed the way panels are.

    The summary writes one distribution per objective plus ``across`` and
    ``random``, each under its own name with ``_shape`` appended. Reading them
    back by suffix keeps the figures free of any fixed list of kinds, which is
    what let an objective's curve be dropped without anything raising.
    """
    return {key[: -len("_shape")]: value for key, value in summary.items()
            if key.endswith("_shape") and value}

#: One sequential map for everything that shades a feature by how well it
#: matched, so a colour means the same thing in every figure that uses one.
_MATCH_MAP = "magma"


def _figure_out(figure, path: Path) -> Path:
    """Write a finished figure where it was asked for."""
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    log.info("wrote %s", path)
    return path


def plot_feature_squares(scores: dict[str, list[float]], path: Path,
                         title: str | None = None, columns: int = 40) -> Path:
    """One square per feature, shaded by how well that feature matched.

    The figure to show first, because it is the only one whose subject is the
    dictionary rather than the comparison. A reader who has never met a sparse
    dictionary can see from it that the thing being compared is a few thousand
    separate parts, and that the question is asked of each part on its own.

    Nothing is sorted. A sorted version reads as a smooth gradient, which is a
    picture of a distribution; left in the dictionary's own order it reads as
    individual features, some of which found a counterpart and some of which
    did not.

    Args:
        scores: kind to per-feature matched scores, from
            ``stitch.compare.detail``.
        path: where the image goes.
        title: a heading, usually which depth this is.
        columns: squares per row.

    Returns:
        The path written.

    Raises:
        ValueError: if no kind carries any scores.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    kinds = _panels(scores)
    if not kinds:
        raise ValueError("nothing to plot: no per-feature scores")

    figure, axes = plt.subplots(len(kinds), 1,
                                figsize=(11, 2.7 * len(kinds) + 0.6),
                                squeeze=False)

    image = None
    for axis, kind in zip(axes[:, 0], kinds):
        values = list(scores[kind])
        # A dictionary smaller than one row gets a shorter row rather than a
        # row padded out with squares that stand for no feature.
        width = min(columns, len(values))
        rows = max(len(values) // width, 1)
        grid = np.array(values[:rows * width]).reshape(rows, width)

        image = axis.imshow(grid, cmap=_MATCH_MAP, vmin=0, vmax=1,
                            interpolation="nearest")
        # Borders between the cells, so the eye reads discrete features rather
        # than a surface.
        axis.set_xticks(np.arange(-0.5, width, 1), minor=True)
        axis.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=1.1)
        axis.tick_params(which="minor", length=0)
        axis.set_title(_label(kind), fontsize=11, pad=8)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)

    figure.colorbar(image, ax=axes, fraction=0.02, pad=0.02,
                    label="how well this feature matched")
    figure.suptitle(title or "Every feature in the dictionary, one square each",
                    fontsize=13)
    return _figure_out(figure, path)


def plot_pairing(pairs: dict[str, list], path: Path,
                 title: str | None = None) -> Path:
    """Which feature became which, as a ribbon per pair.

    Both dictionaries are laid out in order of how often their features fire,
    heaviest on the left, because their own slot numbering is arbitrary and a
    figure drawn against it would show a perfect pairing as a tangle. A ribbon
    that runs straight down is a feature that carried the same share of the
    work in both; one that crosses the figure is a feature that mattered in one
    dictionary and barely fired in the other. Colour and weight carry how well
    the two actually matched.

    The two things a reader should take from it are separate: a bright ribbon
    is a feature the other dictionary also has, and a steep ribbon is a feature
    the other dictionary uses differently.

    Args:
        pairs: kind to triples of position in its own dictionary, position of
            its counterpart in the other, and matched score, from
            ``stitch.compare.detail``.
        path: where the image goes.
        title: a heading, usually which depth this is.

    Returns:
        The path written.

    Raises:
        ValueError: if no kind carries any pairs.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import PathPatch, Rectangle
    from matplotlib.path import Path as MplPath

    kinds = _panels(pairs)
    if not kinds:
        raise ValueError("nothing to plot: no pairs")

    figure, axes = plt.subplots(1, len(kinds), figsize=(5.8 * len(kinds), 5.2),
                                squeeze=False)
    cmap = plt.get_cmap(_MATCH_MAP)

    for axis, kind in zip(axes[0], kinds):
        rows = sorted(pairs[kind], key=lambda row: row[2])
        for left, right, score in rows:
            quality = min(max(float(score), 0.0), 1.0)
            patch = PathPatch(
                MplPath([(left, 1.0), (left, 0.62), (right, 0.38), (right, 0.0)],
                        [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4,
                         MplPath.CURVE4]),
                fill=False, linewidth=1.5, alpha=0.25 + 0.65 * quality,
                color=cmap(0.15 + 0.7 * quality))
            axis.add_patch(patch)

        axis.add_patch(Rectangle((0.0, 1.0), 1.0, 0.05, color="#2a2a33"))
        axis.add_patch(Rectangle((0.0, -0.05), 1.0, 0.05, color="#2a2a33"))
        axis.text(0.5, 1.12, "one dictionary", ha="center", fontsize=10.5)
        axis.text(0.5, -0.18, "the other", ha="center", fontsize=10.5)
        axis.set_title(_label(kind), fontsize=11, pad=14)
        axis.set_xlim(-0.03, 1.03)
        axis.set_ylim(-0.26, 1.2)
        axis.axis("off")

    figure.suptitle(title or "Which feature became which", fontsize=13, y=1.02)
    figure.text(0.5, -0.03,
                "Both dictionaries run from most used on the left to least "
                "used on the right.\nBright ribbons matched well. Steep ones "
                "changed how much work they do.",
                ha="center", fontsize=10, color="#5f6368")
    return _figure_out(figure, path)


def plot_match_matrix(blocks: dict[str, list[list[float]]], path: Path,
                      title: str | None = None) -> Path:
    """Every drawn feature against every drawn feature, pairing on the diagonal.

    The ribbons make the claim and this one lets it be checked. The features
    are ordered so that each pair sits on the diagonal, which means a clean
    one-to-one correspondence shows as a bright diagonal line and nothing else.

    What it can show that no summary can: a bright band around the diagonal
    rather than a line is crowding, where several features of one dictionary
    answer to the same feature of the other. That is a different situation from
    a weak pairing and the mean score cannot tell them apart.

    Only the most heavily used features are drawn, since the whole matrix is
    both unreadable at this size and far larger than a figure.

    Args:
        blocks: kind to a square block of the similarity matrix, from
            ``stitch.compare.detail``.
        path: where the image goes.
        title: a heading, usually which depth this is.

    Returns:
        The path written.

    Raises:
        ValueError: if no kind carries a block.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kinds = _panels(blocks)
    if not kinds:
        raise ValueError("nothing to plot: no similarity blocks")

    figure, axes = plt.subplots(1, len(kinds), figsize=(5.4 * len(kinds), 5.4),
                                squeeze=False)

    image = None
    for axis, kind in zip(axes[0], kinds):
        image = axis.imshow(blocks[kind], cmap=_MATCH_MAP, vmin=0, vmax=1,
                            interpolation="nearest")
        axis.set_title(_label(kind), fontsize=11, pad=10)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)

    figure.colorbar(image, ax=axes, fraction=0.03, pad=0.02, label="similarity")
    figure.suptitle(title or "Every feature against every feature, "
                             "ordered by its partner", fontsize=13)
    figure.text(0.5, 0.02,
                "A bright diagonal is one feature answering to one feature. "
                "A bright band is several\nfeatures of one dictionary sharing "
                "a counterpart in the other.",
                ha="center", fontsize=10, color="#5f6368")
    return _figure_out(figure, path)


def plot_distributions(summary: dict, path: Path,
                       centres: list[float] | None = None,
                       title: str | None = None) -> Path:
    """The three comparisons as separate distributions, stacked.

    The same three shapes ``plot_comparison`` overlays. Separated they are
    easier to read one at a time, and a shape a mean cannot report stands out:
    an across distribution with two humps is a set of features that transferred
    and a set that did not, which is a different claim from everything
    transferring halfway.

    Args:
        summary: output of ``stitch.compare.summarise``, built from records
            carrying distributions.
        path: where the image goes.
        centres: the cosine value each bin sits at. Defaults to the bins
            ``stitch.compare`` records.
        title: a heading, usually which depth this is.

    Returns:
        The path written.

    Raises:
        ValueError: if the summary carries no distributions.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shapes = shapes_of(summary)
    present = _stack(shapes)
    if not present:
        raise ValueError("nothing to plot: the summary carries no distributions")

    colours = _colours(present)

    if centres is None:
        from stitch.compare import bin_centres
        centres = bin_centres(len(shapes[present[0]]))

    figure, axis = plt.subplots(figsize=(9, 1.5 * len(present) + 1.8))

    for position, kind in enumerate(present):
        shape = shapes[kind]
        peak = max(shape) or 1.0
        height = [position + 0.85 * value / peak for value in shape]
        axis.fill_between(centres, position, height,
                          color=colours[kind], alpha=0.85,
                          linewidth=0)
        axis.plot(centres, height, color="white", linewidth=1.4)
        axis.text(-0.12, position + 0.12, _label(kind), ha="right",
                  fontsize=10.5)

    axis.set_xlim(-0.1, 1.0)
    axis.set_ylim(-0.1, len(present))
    axis.set_yticks([])
    axis.set_xlabel("how closely each feature is matched in the other dictionary\n"
                    "(1.0 is the same feature, 0.0 is nothing in common)")
    for side in ("left", "right", "top"):
        axis.spines[side].set_visible(False)
    axis.set_title(title or "Where the comparisons sit", fontsize=13,
                   pad=14)
    return _figure_out(figure, path)


def plot_match_shares(shares: dict[str, dict], path: Path,
                      title: str | None = None) -> Path:
    """The distributions reduced to how much of the dictionary matched.

    For a reader who will not read a density plot. The two cuts are the two
    numbers the method already has: a feature counts as matched when it clears
    the threshold, and as unmatched when it does no better than two untrained
    dictionaries, which is where a feature that found nothing lands.

    Args:
        shares: kind to ``strong``, ``partial`` and ``absent``, from
            ``stitch.compare.shares``.
        path: where the image goes.
        title: a heading, usually which depth this is.

    Returns:
        The path written.

    Raises:
        ValueError: if no kind carries shares.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    kinds = _stack(shares)
    if not kinds:
        raise ValueError("nothing to plot: no shares")

    bands = (("strong", "the same feature", "#2a6f97"),
             ("partial", "something in common", "#e8a33d"),
             ("absent", "no counterpart", "#b9b9c3"))

    figure, axis = plt.subplots(figsize=(9.5, 1.1 * len(kinds) + 2.0))

    for position, kind in enumerate(kinds):
        y = len(kinds) - 1 - position
        left = 0.0
        for name, _, colour in bands:
            width = float(shares[kind].get(name, 0.0))
            axis.add_patch(Rectangle((left, y - 0.3), width, 0.6, color=colour))
            if width > 0.08:
                axis.text(left + width / 2, y, f"{width:.0%}", ha="center",
                          va="center", fontsize=10.5,
                          color="white" if colour != "#b9b9c3" else "#1b1b1f")
            left += width
        axis.text(-0.015, y, _label(kind), ha="right", va="center",
                  fontsize=10.5)

    handles = [Rectangle((0, 0), 1, 1, color=colour) for _, _, colour in bands]
    axis.legend(handles, [label for _, label, _ in bands], frameon=False,
                ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.02),
                fontsize=10)

    axis.set_xlim(0, 1)
    axis.set_ylim(-0.7, len(kinds) - 0.3)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_title(title or "How much of one dictionary is in the other",
                   fontsize=13, pad=14)
    return _figure_out(figure, path)


def plot_position(summary: dict, path: Path, title: str | None = None) -> Path:
    """The conclusion on its own, with nothing else in the frame.

    The lower panel of ``plot_comparison``, pulled out large. A similarity of
    0.6 means nothing by itself; what it means is where it falls between two
    dictionaries that learned nothing and two runs of the same objective. That
    is one sentence and this is the figure to stop on while saying it.

    Args:
        summary: output of ``stitch.compare.summarise``.
        path: where the image goes.
        title: a heading, usually which depth this is.

    Returns:
        The path written.

    Raises:
        ValueError: if the summary has no floor, ceiling and measurement to
            place, which means there were not enough runs of each kind.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    floor = summary.get("random")
    ceilings = ceilings_of(summary)
    measured = (summary.get("across") or {}).get("mean")
    if floor is None or not ceilings or measured is None:
        raise ValueError("nothing to plot: no measurement to place")

    figure, axis = plt.subplots(figsize=(9.5, 3.2))

    highest = ceilings[-1][1]
    axis.plot([floor, highest], [0, 0], color="#c7ccd1", linewidth=10,
              solid_capstyle="round", zorder=1)
    axis.scatter([floor], [0], s=150, color=_COMPARISON_COLOURS["random"],
                 zorder=2)
    axis.scatter([ceiling for _, ceiling, _ in ceilings],
                 [0] * len(ceilings), s=150,
                 color=_COMPARISON_COLOURS["within"], zorder=2)
    axis.scatter([measured], [0], s=320, marker="D",
                 color=_COMPARISON_COLOURS["across"], zorder=3)

    span = max(highest - floor, 1e-9)
    axis.annotate("nothing in common", (floor, 0), xytext=(0, -30),
                  textcoords="offset points", ha="center", fontsize=10,
                  color="#5f6368")

    # One ceiling per objective, each labelled, because they differ and the
    # measurement sits at a different fraction of each.
    for index, (name, ceiling, _) in enumerate(ceilings):
        axis.annotate(f"two runs of\n{name}", (ceiling, 0),
                      xytext=(0, -38 - 26 * index), textcoords="offset points",
                      ha="center", fontsize=10, color="#5f6368")

    places = ", ".join(f"{position:.0%} of {name}"
                       for name, _, position in ceilings
                       if position is not None)
    axis.annotate(f"different goals: {places}",
                  (measured, 0), xytext=(0, 24), textcoords="offset points",
                  ha="center", fontsize=12, fontweight="bold",
                  color=_COMPARISON_COLOURS["across"])

    pad = span * 0.2
    axis.set_xlim(floor - pad, highest + pad)
    axis.set_ylim(-1.0, 0.8)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    if title:
        axis.set_title(title, fontsize=13, pad=16)
    return _figure_out(figure, path)


def plot_frequency_bands(bands: dict[str, list[float]], path: Path,
                         title: str | None = None) -> Path:
    """The gap between the two readings, split by how often a feature fires.

    Rare features are harder to match for an ordinary reason: a direction
    estimated from few tokens is estimated badly. So a gap that only appears
    among the rare features is a statement about how heavily each dictionary
    was used, and a gap that stays open across every band is a statement about
    what was learned. The figure exists to keep those two apart, which is why
    the gap itself is shaded rather than left for the reader to measure.

    Args:
        bands: one entry per objective plus ``across``, each a list of
            per-band means with the rarest band first, from
            ``stitch.compare.by_kind``. Whatever names the runs carry are
            drawn, so an objective is never dropped for not being on a list.
        path: where the image goes.
        title: a heading, usually which depth this is.

    Returns:
        The path written.

    Raises:
        ValueError: if neither kind carries any bands.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    present = {kind: list(values) for kind, values in bands.items() if values}
    if not present:
        raise ValueError("nothing to plot: no frequency bands")

    # Every kind that is not the comparison is an objective's own ceiling.
    # Named rather than listed, because the names come from the runs and a
    # fixed list silently drops whatever it does not recognise.
    ceilings = sorted(kind for kind in present if kind != "across")
    colours = _colours(["across"] + ceilings)

    count = min(len(values) for values in present.values())
    positions = list(range(count))

    figure, axis = plt.subplots(figsize=(9.5, 4.8))

    # The gap is shaded only where there is one ceiling to measure it against.
    # With two objectives on the axes a single shaded band would be the gap to
    # neither of them, which is the pooling this figure exists to avoid, so
    # the curves are left to be read against each other.
    if "across" in present and len(ceilings) == 1:
        ceiling = present[ceilings[0]]
        axis.fill_between(positions, present["across"][:count],
                          ceiling[:count], color="#f0d9dc",
                          alpha=0.8, zorder=1)
        for position in positions:
            axis.annotate("", xy=(position, ceiling[position]),
                          xytext=(position, present["across"][position]),
                          arrowprops=dict(arrowstyle="<->", color="#8a5f63",
                                          linewidth=1.2), zorder=2)

    for kind in ceilings + (["across"] if "across" in present else []):
        axis.plot(positions, present[kind][:count], "-o",
                  color=colours[kind], linewidth=3,
                  markersize=10, zorder=3, label=_label(kind))

    labels = [""] * count
    if count:
        labels[0] = "rarest"
        labels[-1] = "most frequent"
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("features grouped by how often they fire", labelpad=12)
    axis.set_ylabel("how well features matched")
    # Top left. Every curve rises to the right and none of them reaches the
    # upper band, so the legend sits clear of the lines wherever the gap
    # happens to open.
    axis.legend(frameon=False, loc="upper left", fontsize=10)
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.set_title(title or "The gap, split by how often a feature fires",
                   fontsize=13, pad=14)
    # Below the axis label, not level with it. The figure is written with a
    # tight bounding box, so the space is made here and not by the margin.
    figure.text(0.5, -0.09,
                "A gap open in every band is a difference in what was learned. "
                "A gap only on the left\nis a difference in how heavily the "
                "dictionary was used.",
                ha="center", fontsize=10, color="#5f6368")
    return _figure_out(figure, path)