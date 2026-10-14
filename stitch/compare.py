"""Comparing what two trained dictionaries learned.

Every other measurement in this project describes a single coder: how much it
reconstructs, how many features it uses, how many are dead. None of them
compares one coder against another, and the question being asked is a
comparison: does a dictionary trained to reproduce one target differ from a
dictionary trained to reproduce another?

What is compared
----------------

A feature is a direction in the output space, held in a column of the decoder.
Two coders have learned the same feature when some column of one points the
same way as some column of the other, whatever position it occupies in either
dictionary, since nothing fixes the order.

So for each feature in one dictionary, find its closest counterpart in the
other by cosine, and look at the distribution of those similarities. A
dictionary whose features all have close counterparts has learned the same
things; one whose features have no counterparts has learned something else.

The measure is one-sided and the two sides differ, so both are reported. A
small dictionary contained inside a large one scores high read one way and low
read the other.

Why three numbers and not one
-----------------------------

A similarity of 0.6 means nothing on its own. It has to be read against two
references:

``random``
    two dictionaries of random directions. In high dimensions the closest of
    many random vectors is not close, but it is not zero either, and the value
    grows with dictionary size, so it is computed rather than assumed.

``within``
    two coders trained on the same objective, differing only by seed. This is
    the ceiling: no comparison between objectives can be expected to exceed
    what the same objective reproduces of itself.

``across``
    the comparison the project is about.

The claim the project can make is about where ``across`` falls between the
other two. Close to ``within`` means the objective did not matter. Close to
``random`` means it entirely determined what was learned.

Dead features
-------------

A feature that never fired is an initialisation that was never trained, and
its direction is noise. Including dead features would drag any comparison
toward the random baseline in proportion to how many there are, which would
make a coder look more objective-dependent simply for having wasted capacity.
They are therefore excluded, and the count that remains is reported alongside.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import torch
from torch import Tensor

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# reading a dictionary off disk
# ----------------------------------------------------------------------

#: Where the training loop puts them, relative to a run directory.
CHECKPOINTS = "checkpoints"


def latest_checkpoint(run_dir: Path) -> Path:
    """The last checkpoint a run wrote.

    Args:
        run_dir: a run directory, or the checkpoint directory inside one.

    Returns:
        Path to the newest ``step_*.pt``, by step number rather than by name.

    Raises:
        FileNotFoundError: if the run holds no checkpoint, which means it died
            before the first one or they were pruned away.
    """
    run_dir = Path(run_dir)
    directory = run_dir if run_dir.name == CHECKPOINTS else run_dir / CHECKPOINTS
    found = sorted(directory.glob("step_*.pt"),
                   key=lambda path: int(path.stem.split("_")[1]))
    if not found:
        raise FileNotFoundError(f"no checkpoint under {directory}")
    return found[-1]


def load_dictionary(path: Path, drop_dead: bool = True) -> tuple[Tensor, dict]:
    """The dictionary from a checkpoint, as unit column vectors.

    Args:
        path: a checkpoint file, or a run directory, in which case its last
            checkpoint is used.
        drop_dead: leave out features that never fired.

    Returns:
        ``(directions, info)``. ``directions`` is ``(width, features)`` with
        every column at unit length. ``info`` carries the step, the dictionary
        size before dropping, and how many were dropped.
    """
    path = Path(path)
    if path.is_dir():
        path = latest_checkpoint(path)

    state = torch.load(path, map_location="cpu", weights_only=False)
    weight = state["coder"]["decoder.weight"].float()      # (width, features)

    total = weight.shape[1]
    dropped = 0
    if drop_dead:
        silence = state.get("tracker", {}).get("steps_since_fired")
        if silence is not None:
            # The window is not stored, so a feature counts as dead only if it
            # never fired at all, which is the conservative reading.
            alive = silence < state["step"]
            if alive.any():
                weight = weight[:, alive]
                dropped = int((~alive).sum())
            else:
                log.warning("%s has no live features, keeping all of them", path)

    norms = weight.norm(dim=0, keepdim=True)
    norms = torch.where(norms > 0, norms, torch.ones_like(norms))

    return weight / norms, {
        "checkpoint": str(path),
        "step": int(state.get("step", 0)),
        "features": total,
        "dropped": dropped,
        "kept": weight.shape[1],
    }


# ----------------------------------------------------------------------
# the measure
# ----------------------------------------------------------------------

def best_match(a: Tensor, b: Tensor, chunk: int = 1024) -> Tensor:
    """For each column of ``a``, its highest cosine with any column of ``b``.

    Both are expected to have unit columns, so the cosine is a dot product.
    Done in chunks: the full matrix at a dictionary of twelve thousand is
    several hundred megabytes, and only the row maxima are wanted.

    Args:
        a: ``(width, features_a)``, unit columns.
        b: ``(width, features_b)``, unit columns.
        chunk: columns of ``a`` per pass.

    Returns:
        ``(features_a,)``, the best cosine for each.

    Raises:
        ValueError: if the two are not in the same space, or either is empty.
    """
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"different widths, {a.shape[0]} and {b.shape[0]}")
    if a.shape[1] == 0 or b.shape[1] == 0:
        raise ValueError("an empty dictionary cannot be compared")

    out = torch.empty(a.shape[1], dtype=torch.float32)
    for start in range(0, a.shape[1], chunk):
        block = a[:, start:start + chunk]
        out[start:start + block.shape[1]] = (block.T @ b).max(dim=1).values
    return out


#: Bins the match distributions are recorded in, spanning the whole cosine
#: range so that histograms from different runs can be added together.
BINS = 200


def histogram(values: Tensor, bins: int = BINS) -> list[float]:
    """The distribution of match scores, as fractions summing to one.

    Recorded rather than the raw values: a dictionary has thousands of
    features, the figure only needs the shape, and fractions let histograms
    from dictionaries of different sizes be averaged together.
    """
    counts = torch.histc(values.float(), bins=bins, min=-1.0, max=1.0)
    total = float(counts.sum())
    return (counts / total).tolist() if total else counts.tolist()


def bin_centres(bins: int = BINS) -> list[float]:
    """Where each bin of ``histogram`` sits on the cosine axis."""
    width = 2.0 / bins
    return [-1.0 + width * (index + 0.5) for index in range(bins)]


def similarity(a: Tensor, b: Tensor, close: float = 0.9,
               keep_distribution: bool = False) -> dict:
    """How much of each dictionary the other one contains.

    Args:
        a: ``(width, features_a)``, unit columns.
        b: ``(width, features_b)``, unit columns.
        close: cosine above which two features count as the same feature.
        keep_distribution: also return the shape of the match scores, which is
            what the figure draws.

    Returns:
        Mean best match in each direction, the symmetric mean of the two, the
        median, and the share of features with a counterpart above ``close``.
    """
    forward = best_match(a, b)
    backward = best_match(b, a)

    out = {
        "mean_a_to_b": float(forward.mean()),
        "mean_b_to_a": float(backward.mean()),
        "mean": float((forward.mean() + backward.mean()) / 2),
        "median_a_to_b": float(forward.median()),
        "matched_a_to_b": float((forward >= close).float().mean()),
        "matched_b_to_a": float((backward >= close).float().mean()),
    }
    if keep_distribution:
        out["histogram"] = histogram(torch.cat([forward, backward]))
    return out


def random_baseline(width: int, features: int, trials: int = 3,
                    seed: int = 0, keep_distribution: bool = False) -> dict:
    """The same measure on dictionaries that learned nothing.

    Two sets of random unit directions still match better than zero, and the
    value rises with dictionary size, because the more directions there are the
    closer the nearest one happens to be. Measured rather than assumed, since
    the analytic approximation is only good for large widths.

    Args:
        width: the space both dictionaries live in.
        features: dictionary size.
        trials: how many random pairs to average over.
        seed: generator seed, so the baseline is reproducible.

    Returns:
        The same fields ``similarity`` returns, averaged over the trials.
    """
    generator = torch.Generator().manual_seed(seed)
    totals: dict[str, float] = {}
    shape = [0.0] * BINS

    for _ in range(trials):
        a = torch.randn(width, features, generator=generator)
        b = torch.randn(width, features, generator=generator)
        a = a / a.norm(dim=0, keepdim=True)
        b = b / b.norm(dim=0, keepdim=True)
        scores = similarity(a, b, keep_distribution=keep_distribution)
        if keep_distribution:
            shape = [held + new for held, new in zip(shape, scores.pop("histogram"))]
        for key, value in scores.items():
            totals[key] = totals.get(key, 0.0) + value

    out = {key: value / trials for key, value in totals.items()}
    if keep_distribution:
        out["histogram"] = [value / trials for value in shape]
    return out


# ----------------------------------------------------------------------
# comparing runs
# ----------------------------------------------------------------------

def from_grids(paths: list[Path], run_root: Path) -> list[Path]:
    """The runs a grid produced, read from the grid's own record of them.

    A run directory name carries the objective, the tag, the expansion, the
    sparsity mode, the seed and a hash. It does not carry the coefficient, the
    budget or the depth, so a pattern over names cannot pick out one grid's
    runs: every control run ever trained matches the same pattern. The grid
    file knows exactly which runs it produced, so it is asked instead.

    Args:
        paths: grid result files.
        run_root: where run directories live.

    Returns:
        The run directories, in grid order, deduplicated.
    """
    directories: list[Path] = []
    for path in paths:
        records = json.loads(Path(path).read_text())
        for record in records:
            name = record.get("run_name")
            if not name:
                log.warning("%s has a record with no run name, skipping it", path)
                continue
            directory = Path(run_root) / name
            if not directory.is_dir():
                log.warning("%s is named by %s but is not on disk", directory, path)
                continue
            if directory not in directories:
                directories.append(directory)

    if not directories:
        raise SystemExit(f"no runs found from {[str(path) for path in paths]}")
    return directories


def compare_runs(path_a: Path, path_b: Path, close: float = 0.9,
                 drop_dead: bool = True, keep_distribution: bool = False) -> dict:
    """Two runs, read from disk and compared.

    Returns:
        The similarity fields, plus what was loaded from each side.
    """
    a, info_a = load_dictionary(path_a, drop_dead)
    b, info_b = load_dictionary(path_b, drop_dead)

    record = similarity(a, b, close, keep_distribution)
    record.update({
        "a": str(Path(path_a).name),
        "b": str(Path(path_b).name),
        "a_kept": info_a["kept"],
        "b_kept": info_b["kept"],
        "width": a.shape[0],
    })
    return record


def group_of(name: str) -> str:
    """What a run directory name says the run was.

    Run names are ``<objective>-<tag>-x<expansion>-<sparsity>-s<seed>-<hash>``,
    so everything but the seed and the hash identifies the configuration a run
    belongs to. Two runs in the same group differ only by seed.
    """
    parts = str(name).split("-")
    return "-".join(part for part in parts[:-1] if not part.startswith("s")) or str(name)


def pooled_shapes(records: list[dict]) -> dict[str, list[float]]:
    """The average match distribution for seed repeats and for the comparison.

    Averaged over pairs rather than concatenated, so that a configuration with
    more surviving features does not weigh more heavily than one with fewer.
    """
    groups: dict[str, list[list[float]]] = {"within_shape": [], "across_shape": []}
    for record in records:
        if "histogram" not in record:
            continue
        key = ("within_shape" if group_of(record["a"]) == group_of(record["b"])
               else "across_shape")
        groups[key].append(record["histogram"])

    out = {}
    for key, shapes in groups.items():
        if shapes:
            out[key] = [sum(column) / len(shapes) for column in zip(*shapes)]
    return out


def summarise(records: list[dict], baseline: dict | None = None) -> dict:
    """Within-objective and across-objective, side by side.

    A pair of runs from the same group is a seed repeat and sets the ceiling.
    A pair from different groups is the comparison the project is about.

    Args:
        records: output of ``compare_runs``, one per pair.
        baseline: what random dictionaries score, if it was computed.

    Returns:
        Mean and spread of each kind, and where ``across`` falls between
        ``random`` and ``within`` as a fraction, where 1 means
        indistinguishable from a seed repeat and 0 means no better than random.
    """
    within = [r["mean"] for r in records if group_of(r["a"]) == group_of(r["b"])]
    across = [r["mean"] for r in records if group_of(r["a"]) != group_of(r["b"])]

    def stats(values: list[float]) -> dict:
        if not values:
            return {"n": 0}
        mean = sum(values) / len(values)
        spread = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
        return {"n": len(values), "mean": mean, "std": spread,
                "min": min(values), "max": max(values)}

    out = {"within": stats(within), "across": stats(across)}
    out.update(pooled_shapes(records))
    if baseline is not None:
        out["random"] = baseline["mean"]
        if "histogram" in baseline:
            out["random_shape"] = baseline["histogram"]

    if within and across and baseline is not None:
        floor = baseline["mean"]
        ceiling = out["within"]["mean"]
        if ceiling > floor:
            out["position"] = (out["across"]["mean"] - floor) / (ceiling - floor)

    return out


def write_comparison(records: list[dict], summary: dict, path: Path) -> None:
    """Write the comparisons where a plot or a table can read them.

    Per-pair distributions are left out: the summary holds their average,
    which is what anything downstream reads, and two hundred numbers per pair
    would bury the fields a person opening the file is looking for.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    trimmed = [{key: value for key, value in record.items() if key != "histogram"}
               for record in records]
    path.write_text(json.dumps({"summary": summary, "pairs": trimmed}, indent=2))
    log.info("wrote %d comparisons to %s", len(records), path)