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
import re
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


#: Which side of a coder a reading describes. The decoder holds what a feature
#: writes out, the encoder what it looks for. They are the same features either
#: way, so the same live mask applies to both.
SIDES = ("decoder", "encoder")


def _directions(state: dict, side: str, drop_dead: bool,
                path: Path) -> tuple[Tensor, dict]:
    """One side of a loaded checkpoint, as unit column vectors.

    The decoder is stored as ``(width out, features)``, so its columns are
    already the directions. The encoder is stored as ``(features, width in)``,
    one row per feature, so it is transposed into the same shape and
    everything downstream is indifferent to which side it was given.
    """
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")

    weight = state["coder"][f"{side}.weight"].float()
    if side == "encoder":
        weight = weight.T                                  # (width in, features)

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


def load_dictionary(path: Path, drop_dead: bool = True,
                    side: str = "decoder") -> tuple[Tensor, dict]:
    """One side of a coder from a checkpoint, as unit column vectors.

    Args:
        path: a checkpoint file, or a run directory, in which case its last
            checkpoint is used.
        drop_dead: leave out features that never fired.
        side: ``decoder`` for the directions a feature writes out, ``encoder``
            for the directions it reads. Which one is meaningful depends on
            what is being compared; see :func:`compare_runs`.

    Returns:
        ``(directions, info)``. ``directions`` is ``(width, features)`` with
        every column at unit length. ``info`` carries the step, the dictionary
        size before dropping, and how many were dropped.
    """
    path = Path(path)
    if path.is_dir():
        path = latest_checkpoint(path)

    state = torch.load(path, map_location="cpu", weights_only=False)
    return _directions(state, side, drop_dead, path)


def load_sides(path: Path, drop_dead: bool = True) -> dict[str, tuple[Tensor, dict]]:
    """Both sides of a coder, from one read of the checkpoint."""
    path = Path(path)
    if path.is_dir():
        path = latest_checkpoint(path)

    state = torch.load(path, map_location="cpu", weights_only=False)
    return {side: _directions(state, side, drop_dead, path) for side in SIDES}


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

    The size passed must be the number of features actually compared, not the
    configured dictionary size. Dead features are dropped before two coders are
    compared, so a floor measured over the full dictionary is a floor for a
    larger problem than the one being solved, and it is one of the two anchors
    the reported position sits between.

    Args:
        width: the space both dictionaries live in.
        features: dictionary size, meaning the live count.
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
    """Two runs, read from disk and compared on both sides of the coder.

    The decoder reading is the one the unprefixed fields carry, and it is the
    right one whenever both coders write into the same space: two runs of the
    single-model objective, or two of the cross-model objective.

    It is the wrong one across the two objectives. A coder reproducing its own
    input has decoder columns in the space it read; a coder reconstructing the
    other model has them in that model's space. Those are two independently
    trained networks and nothing ever aligned their coordinates, so a cosine
    between them compares a direction measured in one basis against a direction
    measured in another. Both halves of each pair here are the same width, so
    that comparison does not fail. It returns a number near the floor, which
    would read as two objectives having learned different features when the
    reason is that the question has no content.

    The encoder reading, under ``encoder_`` names, is the one that survives
    that. Both objectives read the same activation, so the directions a feature
    looks for are in one space whatever it writes out, and the question of
    whether two coders carved the same features out of it is answerable.

    What the encoder reading does not see: under TopK a feature's direction
    does not by itself decide whether it fires, since the active set is chosen
    in competition across the dictionary. Two coders can look for similar
    directions and still fire on different tokens.

    Returns:
        The similarity fields for the decoder, the same fields prefixed
        ``encoder_``, and what was loaded from each run.
    """
    sides_a = load_sides(path_a, drop_dead)
    sides_b = load_sides(path_b, drop_dead)

    a, info_a = sides_a["decoder"]
    b, info_b = sides_b["decoder"]

    record = similarity(a, b, close, keep_distribution)

    encoder_a = sides_a["encoder"][0]
    encoder_b = sides_b["encoder"][0]
    for key, value in similarity(encoder_a, encoder_b, close,
                                 keep_distribution).items():
        record[f"encoder_{key}"] = value

    record.update({
        "a": str(Path(path_a).name),
        "b": str(Path(path_b).name),
        # Resolved here, where both run directories are in hand, rather than
        # downstream from the names alone, which cannot see the depth.
        "a_group": group_of(path_a),
        "b_group": group_of(path_b),
        "a_kept": info_a["kept"],
        "b_kept": info_b["kept"],
        "width": a.shape[0],
        "encoder_width": encoder_a.shape[0],
    })
    return record


def same_configuration(record: dict) -> bool:
    """Whether a pair differs only by seed, so sets the ceiling rather than
    being the comparison.

    Reads the groups the pair was recorded with. A record written before those
    were carried falls back to its names, which is the older and weaker rule.
    """
    if "a_group" in record and "b_group" in record:
        return record["a_group"] == record["b_group"]
    return group_of(record["a"]) == group_of(record["b"])


def _group_from_params(path: Path) -> str:
    """A configuration's identity, from the settings the run recorded.

    Everything except the seed. Two runs are a seed repeat exactly when every
    other setting matches, which is what the ceiling is supposed to be measured
    from.
    """
    import hashlib

    stored = json.loads(Path(path).read_text())
    stored.pop("train.seed", None)
    blob = json.dumps(stored, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


#: A seed, as a run name spells it: the letter followed by digits and nothing
#: else. Matching on the prefix alone would also swallow a tag beginning with
#: the same letter and merge two configurations into one group.
SEED_PART = re.compile(r"^s\d+$")


def _group_from_name(name: str) -> str:
    """The same identity, from the directory name, where that is all there is.

    Names are ``<objective>-<tag>-x<expansion>-<sparsity>-s<seed>-<hash>``. The
    name carries neither the depth nor the sparsity setting, so two runs that
    differ only in those are indistinguishable here. It is a fallback for a run
    whose parameters are missing, and the caller is warned.
    """
    parts = str(name).split("-")
    return "-".join(part for part in parts[:-1] if not SEED_PART.match(part)) or str(name)


def group_of(run: str | Path) -> str:
    """Which configuration a run belongs to, for telling a seed repeat apart.

    Args:
        run: a run directory, or a bare run name.

    Returns:
        A key that is equal for two runs differing only by seed.

    The directory is preferred because the name is not enough. It carries the
    objective, the tag, the expansion, the sparsity mode, the seed and a hash,
    and not the depth, the sparsity setting or the budget. In the main grid the
    depths are separate configurations differing only in which layers they
    read, so two runs at different depths share a name shape, and a ceiling
    measured over them would be measuring depth rather than seed.
    """
    path = Path(run)
    params = path / "params.json"
    if params.is_file():
        return _group_from_params(params)

    if path.is_dir():
        log.warning("%s has no params.json, so its configuration is read from its "
                    "name, which cannot see the depth", path)
    return _group_from_name(path.name)


def pooled_shapes(records: list[dict],
                  shape_key: str = "histogram") -> dict[str, list[float]]:
    """The average match distribution for seed repeats and for the comparison.

    Averaged over pairs rather than concatenated, so that a configuration with
    more surviving features does not weigh more heavily than one with fewer.
    """
    groups: dict[str, list[list[float]]] = {"within_shape": [], "across_shape": []}
    for record in records:
        if shape_key not in record:
            continue
        key = "within_shape" if same_configuration(record) else "across_shape"
        groups[key].append(record[shape_key])

    out = {}
    for key, shapes in groups.items():
        if shapes:
            out[key] = [sum(column) / len(shapes) for column in zip(*shapes)]
    return out


def summarise(records: list[dict], baseline: dict | None = None,
              key: str = "mean", shape_key: str = "histogram") -> dict:
    """Within-objective and across-objective, side by side.

    A pair of runs from the same group is a seed repeat and sets the ceiling.
    A pair from different groups is the comparison the project is about.

    Args:
        records: output of ``compare_runs``, one per pair.
        baseline: what random dictionaries score, if it was computed.
        key: which reading to summarise, ``mean`` for the decoder and
            ``encoder_mean`` for the encoder. A floor, a ceiling and an across
            number only sit on one scale if all three are the same reading, so
            this is what keeps a chain from being assembled out of two.
        shape_key: the matching distribution field, which moves with ``key``.

    Returns:
        Mean and spread of each kind, and where ``across`` falls between
        ``random`` and ``within`` as a fraction, where 1 means
        indistinguishable from a seed repeat and 0 means no better than random.
    """
    within = [r[key] for r in records if same_configuration(r)]
    across = [r[key] for r in records if not same_configuration(r)]

    def stats(values: list[float]) -> dict:
        if not values:
            return {"n": 0}
        mean = sum(values) / len(values)
        spread = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
        return {"n": len(values), "mean": mean, "std": spread,
                "min": min(values), "max": max(values)}

    out = {"within": stats(within), "across": stats(across)}
    out.update(pooled_shapes(records, shape_key))
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
    trimmed = [{key: value for key, value in record.items()
                if key not in ("histogram", "encoder_histogram")}
               for record in records]
    path.write_text(json.dumps({"summary": summary, "pairs": trimmed}, indent=2))
    log.info("wrote %d comparisons to %s", len(records), path)