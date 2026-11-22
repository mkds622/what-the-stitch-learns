#!/usr/bin/env python3
"""Rename the silent-feature metric in runs that predate the new name.

``eval_never_fired`` counted features that fired on none of the evaluation
tokens. That is two groups together: features that never fired in the whole
training run, and features that fired during training and not inside the
evaluation window. The evaluation cannot tell them apart, and the old name
claimed a distinction it could not make, so the metric is now
``eval_dead_or_dormant`` and its threshold is ``eval.max_allowed_silent``.

    python scripts/rename_silent_metric.py \\
        --results grids/topk_*.json

Nothing is recomputed and no value changes. The same number is written under
the new key, in the grid files named, in each run's summary, and in each run's
recorded configuration so a run can still be rebuilt from what it saved.

Scoped by grid file so only the runs named in the grids given are touched, and
safe to run more than once: a record already carrying the new name is left
alone.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

log = logging.getLogger("rename")

#: Metric keys, in results files and run summaries.
METRICS = {"eval_never_fired": "eval_dead_or_dormant"}

#: Configuration keys, in each run's recorded parameters. Dotted, as a run
#: writes them.
SETTINGS = {"eval.max_never_fired": "eval.max_allowed_silent"}


def rename_in(payload: dict, mapping: dict[str, str]) -> int:
    """Rename keys in one mapping, in place. Returns how many keys changed.

    A key already under its new name is left as it is, and an old key sitting
    beside a new one is dropped rather than overwriting it, since the new one
    is the value anything downstream has been reading.

    A drop counts as a change. It is the case a half-finished earlier pass
    leaves behind, and reporting it as nothing to do would write the file back
    with the stale key still in it.
    """
    changed = 0
    for old, new in mapping.items():
        if old not in payload:
            continue
        if new in payload:
            payload.pop(old)
        else:
            payload[new] = payload.pop(old)
        changed += 1
    return changed


def rewrite(path: Path, mapping: dict[str, str], dry_run: bool) -> int:
    """One JSON file holding a single object."""
    if not path.is_file():
        return 0
    payload = json.loads(path.read_text())
    moved = rename_in(payload, mapping)
    if moved and not dry_run:
        path.write_text(json.dumps(payload, indent=2))
    return moved


def rename_grid(path: Path, runs_root: Path, dry_run: bool) -> dict[str, int]:
    """One grid file, and the runs it names."""
    records = json.loads(path.read_text())
    counts = {"records": 0, "summaries": 0, "params": 0, "missing": 0}

    for record in records:
        if rename_in(record, METRICS):
            counts["records"] += 1

        name = record.get("run_name")
        if not name:
            continue
        run_dir = runs_root / name
        if not run_dir.is_dir():
            counts["missing"] += 1
            continue

        counts["summaries"] += bool(rewrite(run_dir / "summary.json", METRICS, dry_run))
        counts["params"] += bool(rewrite(run_dir / "params.json", SETTINGS, dry_run))

    if counts["records"] and not dry_run:
        path.write_text(json.dumps(records, indent=2))

    return counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, nargs="+", required=True,
                        help="grid files to rename in, and the runs they name")
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be renamed, writing nothing")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    totals = {"records": 0, "summaries": 0, "params": 0, "missing": 0}
    for path in args.results:
        if not path.is_file():
            log.warning("%s does not exist", path)
            continue
        counts = rename_grid(path, args.runs, args.dry_run)
        log.info("%s: %d records, %d summaries, %d configurations%s",
                 path, counts["records"], counts["summaries"], counts["params"],
                 f", {counts['missing']} runs not found" if counts["missing"] else "")
        for key, value in counts.items():
            totals[key] += value

    log.info("%s%d records, %d summaries, %d configurations renamed",
             "would rename: " if args.dry_run else "",
             totals["records"], totals["summaries"], totals["params"])
    if totals["missing"]:
        log.warning("%d runs named in a grid had no directory", totals["missing"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())