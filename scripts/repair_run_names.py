#!/usr/bin/env python3
"""Make recorded run names agree with the directories they describe.

A run's directory was named at the start of the run and its summary recorded a
name computed at the end. Opening the dataset writes the class count it found
back into the configuration, the hash covers every field, so the two names
differ for every run trained before that was fixed. Grid files copied the
second name, so nothing downstream could find the run on disk.

The directory name is the one to keep. It is what the run wrote under, what
every log and checkpoint sits inside, and what a run produced after the fix
will record. So this rewrites the records to match the directories, not the
other way round, and nothing is moved or renamed on disk.

    python scripts/repair_run_names.py --dry-run
    python scripts/repair_run_names.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

log = logging.getLogger("repair")


def summaries(run_root: Path) -> list[Path]:
    return sorted(run_root.glob("*/summary.json"))


def repair_summaries(run_root: Path, dry_run: bool) -> dict[str, str]:
    """Point every summary at its own directory.

    Returns:
        A map from the name that was recorded to the directory's name, for the
        grid files to be rewritten with.
    """
    renames: dict[str, str] = {}

    for path in summaries(run_root):
        directory = path.parent.name
        try:
            summary = json.loads(path.read_text())
        except json.JSONDecodeError as error:
            log.error("%s is not readable: %s", path, error)
            continue

        recorded = summary.get("run_name")
        if recorded == directory:
            continue
        if recorded:
            renames[recorded] = directory

        log.info("%s recorded itself as %s", directory, recorded)
        if dry_run:
            continue

        summary["run_name"] = directory
        if recorded and "recorded_run_name" not in summary:
            # Kept so the repair is reversible and the old grids stay readable.
            summary["recorded_run_name"] = recorded
        path.write_text(json.dumps(summary, indent=2))

    return renames


def repair_grids(paths: list[Path], renames: dict[str, str], dry_run: bool) -> int:
    """Rewrite the run names a grid recorded, leaving its metrics untouched."""
    changed = 0

    for path in paths:
        try:
            records = json.loads(path.read_text())
        except json.JSONDecodeError as error:
            log.error("%s is not readable: %s", path, error)
            continue
        if not isinstance(records, list):
            log.warning("%s is not a list of records, skipping", path)
            continue

        hits = 0
        for record in records:
            new = renames.get(record.get("run_name"))
            if new:
                record["run_name"] = new
                hits += 1

        if not hits:
            continue
        changed += 1
        log.info("%s: %d of %d names repointed", path.name, hits, len(records))
        if not dry_run:
            path.write_text(json.dumps(records, indent=2))

    return changed


def missing(run_root: Path, paths: list[Path]) -> list[str]:
    """Grid entries that still name no directory, after the repair."""
    out = []
    for path in paths:
        try:
            records = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(records, list):
            continue
        for record in records:
            name = record.get("run_name")
            if name and not (run_root / name).is_dir():
                out.append(f"{path.name}: {name}")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--grids", type=Path, default=Path("grids"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    grid_files = sorted(args.grids.glob("*.json"))
    renames = repair_summaries(args.run_root, args.dry_run)
    changed = repair_grids(grid_files, renames, args.dry_run)

    print(f"\n{len(renames)} summaries disagreed with their directory")
    print(f"{changed} of {len(grid_files)} grid files {'would be' if args.dry_run else ''} updated")

    if not args.dry_run:
        left = missing(args.run_root, grid_files)
        if left:
            print(f"\n{len(left)} grid entries still name no directory:")
            for item in left[:20]:
                print(f"  {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())