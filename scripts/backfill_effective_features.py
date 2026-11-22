#!/usr/bin/env python3
"""Add the effective feature count to runs trained before it existed.

The firing distribution behind the concentration metrics is written out per
run as an artifact, so the count is arithmetic on a file each run already has
and needs neither the checkpoint nor a forward pass. Nothing is retrained and
no GPU is touched.

    python scripts/backfill_effective_features.py \
        --results grids/v2_coder_case1_control_50M.json \
                  grids/v2_coder_case1_control_budget.json

Scoped by grid file rather than by walking the run directory, so only the runs
named in the grids given are touched.

The recomputation is also a check. The Gini coefficient and the median are
computed here from the saved distribution and compared against what the run
recorded. The distribution is the only input to all three numbers, so a run
whose saved artifact is the one its metrics were computed from has to agree.
Beyond the tolerance the run is left alone and reported, because the artifact
is then not the distribution behind the stored numbers and a count derived
from it would not mean what the other two mean.

A small disagreement is expected rather than alarming. The metrics and the
artifact come from two evaluation passes over the same split at the same batch
count, so they agree exactly only if both passes drew the same tokens. The
observed size of the disagreement is logged for every run, which is the way to
find out whether they did.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stitch.evaluate import frequency_concentration                    # noqa: E402

log = logging.getLogger("backfill")

#: The field this script adds. Anything already carrying it is skipped.
ADDED = ("eval_effective_features",)

#: Recomputed from the saved distribution and compared against what the run
#: stored. Both come from the same function over the same input, so they are a
#: statement about the artifact rather than about the arithmetic.
CHECKED = {"gini": "eval_frequency_gini", "median": "eval_median_feature_frequency"}


def load_frequencies(run_dir: Path) -> torch.Tensor | None:
    """The firing distribution a run wrote out, or None if it did not.

    Written under ``artifacts`` by the launcher and by the earlier backfill
    alike. Runs predating both have no distribution saved and cannot be given
    the count without an evaluation pass, which is a different script.
    """
    path = run_dir / "artifacts" / "feature_frequencies.json"
    if not path.is_file():
        return None
    saved = json.loads(path.read_text())
    return torch.tensor(saved["frequencies"], dtype=torch.float64)


def disagreements(fresh: dict[str, float], record: dict,
                  tolerance: float) -> list[str]:
    """Where the recomputation and the stored metric differ by more than allowed.

    Compared in absolute terms. Both quantities are shares in nought to one, so
    a relative comparison would hold a near zero Gini coefficient to a
    precision the metric does not have.
    """
    wrong = []
    for key, stored_key in CHECKED.items():
        if stored_key not in record:
            continue
        gap = abs(fresh[key] - float(record[stored_key]))
        if gap > tolerance:
            wrong.append(f"{stored_key} {record[stored_key]:.6f} "
                         f"against {fresh[key]:.6f} recomputed")
    return wrong


def report(fresh: dict[str, float], record: dict) -> str:
    """How far the recomputation sits from the stored numbers, for the log."""
    gaps = [f"{stored_key.removeprefix('eval_')} {abs(fresh[key] - float(record[stored_key])):.2e}"
            for key, stored_key in CHECKED.items() if stored_key in record]
    return ", ".join(gaps) if gaps else "nothing stored to compare against"


def backfill_run(record: dict, runs_root: Path, tolerance: float) -> dict:
    """The count for one run, measured from the distribution it saved.

    Returns:
        The new field, or a dict holding ``skipped`` with a reason.
    """
    name = record.get("run_name")
    if not name:
        return {"skipped": "the record names no run"}

    run_dir = runs_root / name
    if not run_dir.is_dir():
        return {"skipped": f"no run directory at {run_dir}"}

    frequencies = load_frequencies(run_dir)
    if frequencies is None:
        return {"skipped": "no saved frequency distribution"}

    fresh = frequency_concentration(frequencies)

    wrong = disagreements(fresh, record, tolerance)
    if wrong:
        return {"skipped": "the saved distribution disagrees with the stored "
                           "metrics: " + "; ".join(wrong)}

    return {"eval_effective_features": fresh["effective"],
            "_report": report(fresh, record)}


def backfill_grid(path: Path, runs_root: Path, tolerance: float,
                  force: bool, dry_run: bool) -> dict[str, int]:
    """Every run named in one grid file.

    The grid file and each run's summary are both updated, since the two are
    read by different things and a number in one and not the other is worse
    than the number being absent from both.
    """
    records = json.loads(path.read_text())
    counts = {"done": 0, "already": 0, "skipped": 0, "failed": 0}

    for index, record in enumerate(records, start=1):
        label = record.get("run_name", record.get("point"))

        if "error" in record:
            log.info("[%d/%d] %s: the run failed, nothing to add",
                     index, len(records), label)
            counts["skipped"] += 1
            continue

        if not force and all(key in record for key in ADDED):
            counts["already"] += 1
            continue

        try:
            added = backfill_run(record, runs_root, tolerance)
        except Exception:                               # noqa: BLE001
            log.exception("[%d/%d] %s: failed", index, len(records), label)
            counts["failed"] += 1
            continue

        if "skipped" in added:
            log.warning("[%d/%d] %s: %s", index, len(records), label, added["skipped"])
            counts["skipped"] += 1
            continue

        checked = added.pop("_report")
        log.info("[%d/%d] %s: effective features %.1f (recomputation differs by %s)",
                 index, len(records), label, added["eval_effective_features"], checked)

        if not dry_run:
            record.update(added)
            summary_path = runs_root / record["run_name"] / "summary.json"
            if summary_path.is_file():
                summary = json.loads(summary_path.read_text())
                summary.update(added)
                summary_path.write_text(json.dumps(summary, indent=2))

        counts["done"] += 1

    if not dry_run and counts["done"]:
        path.write_text(json.dumps(records, indent=2))

    return counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, nargs="+", required=True,
                        help="grid files whose runs to update, and only those")
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--tolerance", type=float, default=1e-3,
                        help="absolute disagreement allowed between a stored "
                             "concentration metric and its recomputation from "
                             "the saved distribution")
    parser.add_argument("--force", action="store_true",
                        help="redo runs that already carry the count")
    parser.add_argument("--dry-run", action="store_true",
                        help="measure and report, writing nothing")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    totals = {"done": 0, "already": 0, "skipped": 0, "failed": 0}
    for path in args.results:
        log.info("--- %s ---", path)
        counts = backfill_grid(path, args.runs, args.tolerance,
                               args.force, args.dry_run)
        for key, value in counts.items():
            totals[key] += value

    log.info("%d updated, %d already current, %d skipped, %d failed",
             totals["done"], totals["already"], totals["skipped"], totals["failed"])
    return 1 if totals["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())