#!/usr/bin/env python3
"""Load finished runs into MLflow.

Runs write JSONL to their own directory, which is deliberate: the record
survives whether or not a tracking server was up, and a crashed run is still
diagnosable when the thing that failed was the tracker. This reads those
directories and replays them into MLflow, so the viewer is a view of the
record rather than the record itself.

It can be run at any time, including on runs that finished before MLflow
existed, and run again later without duplicating anything.

Crashes are loaded into an experiment of their own, so that the table view is
one row per failure with the exception, the message and the configuration hash
as columns, rather than something to be found by filtering a list of runs.

    python scripts/import_runs.py                       # everything in runs/
    python scripts/import_runs.py --run runs/case1-control-x16-l1-s0-c2a98d
    python scripts/import_runs.py --tracking-uri sqlite:///mlflow.db
    python scripts/import_runs.py --no-crashes          # runs only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

log = logging.getLogger("import_runs")


def read_json(path: Path, default=None):
    """One JSON file, or the default if it is absent or unreadable."""
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        log.warning("%s is not valid JSON, skipping it", path)
        return default


def read_metrics(path: Path) -> list[dict]:
    """The per-step metric lines, skipping any that are truncated.

    A run killed mid-write leaves a partial last line. That is one step out of
    thousands, and losing it is better than refusing to import the run.
    """
    if not path.exists():
        return []

    rows = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            log.warning("%s line %d is incomplete, stopping there", path, number)
            break
    return rows


def already_imported(client, experiment_id: str, run_name: str) -> bool:
    """Whether this run directory has been loaded before.

    Matched on the directory name, which carries the configuration hash, so
    two runs of different configurations cannot be mistaken for each other.
    """
    found = client.search_runs(
        [experiment_id], filter_string=f"tags.source_run = '{run_name}'", max_results=1)
    return len(found) > 0


def import_run(directory: Path, experiment: str, dry_run: bool = False) -> bool:
    """Replay one run directory into MLflow.

    Returns:
        True if it was imported, False if it was skipped.
    """
    import mlflow

    params = read_json(directory / "params.json", {})
    metrics = read_metrics(directory / "metrics.jsonl")
    summary = read_json(directory / "summary.json", {})
    status = read_json(directory / "status.json", {})

    if not params and not metrics:
        log.info("%s holds no params or metrics, skipping", directory.name)
        return False

    client = mlflow.tracking.MlflowClient()
    existing = client.get_experiment_by_name(experiment)
    experiment_id = existing.experiment_id if existing else client.create_experiment(experiment)

    if already_imported(client, experiment_id, directory.name):
        log.info("%s already imported", directory.name)
        return False

    if dry_run:
        log.info("%s: %d params, %d metric rows, %d summary values",
                 directory.name, len(params), len(metrics), len(summary))
        return False

    with mlflow.start_run(experiment_id=experiment_id, run_name=directory.name):
        mlflow.set_tag("source_run", directory.name)
        mlflow.set_tag("status", status.get("status", "UNKNOWN"))

        # MLflow caps a parameter value's length, and a long path or note would
        # otherwise fail the whole import.
        mlflow.log_params({key: str(value)[:480] for key, value in params.items()})

        for row in metrics:
            step = int(row.get("step", 0))
            numbers = {key: float(value) for key, value in row.items()
                       if key not in ("step", "wall") and isinstance(value, (int, float))}
            if numbers:
                mlflow.log_metrics(numbers, step=step)

        # The final numbers again, without a step, so they sort as the run's result.
        finals = {f"final_{key}": float(value) for key, value in summary.items()
                  if isinstance(value, (int, float))}
        if finals:
            mlflow.log_metrics(finals)

        for name in ("config.json", "scales.json", "summary.json", "run.log", "crash.json"):
            path = directory / name
            if path.exists():
                mlflow.log_artifact(str(path))

    log.info("imported %s: %d metric rows", directory.name, len(metrics))
    return True


def import_crash(path: Path, experiment: str, dry_run: bool = False) -> bool:
    """Load one crash report as a row in the crash experiment.

    Fields become parameters rather than tags, because the table view shows
    parameters as columns and that is the point of putting them here.
    """
    import mlflow

    report = read_json(path, {})
    if not report:
        return False

    client = mlflow.tracking.MlflowClient()
    existing = client.get_experiment_by_name(experiment)
    experiment_id = existing.experiment_id if existing else client.create_experiment(experiment)

    if already_imported(client, experiment_id, path.name):
        return False

    if dry_run:
        log.info("%s: %s", path.name, report.get("exception", "?"))
        return False

    context = report.get("context", {})
    with mlflow.start_run(experiment_id=experiment_id, run_name=path.stem):
        mlflow.set_tag("source_run", path.name)
        mlflow.log_params({
            "exception": str(report.get("exception", ""))[:480],
            "message": str(report.get("message", ""))[:480],
            "when": str(report.get("time", report.get("timestamp", "")))[:480],
            **{f"context.{key}": str(value)[:480] for key, value in context.items()},
        })
        mlflow.log_artifact(str(path))

    log.info("imported crash %s: %s", path.name, report.get("exception", "?"))
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=Path("runs"),
                        help="directory holding run directories")
    parser.add_argument("--run", type=Path, default=None,
                        help="one run directory instead of all of them")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--experiment", default="stitch")
    parser.add_argument("--crash-experiment", default="stitch-crashes")
    parser.add_argument("--no-crashes", action="store_true",
                        help="skip the crash reports")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be imported and stop")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        import mlflow
    except ImportError:
        raise SystemExit("mlflow is not installed: pip install mlflow")

    mlflow.set_tracking_uri(args.tracking_uri)

    if args.run is not None:
        directories = [args.run]
    elif args.runs.is_dir():
        directories = sorted(path for path in args.runs.iterdir() if path.is_dir())
    else:
        raise SystemExit(f"no such directory: {args.runs}")

    directories = [d for d in directories if d.name != "crashes"]
    imported = sum(import_run(d, args.experiment, args.dry_run) for d in directories)
    print(f"\n{imported} of {len(directories)} runs imported into {args.tracking_uri}")

    if not args.no_crashes:
        crash_dir = args.runs / "crashes"
        reports = sorted(crash_dir.glob("*.json")) if crash_dir.is_dir() else []
        crashed = sum(import_crash(p, args.crash_experiment, args.dry_run) for p in reports)
        print(f"{crashed} of {len(reports)} crashes imported into "
              f"experiment {args.crash_experiment!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())