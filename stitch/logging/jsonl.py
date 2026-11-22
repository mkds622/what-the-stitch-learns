"""A dependency-free experiment logger backed by plain files.

This is the default backend and it exists for three reasons:

  - a run works with no tracking server present, which matters when the machine
    is shared and the server is somebody else's
  - tests exercise the same code path the training loop uses, without needing
    MLflow installed
  - the CSV mirror is directly plottable, which is what produces figures for the
    repo without going through a tracking UI

Layout under ``<run_root>/<run_id>/``::

    config.yaml      the resolved configuration
    params.json      flattened parameters, as logged
    metrics.jsonl    one JSON object per log_metrics call
    metrics.csv      the same, as a table, when csv_mirror is on
    status.json      run state, updated on start and end
    run.log          the process log
    artifacts/       anything passed to log_artifact
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from .base import ExperimentLogger


class JsonlLogger(ExperimentLogger):
    """Writes parameters, metrics and artifacts as files in a run directory.

    Args:
        run_root: parent directory for all runs.
        csv_mirror: also maintain ``metrics.csv``. The CSV gains columns as new
            metric names appear, which means the file is rewritten when the
            column set changes. That is cheap at the sizes involved and keeps
            the file readable by anything that reads CSV.
    """

    def __init__(self, run_root: str | Path = "runs", csv_mirror: bool = True):
        self._root = Path(run_root)
        self._csv_mirror = csv_mirror
        self._run_id: str | None = None
        self._metrics_fh = None
        self._csv_columns: list[str] = []
        self._csv_rows: list[dict] = []
        self._started: float = 0.0
        self._ended = False

    # ------------------------------------------------------------------

    def start_run(self, name: str, tags: dict[str, str] | None = None) -> str:
        self._run_id = name
        self.run_dir = self._root / name
        (self.run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        self._metrics_fh = open(self.run_dir / "metrics.jsonl", "a", buffering=1)
        self._started = time.time()
        self._ended = False
        self._write_status("RUNNING", tags or {})
        return name

    def log_params(self, params: dict[str, Any]) -> None:
        self._require_run()
        # default=str so that paths, enums and anything else non-JSON still land
        # rather than losing the whole parameter set to one awkward value.
        (self.run_dir / "params.json").write_text(
            json.dumps(params, indent=2, sort_keys=True, default=str))

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        self._require_run()
        row = {"step": step, "wall": round(time.time() - self._started, 3)}
        row.update({k: _scalar(v) for k, v in metrics.items()})

        self._metrics_fh.write(json.dumps(row) + "\n")

        if self._csv_mirror:
            self._csv_rows.append(row)
            new = [k for k in row if k not in self._csv_columns]
            if new:
                # Column set changed, so the whole file is rewritten with the
                # wider header. Only happens when a new metric first appears.
                self._csv_columns.extend(new)
                self._rewrite_csv()
            else:
                with open(self.run_dir / "metrics.csv", "a", newline="") as fh:
                    csv.DictWriter(fh, self._csv_columns).writerow(row)

    def log_artifact(self, path: str | Path, subdir: str | None = None) -> None:
        self._require_run()
        src = Path(path)
        if not src.exists():
            # An artifact that does not exist is a bug in the caller, but losing
            # the run over it would be worse than carrying on without the file.
            return
        dest_dir = self.run_dir / "artifacts" / (subdir or "")
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest_dir / src.name)

    def log_dict(self, obj: Any, name: str) -> None:
        self._require_run()
        dest = self.run_dir / "artifacts" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))

    def end_run(self, status: str = "FINISHED") -> None:
        if self._ended or self._run_id is None:
            return          # idempotent: the crash handler may double-call
        self._ended = True
        if self._metrics_fh is not None:
            self._metrics_fh.flush()
            os.fsync(self._metrics_fh.fileno())
            self._metrics_fh.close()
            self._metrics_fh = None
        self._write_status(status, {})

    # ------------------------------------------------------------------

    def _require_run(self) -> None:
        if self._run_id is None:
            raise RuntimeError("start_run must be called before logging")

    def _write_status(self, status: str, tags: dict[str, str]) -> None:
        """Record run state on disk.

        Read by the crash viewer and by anyone wondering whether a run is still
        going. Written on start and on end, so a process killed outright leaves
        RUNNING behind, which is itself the signal that it did not exit cleanly.
        """
        payload = {
            "run_id": self._run_id,
            "status": status,
            "started": self._started,
            "started_iso": time.strftime("%Y-%m-%dT%H:%M:%S",
                                         time.localtime(self._started)),
            "elapsed_s": round(time.time() - self._started, 1),
            "pid": os.getpid(),
        }
        if tags:
            payload["tags"] = tags
        (self.run_dir / "status.json").write_text(json.dumps(payload, indent=2))

    def _rewrite_csv(self) -> None:
        with open(self.run_dir / "metrics.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, self._csv_columns)
            w.writeheader()
            w.writerows(self._csv_rows)


def _scalar(v) -> float:
    """Coerce a metric value to float, unwrapping single-element tensors.

    Keeps the logger free of a torch import while still accepting tensors, which
    is what the training loop naturally produces.
    """
    item = getattr(v, "item", None)
    if callable(item):
        try:
            return float(item())
        except Exception:
            pass
    return float(v)
