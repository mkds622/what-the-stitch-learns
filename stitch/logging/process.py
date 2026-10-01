"""Process logging and crash capture.

Separate from experiment tracking, and deliberately so. Experiment tracking
answers "what did this run measure". Process logging answers "what did this run
do, and if it stopped, why". They share a directory and nothing else.

Two things are set up per run:

  setup_logging      the standard library logger, writing to both the console
                     and ``<run_dir>/run.log``
  install_crash_handler
                     an excepthook that captures the traceback to a timestamped
                     file under ``crashes/``, marks the run FAILED, and flushes
                     everything before the process dies

The crash file is written to a shared ``crashes/`` directory as well as to the
run directory, so that every failure across every run can be listed in one place
without walking the run tree. ``scripts/crash_viewer.py`` renders that directory
as a single HTML page, newest first.

A run killed outright, by the OOM killer or SIGKILL, cannot write anything. That
case is detectable instead by a ``status.json`` that still says RUNNING with no
process behind the recorded pid.
"""

from __future__ import annotations

import faulthandler
import json
import logging
import os
import signal
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)-22s %(message)s"
DATE_FORMAT = "%H:%M:%S"


def setup_logging(run_dir: str | Path, level: str = "INFO",
                  console: bool = True) -> logging.Logger:
    """Configure the root logger for one run.

    Writes to ``<run_dir>/run.log`` and, by default, to stderr. Existing
    handlers are removed first so that repeated calls within one process, which
    happens when a sweep drives several runs, do not duplicate every line.

    Args:
        run_dir: the run's own directory. Created if absent.
        level: threshold for both handlers.
        console: also write to stderr. Turned off by tests.

    Returns:
        The package logger, for the caller's convenience.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()

    root.setLevel(level.upper())
    fmt = logging.Formatter(LOG_FORMAT, DATE_FORMAT)

    fh = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    if console:
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(fmt)
        root.addHandler(sh)

    # Third-party libraries are noisy at DEBUG and rarely informative here.
    for noisy in ("urllib3", "matplotlib", "PIL", "git", "botocore", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logging.getLogger("stitch")


def install_crash_handler(run_dir: str | Path,
                          crash_dir: str | Path = "crashes",
                          logger: Any = None,
                          context: dict | None = None) -> None:
    """Capture unhandled exceptions and fatal signals.

    On an unhandled exception the traceback is written to a timestamped file in
    both the run directory and the shared crash directory, the experiment logger
    is closed with status FAILED, and handlers are flushed before the default
    excepthook runs.

    ``faulthandler`` is also enabled, which catches segfaults and similar faults
    that never reach Python's exception machinery. Those produce a raw C-level
    traceback in ``run.log``, which is usually enough to identify the offending
    call.

    Args:
        run_dir: the run's own directory.
        crash_dir: shared directory collecting crashes from every run.
        logger: an ExperimentLogger to mark FAILED. Optional.
        context: extra fields recorded alongside the traceback, typically the
            run name and the configuration hash.
    """
    run_dir = Path(run_dir)
    crash_dir = Path(crash_dir)
    crash_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("stitch.crash")

    # Faults that bypass Python exceptions, such as a segfault in a CUDA kernel.
    fault_log = open(run_dir / "faults.log", "a", buffering=1)
    faulthandler.enable(file=fault_log, all_threads=True)

    previous_hook = sys.excepthook

    def _record(exc_type, exc, tb, cause: str) -> Path:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        name = f"{stamp}_{run_dir.name}.json"
        payload = {
            "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "run": run_dir.name,
            "run_dir": str(run_dir.resolve()),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "cause": cause,
            "exception": exc_type.__name__ if exc_type else cause,
            "message": str(exc) if exc else "",
            "traceback": "".join(traceback.format_exception(exc_type, exc, tb))
            if exc_type else "",
            "context": context or {},
        }
        body = json.dumps(payload, indent=2)
        (crash_dir / name).write_text(body)
        (run_dir / "crash.json").write_text(body)
        return crash_dir / name

    def _excepthook(exc_type, exc, tb):
        # KeyboardInterrupt is a deliberate stop, not a crash. Recorded as such
        # so it does not clutter the crash list.
        cause = "interrupt" if issubclass(exc_type, KeyboardInterrupt) else "exception"
        try:
            path = _record(exc_type, exc, tb, cause)
            log.error("run failed: %s: %s", exc_type.__name__, exc)
            log.error("crash report written to %s", path)
        except Exception:
            # Never let the crash handler itself hide the original failure.
            traceback.print_exc()
        finally:
            _shutdown(logger, "KILLED" if cause == "interrupt" else "FAILED")
            previous_hook(exc_type, exc, tb)

    sys.excepthook = _excepthook

    # SIGTERM arrives when a scheduler or a user stops the job. It is not an
    # exception, so without this the run would simply vanish mid-training with
    # status still RUNNING.
    def _on_term(signum, frame):
        log.warning("received signal %s, shutting down", signum)
        _record(None, None, None, f"signal {signum}")
        _shutdown(logger, "KILLED")
        sys.exit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, _on_term)
        except (ValueError, OSError, AttributeError):
            # Not the main thread, or the platform lacks the signal. Neither is
            # worth failing the run over.
            pass


def _shutdown(logger, status: str) -> None:
    """Close the experiment logger and flush log handlers, tolerating failure."""
    try:
        if logger is not None:
            logger.end_run(status)
    except Exception:
        traceback.print_exc()
    logging.shutdown()
