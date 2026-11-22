"""Logging for the stitching experiments.

Two independent concerns, kept apart:

  experiment tracking   parameters, metrics and artifacts, through
                        :class:`~stitch.logging.base.ExperimentLogger`
  process logging       what the run did and why it stopped, through
                        :mod:`stitch.logging.process`

The training code depends on the abstract interface, never on a specific
tracker. To add a backend, implement :class:`ExperimentLogger` and register it
in :func:`get_logger`; nothing else changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import ExperimentLogger
from .jsonl import JsonlLogger
from .process import setup_logging, install_crash_handler

if TYPE_CHECKING:                       # avoids a circular import at runtime
    from stitch.config import Config

__all__ = [
    "ExperimentLogger",
    "JsonlLogger",
    "get_logger",
    "setup_logging",
    "install_crash_handler",
]


def get_logger(cfg: "Config") -> ExperimentLogger:
    """Build the experiment logger named by ``cfg.logging.backend``.

    The MLflow backend is imported lazily so that a run using the default
    backend needs neither the package nor a server.
    """
    backend = cfg.logging.backend

    if backend == "jsonl":
        return JsonlLogger(run_root=cfg.logging.run_root,
                           csv_mirror=cfg.logging.csv_mirror)

    if backend == "mlflow":
        from .mlflow_logger import MlflowLogger
        return MlflowLogger(
            tracking_uri=cfg.logging.tracking_uri,
            experiment=cfg.logging.experiment,
            run_root=cfg.logging.run_root,
            csv_mirror=cfg.logging.csv_mirror,
        )

    raise ValueError(
        f"unknown logging backend {backend!r}. Known backends: jsonl, mlflow")
