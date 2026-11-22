"""The experiment logger interface.

Nothing outside this package imports MLflow, or any other tracking library. The
training code holds an :class:`ExperimentLogger` and calls six methods on it.
Swapping backend is then a matter of writing one class and adding a line to
:func:`stitch.logging.get_logger`, rather than editing the training loop.

The interface is deliberately small. Every method here maps onto something every
mainstream tracker supports, so nothing in it constrains the choice of backend.
Anything a particular backend can do beyond this belongs behind that backend's
own configuration, not in this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class ExperimentLogger(ABC):
    """Records parameters, metrics and artifacts for a single run.

    Lifecycle: :meth:`start_run` once, then any number of :meth:`log_params`,
    :meth:`log_metrics`, :meth:`log_artifact` and :meth:`log_dict` calls, then
    exactly one :meth:`end_run`. Implementations must tolerate ``end_run`` being
    called twice, since the crash handler calls it on a path where the normal
    call may also have run.
    """

    #: Directory for this run's own files. Set by start_run. The process log and
    #: any crash reports are written here regardless of backend, so a crash
    #: remains diagnosable even when the backend itself is what failed.
    run_dir: Path

    @abstractmethod
    def start_run(self, name: str, tags: dict[str, str] | None = None) -> str:
        """Begin a run and return its identifier."""

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:
        """Record settings that do not change during the run.

        Called once, with the flattened configuration. Backends generally treat
        parameters as immutable, so this must not be used for anything that
        varies over training.
        """

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Record measurements at a point in training.

        Values must be scalar. Anything structured goes through
        :meth:`log_dict` or :meth:`log_artifact` instead.
        """

    @abstractmethod
    def log_artifact(self, path: str | Path, subdir: str | None = None) -> None:
        """Attach a file to the run, such as a checkpoint or the process log."""

    @abstractmethod
    def log_dict(self, obj: Any, name: str) -> None:
        """Attach structured data to the run, serialised as JSON."""

    @abstractmethod
    def end_run(self, status: str = "FINISHED") -> None:
        """Close the run.

        ``status`` is ``FINISHED``, ``FAILED`` or ``KILLED``. Marking a crashed
        run as FAILED is what distinguishes it from one that is still going,
        which is otherwise impossible to tell apart from outside.
        """

    # ------------------------------------------------------------------
    # convenience, shared by every backend

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.end_run("FAILED" if exc_type is not None else "FINISHED")
        return False    # never swallow the exception
