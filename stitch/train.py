"""The training loop.

Pulls token batches from an activation source, takes an optimiser step for each
one, and stops when the token budget is spent. Everything it needs is passed
in, so the same loop trains every objective and the only thing that changes
between them is which layer is the input and which is the target.

Four calls happen around each step, in this order, and the order is not
incidental:

    loss.backward()
    coder.project_decoder_gradient()   # drop the part that changes vector length
    optimiser.step()
    coder.normalize_decoder_columns()  # return the vectors to unit length

Without the projection the optimiser proposes a step the renormalisation then
partly discards, and with momentum the discarded part accumulates in the
optimiser's state.

Checkpoints hold everything needed to continue: the coder, the optimiser, the
dead feature counts, the step reached and the activation scales. A resumed run
is the same run, not a new one that happens to start from the same weights.
"""

from __future__ import annotations

import logging
import math
import random
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import torch
from torch import nn

from stitch.coder import DeadFeatureTracker, SparseCoder, compute_loss

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# setup
# ----------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed every generator a run draws from.

    The comparison between objectives is read against the spread between runs
    that differ by seed alone, so a seed has to control everything that varies.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_layers(cfg) -> tuple[str, str]:
    """Which layer the coder reads and which it reconstructs.

    One layer means the coder reproduces its own input. Two mean the first is
    read and the second is the target, in that order.

    Raises:
        ValueError: on any other number, which would leave the target ambiguous.
    """
    layers = cfg.activations.layers
    if len(layers) == 1:
        return layers[0], layers[0]
    if len(layers) == 2:
        return layers[0], layers[1]
    raise ValueError(
        f"activations.layers must hold one or two entries, got {len(layers)}: {layers}")


def build_optimizer(cfg, coder: SparseCoder) -> torch.optim.Optimizer:
    """The optimiser named in the configuration.

    Raises:
        ValueError: on an unknown name.
    """
    name = cfg.train.optimizer.lower()
    if name == "adam":
        return torch.optim.Adam(coder.parameters(), lr=cfg.train.lr)
    if name == "adamw":
        return torch.optim.AdamW(coder.parameters(), lr=cfg.train.lr)
    raise ValueError(f"unknown train.optimizer {cfg.train.optimizer!r}, expected adam or adamw")


# ----------------------------------------------------------------------
# checkpoints
# ----------------------------------------------------------------------

def save_checkpoint(path: Path, step: int, tokens: int, coder: SparseCoder,
                    tracker: DeadFeatureTracker, optimiser: torch.optim.Optimizer,
                    scales: dict[str, float]) -> None:
    """Write everything needed to continue this run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step,
        "tokens": tokens,
        "coder": coder.state_dict(),
        "tracker": tracker.state_dict(),
        "optimizer": optimiser.state_dict(),
        "scales": scales,
    }, path)


def load_checkpoint(path: Path, coder: SparseCoder, tracker: DeadFeatureTracker,
                    optimiser: torch.optim.Optimizer) -> tuple[int, int]:
    """Restore a run from a checkpoint.

    Returns:
        ``(step, tokens)`` reached when it was written.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    coder.load_state_dict(state["coder"])
    tracker.load_state_dict(state["tracker"])
    optimiser.load_state_dict(state["optimizer"])
    return state["step"], state["tokens"]


def prune_checkpoints(directory: Path, keep: int) -> list[Path]:
    """Delete all but the newest few, so a long run does not fill the disk.

    A coder at the usual width is a few hundred megabytes, and a run writes one
    every few thousand steps. A keep of zero keeps all of them, since keeping
    none would throw away the one the run finished at.

    Returns:
        The checkpoints that remain, oldest first.
    """
    existing = sorted(directory.glob("step_*.pt"))
    if keep > 0:
        for path in existing[:-keep]:
            path.unlink()
    return sorted(directory.glob("step_*.pt"))


# ----------------------------------------------------------------------
# the loop
# ----------------------------------------------------------------------

def train(cfg, source, coder: SparseCoder, tracker: DeadFeatureTracker,
          logger, run_dir: Path,
          evaluate: Callable[[SparseCoder, int], dict] | None = None,
          resume: Path | None = None) -> dict:
    """Train one coder until the token budget is spent.

    Args:
        cfg: the run configuration.
        source: an activation source, supplying token batches and its scales.
        coder: the coder to train.
        tracker: its dead feature tracker.
        logger: anything with ``log_metrics(dict, step)``.
        run_dir: where checkpoints go.
        evaluate: called every ``train.eval_every`` steps with the coder and the
            step, returning metrics to log. Left out, nothing is evaluated
            during training.
        resume: a checkpoint to continue from.

    Returns:
        The metrics from the final step.
    """
    device = torch.device(cfg.train.device)
    coder.to(device)
    tracker.to(device)

    optimiser = build_optimizer(cfg, coder)

    # Parameters stay in float32 whatever train.dtype says. Half precision is a
    # property of the arithmetic, not of the weights; keeping master weights in
    # half loses updates smaller than the gap between representable values.
    half = cfg.train.dtype == "float16" and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=half)

    step, tokens = 0, 0
    if resume is not None:
        step, tokens = load_checkpoint(resume, coder, tracker, optimiser)
        log.info("resumed from %s at step %d", resume, step)

    source_key, target_key = resolve_layers(cfg)
    total_steps = math.ceil(cfg.train.tokens / cfg.train.token_batch)
    checkpoints = run_dir / "checkpoints"

    log.info("training %d steps of %d tokens, reading %s, reconstructing %s",
             total_steps, cfg.train.token_batch, source_key, target_key)

    metrics: dict[str, float] = {}
    batches: Iterator = source.batches()

    try:
        while step < total_steps:
            batch = next(batches)
            x = batch[source_key].to(device, non_blocking=True)
            target = batch[target_key].to(device, non_blocking=True)

            optimiser.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=half):
                loss, metrics = compute_loss(coder, x, target, cfg.coder.sparsity.coeff,
                                             tracker=tracker)

            scaler.scale(loss).backward()

            # Unscaled first, so the projection and the clip see true gradients.
            scaler.unscale_(optimiser)
            coder.project_decoder_gradient()
            if cfg.train.grad_clip is not None:
                grad_norm = nn.utils.clip_grad_norm_(coder.parameters(), cfg.train.grad_clip)
                metrics["grad_norm"] = float(grad_norm)

            scaler.step(optimiser)
            scaler.update()
            coder.normalize_decoder_columns()

            step += 1
            tokens += x.shape[0]

            if step % cfg.train.log_every == 0 or step == total_steps:
                metrics["tokens"] = float(tokens)
                metrics["dead_fraction"] = tracker.dead_fraction
                logger.log_metrics(metrics, step=step)

            if evaluate is not None and step % cfg.train.eval_every == 0:
                logger.log_metrics(evaluate(coder, step), step=step)

            if step % cfg.train.checkpoint_every == 0:
                save_checkpoint(checkpoints / f"step_{step:09d}.pt", step, tokens,
                                coder, tracker, optimiser, source.scales())
                prune_checkpoints(checkpoints, cfg.train.keep_checkpoints)
    finally:
        # Generators hold the loader open, and a loader with workers holds
        # processes open. Closing here covers the crash as well as the finish.
        batches.close()

    save_checkpoint(checkpoints / f"step_{step:09d}.pt", step, tokens,
                    coder, tracker, optimiser, source.scales())
    prune_checkpoints(checkpoints, cfg.train.keep_checkpoints)

    log.info("finished at step %d, %d tokens, mse %.5f, l0 %.1f, dead %.1f%%",
             step, tokens, metrics.get("mse", float("nan")),
             metrics.get("l0", float("nan")), 100 * tracker.dead_fraction)
    return metrics