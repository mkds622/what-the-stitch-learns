#!/usr/bin/env python3
"""Prove the configuration and logging plumbing without training anything.

Runs the same sequence a real training run will: load the configuration, report
unresolved placeholders, open a run, arm crash capture, log parameters, emit
metrics on a loop, close cleanly. The loop computes a decaying number rather
than a loss; nothing here is a model.

This exists so that failures in the plumbing surface on a laptop in seconds
rather than on a GPU node twenty minutes into a real run.

Usage:
    python scripts/smoke_plumbing.py
    python scripts/smoke_plumbing.py --config configs/case1_smoke.yaml
    python scripts/smoke_plumbing.py --steps 50 train.seed=3
    python scripts/smoke_plumbing.py --crash        # exercise the crash path
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stitch.config import load_config, dump_config
from stitch.logging import get_logger, setup_logging, install_crash_handler


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None, help="YAML configuration file")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--crash", action="store_true",
                   help="raise partway through, to exercise crash capture")
    p.add_argument("overrides", nargs="*", help="dotted overrides, e.g. train.seed=3")
    args = p.parse_args()

    # 1. Configuration. Unknown keys raise here rather than being ignored.
    cfg = load_config(args.config, args.overrides)

    # 2. Experiment logger, which owns the run directory.
    logger = get_logger(cfg)
    run_id = logger.start_run(cfg.run_name(), tags={"objective": cfg.objective})

    # 3. Process logging, into the same directory.
    log = setup_logging(logger.run_dir, level=cfg.logging.level)

    # 4. Crash capture, armed before anything can fail.
    install_crash_handler(
        logger.run_dir,
        crash_dir=Path(cfg.logging.run_root).parent / "crashes",
        logger=logger,
        context={"run": run_id, "config_hash": cfg.hash(),
                 "objective": cfg.objective},
    )

    log.info("run %s", run_id)
    log.info("config hash %s", cfg.hash())

    # Placeholders are announced, not hidden, so that a result is never quietly
    # produced under a value nobody chose.
    for item in cfg.pending_decisions():
        log.warning("PENDING: %s", item)

    # 5. Parameters and the resolved config, both recorded with the run.
    logger.log_params(cfg.flat())
    dump_config(cfg, logger.run_dir / "config.yaml")
    logger.log_artifact(logger.run_dir / "config.yaml")

    # 6. The loop. A stand-in for training, with the same logging cadence.
    log.info("stepping %d times", args.steps)
    for step in range(args.steps):
        fake_loss = 1.0 / (1.0 + 0.1 * step)
        fake_l0 = 64.0 - 0.5 * step

        if args.crash and step == args.steps // 2:
            raise RuntimeError("deliberate failure, to exercise crash capture")

        if step % max(1, cfg.train.log_every // 10) == 0:
            logger.log_metrics({"loss": fake_loss, "l0": fake_l0}, step=step)

    # 7. Close cleanly. The process log becomes an artifact of the run.
    logger.log_artifact(logger.run_dir / "run.log")
    logger.end_run("FINISHED")
    log.info("done, outputs in %s", logger.run_dir)
    logging.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
