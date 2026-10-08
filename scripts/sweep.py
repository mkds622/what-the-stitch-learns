#!/usr/bin/env python3
"""Rank layer pairs by how well a linear map connects them.

Run before committing a training budget to any depth. What the ranking means
is in ``stitch/sweep.py``.

    python scripts/sweep.py --config configs/case2.yaml
    python scripts/sweep.py --config configs/case2.yaml --set sweep.fit_tokens=100000
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stitch.config import load_config  # noqa: E402
from stitch.sweep import run_sweep     # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config, args.overrides)
    records = run_sweep(cfg)

    print(f"\n{'rank':<6} {'reads':<22} {'reconstructs':<22} explained")
    print("-" * 64)
    for record in records:
        print(f"{record['rank']:<6} {record['source']:<22} {record['target']:<22} "
              f"{record['explained_variance']:.4f}")
    print(f"\nwritten to {cfg.sweep.out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())