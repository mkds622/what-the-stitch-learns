#!/usr/bin/env python3
"""Train one sparse coder.

A thin wrapper. What a run consists of is in ``stitch/run.py``.

    python scripts/run.py --config configs/case1_smoke.yaml
    python scripts/run.py --config configs/case1_smoke.yaml --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stitch.run import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())