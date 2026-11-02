#!/usr/bin/env python3
"""Whether the mean or the geometric median should centre an activation.

The coder subtracts a fixed vector before encoding and adds one back after
decoding. Which statistic those should be depends on whether the activation
has outlier tokens: the mean is dragged by them and the geometric median is
not. Transformer residual streams sometimes carry a handful of tokens with
enormous norm, and whether these do is a property of the pair and the depth
rather than something to assume.

Prints both statistics for every configured layer, and the distribution of
token norms behind them, which is the part that says whether outliers exist at
all.

Writes nothing and trains nothing.

Usage:
    python scripts/check_activation_statistics.py
    python scripts/check_activation_statistics.py --tokens 500000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from stitch.config import load_config
from stitch.run import build_streams, load_models


def geometric_median(x: torch.Tensor, iterations: int = 50,
                     eps: float = 1e-8) -> torch.Tensor:
    """Weiszfeld's algorithm, started from the mean."""
    m = x.mean(dim=0)
    for _ in range(iterations):
        distance = (x - m).norm(dim=1).clamp_min(eps)
        weight = 1.0 / distance
        m = (x * weight[:, None]).sum(dim=0) / weight.sum()
    return m


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/case2_main.yaml"))
    parser.add_argument("--layers", default='["A@add_12","B@add_12"]')
    parser.add_argument("--tokens", type=int, default=100_000)
    args = parser.parse_args(argv)

    cfg = load_config(args.config, [f"activations.layers={args.layers}"])
    models, backbones = load_models(cfg)
    training, _, _ = build_streams(cfg, models, backbones)

    held: dict[str, list[torch.Tensor]] = {}
    seen = 0
    for batch in training.batches():
        for key, value in batch.items():
            held.setdefault(key, []).append(value.float().cpu())
        seen += next(iter(batch.values())).shape[0]
        if seen >= args.tokens:
            break

    for key, chunks in held.items():
        x = torch.cat(chunks)[:args.tokens]
        mean = x.mean(dim=0)
        median = geometric_median(x)
        norms = x.norm(dim=1)

        print("=" * 62)
        print(f"{key}   {x.shape[0]} tokens, width {x.shape[1]}")
        print(f"  ||mean||                      {mean.norm():.4f}")
        print(f"  ||geometric median||          {median.norm():.4f}")
        print(f"  cos(mean, median)             "
              f"{float(torch.dot(mean, median) / (mean.norm() * median.norm())):+.5f}")
        print(f"  ||mean - median|| / ||mean||  "
              f"{float((mean - median).norm() / mean.norm()):.5f}")
        print()
        print(f"  token norm, median            {norms.median():.3f}")
        print(f"  token norm, 99th pct          {norms.quantile(0.99):.3f}")
        print(f"  token norm, max               {norms.max():.3f}")
        print(f"  max / median                  {float(norms.max() / norms.median()):.2f}")
        print(f"  tokens over 3x median norm    "
              f"{int((norms > 3 * norms.median()).sum())} of {x.shape[0]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())