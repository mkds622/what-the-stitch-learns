#!/usr/bin/env python3
"""How strongly each activation dimension carries a constant offset.

A feature's pre-activation splits into a part that is the same for every token,
the encoder row's projection onto the activation's mean, and a part that
depends on the token. When the first dominates, which features fire is decided
before any token is seen: rows anti-aligned with the mean never clear the ReLU,
and rows only weakly aligned with it lose the top-k competition to rows that
are strongly aligned. Neither kind ever trains.

The severity of that is a ratio of the mean offset to the variation around it,
taken per dimension and then over the whole vector:

    gamma = ||mu|| / ||sigma||

with ``mu`` the per-dimension mean over tokens and ``sigma`` the per-dimension
standard deviation over tokens. Large gamma means the dimensions sit a long way
from zero compared with how much they move, which is the geometry that kills
features at initialisation.

Measured on per-token layer-normalised activations. Without that step a
dimension that is large in every token looks like signal rather than offset,
and the ratio understates the problem. Both readings are printed, so the size
of that correction is visible rather than assumed.

The dimension-level breakdown underneath the ratio is the part that stands on
its own. How many dimensions have an offset larger than their own variation is
a direct statement about the activation, and needs no threshold borrowed from
anywhere to be read.

Loads the backbones and reads tokens. Writes nothing and trains nothing.

Usage:
    python scripts/check_outlier_severity.py
    python scripts/check_outlier_severity.py --tokens 500000
    python scripts/check_outlier_severity.py --layers '["A@add_20","B@add_20"]'
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from stitch.config import load_config
from stitch.run import build_streams, load_models


def layer_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Normalise each token to zero mean and unit variance across dimensions.

    No learned scale or shift. This is the token-wise standardisation the
    severity ratio is defined on, not a layer of the model.
    """
    mean = x.mean(dim=1, keepdim=True)
    variance = x.var(dim=1, keepdim=True, unbiased=False)
    return (x - mean) / (variance + eps).sqrt()


def severity(x: torch.Tensor) -> dict[str, float]:
    """The ratio and the dimension-level detail behind it.

    Args:
        x: ``(tokens, width)``.

    Returns:
        ``gamma``, the ratio of the two norms; ``mu_norm`` and ``sigma_norm``
        themselves; ``worst``, the largest per-dimension offset measured in its
        own standard deviations; ``over_one`` and ``over_two``, how many
        dimensions have an offset larger than one and two of their own standard
        deviations; and ``median_ratio``, the typical dimension's.
    """
    mu = x.mean(dim=0)
    sigma = x.std(dim=0, unbiased=False)
    ratio = mu.abs() / sigma.clamp_min(1e-12)

    return {
        "gamma": float(mu.norm() / sigma.norm()),
        "mu_norm": float(mu.norm()),
        "sigma_norm": float(sigma.norm()),
        "worst": float(ratio.max()),
        "median_ratio": float(ratio.median()),
        "over_one": int((ratio > 1.0).sum()),
        "over_two": int((ratio > 2.0).sum()),
        "width": int(x.shape[1]),
    }


def report(label: str, raw: dict[str, float], normalised: dict[str, float]) -> None:
    print("=" * 66)
    print(f"{label}   width {raw['width']}")
    print()
    print(f"  {'':<28} {'raw':>12} {'layer-normed':>14}")
    print(f"  {'gamma = ||mu||/||sigma||':<28} "
          f"{raw['gamma']:>12.5f} {normalised['gamma']:>14.5f}")
    print(f"  {'||mu||':<28} {raw['mu_norm']:>12.4f} {normalised['mu_norm']:>14.4f}")
    print(f"  {'||sigma||':<28} {raw['sigma_norm']:>12.4f} "
          f"{normalised['sigma_norm']:>14.4f}")
    print()
    print(f"  per-dimension offset, measured in that dimension's own sigma")
    print(f"  {'largest':<28} {raw['worst']:>12.3f} {normalised['worst']:>14.3f}")
    print(f"  {'median':<28} {raw['median_ratio']:>12.3f} "
          f"{normalised['median_ratio']:>14.3f}")
    print(f"  {'dimensions over 1 sigma':<28} {raw['over_one']:>12d} "
          f"{normalised['over_one']:>14d}")
    print(f"  {'dimensions over 2 sigma':<28} {raw['over_two']:>12d} "
          f"{normalised['over_two']:>14d}")


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
        report(key, severity(x), severity(layer_norm(x)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())