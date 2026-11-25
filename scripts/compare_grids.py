#!/usr/bin/env python3
"""One run from each of two grids, metric by metric.

Prints every metric of the run with the given seed in each grid, the
difference and the relative difference, and draws the relative differences.

    python scripts/compare_grids.py \\
        --old grids/main2_case1_add_12.json \\
        --new grids/verify_case1_add_12.json \\
        --seed 0 \\
        --out figures/verify_case1_add_12.png
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

METRICS = [
    "eval_explained_variance",
    "eval_l0",
    "eval_l0_frac",
    "eval_effective_expansion",
    "eval_effective_features",
    "eval_frequency_gini",
    "eval_max_feature_frequency",
    "eval_dead_or_dormant",
    "eval_dead_or_dormant_wider",
    "train_dead",
    "eval_linear_fit",
    "eval_fit_ratio",
    "baseline_ratio",
    "agreement",
    "plain_top1",
    "substituted_top1",
    "top1_drop",
    "gate_passed",
]


def run_for_seed(path: Path, seed: int) -> dict:
    entries = json.loads(Path(path).read_text())
    for entry in entries:
        match = re.search(r"-s(\d+)-", entry.get("run_name", ""))
        if match and int(match.group(1)) == seed:
            return entry
    raise SystemExit(f"no run with seed {seed} in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    old = run_for_seed(args.old, args.seed)
    new = run_for_seed(args.new, args.seed)
    print(f"old: {old['run_name']}")
    print(f"new: {new['run_name']}")
    print()
    print(f"{'metric':<30} {'old':>12} {'new':>12} {'difference':>12} {'relative':>10}")

    names, relative = [], []
    for metric in METRICS:
        if metric not in old or metric not in new:
            print(f"{metric:<30} {'missing':>12}")
            continue
        a, b = float(old[metric]), float(new[metric])
        change = b - a
        ratio = change / abs(a) if a != 0 else float("nan")
        print(f"{metric:<30} {a:>12.5f} {b:>12.5f} {change:>+12.5f} {ratio:>+10.1%}")
        names.append(metric)
        relative.append(ratio)

    import math

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = [(n, r) for n, r in zip(names, relative) if not math.isnan(r)]
    figure, axis = plt.subplots(figsize=(9, 0.38 * len(shown) + 1.5))
    labels = [n for n, _ in shown][::-1]
    values = [r * 100 for _, r in shown][::-1]
    axis.barh(labels, values, color=["#2b6cb0" if v >= 0 else "#c0504d" for v in values])
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("new relative to old (%)")
    axis.set_title(f"seed {args.seed}: {args.new.name} against {args.old.name}")
    axis.grid(alpha=0.3, axis="x")
    figure.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=150)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()