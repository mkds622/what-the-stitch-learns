#!/usr/bin/env python3
"""Two layer sweeps side by side, depth by depth.

Prints the explained variance of the linear map at every matched depth from
each sweep and the difference, and draws both against depth with the
difference beneath.

    python scripts/compare_sweeps.py \\
        --old sweeps/v1/layer_sweep_main.json \\
        --new sweeps/layer_sweep_main.json \\
        --out figures/sweep_v1_vs_v2.png
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def load(path: Path) -> tuple[dict, dict[int, dict]]:
    """The sweep's context and its records keyed by block depth."""
    data = json.loads(Path(path).read_text())
    records = {}
    for record in data["ranking"]:
        match = re.search(r"add_(\d+)$", record["source"])
        if match:
            records[int(match.group(1))] = record
    return data.get("context", {}), records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    old_context, old = load(args.old)
    new_context, new = load(args.new)
    depths = sorted(set(old) & set(new))
    if not depths:
        raise SystemExit("the two sweeps share no depths")

    for name, context in (("old", old_context), ("new", new_context)):
        print(f"{name}: fit on {context.get('tokens', '?'):,} tokens, "
              f"scored on {context.get('held_out_tokens', '?'):,} held-out tokens")
    print()
    print(f"{'depth':>8} {'old held-out':>13} {'new held-out':>13} {'difference':>11}"
          f" {'old in-sample':>14} {'new in-sample':>14}")
    for depth in depths:
        o, n = old[depth], new[depth]
        print(f"{'add_' + str(depth):>8} {o['explained_variance']:>13.4f} "
              f"{n['explained_variance']:>13.4f} "
              f"{n['explained_variance'] - o['explained_variance']:>+11.4f} "
              f"{o.get('explained_variance_in_sample', float('nan')):>14.4f} "
              f"{n.get('explained_variance_in_sample', float('nan')):>14.4f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    old_values = [old[d]["explained_variance"] for d in depths]
    new_values = [new[d]["explained_variance"] for d in depths]
    differences = [n - o for o, n in zip(old_values, new_values)]

    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.4]})

    top.plot(depths, old_values, marker="o", color="#8a8a8a", label=f"old ({args.old})")
    top.plot(depths, new_values, marker="o", color="#2b6cb0", label=f"new ({args.new})")
    top.set_ylabel("explained variance, held out")
    top.set_title("Linear map from A to B at matched depth")
    top.grid(alpha=0.3)
    top.legend(loc="lower left", fontsize=9)

    bottom.bar(depths, differences, color=["#2b6cb0" if d >= 0 else "#c0504d" for d in differences],
               width=1.2)
    bottom.axhline(0, color="black", linewidth=0.8)
    bottom.set_ylabel("new minus old")
    bottom.set_xlabel("cut point (add_N)")
    bottom.set_xticks(depths)
    bottom.grid(alpha=0.3, axis="y")

    figure.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=150)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()