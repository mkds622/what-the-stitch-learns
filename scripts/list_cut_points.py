#!/usr/bin/env python3
"""Print the places a model can be cut, with the node names and shapes.

Node names are generated during tracing. They are not the names in the model's
source, and they differ between library versions, so they have to be read off
the model actually installed rather than assumed. Those names go into the
``activations.layers`` setting as ``<label>@<node>``.

By default only one cut point per transformer block is shown, the residual
stream leaving it, which is where a coder belongs. ``--all`` shows every node
carrying a token stream, and ``--raw`` shows the entire graph.

Usage:
    python scripts/list_cut_points.py
    python scripts/list_cut_points.py --model vit_small_patch16_224
    python scripts/list_cut_points.py --source torchvision --model vit_b_16
    python scripts/list_cut_points.py --all
    python scripts/list_cut_points.py --raw | less
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from stitch.models import (
    block_outputs, cut_point_candidates, layer_key, list_nodes, load_backbone,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="vit_base_patch16_224")
    p.add_argument("--source", default="timm", choices=["timm", "torchvision"])
    p.add_argument("--label", default="A",
                   help="model label used in the printed layer keys")
    p.add_argument("--resolution", type=int, default=224)
    p.add_argument("--pretrained", action="store_true",
                   help="download weights. Unnecessary here, since the graph "
                        "and the shapes do not depend on the weights.")
    p.add_argument("--all", action="store_true",
                   help="every node carrying a token stream, not one per block")
    p.add_argument("--raw", action="store_true", help="every node in the graph")
    p.add_argument("--graph-surgery", action="store_true",
                   help="trace with the graph surgery library rather than "
                        "torch.fx directly. Node names should agree; use this "
                        "to confirm they do before relying on them.")
    args = p.parse_args()

    model = load_backbone(args.model, pretrained=args.pretrained,
                          source=args.source)

    if args.graph_surgery:
        from stitch.models import trace
        gm = trace(model)
    else:
        from torch.fx import symbolic_trace
        gm = symbolic_trace(model)

    batch = torch.randn(2, 3, args.resolution, args.resolution)
    nodes = list_nodes(gm, batch)

    if args.raw:
        print(f"{'node':<38} {'op':<16} {'target':<34} shape")
        print("-" * 110)
        for n in nodes:
            print(f"{n.name:<38} {n.op:<16} {n.target[:33]:<34} {n.shape}")
        return 0

    chosen = cut_point_candidates(nodes) if args.all else block_outputs(gm, batch)

    if not chosen:
        print(f"No cut points found in {args.model}. It may not be a "
              f"transformer, or its blocks may not end in a residual addition. "
              f"Use --raw to inspect the graph.", file=sys.stderr)
        return 1

    print(f"{args.model} via {args.source}: {len(nodes)} nodes, "
          f"{len(chosen)} cut point{'s' if len(chosen) != 1 else ''}")
    print()
    print(f"{'depth':<7} {'layer key':<46} shape")
    print("-" * 84)
    for i, n in enumerate(chosen):
        depth = str(i) if not args.all else ""
        print(f"{depth:<7} {layer_key(args.label, n.name):<46} {n.shape}")

    print()
    print("Copy a layer key into activations.layers, for example:")
    print(f"    layers: [\"{layer_key(args.label, chosen[len(chosen) // 2].name)}\"]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
