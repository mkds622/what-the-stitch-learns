#!/usr/bin/env python3
"""Show what happens to a model's input when the model is cut in half.

Cutting a model means taking everything after a chosen point and turning that
point into the new input. The original input should then be unnecessary, and
for a convolutional network it is. For a vision transformer it is not: the
lower half still asks for the image.

The cause is a size check. A transformer divides the image into a fixed grid of
patches, so it verifies the image is the size it expects before doing anything
else. That check reads the image and ends in an assertion, which produces no
value. Removing nodes that nothing uses therefore does not remove it, because
every node in that chain is used, right up to an assertion that leads nowhere.
The image stays reachable, and everything still attached to it stays too.

Run before and after applying the fix to the graph library. The script reports
which state it is in, so no record of the expected output is needed.

Needs no GPU, no pretrained weights and no dataset.

Usage:
    python scripts/reproduce_subgraph_input.py
    python scripts/reproduce_subgraph_input.py --timing
"""

from __future__ import annotations

import argparse
import sys
import time

import torch
import torchvision
from torch import nn

try:
    from nn_lib.models.graph_module_plus import GraphModulePlus
except ImportError:
    sys.exit("nn_lib is not importable. Install it into this environment first.")


def trace(model: nn.Module) -> GraphModulePlus:
    """Convert a model into a graph that can be cut."""
    return GraphModulePlus.new_from_trace(model.eval()).squash_all_conv_batchnorm_pairs().eval()


def residual_adds(graph_module) -> list[str]:
    """Names of the residual additions, which are where a model is cut."""
    return [
        node.name
        for node in graph_module.graph.nodes
        if node.op == "call_function" and node.name.startswith("add")
    ]


def placeholders(graph_module) -> list[str]:
    """Names of the graph's inputs, which become the rebuilt function's parameters."""
    return [node.name for node in graph_module.graph.nodes if node.op == "placeholder"]


def node_count(graph_module) -> int:
    return len(list(graph_module.graph.nodes))


def examine(label: str, model: nn.Module, cut_index: int, resolution: int = 224) -> bool:
    """Cut one model in half and report what the lower half asks for.

    Returns True when the lower half takes only the activation, which is the
    correct result.
    """
    whole = trace(model)
    adds = residual_adds(whole)
    if cut_index >= len(adds):
        print(f"{label}: only {len(adds)} cut points, cannot cut at {cut_index}")
        return False
    cut = adds[cut_index]

    upper = GraphModulePlus.new_from_copy(whole).extract_subgraph(output=cut).eval()
    lower = GraphModulePlus.new_from_copy(whole).extract_subgraph(inputs=[cut]).eval()

    lower_inputs = placeholders(lower)
    clean = lower_inputs == [cut]

    print(f"\n{label}")
    print(f"  cut at                {cut}")
    print(f"  lower half asks for   {lower_inputs}")
    print(f"  nodes: whole {node_count(whole)}, upper {node_count(upper)}, "
          f"lower {node_count(lower)}, upper+lower {node_count(upper) + node_count(lower)}")

    # Confirm the halves still reproduce the original, whichever state we are in.
    batch = torch.randn(2, 3, resolution, resolution)
    with torch.no_grad():
        activation = upper(batch)
        output = lower(activation) if clean else lower(batch, activation)
        matches = torch.allclose(whole(batch), output, atol=1e-4)
    print(f"  halves recompose      {matches}")

    if clean:
        print("  VERDICT               clean, takes the activation only")
    else:
        extra = [name for name in lower_inputs if name != cut]
        print(f"  VERDICT               leaks, also demands {extra}")
    return clean


def time_halves(resolution: int = 224) -> None:
    """Show that the leaked input is not free.

    The lower half of a transformer holds a working copy of the upper half and
    runs it, to satisfy a size check whose answer is discarded.
    """
    whole = trace(torchvision.models.vit_b_16(weights=None))
    cut = residual_adds(whole)[7]
    upper = GraphModulePlus.new_from_copy(whole).extract_subgraph(output=cut).eval()
    lower = GraphModulePlus.new_from_copy(whole).extract_subgraph(inputs=[cut]).eval()
    clean = placeholders(lower) == [cut]

    batch = torch.randn(4, 3, resolution, resolution)
    print("\nTiming, vit_b_16, batch of 4")
    with torch.no_grad():
        activation = upper(batch)
        cases = [
            ("whole model", lambda: whole(batch)),
            ("upper half", lambda: upper(batch)),
            ("lower half", (lambda: lower(activation)) if clean else
                           (lambda: lower(batch, activation))),
        ]
        for name, call in cases:
            call()
            start = time.perf_counter()
            for _ in range(5):
                call()
            print(f"  {name:<14} {(time.perf_counter() - start) / 5 * 1000:8.1f} ms")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--timing", action="store_true",
                        help="also time the halves against the whole model")
    args = parser.parse_args()

    print("Cutting four models in half and reporting what the lower half requires.")

    results = {
        "resnet18": examine("resnet18 (convolutional)",
                            torchvision.models.resnet18(weights=None), 3),
        "resnet34": examine("resnet34 (convolutional)",
                            torchvision.models.resnet34(weights=None), 5),
        "vit_b_16": examine("vit_b_16 (transformer)",
                            torchvision.models.vit_b_16(weights=None), 7),
    }

    try:
        import timm

        class SingleInput(nn.Module):
            """Expose only the image input.

            A timm transformer's forward also takes attn_mask and is_causal.
            Each becomes a separate graph input when traced, which is a
            different problem from the one this script is about.
            """

            def __init__(self, inner: nn.Module):
                super().__init__()
                self.inner = inner

            def forward(self, x):
                return self.inner(x)

        results["timm vit"] = examine(
            "vit_base_patch16_224 (transformer, timm)",
            SingleInput(timm.create_model("vit_base_patch16_224", pretrained=False)), 7)
    except ImportError:
        print("\ntimm not installed, skipping its transformer")

    if args.timing:
        time_halves()

    convolutional = [name for name in ("resnet18", "resnet34") if results.get(name)]
    transformers = [name for name in ("vit_b_16", "timm vit")
                    if name in results and not results[name]]

    print("\n" + "=" * 60)
    if transformers:
        print("UNFIXED. Convolutional models cut cleanly; transformers leak the input.")
        print(f"  clean: {convolutional}")
        print(f"  leaking: {transformers}")
        print("\nApply the fix to the graph library and run this again.")
        return 1

    print("FIXED. Every model cuts cleanly and every pair recomposes correctly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())