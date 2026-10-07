#!/usr/bin/env python3
"""Show what happens when models are merged into one.

Merging joins several models into a single module with a single forward. The
inputs of the parts are supposed to become one shared input, so the merged
model is called with one image the way its parts were.

That works when every part takes exactly one argument. It does not when a part
takes more. A recent timm transformer's forward takes an attention mask and a
causal flag alongside the image, so a traced one has three inputs, and the
merge refuses because an image and a boolean flag cannot become the same input.

The second thing checked here is subtler. Two parts can name the same argument
differently, since the name is whatever each author chose. Those are still one
input, and a merge that decides by name rather than by position will split them
and produce a model that demands two images.

Run before and after applying the fix to the graph library. The script reports
which state it is in, so no record of the expected output is needed.

Needs no GPU, no pretrained weights and no dataset.

Usage:
    python scripts/reproduce_merge_inputs.py
"""

from __future__ import annotations

import sys

import torch
import torchvision
from torch import nn

try:
    from nn_lib.models.graph_module_plus import GraphModulePlus
except ImportError:
    sys.exit("nn_lib is not importable. Install it into this environment first.")


class DifferentlyNamedInput(nn.Module):
    """A model whose forward argument is called something other than 'input'.

    nn.Sequential calls its argument 'input'. This one calls it 'x'. They are
    the same kind of input, and merging the two should produce a model with one.
    """

    def __init__(self, width_in: int, width_out: int):
        super().__init__()
        self.fc = nn.Linear(width_in, width_out)

    def forward(self, x):
        return self.fc(x)


def placeholders(graph_module) -> list[str]:
    """Names of the graph's inputs, which become the rebuilt function's parameters."""
    return [node.name for node in graph_module.graph.nodes if node.op == "placeholder"]


def merge(modules: dict[str, nn.Module]) -> tuple[GraphModulePlus | None, str]:
    """Merge some modules, returning either the result or the reason it refused."""
    try:
        return GraphModulePlus.new_from_merge(modules, rewire_inputs={}).eval(), ""
    except Exception as error:  # noqa: BLE001 - the point is to report whatever it was
        return None, f"{type(error).__name__}: {error}"


def examine(label: str, modules: dict[str, nn.Module], batch, expected: list[str]) -> bool:
    """Merge one set of modules and report what the merged model asks for.

    Returns True when the merge succeeded and produced the expected inputs.
    """
    merged, failure = merge(modules)

    print(f"\n{label}")
    if merged is None:
        print(f"  merge                 refused")
        print(f"  reason                {failure}")
        print(f"  VERDICT               refuses to merge")
        return False

    found = placeholders(merged)
    print(f"  merged model takes    {found}")
    print(f"  expected              {expected}")

    # The merged model's output is the output of the last part, so compare against it.
    last = list(modules.values())[-1].eval()
    with torch.no_grad():
        try:
            matches = torch.allclose(last(batch), merged(batch), atol=1e-4)
        except TypeError as error:
            print(f"  called with one input {type(error).__name__}: {error}")
            print(f"  VERDICT               demands inputs it should not")
            return False
    print(f"  output preserved      {matches}")

    if found == expected and matches:
        print("  VERDICT               clean")
        return True
    print("  VERDICT               wrong inputs")
    return False


def main() -> int:
    print("Merging several sets of models and reporting what the merged model requires.")

    results = {}

    # A single-argument convolutional model. This has always worked and must keep working.
    results["resnet18"] = examine(
        "resnet18 (one argument)",
        {"a": torchvision.models.resnet18(weights=None)},
        torch.randn(2, 3, 224, 224),
        ["x"],
    )

    # Two parts that name their argument differently. One input, two names.
    results["mixed names"] = examine(
        "nn.Sequential plus a module that calls its argument x",
        {
            "a": nn.Sequential(nn.Linear(2, 4), nn.Linear(4, 6)),
            "b": DifferentlyNamedInput(2, 6),
        },
        torch.randn(3, 2),
        ["x"],
    )

    # A model whose forward takes optional arguments.
    try:
        import timm

        results["timm vit"] = examine(
            "vit_base_patch16_224 (three arguments, timm)",
            {"a": timm.create_model("vit_base_patch16_224", pretrained=False)},
            torch.randn(2, 3, 224, 224),
            ["x", "attn_mask", "is_causal"],
        )
    except ImportError:
        print("\ntimm not installed, skipping its transformer")

    failing = [name for name, ok in results.items() if not ok]

    print("\n" + "=" * 60)
    if failing:
        print("UNFIXED.")
        print(f"  clean: {[name for name, ok in results.items() if ok]}")
        print(f"  broken: {failing}")
        print("\nApply the fix to the graph library and run this again.")
        return 1

    print("FIXED. Every merge succeeds with the inputs it should have.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
    