"""Loading backbones, tracing them, and cutting them at a chosen depth.

A stitch replaces part of one network with another. Expressing "replace
everything after block 6" requires the model as a graph of named operations
rather than as a call stack, so every model is converted with ``torch.fx`` the
moment it is loaded and referred to by node name from then on.

Node names are generated during tracing and are not the names in the model's
source. A block written as ``x = x + branch(x)`` traces to a bare ``add_14``,
carrying no block index at all, and the generated names differ between model
libraries and versions. ``scripts/list_cut_points.py`` prints them for the model
actually installed, and no name is hard-coded anywhere in this package.

Layer keys
----------
Layers are referred to throughout the configuration as ``<label>@<node>``, for
example ``A@add_14``. The label selects which model and the node selects where
in it. :func:`parse_layer_key` and :func:`layer_key` are the only places that
format is interpreted.

External dependency
-------------------
Graph surgery comes from ``nn_lib`` rather than being reimplemented here. The
import is deferred to the point of use so that the parts of this module that do
not need it, parsing and shape inspection, work without it installed.
"""

from __future__ import annotations

# nn_lib targets Python 3.13 and decorates with warnings.deprecated, added in
# 3.13 by PEP 702. The decorator is available for earlier versions through
# typing_extensions, so the attribute is supplied here before nn_lib is
# imported anywhere in this module.
import warnings
if not hasattr(warnings, "deprecated"):
    from typing_extensions import deprecated as _deprecated
    warnings.deprecated = _deprecated

import logging
from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch import nn

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# layer keys
# ----------------------------------------------------------------------

def parse_layer_key(key: str) -> tuple[str, str]:
    """Split ``"A@blocks_6_add_1"`` into ``("A", "blocks_6_add_1")``.

    Raises:
        ValueError: if the key is not of that form. Node names may contain
            underscores and dots but never ``@``, so a single split is safe.
    """
    if "@" not in key:
        raise ValueError(
            f"layer key {key!r} must be '<model label>@<node name>', "
            f"for example 'A@blocks_6_add_1'")
    label, node = key.split("@", 1)
    if not label or not node:
        raise ValueError(f"layer key {key!r} has an empty label or node name")
    return label, node


def layer_key(label: str, node: str) -> str:
    """Inverse of :func:`parse_layer_key`."""
    return f"{label}@{node}"


# ----------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------

def load_backbone(name: str,
                  pretrained: bool = True,
                  checkpoint: str | None = None,
                  source: str = "timm") -> nn.Module:
    """Load a backbone by name, optionally overriding its weights.

    Args:
        name: identifier in the chosen source's namespace. The two namespaces
            disagree: ``vit_base_patch16_224`` is timm, ``vit_b_16`` is
            torchvision, and they are the same architecture.
        pretrained: load the source's own weights. Set False for benchmarking,
            where random weights cost the same arithmetic.
        checkpoint: path to a local state dict, loaded after construction. This
            is how a pair of models trained locally and differing only by seed
            is supplied, since neither source can provide those.
        source: ``timm`` or ``torchvision``.

    Returns:
        The model in eval mode. Not yet traced; pass it to :func:`trace`.
    """
    if source == "timm":
        import timm
        model = timm.create_model(name, pretrained=pretrained)
    elif source == "torchvision":
        from torchvision.models import get_model, get_model_weights
        weights = get_model_weights(name).DEFAULT if pretrained else None
        model = get_model(name, weights=weights)
    else:
        raise ValueError(f"unknown model source {source!r}, expected 'timm' or 'torchvision'")

    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        # Checkpoints are saved in several shapes depending on what wrote them.
        for key in ("state_dict", "model", "model_state_dict"):
            if isinstance(state, dict) and key in state:
                state = state[key]
                break
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            # Not fatal, but silently loading half a checkpoint would produce a
            # model that runs and means nothing.
            log.warning("checkpoint %s: %d missing and %d unexpected keys",
                        checkpoint, len(missing), len(unexpected))

    return model.eval()


def trace(model: nn.Module):
    """Convert a model into a graph that can be cut.

    Folds any convolution and batch-norm pairs first. A batch-norm layer depends
    on the convolution immediately before it, so cutting between them would
    leave two halves that are each wrong; folding removes the possibility.
    Transformers use layer norm and are unaffected, but the step is harmless and
    keeps the path identical for both families.

    Returns:
        A ``GraphModulePlus`` in eval mode.
    """
    from nn_lib.models import GraphModulePlus
    return GraphModulePlus.new_from_trace(model).squash_all_conv_batchnorm_pairs().eval()


# ----------------------------------------------------------------------
# inspecting the graph
# ----------------------------------------------------------------------

@dataclass
class NodeInfo:
    """One operation in a traced model.

    Attributes:
        name: the generated node name, which is what cut points are specified by.
        op: the fx node kind, such as ``call_module`` or ``call_function``.
        target: what it calls, which is the closest thing to a source-level name.
        shape: output shape including the batch dimension, or None if unprobed.
    """

    name: str
    op: str
    target: str
    shape: tuple[int, ...] | None = None

    @property
    def is_token_stream(self) -> bool:
        """True when the output looks like a transformer residual stream.

        Three dimensions, batch by tokens by width, with more tokens than one.
        This is the shape a coder can be inserted into, so it is the filter that
        turns a few hundred nodes into a short list of candidate cut points.
        """
        return self.shape is not None and len(self.shape) == 3 and self.shape[1] > 1


def list_nodes(gm, example_input: torch.Tensor | None = None) -> list[NodeInfo]:
    """Every node in a traced model, in execution order.

    Args:
        gm: a traced model.
        example_input: one batch. When given, each node's output shape is
            recorded by running the batch through and reading the shapes off,
            which is the only reliable way to learn them.

    Returns:
        Nodes in execution order.
    """
    shapes: dict[str, tuple[int, ...]] = {}
    if example_input is not None:
        shapes = probe_shapes(gm, example_input)

    out = []
    for node in gm.graph.nodes:
        out.append(NodeInfo(name=node.name, op=node.op,
                            target=str(node.target), shape=shapes.get(node.name)))
    return out


def probe_shapes(gm, example_input: torch.Tensor) -> dict[str, tuple[int, ...]]:
    """Output shape of every node, found by running one batch through.

    Uses fx's shape propagation, which executes the graph once and annotates
    each node. Nodes producing something other than a single tensor, such as a
    tuple or a module with no output, are omitted rather than guessed at.
    """
    from torch.fx.passes.shape_prop import ShapeProp

    device = next(gm.parameters()).device
    with torch.no_grad():
        ShapeProp(gm).propagate(example_input.to(device))

    shapes = {}
    for node in gm.graph.nodes:
        meta = node.meta.get("tensor_meta")
        if meta is not None and hasattr(meta, "shape"):
            shapes[node.name] = tuple(meta.shape)
    return shapes


def cut_point_candidates(nodes: Iterable[NodeInfo]) -> list[NodeInfo]:
    """Nodes whose output is a residual stream of the expected shape.

    Filters out everything whose output a coder could not be inserted into. The
    result is still longer than the number of transformer blocks, since several
    nodes inside each block carry the same shape, so a selection rule is applied
    on top of this by :func:`block_outputs`.
    """
    return [n for n in nodes if n.is_token_stream]


BLOCK_INDEX_PATTERN = r"(?:blocks?|layers?|encoder_layer)[._]?(\d+)"


def block_outputs(gm, example_input: torch.Tensor) -> list[NodeInfo]:
    """One cut point per transformer block: the residual stream leaving it.

    A transformer block ends in ``x = x + branch(x)``. That addition is the
    residual stream and is the only correct place to cut, because the branch
    output alone carries the block's contribution without the stream it is
    added to. Inserting a coder at the branch instead produces a model that
    runs and reconstructs something other than the representation.

    The addition cannot be found by name. Operators written with ``+`` are
    traced as bare function calls named ``add_2``, ``add_4`` and so on, with no
    block prefix, so grouping nodes by the block index in their name finds only
    the branch and silently picks the wrong one.

    It is found through the graph instead. The last prefixed node in each block
    is located, and the node consuming it is the addition. That holds wherever
    a block ends in a residual addition, regardless of naming.

    Args:
        gm: a traced model.
        example_input: one batch, used to establish shapes.

    Returns:
        One node per block, in depth order. Empty if the model has no nodes
        carrying a block index, in which case the model is not of this shape and
        :func:`cut_point_candidates` should be used directly.
    """
    import re

    nodes = list_nodes(gm, example_input)
    info = {n.name: n for n in nodes}

    # last prefixed token-stream node within each block
    last_in_block: dict[int, str] = {}
    for n in cut_point_candidates(nodes):
        m = re.search(BLOCK_INDEX_PATTERN, n.name)
        if m is not None:
            last_in_block[int(m.group(1))] = n.name

    by_name = {node.name: node for node in gm.graph.nodes}

    out: list[NodeInfo] = []
    for idx in sorted(last_in_block):
        name = last_in_block[idx]
        node = by_name[name]
        users = list(node.users)
        # The residual addition is the single consumer of the branch output and
        # carries the same shape. Anything else means the block does not end the
        # way this assumes, so the prefixed node is kept and the caller sees a
        # shape that still makes sense.
        chosen = info[name]
        if len(users) == 1:
            candidate = info.get(users[0].name)
            if candidate is not None and candidate.shape == chosen.shape:
                chosen = candidate
        out.append(chosen)
    return out


def describe(gm, example_input: torch.Tensor) -> str:
    """A printable table of candidate cut points.

    Written for use at a terminal when choosing where to stitch, which needs the
    real node names and shapes for the model actually installed.
    """
    nodes = list_nodes(gm, example_input)
    cands = cut_point_candidates(nodes)
    blocks = block_outputs(gm, example_input)

    lines = [f"{len(nodes)} nodes, {len(cands)} carrying a token stream, "
             f"{len(blocks)} block outputs",
             "",
             f"{'node':<34} {'op':<16} shape",
             "-" * 72]
    for n in (blocks or cands):
        lines.append(f"{n.name:<34} {n.op:<16} {n.shape}")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# cutting
# ----------------------------------------------------------------------

def upstream(gm, node: str):
    """The part of the model from its input up to and including ``node``.

    Running this returns the activation at the cut point, which is what the
    coder reads and, for a same-layer target, what it is trained to reproduce.

    The returned module shares parameters with the original, so freezing one
    freezes the other and no weights are duplicated in memory.
    """
    from nn_lib.models import GraphModulePlus
    return GraphModulePlus.new_from_copy(gm).extract_subgraph(output=node).eval()


def downstream(gm, node: str):
    """The part of the model after ``node``, taking that activation as input.

    This is the half a reconstruction is fed into to see whether the model still
    works, and the half gradients travel back through when the coder is trained
    on behaviour rather than on reconstruction.
    """
    from nn_lib.models import GraphModulePlus
    return GraphModulePlus.new_from_copy(gm).extract_subgraph(inputs=[node]).eval()


def make_extractor(gm, nodes: list[str]):
    """A model whose forward returns a dict of several nodes' activations.

    One pass captures every requested layer, so extracting six layers costs the
    same as extracting one. Without this, each layer would need its own pass
    over the data and the cost would multiply by the number of layers.

    Args:
        gm: a traced model.
        nodes: node names to return.

    Returns:
        A module returning ``{node_name: activation}``. The keys are the node
        names as fx reports them, which may differ from the names passed in if
        fx resolves them to canonical nodes.
    """
    from nn_lib.models import GraphModulePlus
    if not nodes:
        raise ValueError("make_extractor needs at least one node")
    return GraphModulePlus.new_from_copy(gm).set_dict_outputs(nodes).eval()


def stitch(model_a, node_a: str, connector: nn.Module, model_b, node_b: str):
    """Assemble model A's upstream half, a connector, and model B's downstream half.

    The result is one model that runs end to end. Both backbones keep their own
    parameters, so freezing them leaves the connector as the only trainable
    part, and gradients from a loss at the output reach the connector through
    B's remaining layers.

    A no-op node is inserted at B's cut point first. Rewiring needs somewhere to
    attach that is not the original computation, and inserting a pass-through
    node provides it without changing what B computes.

    Args:
        model_a: traced model supplying the upstream half.
        node_a: where to cut A.
        connector: the module between them. Passed through untraced, so it stays
            a normal module and keeps any methods of its own.
        model_b: traced model supplying the downstream half.
        node_b: where to cut B.

    Returns:
        The merged model in eval mode. Freezing the backbones is the caller's
        responsibility; use ``nn_lib.utils.frozen``.
    """
    from nn_lib.models import GraphModulePlus

    node_b_noop = model_b.insert_noop(node_b)
    return GraphModulePlus.new_from_merge(
        modules={"model_a": model_a, "connector": connector, "model_b": model_b},
        rewire_inputs={
            "connector": f"model_a_{node_a}",
            f"model_b_{node_b_noop}": "connector",
        },
        auto_trace=False,
    ).eval()


def activation_width(gm, node: str, example_input: torch.Tensor) -> int:
    """Width of the activation at a cut point, which sets the coder's input size.

    Read from the model rather than configured, so that a coder reading one
    model and writing into another needs no extra setting when the two differ.
    """
    shape = probe_shapes(gm, example_input).get(node)
    if shape is None:
        raise KeyError(f"node {node!r} produced no tensor, so it has no width")
    return int(shape[-1])


def trainable_parameters(model: nn.Module) -> list[str]:
    """Names of parameters that would receive gradients.

    Called before training to confirm that only the connector is trainable.
    Merged models share parameters with their sources, so a missing freeze
    silently fine-tunes a backbone and produces results that look fine and are
    not comparable with anything.
    """
    return [n for n, p in model.named_parameters() if p.requires_grad]
