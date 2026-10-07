"""Tests for loading, tracing and cutting backbones.

Most of these run on a real ViT traced with ``torch.fx`` directly, with random
weights and a two-image batch, so they need no dataset, no GPU and no network.

The graph surgery helpers need ``nn_lib``, which is skipped when absent rather
than mocked, since a mock of a graph library would test nothing.

    python -m pytest tests/test_models.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")

from torch.fx import symbolic_trace                                   # noqa: E402

from stitch.models import (                                           # noqa: E402
    NodeInfo, block_outputs, cut_point_candidates, describe,
    layer_key, list_nodes, load_backbone, parse_layer_key,
    probe_shapes, activation_width,
)

MODEL = "vit_base_patch16_224"
WIDTH = 768
TOKENS = 197
BLOCKS = 12


@pytest.fixture(scope="module")
def gm():
    """A traced ViT with random weights. Module-scoped; tracing is not free."""
    return symbolic_trace(timm.create_model(MODEL, pretrained=False).eval())


@pytest.fixture(scope="module")
def batch():
    return torch.randn(2, 3, 224, 224)


# ----------------------------------------------------------------------
# layer keys
# ----------------------------------------------------------------------

def test_layer_key_roundtrip():
    assert parse_layer_key(layer_key("A", "add_12")) == ("A", "add_12")


def test_layer_key_splits_on_first_separator_only():
    """Node names may contain anything except the separator."""
    assert parse_layer_key("A@blocks.6_add_1") == ("A", "blocks.6_add_1")


@pytest.mark.parametrize("bad", ["add_12", "@add_12", "A@", ""])
def test_malformed_layer_keys_raise(bad):
    with pytest.raises(ValueError):
        parse_layer_key(bad)


# ----------------------------------------------------------------------
# inspecting the graph
# ----------------------------------------------------------------------

def test_list_nodes_is_in_execution_order(gm):
    nodes = list_nodes(gm)
    assert nodes[0].op == "placeholder"
    assert nodes[-1].op == "output"
    assert len(nodes) > 100


def test_probe_shapes_finds_the_token_stream(gm, batch):
    """Token-stream shapes appear at more than one width.

    The MLP inside each block widens the stream fourfold before narrowing it
    again, so nodes at 3072 are as common as nodes at 768. Only the residual
    width is a valid cut point, which is why a shape filter alone is not enough
    to choose one.
    """
    from collections import Counter
    shapes = probe_shapes(gm, batch)
    stream = [s for s in shapes.values() if len(s) == 3 and s[1] == TOKENS]
    assert stream, "no node produced a token stream"

    widths = Counter(s[2] for s in stream)
    assert WIDTH in widths
    assert WIDTH * 4 in widths, "the MLP hidden width should also be present"


def test_shapes_are_absent_until_probed(gm):
    assert all(n.shape is None for n in list_nodes(gm))


def test_candidates_are_only_token_streams(gm, batch):
    for n in cut_point_candidates(list_nodes(gm, batch)):
        assert len(n.shape) == 3 and n.shape[1] > 1


def test_is_token_stream_rejects_collapsed_shapes():
    assert not NodeInfo("x", "call_module", "x", (2, 768)).is_token_stream
    assert not NodeInfo("x", "call_module", "x", (2, 1, 768)).is_token_stream
    assert not NodeInfo("x", "call_module", "x", None).is_token_stream
    assert NodeInfo("x", "call_module", "x", (2, 197, 768)).is_token_stream


# ----------------------------------------------------------------------
# block outputs
#
# The important property is which node is chosen, not how many. A block ends in
# a residual addition, and the branch feeding that addition carries the same
# shape, so picking the wrong one produces a model that runs and reconstructs
# the wrong thing.
# ----------------------------------------------------------------------

def test_one_cut_point_per_block(gm, batch):
    blocks = block_outputs(gm, batch)
    assert len(blocks) == BLOCKS
    assert all(b.shape == (2, TOKENS, WIDTH) for b in blocks)


def test_block_output_is_the_residual_addition_not_the_branch(gm, batch):
    """Regression: name-based grouping selected the branch, which is wrong.

    Operators written with + trace to bare calls with no block prefix, so any
    rule that groups by the block index in a node's name cannot see the
    addition and lands on the last prefixed node instead.
    """
    blocks = block_outputs(gm, batch)
    for b in blocks:
        assert b.op == "call_function", f"{b.name} is not an operator call"
        assert "drop_path" not in b.name
        assert "mlp" not in b.name
        assert "ls" not in b.name.split("_")


def test_block_outputs_are_in_depth_order(gm, batch):
    """Depth order must match graph order, since the sweep indexes by depth."""
    order = [n.name for n in list_nodes(gm)]
    chosen = [order.index(b.name) for b in block_outputs(gm, batch)]
    assert chosen == sorted(chosen)


def test_block_outputs_are_distinct(gm, batch):
    names = [b.name for b in block_outputs(gm, batch)]
    assert len(set(names)) == len(names)


def test_narrower_model_reports_its_own_width(batch):
    """Width is read from the model, never configured."""
    small = symbolic_trace(timm.create_model("vit_small_patch16_224",
                                             pretrained=False).eval())
    blocks = block_outputs(small, batch)
    assert len(blocks) == BLOCKS
    assert blocks[0].shape[2] == 384


def test_torchvision_model_gives_the_same_cut_points(batch):
    """The two libraries name nodes differently but must cut in the same places."""
    tv = pytest.importorskip("torchvision")
    gm2 = symbolic_trace(tv.models.vit_b_16(weights=None).eval())
    blocks = block_outputs(gm2, batch)
    assert len(blocks) == BLOCKS
    assert all(b.shape == (2, TOKENS, WIDTH) for b in blocks)


def test_model_without_blocks_yields_no_block_outputs(batch):
    """A model of another shape must return nothing rather than guess."""
    plain = symbolic_trace(torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3), torch.nn.ReLU()).eval())
    assert block_outputs(plain, batch) == []


def test_describe_mentions_every_block(gm, batch):
    text = describe(gm, batch)
    assert f"{BLOCKS} block outputs" in text
    for b in block_outputs(gm, batch):
        assert b.name in text


# ----------------------------------------------------------------------
# widths
# ----------------------------------------------------------------------

def test_activation_width_matches_the_model(gm, batch):
    node = block_outputs(gm, batch)[6].name
    assert activation_width(gm, node, batch) == WIDTH


def test_activation_width_rejects_a_node_with_no_tensor(gm, batch):
    with pytest.raises(KeyError):
        activation_width(gm, "no_such_node", batch)


# ----------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------

def test_load_backbone_returns_an_eval_model():
    m = load_backbone(MODEL, pretrained=False, source="timm")
    assert not m.training
    assert m.embed_dim == WIDTH


def test_load_backbone_rejects_an_unknown_source():
    with pytest.raises(ValueError, match="unknown model source"):
        load_backbone(MODEL, pretrained=False, source="keras")


# ----------------------------------------------------------------------
# graph surgery
#
# These need the external graph library. Skipped rather than mocked when it is
# absent, since a mocked graph library would assert nothing about the surgery.
# ----------------------------------------------------------------------

try:
    import nn_lib                                                     # noqa: F401
    HAVE_NN_LIB = True
except ImportError:
    HAVE_NN_LIB = False

needs_nn_lib = pytest.mark.skipif(
    not HAVE_NN_LIB, reason="graph surgery library not installed")


@pytest.fixture(scope="module")
def traced():
    from stitch.models import trace
    return trace(timm.create_model(MODEL, pretrained=False))


@needs_nn_lib
def test_upstream_returns_the_activation_at_the_cut(traced, batch):
    from stitch.models import upstream
    node = block_outputs(traced, batch)[6].name
    with torch.no_grad():
        out = upstream(traced, node)(batch)
    assert out.shape == (2, TOKENS, WIDTH)


@needs_nn_lib
def test_downstream_consumes_that_activation(traced, batch):
    from stitch.models import upstream, downstream
    node = block_outputs(traced, batch)[6].name
    with torch.no_grad():
        h = upstream(traced, node)(batch)
        y = downstream(traced, node)(h)
    assert y.shape[0] == 2 and y.ndim == 2


@needs_nn_lib
def test_halves_recompose_into_the_original(traced, batch):
    """Cutting and rejoining must not change what the model computes."""
    from stitch.models import upstream, downstream
    node = block_outputs(traced, batch)[6].name
    with torch.no_grad():
        direct = traced(batch)
        split = downstream(traced, node)(upstream(traced, node)(batch))
    assert torch.allclose(direct, split, atol=1e-4)


@needs_nn_lib
def test_extractor_returns_every_requested_layer_in_one_pass(traced, batch):
    from stitch.models import make_extractor
    names = [b.name for b in block_outputs(traced, batch)[:3]]
    with torch.no_grad():
        out = make_extractor(traced, names)(batch)
    assert isinstance(out, dict) and len(out) == 3
    assert all(v.shape == (2, TOKENS, WIDTH) for v in out.values())


@needs_nn_lib
def test_extractor_rejects_an_empty_request(traced):
    from stitch.models import make_extractor
    with pytest.raises(ValueError):
        make_extractor(traced, [])


@needs_nn_lib
def test_stitched_model_runs_and_only_the_connector_trains(traced, batch):
    """A merged model shares parameters with its sources.

    Without freezing, a loss at the output fine-tunes the backbones, which
    produces results that look reasonable and are not comparable with anything.
    """
    from nn_lib.utils import frozen
    from stitch.models import stitch, trainable_parameters

    other = __import__("stitch.models", fromlist=["trace"]).trace(
        timm.create_model(MODEL, pretrained=False))
    node = block_outputs(traced, batch)[6].name
    node_b = block_outputs(other, batch)[6].name
    connector = torch.nn.Linear(WIDTH, WIDTH)

    merged = stitch(traced, node, connector, other, node_b)
    with torch.no_grad():
        out = merged(batch)
    assert out.shape[0] == 2

    with frozen(traced, other):
        trainable = trainable_parameters(merged)
    assert trainable, "the connector must remain trainable"
    assert all("connector" in n for n in trainable), \
        f"a backbone is still trainable: {[n for n in trainable if 'connector' not in n][:3]}"
