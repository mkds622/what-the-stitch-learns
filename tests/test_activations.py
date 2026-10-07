"""Tests for activation supply.

A four-channel toy model over random images, so these run on CPU in under a
second with no dataset and no backbone download. The properties under test are
about the stream, not about any particular model: batch shape, alignment
between layers, and reproducibility from the seed.

    python -m pytest tests/test_activations.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")

from torch import nn                                                  # noqa: E402
from torch.utils.data import TensorDataset                            # noqa: E402

from stitch.config import Config                                      # noqa: E402
from stitch.data import build_loader                                  # noqa: E402

try:
    import nn_lib                                                     # noqa: F401
    HAVE_NN_LIB = True
except ImportError:
    HAVE_NN_LIB = False

needs_nn_lib = pytest.mark.skipif(
    not HAVE_NN_LIB, reason="graph surgery library not installed")

WIDTH = 4
TOKENS = 16          # a 32 by 32 image at stride 8
IMAGES = 24


class TokenNet(nn.Module):
    """A model producing a token stream, with a second layer derived from the first.

    The second is exactly twice the first, which makes it possible to assert
    that shuffling keeps corresponding tokens together: after any permutation,
    one must still be double the other row for row.
    """

    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, WIDTH, kernel_size=8, stride=8)

    def forward(self, x):
        stream = self.proj(x).flatten(2).transpose(1, 2)
        return stream * 2


@pytest.fixture(scope="module")
def traced():
    from stitch.models import trace
    return trace(TokenNet())


@pytest.fixture(scope="module")
def node_names(traced):
    """The two node names this model traces to, since fx generates them."""
    names = [node.name for node in traced.graph.nodes]
    assert "transpose" in names and "mul" in names, names
    return "transpose", "mul"


@pytest.fixture
def cfg(node_names):
    first, second = node_names
    c = Config()
    c.dataset.num_workers = 0
    c.train.device = "cpu"
    c.train.token_batch = 8
    c.activations.mode = "on_the_fly"
    c.activations.layers = [f"A@{first}", f"A@{second}"]
    c.activations.shuffle_buffer_tokens = 64
    return c


@pytest.fixture
def loader(cfg):
    images = torch.randn(IMAGES, 3, 32, 32)
    labels = torch.zeros(IMAGES, dtype=torch.long)
    return build_loader(cfg, TensorDataset(images, labels), batch_size=4, shuffle=False)


def first_batches(source, count: int) -> list[dict]:
    stream = source.batches()
    return [next(stream) for _ in range(count)]


# ----------------------------------------------------------------------
# shape and content
# ----------------------------------------------------------------------

@needs_nn_lib
def test_batches_are_keyed_by_layer(cfg, traced, loader):
    from stitch.activations import build_source

    batch = first_batches(build_source(cfg, {"A": traced}, loader), 1)[0]
    assert sorted(batch) == sorted(cfg.activations.layers)


@needs_nn_lib
def test_batch_is_tokens_by_width(cfg, traced, loader):
    """The coder consumes tokens, so the image dimension must be gone."""
    from stitch.activations import build_source

    batch = first_batches(build_source(cfg, {"A": traced}, loader), 1)[0]
    for value in batch.values():
        assert value.shape == (cfg.train.token_batch, WIDTH)


@needs_nn_lib
def test_layers_stay_aligned_through_shuffling(cfg, traced, loader):
    """A token in one layer must still be the same token in another.

    Permuting each layer independently would destroy the pairing silently, and
    every number computed from a second layer afterwards would be meaningless.
    """
    from stitch.activations import build_source

    first, second = cfg.activations.layers
    for batch in first_batches(build_source(cfg, {"A": traced}, loader), 3):
        assert torch.allclose(batch[second], batch[first] * 2, atol=1e-5)


@needs_nn_lib
def test_the_stream_does_not_end(cfg, traced, loader):
    """Twenty-four images cannot supply this many batches without repeating."""
    from stitch.activations import build_source

    wanted = (IMAGES * TOKENS) // cfg.train.token_batch * 3
    assert len(first_batches(build_source(cfg, {"A": traced}, loader), wanted)) == wanted


@needs_nn_lib
def test_widths_match_the_model(cfg, traced, loader):
    from stitch.activations import build_source

    widths = build_source(cfg, {"A": traced}, loader).widths()
    assert set(widths) == set(cfg.activations.layers)
    assert all(width == WIDTH for width in widths.values())


# ----------------------------------------------------------------------
# reproducibility
# ----------------------------------------------------------------------

@needs_nn_lib
def test_the_same_seed_gives_the_same_tokens(cfg, traced, loader):
    from stitch.activations import build_source

    key = cfg.activations.layers[0]
    one = first_batches(build_source(cfg, {"A": traced}, loader), 2)
    two = first_batches(build_source(cfg, {"A": traced}, loader), 2)

    assert all(torch.equal(a[key], b[key]) for a, b in zip(one, two))


@needs_nn_lib
def test_a_different_seed_gives_different_tokens(cfg, traced, loader):
    """Seeds are what the noise floor is measured across, so they must bite."""
    from stitch.activations import build_source

    key = cfg.activations.layers[0]
    one = first_batches(build_source(cfg, {"A": traced}, loader), 2)

    cfg.train.seed = 1
    two = first_batches(build_source(cfg, {"A": traced}, loader), 2)

    assert not torch.equal(one[0][key], two[0][key])


@needs_nn_lib
def test_tokens_are_not_in_model_order(cfg, traced, loader):
    """Unshuffled, a batch would be consecutive tokens from one or two images."""
    from stitch.activations import build_source

    key = cfg.activations.layers[0]
    shuffled = first_batches(build_source(cfg, {"A": traced}, loader), 1)[0][key]

    with torch.no_grad():
        images, _ = next(iter(loader))
        straight = traced(images).reshape(-1, WIDTH)[:cfg.train.token_batch] * 1

    assert not torch.allclose(shuffled, straight, atol=1e-5)


# ----------------------------------------------------------------------
# selection and errors
# ----------------------------------------------------------------------

@needs_nn_lib
def test_a_layer_naming_an_absent_model_raises(cfg, traced, loader):
    from stitch.activations import build_source

    cfg.activations.layers = cfg.activations.layers + ["B@transpose"]
    with pytest.raises(KeyError, match="B"):
        build_source(cfg, {"A": traced}, loader)


def test_cached_mode_is_not_implemented_yet(cfg):
    from stitch.activations import build_source

    cfg.activations.mode = "cached"
    with pytest.raises(NotImplementedError):
        build_source(cfg, {}, None)


def test_unknown_mode_raises(cfg):
    from stitch.activations import build_source

    cfg.activations.mode = "streamed"
    with pytest.raises(ValueError, match="activations.mode"):
        build_source(cfg, {}, None)