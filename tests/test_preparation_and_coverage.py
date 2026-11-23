"""Each model reads images prepared its own way, and evaluation reads all of them.

Two properties, both checked at every place images or evaluation tokens enter.

**Preparation.** One image is decoded, resized and cropped once, and each model
applies its own normalisation to it. The two main backbones agree on size,
interpolation and crop and differ in mean and standard deviation, so the check
that matters is that the shared pipeline followed by a model's own
normalisation is exactly that model's own transform, and that every path into a
model (the training and evaluation streams, the sweep, the substitution halves)
applies the right model's normalisation.

**Coverage.** Evaluation reads every token of the evaluation split exactly
once: the held-out metrics, the feature frequencies, the sweep's held-out half
and the firing comparison. Nothing is skipped because a buffer filled first,
and nothing is read twice because a stream wrapped.

    python -m pytest tests/test_preparation_and_coverage.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")

from PIL import Image                                                 # noqa: E402
from torch import nn                                                  # noqa: E402
from torch.utils.data import TensorDataset                            # noqa: E402

from stitch.config import Config                                      # noqa: E402
from stitch.data import (                                             # noqa: E402
    build_loader, build_transform, normalizer_for, shared_transform,
)

try:
    import nn_lib                                                     # noqa: F401
    HAVE_NN_LIB = True
except ImportError:
    HAVE_NN_LIB = False

needs_nn_lib = pytest.mark.skipif(
    not HAVE_NN_LIB, reason="graph surgery library not installed")

MODEL_A = "vit_base_patch16_224.augreg_in21k_ft_in1k"
MODEL_B = "deit_base_patch16_224.fb_in1k"

WIDTH = 4
TOKENS = 16          # a 32 by 32 image at stride 8
IMAGES = 24


# ----------------------------------------------------------------------
# small stand-ins
# ----------------------------------------------------------------------

class TokenNet(nn.Module):
    """A model producing a token stream, declaring its own preprocessing."""

    def __init__(self, seed: int, mean, std):
        super().__init__()
        torch.manual_seed(seed)
        self.proj = nn.Conv2d(3, WIDTH, kernel_size=8, stride=8)
        self.pretrained_cfg = {
            "input_size": (3, 32, 32), "interpolation": "bilinear",
            "crop_pct": 1.0, "crop_mode": "center",
            "mean": tuple(mean), "std": tuple(std),
        }

    def forward(self, x):
        stream = self.proj(x).flatten(2).transpose(1, 2)
        return stream * 2


MEAN_A, STD_A = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
MEAN_B, STD_B = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


@pytest.fixture(scope="module")
def backbones():
    return {"A": TokenNet(0, MEAN_A, STD_A), "B": TokenNet(1, MEAN_B, STD_B)}


@pytest.fixture(scope="module")
def traced(backbones):
    from stitch.models import trace
    return {label: trace(model) for label, model in backbones.items()}


@pytest.fixture
def cfg():
    c = Config()
    c.dataset.num_workers = 0
    c.train.device = "cpu"
    c.train.token_batch = 50          # does not divide the split, so the last batch is short
    c.activations.mode = "on_the_fly"
    c.activations.layers = ["A@transpose", "B@transpose"]
    c.activations.shuffle_buffer_tokens = 64
    c.activations.scaling = "none"
    c.activations.scale_tokens = 64
    return c


@pytest.fixture(scope="module")
def images():
    torch.manual_seed(2)
    return torch.rand(IMAGES, 3, 32, 32)


@pytest.fixture
def loader(cfg, images):
    labels = torch.arange(IMAGES) % 4
    return build_loader(cfg, TensorDataset(images, labels), batch_size=5, shuffle=False)


def expected(traced, backbones, label: str, images: torch.Tensor) -> torch.Tensor:
    """What one model produces at the cut, fed its own normalisation, computed directly."""
    from stitch.models import make_extractor
    with torch.no_grad():
        out = make_extractor(traced[label], ["transpose"])(normalizer_for(backbones[label])(images))
    return out["transpose"].reshape(-1, WIDTH)


# ----------------------------------------------------------------------
# preparation, on the two main backbones
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def main_backbones():
    return {"A": timm.create_model(MODEL_A, pretrained=False).eval(),
            "B": timm.create_model(MODEL_B, pretrained=False).eval()}


def test_the_two_main_backbones_normalise_differently(main_backbones):
    """Otherwise the checks below could not tell one model's preparation from the other's."""
    a = normalizer_for(main_backbones["A"])
    b = normalizer_for(main_backbones["B"])
    assert not torch.allclose(a.mean, b.mean) and not torch.allclose(a.std, b.std)


@pytest.mark.parametrize("label", ["A", "B"])
def test_shared_preparation_then_own_normalisation_is_the_models_own_transform(
        main_backbones, label):
    """The shared pipeline plus a model's normalisation is exactly that model's transform."""
    torch.manual_seed(3)
    picture = Image.fromarray(
        (torch.rand(300, 400, 3) * 255).to(torch.uint8).numpy(), mode="RGB")

    model = main_backbones[label]
    own = build_transform(model)(picture)[None]
    shared = normalizer_for(model)(shared_transform(main_backbones)(picture)[None])

    assert shared.shape == own.shape
    assert torch.allclose(shared, own, atol=1e-6), \
        f"largest difference {(shared - own).abs().max():.3e}"


def test_backbones_that_crop_differently_cannot_share_an_image(main_backbones):
    other = timm.create_model(MODEL_B, pretrained=False).eval()
    other.pretrained_cfg = {**other.pretrained_cfg, "crop_pct": 0.875}
    with pytest.raises(ValueError, match="crop_pct"):
        shared_transform({"A": main_backbones["A"], "B": other})


# ----------------------------------------------------------------------
# preparation, at every path into a model
# ----------------------------------------------------------------------

@needs_nn_lib
def test_each_model_in_the_stream_reads_its_own_normalisation(cfg, traced, backbones, loader, images):
    from stitch.activations import build_source

    normalizers = {label: normalizer_for(model) for label, model in backbones.items()}
    source = build_source(cfg, traced, loader, normalizers)
    batches = list(source.single_pass())

    for label in ("A", "B"):
        got = torch.cat([batch[f"{label}@transpose"] for batch in batches])
        assert torch.allclose(got, expected(traced, backbones, label, images), atol=1e-6), label


@needs_nn_lib
def test_a_stream_given_normalisers_must_have_one_for_every_model(cfg, traced, backbones, loader):
    from stitch.activations import build_source
    with pytest.raises(KeyError, match="B"):
        build_source(cfg, traced, loader, {"A": normalizer_for(backbones["A"])})


@needs_nn_lib
def test_each_upper_half_normalises_for_its_own_model(cfg, traced, backbones, images):
    """The coder reads A prepared A's way; B's unmodified prediction starts from B's way."""
    from stitch.run import build_halves

    upper, lower, reference_upper = build_halves(cfg, traced, backbones)
    with torch.no_grad():
        a = upper(images).reshape(-1, WIDTH)
        b = reference_upper(images).reshape(-1, WIDTH)

    assert torch.allclose(a, expected(traced, backbones, "A", images), atol=1e-6)
    assert torch.allclose(b, expected(traced, backbones, "B", images), atol=1e-6)


@needs_nn_lib
def test_the_sweep_extracts_each_model_with_its_own_normalisation(cfg, traced, backbones, loader, images):
    """The sweep's held-out sums equal sums over each model's own activations."""
    from stitch.models import make_extractor
    from stitch.sweep import accumulate

    extractors = {label: nn.Sequential(normalizer_for(backbones[label]),
                                       make_extractor(traced[label], ["transpose"]))
                  for label in ("A", "B")}

    def extract(batch):
        return {f"{label}@{node}": value
                for label, extractor in extractors.items()
                for node, value in extractor(batch).items()}

    keys = ("A@transpose", "B@transpose")
    moments = accumulate(extract, loader, [keys], {key: WIDTH for key in keys}, None)

    # A is the side read and B the side reconstructed, so their sums are kept
    # under the source and target halves of the moments.
    for sums, key, label in ((moments.sum_x, keys[0], "A"), (moments.sum_y, keys[1], "B")):
        direct = expected(traced, backbones, label, images).to(sums[key].dtype)
        assert torch.allclose(sums[key], direct.sum(dim=0), atol=1e-4), key


# ----------------------------------------------------------------------
# coverage
# ----------------------------------------------------------------------

@needs_nn_lib
def test_a_single_pass_reads_every_token_once_in_order(cfg, traced, backbones, loader, images):
    from stitch.activations import build_source

    normalizers = {label: normalizer_for(model) for label, model in backbones.items()}
    batches = list(build_source(cfg, traced, loader, normalizers).single_pass())

    sizes = [batch["A@transpose"].shape[0] for batch in batches]
    total = IMAGES * TOKENS
    assert sum(sizes) == total
    assert all(size == cfg.train.token_batch for size in sizes[:-1])
    assert sizes[-1] == total - cfg.train.token_batch * (len(sizes) - 1)

    got = torch.cat([batch["A@transpose"] for batch in batches])
    assert torch.equal(got, expected(traced, backbones, "A", images))


@needs_nn_lib
def test_held_out_metrics_read_the_whole_split_when_batches_is_unset(cfg, traced, backbones, loader):
    from stitch.activations import build_source
    from stitch.coder import SparseCoder
    from stitch.evaluate import feature_frequencies, reconstruction_metrics
    from stitch.run import held_out

    cfg.eval.batches = None
    normalizers = {label: normalizer_for(model) for label, model in backbones.items()}
    source = build_source(cfg, traced, loader, normalizers)
    coder = SparseCoder(d_in=WIDTH, d_out=WIDTH, expansion=2, sparsity="topk", k=2)

    metrics = reconstruction_metrics(coder, held_out(source, cfg), "A@transpose",
                                     "B@transpose", cfg.eval.batches)
    assert metrics["eval_tokens"] == IMAGES * TOKENS

    frequencies = feature_frequencies(coder, held_out(source, cfg), "A@transpose",
                                      cfg.eval.batches)
    assert frequencies.shape == (coder.d_latent,)


def test_an_endless_source_has_no_single_pass(cfg):
    from stitch.activations import ActivationSource

    class Endless(ActivationSource):
        def batches(self):
            while True:
                yield {}
        def widths(self):
            return {}
        def scales(self):
            return {}
        def set_scales(self, scales):
            pass
        def centres(self):
            return {}

    with pytest.raises(NotImplementedError):
        next(Endless().single_pass())


def test_the_sweep_held_out_half_reads_every_image():
    from stitch.sweep import accumulate

    keys = ("A@x", "B@x")
    batches = [(torch.randn(3, TOKENS, WIDTH), None) for _ in range(7)]

    def extract(images):
        return {key: images for key in keys}

    moments = accumulate(extract, batches, [keys], {key: WIDTH for key in keys}, None)
    assert moments.tokens == 7 * 3 * TOKENS


def test_the_firing_pass_reads_every_token_once_when_uncapped():
    from stitch.coder import SparseCoder
    from stitch.compare import firing_correlation

    torch.manual_seed(4)
    a = SparseCoder(d_in=WIDTH, d_out=WIDTH, expansion=2, sparsity="topk", k=2)
    b = SparseCoder(d_in=WIDTH, d_out=WIDTH, expansion=2, sparsity="topk", k=2)
    sizes = [50, 50, 50, 17]
    stream = [{"A@x": torch.randn(size, WIDTH)} for size in sizes]

    result = firing_correlation(a, b, iter(stream), "A@x", None)
    assert result["tokens"] == sum(sizes)


def test_the_main_configs_evaluate_on_the_whole_split():
    from stitch.config import load_config

    root = Path(__file__).resolve().parents[1]
    for name in ("case1_main.yaml", "case2_main.yaml"):
        assert load_config(root / "configs" / name).eval.batches is None, name