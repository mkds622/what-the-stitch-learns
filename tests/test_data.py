"""Tests for datasets and loaders.

A handful of generated images standing in for the real thing, so these run on
CPU with no dataset, no GPU and no network. What they check is the plumbing
that would otherwise only fail several minutes into a run: that preprocessing
comes from the model, that a bad path fails immediately, and that the data
order follows the seed.

    python -m pytest tests/test_data.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("PIL")

from stitch.config import Config                                      # noqa: E402
from stitch.data import (                                             # noqa: E402
    build_dataset, build_loader, build_transform, image_loaders,
)

CLASSES = ("n01440764", "n01443537", "n01484850")
PER_CLASS = 4

@pytest.fixture(scope="module")
def transform():
    """The real preprocessing, built once for the whole file."""
    timm = pytest.importorskip("timm")
    return build_transform(
        timm.create_model("vit_base_patch16_224", pretrained=False).eval())

@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    """A tiny ImageFolder tree with the same layout as the real dataset."""
    from PIL import Image

    root = tmp_path_factory.mktemp("imagenet")
    for split in ("train", "val"):
        for index, name in enumerate(CLASSES):
            directory = root / split / name
            directory.mkdir(parents=True)
            for n in range(PER_CLASS):
                shade = (index * 60 + n * 10) % 256
                Image.new("RGB", (256, 320), (shade, 128, 255 - shade)).save(
                    directory / f"{name}_{n}.JPEG")
    return root


@pytest.fixture
def cfg(tree):
    """A configuration pointing at the generated tree, with workers off.

    Workers are disabled so the tests are deterministic and do not pay process
    startup for twelve images.
    """
    c = Config()
    c.dataset.root = str(tree)
    c.dataset.num_workers = 0
    c.train.device = "cpu"
    return c


# ----------------------------------------------------------------------
# preprocessing
# ----------------------------------------------------------------------

def test_transform_matches_the_model_input_size():
    """The transform must produce what the backbone expects, not a guess."""
    timm = pytest.importorskip("timm")
    from PIL import Image

    model = timm.create_model("vit_base_patch16_224", pretrained=False).eval()
    out = build_transform(model)(Image.new("RGB", (400, 300)))

    assert out.shape == (3, 224, 224)
    assert out.dtype == torch.float32


def test_transform_follows_the_model_rather_than_a_fixed_size():
    """A model at another resolution must get that resolution."""
    timm = pytest.importorskip("timm")
    from PIL import Image

    model = timm.create_model("vit_base_patch16_384", pretrained=False).eval()
    out = build_transform(model)(Image.new("RGB", (400, 300)))

    assert out.shape == (3, 384, 384)


def test_transform_refuses_a_model_that_declares_nothing():
    """Silently guessing would produce wrong activations with no symptom."""
    with pytest.raises(ValueError, match="preprocessing"):
        build_transform(torch.nn.Linear(4, 4))


def test_normalisation_is_applied():
    """An unnormalised tensor would sit in [0, 1] and never go negative."""
    timm = pytest.importorskip("timm")
    from PIL import Image

    model = timm.create_model("vit_base_patch16_224", pretrained=False).eval()
    out = build_transform(model)(Image.new("RGB", (256, 256), (10, 10, 10)))

    assert out.min() < 0.0


def test_preprocessing_is_deterministic():
    """The same image must give the same activation on every pass."""
    timm = pytest.importorskip("timm")
    from PIL import Image

    model = timm.create_model("vit_base_patch16_224", pretrained=False).eval()
    transform = build_transform(model)
    image = Image.new("RGB", (400, 300), (30, 90, 150))

    assert torch.equal(transform(image), transform(image))


# ----------------------------------------------------------------------
# datasets
# ----------------------------------------------------------------------

def test_classes_are_discovered_in_sorted_order(cfg, transform):
    dataset = build_dataset(cfg, "train", transform)
    assert dataset.classes == sorted(CLASSES)
    assert len(dataset) == len(CLASSES) * PER_CLASS


def test_num_classes_is_filled_when_unset(cfg):
    assert cfg.dataset.num_classes is None
    build_dataset(cfg, "train")
    assert cfg.dataset.num_classes == len(CLASSES)


def test_num_classes_is_kept_when_it_agrees(cfg):
    cfg.dataset.num_classes = len(CLASSES)
    build_dataset(cfg, "train")
    assert cfg.dataset.num_classes == len(CLASSES)


def test_num_classes_mismatch_raises(cfg):
    """A configured count that disagrees with the disk is a wrong run, not a warning."""
    cfg.dataset.num_classes = 100
    with pytest.raises(ValueError, match="num_classes"):
        build_dataset(cfg, "train")


def test_unset_root_raises(cfg):
    cfg.dataset.root = ""
    with pytest.raises(ValueError, match="dataset.root"):
        build_dataset(cfg, "train")


def test_missing_split_raises_immediately(cfg):
    """Failing here costs a second; failing at the first batch costs minutes."""
    with pytest.raises(FileNotFoundError, match="test"):
        build_dataset(cfg, "test")


def test_both_splits_are_present(cfg):
    assert len(build_dataset(cfg, "train")) == len(build_dataset(cfg, "val"))


# ----------------------------------------------------------------------
# loaders
# ----------------------------------------------------------------------

def _label_order(loader) -> list[int]:
    return [int(label) for _, labels in loader for label in labels]


def test_batch_shape_and_dtype(cfg):
    timm = pytest.importorskip("timm")
    model = timm.create_model("vit_base_patch16_224", pretrained=False).eval()

    dataset = build_dataset(cfg, "train", build_transform(model))
    images, labels = next(iter(build_loader(cfg, dataset, 4, shuffle=False)))

    assert images.shape == (4, 3, 224, 224)
    assert images.dtype == torch.float32
    assert labels.shape == (4,)


def test_the_same_seed_gives_the_same_order(cfg, transform):
    dataset = build_dataset(cfg, "train", transform)
    first = _label_order(build_loader(cfg, dataset, 4, shuffle=True))
    second = _label_order(build_loader(cfg, dataset, 4, shuffle=True))
    assert first == second


def test_a_different_seed_gives_a_different_order(cfg, transform):
    """Seeds are what the noise floor is measured across, so they must bite."""
    dataset = build_dataset(cfg, "train", transform)
    first = _label_order(build_loader(cfg, dataset, 4, shuffle=True))

    cfg.train.seed = 1
    second = _label_order(build_loader(cfg, dataset, 4, shuffle=True))

    assert first != second


def test_shuffling_drops_the_partial_batch(cfg, transform):
    """Every training step must see the same number of tokens."""
    dataset = build_dataset(cfg, "train", transform)
    loader = build_loader(cfg, dataset, 5, shuffle=True)
    assert len(_label_order(loader)) == 10        # twelve images, two full batches


def test_evaluation_keeps_every_image(cfg, transform):
    dataset = build_dataset(cfg, "val", transform)
    loader = build_loader(cfg, dataset, 5, shuffle=False)
    assert len(_label_order(loader)) == len(dataset)


def test_evaluation_order_is_the_dataset_order(cfg, transform):
    dataset = build_dataset(cfg, "val", transform)
    loader = build_loader(cfg, dataset, 4, shuffle=False)
    assert _label_order(loader) == sorted(_label_order(loader))


def test_image_loaders_returns_both_splits(cfg):
    timm = pytest.importorskip("timm")
    model = timm.create_model("vit_base_patch16_224", pretrained=False).eval()
    cfg.activations.extraction_batch_size = 4

    train, evaluation = image_loaders(cfg, model)

    assert train.batch_size == 4 and evaluation.batch_size == 4
    assert train.drop_last and not evaluation.drop_last
    assert len(_label_order(evaluation)) == len(CLASSES) * PER_CLASS