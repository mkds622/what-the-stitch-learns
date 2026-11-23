"""Datasets and loaders.

Images are prepared the way the backbone expects, not the way a setting says.
A checkpoint was trained at a particular resolution, with a particular crop
ratio, interpolation and normalisation, and feeding it anything else produces
activations that are quietly wrong. The backbone carries that information, so
the transform is built from the model and nothing about preprocessing appears
in the configuration. From the point where two models are involved this also
stops being a choice: one setting cannot be correct for two checkpoints that
disagree. Two backbones that agree on size, interpolation and crop share one
decoded image, and each applies its own normalisation to it.

Preprocessing is deterministic for every split, training included. Augmentation
exists to make a classifier generalise; here the classifier is frozen and what
is being measured is its representation of an image. A random crop would mean
the same image produced a different activation on every pass, which is noise
added to the thing under study.

Layout expected on disk, which is what ``ImageFolder`` reads::

    <root>/<split>/<class>/<image files>

Reproducibility: data order comes from ``train.seed`` through an explicit
generator and per-worker seeding, so two runs at the same seed see the same
batches in the same order.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# preprocessing
# ----------------------------------------------------------------------

def data_config(model: nn.Module) -> dict:
    """What the backbone declares about its own preprocessing.

    Raises:
        ValueError: if the model does not declare what it expects. Guessing
            would produce activations that are wrong with nothing to show for
            it, so the caller is told instead.
    """
    if getattr(model, "pretrained_cfg", None) is None:
        raise ValueError(
            f"{type(model).__name__} does not declare its expected preprocessing, "
            f"so it cannot be inferred. Pass a transform explicitly.")

    try:
        from timm.data import resolve_model_data_config
        return dict(resolve_model_data_config(model))
    except ImportError:
        from timm.data import resolve_data_config
        return dict(resolve_data_config(model=model))


def build_transform(model: nn.Module, normalize: bool = True):
    """The preprocessing pipeline the backbone was trained with.

    Args:
        model: a loaded backbone carrying its own data configuration.
        normalize: apply the backbone's own mean and standard deviation. Off,
            the pipeline stops at pixel values in ``[0, 1]``, and the
            normalisation is applied separately per model by
            :func:`normalizer_for`.

    Returns:
        A callable turning a PIL image into a tensor.

    Raises:
        ValueError: if the model does not declare what it expects.
    """
    config = data_config(model)
    if not normalize:
        config["mean"] = (0.0, 0.0, 0.0)
        config["std"] = (1.0, 1.0, 1.0)

    from timm.data import create_transform
    return create_transform(**config, is_training=False)


# What decides which pixels a model sees, as opposed to how they are scaled.
# Models that agree on these can share one decoded and cropped image, each
# normalising it its own way.
SHARED_PREPARATION = ("input_size", "interpolation", "crop_pct", "crop_mode")


def shared_transform(backbones: dict[str, nn.Module]):
    """One transform for several backbones, stopping before normalisation.

    Every backbone sees the same pixels and applies its own normalisation
    afterwards, through :func:`normalizer_for`. One image is read once, and no
    model is fed another model's preprocessing.

    Args:
        backbones: the untraced backbones, by label, which carry their data
            configurations.

    Returns:
        A callable turning a PIL image into a tensor in ``[0, 1]``.

    Raises:
        ValueError: if the backbones disagree on size, interpolation or crop,
            since then no single image is the right one for all of them.
    """
    if not backbones:
        raise ValueError("shared_transform needs at least one backbone")

    configs = {label: data_config(model) for label, model in backbones.items()}
    first_label = next(iter(configs))
    for label, config in configs.items():
        for key in SHARED_PREPARATION:
            if config.get(key) != configs[first_label].get(key):
                raise ValueError(
                    f"{first_label} and {label} disagree on {key} "
                    f"({configs[first_label].get(key)!r} and {config.get(key)!r}), "
                    f"so one image cannot be prepared correctly for both")

    return build_transform(backbones[first_label], normalize=False)


class Normalize(nn.Module):
    """A backbone's own normalisation, applied to a batch of ``[0, 1]`` images."""

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return (images - self.mean) / self.std


def normalizer_for(model: nn.Module) -> Normalize:
    """The normalisation one backbone was trained with, as a module."""
    config = data_config(model)
    return Normalize(config["mean"], config["std"])


# ----------------------------------------------------------------------
# datasets
# ----------------------------------------------------------------------

def build_dataset(cfg, split: str | None = None, transform: Any = None) -> Dataset:
    """Open one split, failing at construction rather than at the first batch.

    Args:
        cfg: the run configuration.
        split: split directory name. Defaults to ``dataset.split``.
        transform: preprocessing, usually from :func:`build_transform`.

    Returns:
        An ``ImageFolder`` over ``<root>/<split>``.

    Raises:
        ValueError: if ``dataset.root`` is unset, the split holds no classes, or
            ``dataset.num_classes`` disagrees with what is on disk.
        FileNotFoundError: if the split directory does not exist.

    Side effect:
        Fills ``dataset.num_classes`` when it was left unset, so the rest of the
        run can rely on it and the recorded configuration says what was used.
    """
    from torchvision.datasets import ImageFolder

    if not cfg.dataset.root:
        raise ValueError("dataset.root is not set, so there is nothing to open")

    directory = Path(cfg.dataset.root) / (split or cfg.dataset.split)
    if not directory.is_dir():
        raise FileNotFoundError(f"no such split directory: {directory}")

    dataset = ImageFolder(directory, transform=transform)

    found = len(dataset.classes)
    if found == 0:
        raise ValueError(f"{directory} contains no class directories")

    if cfg.dataset.num_classes is None:
        cfg.dataset.num_classes = found
        log.info("read %d classes from %s", found, directory)
    elif cfg.dataset.num_classes != found:
        raise ValueError(
            f"dataset.num_classes is {cfg.dataset.num_classes} but {directory} "
            f"holds {found} classes")

    return dataset


# ----------------------------------------------------------------------
# loaders
# ----------------------------------------------------------------------

def _seed_worker(worker_id: int) -> None:
    """Seed a worker's own generators from the seed torch gave it.

    Each worker is a separate process with its own random state. Without this
    they start from whatever the operating system provides and the data order
    is not reproducible, whatever the main process does.
    """
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def build_loader(cfg, dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    """Wrap a dataset in a loader whose order is determined by the seed.

    Args:
        cfg: the run configuration.
        dataset: what to read.
        batch_size: images per batch.
        shuffle: whether to shuffle. The last partial batch is dropped when
            shuffling, so every training step sees the same number of tokens,
            and kept otherwise, so evaluation covers the whole split.

    Returns:
        A ``DataLoader``.
    """
    workers = cfg.dataset.num_workers
    generator = torch.Generator()
    generator.manual_seed(cfg.train.seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=cfg.dataset.pin_memory and cfg.train.device != "cpu",
        drop_last=shuffle,
        generator=generator,
        worker_init_fn=_seed_worker,
        persistent_workers=workers > 0,
    )


def image_loaders(cfg, model: nn.Module) -> tuple[DataLoader, DataLoader]:
    """The training and evaluation loaders for a run.

    Both use the backbone's own preprocessing and the extraction batch size,
    since what passes through them is images on their way into a frozen model
    rather than training examples.

    Returns:
        ``(train_loader, eval_loader)``.
    """
    transform = build_transform(model)
    batch_size = cfg.activations.extraction_batch_size

    train = build_dataset(cfg, cfg.dataset.split, transform)
    evaluation = build_dataset(cfg, cfg.dataset.eval_split, transform)

    return (build_loader(cfg, train, batch_size, shuffle=True),
            build_loader(cfg, evaluation, batch_size, shuffle=False))