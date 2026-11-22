"""Datasets and loaders.

Images are prepared the way the backbone expects, not the way a setting says.
A checkpoint was trained at a particular resolution, with a particular crop
ratio, interpolation and normalisation, and feeding it anything else produces
activations that are quietly wrong. The backbone carries that information, so
the transform is built from the model and nothing about preprocessing appears
in the configuration. From the point where two models are involved this also
stops being a choice: one setting cannot be correct for two checkpoints that
disagree.

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

def build_transform(model: nn.Module):
    """The preprocessing pipeline the backbone was trained with.

    Args:
        model: a loaded backbone carrying its own data configuration.

    Returns:
        A callable turning a PIL image into a normalised tensor.

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
        config = resolve_model_data_config(model)
    except ImportError:
        from timm.data import resolve_data_config
        config = resolve_data_config(model=model)

    from timm.data import create_transform
    return create_transform(**config, is_training=False)


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