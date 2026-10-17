"""Translating a subset's class numbering into the model's.

A classifier trained on ImageNet predicts one of a thousand classes, numbered
by the alphabetical order of the thousand directory names. A loader pointed at
a hundred-class subset numbers its classes zero to ninety-nine, by the
alphabetical order of the hundred it can see.

Those are two different numbering systems over the same classes. Comparing a
prediction in one against a label in the other is meaningless, and because the
hundred kept classes have original indices spread across the whole range, it
comes out at almost exactly zero rather than at chance, which looks like a
broken model rather than a broken comparison.

So the subset's labels are translated before they are compared with anything
the model said. Nothing else in the pipeline is affected: training never looks
at a label, and reconstruction metrics never look at a prediction.

When training on the full set the two numberings coincide and the translation
is the identity, which is what an empty reference produces.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import Tensor

log = logging.getLogger(__name__)


def class_names(directory: Path) -> list[str]:
    """The class directory names under a split, in the order a loader numbers them.

    Args:
        directory: a split directory holding one subdirectory per class.

    Returns:
        The names, sorted, which is the order ``ImageFolder`` assigns indices in.

    Raises:
        FileNotFoundError: if the directory does not exist or holds no classes.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"no such directory: {directory}")

    names = sorted(entry.name for entry in directory.iterdir() if entry.is_dir())
    if not names:
        raise FileNotFoundError(f"{directory} holds no class directories")
    return names


def build_label_map(subset: list[str], reference: list[str]) -> Tensor:
    """Where each of the subset's classes sits in the reference numbering.

    Args:
        subset: the class names the loader can see, in its own order.
        reference: the class names the model was trained on, in its order.

    Returns:
        ``(len(subset),)`` of indices into ``reference``, so that
        ``mapped = label_map[labels]`` turns loader labels into model labels.

    Raises:
        ValueError: if a subset class is absent from the reference, which means
            the two are not the same dataset and no mapping exists.
    """
    positions = {name: index for index, name in enumerate(reference)}

    missing = [name for name in subset if name not in positions]
    if missing:
        raise ValueError(
            f"{len(missing)} of {len(subset)} classes are not in the reference, "
            f"first few: {missing[:5]}")

    return torch.tensor([positions[name] for name in subset], dtype=torch.long)


def label_map_for(cfg, dataset) -> Tensor | None:
    """The translation a run needs, or None when none is needed.

    Args:
        cfg: the run configuration. ``dataset.label_reference`` names a split
            directory of the dataset the model was trained on. Empty means the
            loader's numbering is already the model's.
        dataset: the evaluation dataset, which carries the class names the
            loader assigned indices to.

    Returns:
        The map, or None if no reference is configured or it is the identity.
    """
    reference = getattr(cfg.dataset, "label_reference", "") or ""
    if not reference:
        return None

    subset = list(getattr(dataset, "classes", []))
    if not subset:
        log.warning("the dataset carries no class names, labels left untranslated")
        return None

    mapping = build_label_map(subset, class_names(Path(reference)))

    if torch.equal(mapping, torch.arange(len(subset))):
        return None

    log.info("translating %d class labels into the model's numbering", len(subset))
    return mapping