#!/usr/bin/env python3
"""Build an ImageNet-100 tree from a full ImageNet installation.

ImageNet-100 is a hundred-class subset of ImageNet, so no download is needed
where the full set is already present. This creates a directory of symbolic
links pointing at the original class directories, which costs no disk and
leaves the source untouched.

The class list is read from a file rather than embedded, so the subset is
reproducible and its provenance is a property of the data rather than of this
script.

Expected source layout, which is what ``ImageFolder`` reads::

    <source>/train/<wnid>/<images>
    <source>/val/<wnid>/<images>

An ImageNet validation set distributed as a flat directory of images has to be
reorganised into class directories before this will work. The script says so
rather than producing a tree that fails later.

Usage:
    python scripts/make_imagenet100.py \\
        --source /data/datasets/imagenet \\
        --dest data/imagenet-100 \\
        --classes configs/imagenet100_classes.txt
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def read_classes(path: Path) -> list[str]:
    """Class identifiers, one per line, ignoring blanks and comments."""
    lines = [line.strip() for line in path.read_text().splitlines()]
    names = [line for line in lines if line and not line.startswith("#")]

    if len(set(names)) != len(names):
        raise SystemExit(f"{path} contains duplicates")
    return names


def check_split(source: Path, split: str, names: list[str]) -> list[str]:
    """Report which classes are missing from one split of the source."""
    directory = source / split
    if not directory.is_dir():
        raise SystemExit(f"no such directory: {directory}")

    entries = {entry.name for entry in directory.iterdir() if entry.is_dir()}
    if not entries:
        raise SystemExit(
            f"{directory} holds no class directories. An ImageNet validation set "
            f"distributed as a flat directory of images has to be reorganised into "
            f"one directory per class first.")

    return [name for name in names if name not in entries]


def link_split(source: Path, dest: Path, split: str, names: list[str], copy: bool) -> int:
    """Create one entry per class, either a symlink or a copy."""
    target = dest / split
    target.mkdir(parents=True, exist_ok=True)

    made = 0
    for name in names:
        origin = (source / split / name).resolve()
        link = target / name

        if link.exists() or link.is_symlink():
            continue

        if copy:
            shutil.copytree(origin, link)
        else:
            os.symlink(origin, link, target_is_directory=True)
        made += 1
    return made


def count_images(directory: Path) -> int:
    """Images under a split.

    os.walk with followlinks, because the class directories are symbolic links
    and pathlib's rglob does not descend into those.
    """
    return sum(len(files) for _, _, files in os.walk(directory, followlinks=True))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True,
                        help="full ImageNet root, holding train and val")
    parser.add_argument("--dest", type=Path, required=True,
                        help="where to build the subset")
    parser.add_argument("--classes", type=Path, required=True,
                        help="file of class identifiers, one per line")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--copy", action="store_true",
                        help="copy the images instead of linking to them")
    args = parser.parse_args()

    names = read_classes(args.classes)
    print(f"{len(names)} classes from {args.classes}")

    for split in args.splits:
        missing = check_split(args.source, split, names)
        if missing:
            raise SystemExit(
                f"{len(missing)} of {len(names)} classes are absent from "
                f"{args.source / split}, first few: {missing[:5]}")

    for split in args.splits:
        made = link_split(args.source, args.dest, split, names, args.copy)
        total = count_images(args.dest / split)
        verb = "copied" if args.copy else "linked"
        print(f"{split}: {verb} {made} classes, {total} images")

    print(f"\ndataset.root: {args.dest.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())