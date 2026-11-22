"""Tests for translating a subset's class numbering into the model's.

The bug these exist to prevent: a loader over a hundred-class subset numbers
its classes zero to ninety-nine, a model trained on the full set predicts zero
to nine hundred and ninety-nine, and comparing the two gives almost exactly
zero rather than an error.

    python -m pytest tests/test_labels.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")

from stitch.labels import build_label_map, class_names, label_map_for   # noqa: E402


def make_split(root: Path, names: list[str]) -> Path:
    """A split directory holding one empty subdirectory per class."""
    for name in names:
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


class FakeDataset:
    def __init__(self, classes):
        self.classes = list(classes)


class FakeConfig:
    def __init__(self, reference=""):
        self.dataset = type("D", (), {"label_reference": reference})()


# ----------------------------------------------------------------------
# reading class names
# ----------------------------------------------------------------------

def test_class_names_come_back_sorted(tmp_path):
    """Sorted, because that is the order a loader assigns indices in."""
    make_split(tmp_path, ["n03000000", "n01000000", "n02000000"])
    assert class_names(tmp_path) == ["n01000000", "n02000000", "n03000000"]


def test_files_are_not_classes(tmp_path):
    make_split(tmp_path, ["n01000000"])
    (tmp_path / "README.txt").write_text("not a class")
    assert class_names(tmp_path) == ["n01000000"]


def test_a_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no such directory"):
        class_names(tmp_path / "absent")


def test_a_directory_without_classes_raises(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="no class directories"):
        class_names(tmp_path / "empty")


# ----------------------------------------------------------------------
# the map
# ----------------------------------------------------------------------

def test_a_subset_maps_to_its_positions_in_the_reference():
    reference = [f"n{index:08d}" for index in range(10)]
    subset = ["n00000002", "n00000005", "n00000009"]
    assert build_label_map(subset, reference).tolist() == [2, 5, 9]


def test_the_full_set_maps_to_itself():
    reference = [f"n{index:08d}" for index in range(5)]
    assert build_label_map(reference, reference).tolist() == [0, 1, 2, 3, 4]


def test_a_class_outside_the_reference_raises():
    with pytest.raises(ValueError, match="not in the reference"):
        build_label_map(["n99999999"], ["n00000001"])


def test_the_error_names_the_offenders():
    with pytest.raises(ValueError, match="n00000007"):
        build_label_map(["n00000007"], ["n00000001"])


def test_mapping_labels_gives_the_model_numbering():
    """The whole point: loader labels become labels the model's output can be compared with."""
    reference = [f"n{index:08d}" for index in range(1000)]
    subset = ["n00000042", "n00000207", "n00000931"]
    mapping = build_label_map(subset, reference)

    loader_labels = torch.tensor([0, 2, 1, 0])
    assert mapping[loader_labels].tolist() == [42, 931, 207, 42]


# ----------------------------------------------------------------------
# deciding whether a run needs one
# ----------------------------------------------------------------------

def test_no_reference_means_no_translation():
    assert label_map_for(FakeConfig(""), FakeDataset(["a", "b"])) is None


def test_a_subset_gets_a_translation(tmp_path):
    make_split(tmp_path, [f"n{index:08d}" for index in range(10)])
    cfg = FakeConfig(str(tmp_path))
    mapping = label_map_for(cfg, FakeDataset(["n00000003", "n00000008"]))
    assert mapping.tolist() == [3, 8]


def test_the_full_set_needs_no_translation(tmp_path):
    """The identity is returned as None, so nothing is applied needlessly."""
    names = [f"n{index:08d}" for index in range(10)]
    make_split(tmp_path, names)
    assert label_map_for(FakeConfig(str(tmp_path)), FakeDataset(names)) is None


def test_a_dataset_without_class_names_is_left_alone(tmp_path):
    make_split(tmp_path, ["n00000000"])
    assert label_map_for(FakeConfig(str(tmp_path)), FakeDataset([])) is None


def test_a_reference_missing_a_class_raises(tmp_path):
    make_split(tmp_path, ["n00000000"])
    with pytest.raises(ValueError, match="not in the reference"):
        label_map_for(FakeConfig(str(tmp_path)), FakeDataset(["n00000001"]))