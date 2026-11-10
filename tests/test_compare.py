"""Tests for comparing two dictionaries.

The measure is checked against cases whose answer is known by construction: a
dictionary against itself, a dictionary against its own shuffle, a dictionary
against an unrelated one, and one contained inside a larger one.

    python -m pytest tests/test_compare.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stitch.compare import (                                          # noqa: E402
    best_match, compare_runs, group_of, latest_checkpoint, load_dictionary,
    load_sides, random_baseline, same_configuration, similarity, summarise,
    write_comparison,
)


def unit(width: int, features: int, seed: int = 0) -> torch.Tensor:
    """A dictionary of random unit directions."""
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(width, features, generator=generator)
    return weight / weight.norm(dim=0, keepdim=True)


def write_run(directory: Path, weight: torch.Tensor, step: int = 100,
              silence: torch.Tensor | None = None,
              encoder: torch.Tensor | None = None) -> Path:
    """A run directory holding one checkpoint, as training would leave it.

    Args:
        weight: the decoder, ``(width out, features)``, one column per feature.
        encoder: the encoder, ``(features, width in)``, one ROW per feature,
            which is how the layer stores it. Left out, it is the decoder
            transposed, which is what initialisation gives a coder
            reconstructing its own input, so a run written without saying
            otherwise has two sides that agree.
    """
    checkpoints = directory / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    if silence is None:
        silence = torch.zeros(weight.shape[1], dtype=torch.long)
    if encoder is None:
        encoder = weight.T.clone()
    torch.save({"step": step, "tokens": step * 4096,
                "coder": {"decoder.weight": weight, "encoder.weight": encoder},
                "tracker": {"steps_since_fired": silence}},
               checkpoints / f"step_{step:09d}.pt")
    return directory


# ----------------------------------------------------------------------
# the measure
# ----------------------------------------------------------------------

def test_a_dictionary_matches_itself_perfectly():
    a = unit(16, 32)
    assert best_match(a, a).min() == pytest.approx(1.0, abs=1e-5)


def test_order_does_not_matter():
    """Nothing fixes which column a feature occupies."""
    a = unit(16, 32)
    shuffled = a[:, torch.randperm(32, generator=torch.Generator().manual_seed(1))]
    assert similarity(a, shuffled)["mean"] == pytest.approx(1.0, abs=1e-5)


def test_sign_is_not_ignored():
    """Features are non-negative, so a flipped direction is a different feature."""
    a = unit(16, 32)
    assert similarity(a, -a)["mean"] < 0.5


def test_unrelated_dictionaries_score_low():
    a, b = unit(64, 128, seed=0), unit(64, 128, seed=1)
    assert similarity(a, b)["mean"] < 0.6


def test_the_two_directions_differ_when_one_contains_the_other():
    """Half a dictionary is fully contained, but contains only half."""
    a = unit(32, 64)
    half = a[:, :32]
    scores = similarity(a, half)
    assert scores["mean_b_to_a"] == pytest.approx(1.0, abs=1e-5)
    assert scores["mean_a_to_b"] < 0.9


def test_matched_fraction_counts_features_above_the_threshold():
    a = unit(32, 64)
    mixed = torch.cat([a[:, :16], unit(32, 48, seed=5)], dim=1)
    assert similarity(a, mixed, close=0.99)["matched_b_to_a"] == pytest.approx(0.25, abs=0.1)


def test_mismatched_widths_raise():
    with pytest.raises(ValueError, match="different widths"):
        best_match(unit(16, 8), unit(32, 8))


def test_an_empty_dictionary_raises():
    with pytest.raises(ValueError, match="empty"):
        best_match(unit(16, 8), torch.zeros(16, 0))


def test_chunking_does_not_change_the_answer():
    a, b = unit(32, 70, seed=0), unit(32, 70, seed=1)
    assert torch.allclose(best_match(a, b, chunk=7), best_match(a, b, chunk=1000))


# ----------------------------------------------------------------------
# the one-to-one pairing
#
# Taking each feature's closest counterpart lets several features name the same
# counterpart. The pairing forbids that, so the two readings differ exactly
# when the best matches collide, and the gap between them is how much of the
# best-match score came from crowding.
# ----------------------------------------------------------------------

def test_a_dictionary_pairs_with_itself_perfectly():
    from stitch.compare import assigned_similarity
    a = unit(16, 32)
    assert assigned_similarity(a, a)["assigned_mean"] == pytest.approx(1.0, abs=1e-5)


def test_the_pairing_recovers_a_shuffle():
    """Nothing fixes which column a feature occupies, so the pairing has to
    find the permutation rather than be given it."""
    from stitch.compare import assignment
    a = unit(16, 32)
    order = torch.randperm(32, generator=torch.Generator().manual_seed(1))
    rows, columns, scores = assignment(a, a[:, order])

    assert scores.min() == pytest.approx(1.0, abs=1e-5)
    assert torch.equal(order[columns], rows)


def test_every_counterpart_is_used_once():
    from stitch.compare import assignment
    _, columns, _ = assignment(unit(32, 40, seed=0), unit(32, 40, seed=1))
    assert len(set(columns.tolist())) == 40


def test_the_pairing_is_symmetric():
    """No forward and backward, unlike the best match."""
    from stitch.compare import assigned_similarity
    a, b = unit(32, 48, seed=0), unit(32, 48, seed=1)
    assert (assigned_similarity(a, b)["assigned_mean"]
            == pytest.approx(assigned_similarity(b, a)["assigned_mean"], abs=1e-5))


def test_the_pairing_never_scores_above_the_best_match():
    """Each feature's best counterpart is an upper bound on the one it is
    assigned, so the pairing can only be lower."""
    from stitch.compare import assigned_similarity
    a, b = unit(32, 64, seed=0), unit(32, 64, seed=1)
    assert assigned_similarity(a, b)["assigned_mean"] <= similarity(a, b)["mean_a_to_b"]


def test_crowding_separates_the_two_readings():
    """The case the pairing exists for: a dictionary whose features all point
    at one direction scores well on best match and badly when paired."""
    from stitch.compare import assigned_similarity
    target = unit(32, 64, seed=0)
    crowded = target[:, :1].repeat(1, 64)          # every feature the same one

    assert similarity(crowded, target)["mean_a_to_b"] == pytest.approx(1.0, abs=1e-5)
    assert assigned_similarity(crowded, target)["assigned_mean"] < 0.5


def test_the_pairing_is_as_long_as_the_smaller_dictionary():
    from stitch.compare import assigned_similarity
    scores = assigned_similarity(unit(32, 64, seed=0), unit(32, 20, seed=1))
    assert scores["assigned_pairs"] == 20


def test_mismatched_widths_raise_for_the_pairing():
    from stitch.compare import assignment
    with pytest.raises(ValueError, match="different widths"):
        assignment(unit(16, 8), unit(32, 8))


def test_an_empty_dictionary_raises_for_the_pairing():
    from stitch.compare import assignment
    with pytest.raises(ValueError, match="empty"):
        assignment(unit(16, 8), torch.zeros(16, 0))


# ----------------------------------------------------------------------
# firing frequency
#
# Features that fire often are easier to match than features that fire rarely,
# so two dictionaries that use their features with different evenness differ in
# matched score without differing in what they contain. The bands are what
# tells those two apart.
# ----------------------------------------------------------------------

def write_frequencies(directory: Path, values: torch.Tensor) -> Path:
    """The firing rates, as the evaluation leaves them."""
    import json
    artifacts = directory / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "feature_frequencies.json").write_text(
        json.dumps({"frequencies": values.tolist()}))
    return directory


def test_frequencies_are_read_from_the_run(tmp_path):
    from stitch.compare import load_frequencies
    run = write_run(tmp_path / "run", unit(8, 6))
    write_frequencies(run, torch.arange(6, dtype=torch.float32))
    assert torch.equal(load_frequencies(run), torch.arange(6, dtype=torch.float32))


def test_frequencies_are_read_from_a_checkpoint_path(tmp_path):
    """compare_runs may be given either, so both have to resolve to the run."""
    from stitch.compare import latest_checkpoint, load_frequencies
    run = write_run(tmp_path / "run", unit(8, 6))
    write_frequencies(run, torch.ones(6))
    assert load_frequencies(latest_checkpoint(run)) is not None


def test_a_run_without_frequencies_gives_none(tmp_path):
    from stitch.compare import load_frequencies
    run = write_run(tmp_path / "run", unit(8, 6))
    assert load_frequencies(run) is None


def test_frequencies_are_cut_down_to_the_live_features(tmp_path):
    """They are recorded over the whole dictionary and the comparison drops
    the dead ones, so they would not line up otherwise."""
    from stitch.compare import load_dictionary, load_frequencies
    silence = torch.zeros(6, dtype=torch.long)
    silence[:2] = 100
    run = write_run(tmp_path / "run", unit(8, 6), step=100, silence=silence)
    write_frequencies(run, torch.arange(6, dtype=torch.float32))

    _, info = load_dictionary(run)
    assert torch.equal(load_frequencies(run, info["alive"]),
                       torch.tensor([2.0, 3.0, 4.0, 5.0]))


def test_bands_split_at_quantiles_so_none_is_empty():
    from stitch.compare import by_frequency
    scores = torch.ones(40)
    # Nearly all of them share one rate, which fixed-width bands would put in
    # a single band and leave the rest empty.
    frequencies = torch.cat([torch.full((38,), 0.001), torch.tensor([0.5, 0.9])])
    assert len(by_frequency(scores, frequencies, bands=4)) == 4


def test_bands_run_from_rarest_to_most_frequent():
    from stitch.compare import by_frequency
    frequencies = torch.arange(40, dtype=torch.float32)
    bands = by_frequency(frequencies, frequencies, bands=4)
    assert bands == sorted(bands)


def test_bands_are_absent_without_frequencies():
    from stitch.compare import by_frequency
    assert by_frequency(torch.ones(8), None) is None


def test_bands_are_absent_when_the_counts_disagree():
    """A frequency file from a different dictionary size would otherwise be
    silently zipped against the wrong features."""
    from stitch.compare import by_frequency
    assert by_frequency(torch.ones(8), torch.ones(12)) is None


def test_a_comparison_carries_the_pairing_and_the_bands(tmp_path):
    a = write_run(tmp_path / "case1-main-x16-topk-s0-aaaaaa", unit(16, 32, seed=0))
    b = write_run(tmp_path / "case1-main-x16-topk-s1-bbbbbb", unit(16, 32, seed=1))
    write_frequencies(a, torch.rand(32))

    record = compare_runs(a, b)
    assert record["assigned_mean"] <= record["mean_a_to_b"]
    assert record["encoder_assigned_pairs"] == 32
    assert len(record["assigned_by_frequency"]) == 4
    assert len(record["encoder_assigned_by_frequency"]) == 4


def test_the_pairing_can_be_skipped(tmp_path):
    a = write_run(tmp_path / "case1-main-x16-topk-s0-aaaaaa", unit(16, 32, seed=0))
    b = write_run(tmp_path / "case1-main-x16-topk-s1-bbbbbb", unit(16, 32, seed=1))
    record = compare_runs(a, b, assign=False)
    assert "assigned_mean" not in record and "mean" in record


def test_a_comparison_without_frequencies_still_pairs(tmp_path):
    a = write_run(tmp_path / "case1-main-x16-topk-s0-aaaaaa", unit(16, 32, seed=0))
    b = write_run(tmp_path / "case1-main-x16-topk-s1-bbbbbb", unit(16, 32, seed=1))
    record = compare_runs(a, b)
    assert "assigned_mean" in record and "assigned_by_frequency" not in record


def test_the_random_floor_can_be_measured_under_the_pairing():
    """The one-to-one reading needs its own floor: a pairing cannot reuse a
    counterpart, so random dictionaries score lower under it."""
    scores = random_baseline(32, 64, trials=1, assign=True)
    assert scores["assigned_mean"] < scores["mean"]


def test_the_random_floor_leaves_the_pairing_out_by_default():
    assert "assigned_mean" not in random_baseline(32, 64, trials=1)


def test_the_pairing_distributions_are_not_written_out(tmp_path):
    import json
    a = write_run(tmp_path / "case1-main-x16-topk-s0-aaaaaa", unit(16, 32, seed=0))
    b = write_run(tmp_path / "case1-main-x16-topk-s1-bbbbbb", unit(16, 32, seed=1))
    records = [compare_runs(a, b, keep_distribution=True)]

    out = tmp_path / "comparison.json"
    write_comparison(records, summarise(records), out)
    written = json.loads(out.read_text())["pairs"][0]
    assert "assigned_histogram" not in written
    assert "encoder_assigned_histogram" not in written
    assert "assigned_mean" in written


def test_the_summary_can_be_read_off_the_pairing():
    records = [
        {"a": "case1-main-x16-l1-s0-a", "b": "case1-main-x16-l1-s1-b",
         "assigned_mean": 0.80},
        {"a": "case1-main-x16-l1-s0-a", "b": "case2-main-x16-l1-s0-c",
         "assigned_mean": 0.40},
    ]
    summary = summarise(records, baseline={"mean": 0.20}, key="assigned_mean",
                        shape_key="assigned_histogram")
    assert summary["within"]["mean"] == pytest.approx(0.80)
    assert summary["across"]["mean"] == pytest.approx(0.40)
    assert summary["position"] == pytest.approx(1.0 / 3.0)


# ----------------------------------------------------------------------
# the random reference
# ----------------------------------------------------------------------

def test_random_dictionaries_do_not_score_zero():
    """The closest of many random directions is not close, but it is not zero."""
    assert 0.0 < random_baseline(64, 128, trials=1)["mean"] < 0.7


def test_a_bigger_dictionary_matches_random_better():
    """More directions means the nearest one is nearer, by chance alone.

    Which is why the floor has to be measured at the number of features the
    comparison actually used. Dead features are dropped before two coders are
    compared, so a floor taken over the configured dictionary belongs to a
    larger problem than the one being solved.
    """
    small = random_baseline(64, 64, trials=1)["mean"]
    large = random_baseline(64, 512, trials=1)["mean"]
    assert large > small


def test_the_baseline_is_reproducible():
    assert (random_baseline(32, 64, trials=1, seed=3)
            == random_baseline(32, 64, trials=1, seed=3))


# ----------------------------------------------------------------------
# reading runs
# ----------------------------------------------------------------------

def test_the_newest_checkpoint_is_the_one_read(tmp_path):
    run = write_run(tmp_path / "run", unit(8, 4), step=100)
    write_run(run, unit(8, 4), step=2000)
    assert latest_checkpoint(run).name == "step_000002000.pt"


def test_checkpoints_sort_by_number_not_by_name(tmp_path):
    """step_9 is newer than step_10 only if the names are compared as text."""
    run = write_run(tmp_path / "run", unit(8, 4), step=9)
    write_run(run, unit(8, 4), step=10)
    assert latest_checkpoint(run).name == "step_000000010.pt"


def test_a_run_without_checkpoints_raises(tmp_path):
    (tmp_path / "empty" / "checkpoints").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="no checkpoint"):
        latest_checkpoint(tmp_path / "empty")


def test_columns_come_back_at_unit_length(tmp_path):
    run = write_run(tmp_path / "run", unit(8, 6) * 3.0)
    directions, _ = load_dictionary(run)
    assert torch.allclose(directions.norm(dim=0), torch.ones(6), atol=1e-5)


def test_dead_features_are_dropped(tmp_path):
    silence = torch.zeros(6, dtype=torch.long)
    silence[:2] = 100                       # never fired in a run of 100 steps
    run = write_run(tmp_path / "run", unit(8, 6), step=100, silence=silence)

    _, info = load_dictionary(run)
    assert (info["kept"], info["dropped"]) == (4, 2)


def test_dead_features_are_kept_when_asked(tmp_path):
    silence = torch.full((6,), 100, dtype=torch.long)
    run = write_run(tmp_path / "run", unit(8, 6), step=100, silence=silence)
    _, info = load_dictionary(run, drop_dead=False)
    assert info["kept"] == 6


def test_a_fully_dead_dictionary_is_kept_rather_than_emptied(tmp_path):
    """Nothing can be compared against nothing, so it is loaded and reported."""
    silence = torch.full((6,), 100, dtype=torch.long)
    run = write_run(tmp_path / "run", unit(8, 6), step=100, silence=silence)
    directions, info = load_dictionary(run, drop_dead=True)
    assert directions.shape[1] == 6 and info["dropped"] == 0


def test_two_runs_compare(tmp_path):
    a = write_run(tmp_path / "case1-main-x16-l1-s0-aaaaaa", unit(16, 32, seed=0))
    b = write_run(tmp_path / "case1-main-x16-l1-s1-bbbbbb", unit(16, 32, seed=1))
    record = compare_runs(a, b)
    assert record["width"] == 16 and 0.0 < record["mean"] < 1.0


# ----------------------------------------------------------------------
# the two sides of a coder
#
# The decoder holds what a feature writes out, the encoder what it looks for.
# Across the two objectives only the encoder is comparable, because the two
# decoders write into two independently trained models whose coordinates were
# never aligned. The cosine between them does not fail, it returns a number
# with no content, which is the reason these exist.
# ----------------------------------------------------------------------

def test_the_encoder_is_read_as_one_direction_per_feature(tmp_path):
    """Stored one row per feature, returned as unit columns like the decoder."""
    decoder = unit(16, 32, seed=0)
    encoder = torch.randn(32, 16)                      # (features, width in)
    run = write_run(tmp_path / "r", decoder, encoder=encoder)

    directions, _ = load_dictionary(run, side="encoder")
    assert directions.shape == (16, 32)
    assert torch.allclose(directions.norm(dim=0), torch.ones(32), atol=1e-6)


def test_the_two_sides_are_different_readings_of_one_run(tmp_path):
    decoder = unit(16, 32, seed=0)
    encoder = torch.randn(32, 16)
    run = write_run(tmp_path / "r", decoder, encoder=encoder)

    assert not torch.allclose(load_dictionary(run, side="decoder")[0],
                              load_dictionary(run, side="encoder")[0])


def test_an_unknown_side_raises(tmp_path):
    run = write_run(tmp_path / "r", unit(16, 32))
    with pytest.raises(ValueError, match="side must be"):
        load_dictionary(run, side="latents")


def test_both_sides_come_back_from_one_read(tmp_path):
    run = write_run(tmp_path / "r", unit(16, 32, seed=0), encoder=torch.randn(32, 16))
    sides = load_sides(run)
    assert set(sides) == {"decoder", "encoder"}
    assert sides["decoder"][0].shape == sides["encoder"][0].shape


def test_the_same_dead_features_are_dropped_from_both_sides(tmp_path):
    """They are one dictionary. A feature dead on one side is dead on both."""
    silence = torch.zeros(32, dtype=torch.long)
    silence[:12] = 100                                  # never fired
    run = write_run(tmp_path / "r", unit(16, 32), silence=silence,
                    encoder=torch.randn(32, 16))

    sides = load_sides(run)
    assert sides["decoder"][1]["kept"] == 20
    assert sides["encoder"][1]["kept"] == 20
    assert sides["decoder"][0].shape[1] == sides["encoder"][0].shape[1]


def test_a_comparison_carries_both_readings(tmp_path):
    a = write_run(tmp_path / "case1-main-x16-topk-s0-aaaaaa", unit(16, 32, seed=0))
    b = write_run(tmp_path / "case1-main-x16-topk-s1-bbbbbb", unit(16, 32, seed=1))
    record = compare_runs(a, b)
    assert 0.0 < record["mean"] < 1.0
    assert 0.0 < record["encoder_mean"] < 1.0
    assert record["width"] == record["encoder_width"] == 16


def test_the_encoder_sees_agreement_the_decoder_cannot(tmp_path):
    """The case the reading was added for, in its purest form.

    Two coders that look for exactly the same directions in the activation
    they both read, and write out into unrelated spaces. That is the shape of
    a single-model coder against a cross-model one: same input, different
    output space. The decoder reading calls them strangers, which is not a
    finding about what they learned, and the encoder reading sees the identity
    that is actually there.
    """
    shared = torch.randn(32, 16)                        # one encoder, two runs
    a = write_run(tmp_path / "case1-main-x16-topk-s0-aaaaaa",
                  unit(16, 32, seed=0), encoder=shared)
    b = write_run(tmp_path / "case2-main-x16-topk-s0-bbbbbb",
                  unit(16, 32, seed=1), encoder=shared.clone())

    record = compare_runs(a, b)
    assert record["encoder_mean"] == pytest.approx(1.0, abs=1e-5)
    assert record["mean"] < 0.9


def test_the_summary_can_be_read_in_either_measure(tmp_path):
    """A floor, a ceiling and an across number only sit on one scale if all
    three are the same reading, so the summary takes which one."""
    shared = torch.randn(32, 16)
    records = [
        compare_runs(
            write_run(tmp_path / "case1-main-x16-topk-s0-aaaaaa",
                      unit(16, 32, seed=0), encoder=shared),
            write_run(tmp_path / "case2-main-x16-topk-s0-bbbbbb",
                      unit(16, 32, seed=1), encoder=shared.clone())),
    ]
    decoder = summarise(records)
    encoder = summarise(records, key="encoder_mean", shape_key="encoder_histogram")

    assert decoder["across"]["mean"] == records[0]["mean"]
    assert encoder["across"]["mean"] == records[0]["encoder_mean"]
    assert encoder["across"]["mean"] > decoder["across"]["mean"]


def test_the_distributions_do_not_cross_between_readings(tmp_path):
    """Each reading pools its own histogram, so a figure drawn from one is
    not quietly averaging the other in."""
    a = write_run(tmp_path / "case1-main-x16-topk-s0-aaaaaa",
                  unit(16, 32, seed=0), encoder=torch.randn(32, 16))
    b = write_run(tmp_path / "case1-main-x16-topk-s1-bbbbbb",
                  unit(16, 32, seed=1), encoder=torch.randn(32, 16))
    records = [compare_runs(a, b, keep_distribution=True)]

    decoder = summarise(records)
    encoder = summarise(records, key="encoder_mean", shape_key="encoder_histogram")
    assert decoder["within_shape"] == records[0]["histogram"]
    assert encoder["within_shape"] == records[0]["encoder_histogram"]


def test_neither_distribution_is_written_out(tmp_path):
    """Both are two hundred numbers per pair and the summary holds their
    averages, which is what anything downstream reads."""
    import json
    a = write_run(tmp_path / "case1-main-x16-topk-s0-aaaaaa", unit(16, 32, seed=0))
    b = write_run(tmp_path / "case1-main-x16-topk-s1-bbbbbb", unit(16, 32, seed=1))
    records = [compare_runs(a, b, keep_distribution=True)]

    out = tmp_path / "comparison.json"
    write_comparison(records, summarise(records), out)
    written = json.loads(out.read_text())["pairs"][0]
    assert "histogram" not in written and "encoder_histogram" not in written
    assert "encoder_mean" in written


# ----------------------------------------------------------------------
# grouping and the summary
# ----------------------------------------------------------------------

def test_runs_differing_only_by_seed_are_one_group():
    assert (group_of("case1-main-x16-l1-s0-abc123")
            == group_of("case1-main-x16-l1-s4-def456"))


def test_runs_of_different_objectives_are_different_groups():
    assert (group_of("case1-main-x16-l1-s0-abc123")
            != group_of("case2-main-x16-l1-s0-abc123"))


def test_a_tag_beginning_with_the_seed_letter_survives():
    """The seed is dropped by matching the letter and digits, not the letter
    alone, or a tag like 'smoke' would vanish and merge two configurations."""
    assert (group_of("case1-smoke-x16-l1-s0-abc123")
            != group_of("case1-stable-x16-l1-s0-abc123"))


# ----------------------------------------------------------------------
# grouping from what a run recorded, not from its name
#
# A run's directory name carries the objective, the tag, the expansion, the
# sparsity mode, the seed and a hash. It does not carry the depth. In the main
# grid the depths are separate configurations differing only in which layers
# they read, so two depths share a name shape, and a ceiling measured over them
# would be measuring depth rather than seed.
# ----------------------------------------------------------------------

def write_params(directory: Path, layers: list[str], seed: int, **extra) -> Path:
    """A run's resolved settings, as the logger writes them."""
    import json
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "params.json").write_text(json.dumps(
        {"objective": "case1", "activations.layers": layers,
         "coder.expansion": 32, "train.seed": seed, **extra}))
    return directory


def test_two_seeds_at_one_depth_are_one_group(tmp_path):
    a = write_params(tmp_path / "case1-main-x32-topk-s0-aaa", ["A@add_4"], 0)
    b = write_params(tmp_path / "case1-main-x32-topk-s1-bbb", ["A@add_4"], 1)
    assert group_of(a) == group_of(b)


def test_two_depths_are_not_a_seed_repeat(tmp_path):
    """The defect this exists to prevent: the names are indistinguishable."""
    from stitch.compare import _group_from_name

    shallow = write_params(tmp_path / "case1-main-x32-topk-s0-aaa", ["A@add_4"], 0)
    deep = write_params(tmp_path / "case1-main-x32-topk-s0-ccc", ["A@add_12"], 0)

    assert group_of(shallow) != group_of(deep)
    assert _group_from_name(shallow.name) == _group_from_name(deep.name)


def test_two_sparsity_settings_are_not_a_seed_repeat(tmp_path):
    """The name carries the mode but not the value, so these look alike too."""
    loose = write_params(tmp_path / "case1-main-x32-topk-s0-aaa", ["A@add_4"], 0,
                         **{"coder.sparsity.k_frac": 0.153})
    tight = write_params(tmp_path / "case1-main-x32-topk-s0-bbb", ["A@add_4"], 0,
                         **{"coder.sparsity.k_frac": 0.04})
    assert group_of(loose) != group_of(tight)


def test_a_run_without_parameters_falls_back_to_its_name(tmp_path):
    run = tmp_path / "case1-main-x32-topk-s0-aaa"
    run.mkdir()
    assert group_of(run) == group_of("case1-main-x32-topk-s3-bbb")


def test_a_comparison_records_which_configuration_each_side_was(tmp_path):
    """Resolved where both directories are in hand, so nothing downstream has
    to go back to the names."""
    a = write_run(tmp_path / "case1-main-x32-topk-s0-aaa", unit(16, 32, seed=0))
    b = write_run(tmp_path / "case1-main-x32-topk-s1-bbb", unit(16, 32, seed=1))
    write_params(a, ["A@add_4"], 0)
    write_params(b, ["A@add_4"], 1)

    record = compare_runs(a, b)
    assert record["a_group"] == record["b_group"]
    assert same_configuration(record)


def test_the_recorded_groups_win_over_the_names():
    """Two runs at different depths, whose names say they are a seed repeat."""
    record = {"a": "case1-main-x32-topk-s0-aaa", "b": "case1-main-x32-topk-s1-bbb",
              "a_group": "shallow", "b_group": "deep", "mean": 0.5}
    assert not same_configuration(record)


def test_a_record_without_groups_still_reads():
    """Comparisons written before the groups were carried."""
    assert same_configuration({"a": "case1-x16-l1-s0-a", "b": "case1-x16-l1-s1-b"})


def test_the_summary_uses_the_recorded_groups():
    records = [
        {"a": "case1-main-x32-topk-s0-a", "b": "case1-main-x32-topk-s1-b",
         "a_group": "shallow", "b_group": "deep", "mean": 0.90},
        {"a": "case1-main-x32-topk-s0-a", "b": "case1-main-x32-topk-s2-c",
         "a_group": "shallow", "b_group": "shallow", "mean": 0.50},
    ]
    summary = summarise(records)
    # By name both pairs are seed repeats. By what the runs recorded, the
    # first is a comparison across depths and the second is the repeat.
    assert summary["within"]["mean"] == pytest.approx(0.50)
    assert summary["across"]["mean"] == pytest.approx(0.90)


def test_the_summary_separates_seed_repeats_from_the_comparison():
    records = [
        {"a": "case1-main-x16-l1-s0-a", "b": "case1-main-x16-l1-s1-b", "mean": 0.90},
        {"a": "case1-main-x16-l1-s0-a", "b": "case2-main-x16-l1-s0-c", "mean": 0.50},
    ]
    summary = summarise(records)
    assert summary["within"]["mean"] == pytest.approx(0.90)
    assert summary["across"]["mean"] == pytest.approx(0.50)


def test_position_reports_where_across_falls_between_the_references():
    records = [
        {"a": "case1-main-x16-l1-s0-a", "b": "case1-main-x16-l1-s1-b", "mean": 0.90},
        {"a": "case1-main-x16-l1-s0-a", "b": "case2-main-x16-l1-s0-c", "mean": 0.50},
    ]
    summary = summarise(records, baseline={"mean": 0.10})
    assert summary["position"] == pytest.approx(0.5)


def test_an_objective_that_changed_nothing_sits_at_the_ceiling():
    records = [
        {"a": "case1-main-x16-l1-s0-a", "b": "case1-main-x16-l1-s1-b", "mean": 0.80},
        {"a": "case1-main-x16-l1-s0-a", "b": "case2-main-x16-l1-s0-c", "mean": 0.80},
    ]
    assert summarise(records, baseline={"mean": 0.20})["position"] == pytest.approx(1.0)


def test_a_summary_without_pairs_does_not_raise():
    assert summarise([])["within"] == {"n": 0}


def test_the_comparison_is_written(tmp_path):
    records = [{"a": "x-s0-a", "b": "x-s1-b", "mean": 0.5}]
    path = tmp_path / "out" / "comparison.json"
    write_comparison(records, summarise(records), path)
    assert path.exists() and "pairs" in path.read_text()


# ----------------------------------------------------------------------
# the distribution the figure draws
# ----------------------------------------------------------------------

def test_a_histogram_sums_to_one():
    from stitch.compare import histogram
    assert sum(histogram(torch.rand(500))) == pytest.approx(1.0, abs=1e-5)


def test_an_empty_histogram_does_not_divide_by_zero():
    from stitch.compare import histogram
    assert sum(histogram(torch.zeros(0))) == 0.0


def test_bin_centres_span_the_cosine_range():
    from stitch.compare import bin_centres
    centres = bin_centres(4)
    assert centres[0] == pytest.approx(-0.75) and centres[-1] == pytest.approx(0.75)


def test_a_perfect_match_lands_in_the_top_bin():
    from stitch.compare import BINS, similarity
    a = unit(16, 32)
    shape = similarity(a, a, keep_distribution=True)["histogram"]
    assert shape[-1] == pytest.approx(1.0, abs=1e-5)
    assert sum(shape[:BINS - 1]) == pytest.approx(0.0, abs=1e-5)


def test_the_distribution_is_absent_unless_asked_for():
    assert "histogram" not in similarity(unit(16, 8), unit(16, 8))


def test_pooled_shapes_average_within_and_across_separately():
    from stitch.compare import pooled_shapes
    records = [
        {"a": "case1-x-s0-a", "b": "case1-x-s1-b", "histogram": [1.0, 0.0]},
        {"a": "case1-x-s0-a", "b": "case2-x-s0-c", "histogram": [0.0, 1.0]},
        {"a": "case1-x-s1-b", "b": "case2-x-s0-c", "histogram": [0.0, 1.0]},
    ]
    shapes = pooled_shapes(records)
    assert shapes["within_shape"] == [1.0, 0.0]
    assert shapes["across_shape"] == [0.0, 1.0]


def test_pooled_shapes_are_absent_without_distributions():
    from stitch.compare import pooled_shapes
    assert pooled_shapes([{"a": "x-s0-a", "b": "x-s1-b", "mean": 0.5}]) == {}


def test_the_written_file_leaves_out_the_distributions(tmp_path):
    records = [{"a": "x-s0-a", "b": "x-s1-b", "mean": 0.5, "histogram": [0.5, 0.5]}]
    path = tmp_path / "comparison.json"
    write_comparison(records, {}, path)
    assert "histogram" not in path.read_text()


def test_the_random_baseline_can_carry_its_distribution():
    from stitch.compare import random_baseline as baseline
    shape = baseline(16, 32, trials=2, keep_distribution=True)["histogram"]
    assert sum(shape) == pytest.approx(1.0, abs=1e-5)


# ----------------------------------------------------------------------
# picking out one grid's runs
# ----------------------------------------------------------------------

def write_grid(path: Path, names: list[str]) -> Path:
    """A grid result file, shaped the way the grid writes one."""
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        [{"point": {"train.seed": index}, "run_name": name}
         for index, name in enumerate(names)]))
    return path


def test_runs_come_from_the_grid_that_produced_them(tmp_path):
    """A name pattern cannot do this: the hash is the only thing that differs."""
    from stitch.compare import from_grids
    names = ["case1-control-x16-l1-s0-aaa111", "case1-control-x16-l1-s1-bbb222"]
    for name in names:
        write_run(tmp_path / "runs" / name, unit(8, 4))
    write_run(tmp_path / "runs" / "case1-control-x16-l1-s0-ccc333", unit(8, 4))

    grid = write_grid(tmp_path / "grids" / "noise.json", names)
    found = from_grids([grid], tmp_path / "runs")
    assert [path.name for path in found] == names


def test_several_grids_combine_in_order(tmp_path):
    from stitch.compare import from_grids
    for name in ("case1-main-x16-l1-s0-a", "case2-main-x16-l1-s0-b"):
        write_run(tmp_path / "runs" / name, unit(8, 4))
    one = write_grid(tmp_path / "g1.json", ["case1-main-x16-l1-s0-a"])
    two = write_grid(tmp_path / "g2.json", ["case2-main-x16-l1-s0-b"])
    assert len(from_grids([one, two], tmp_path / "runs")) == 2


def test_a_run_named_but_missing_is_skipped(tmp_path):
    from stitch.compare import from_grids
    write_run(tmp_path / "runs" / "case1-main-x16-l1-s0-a", unit(8, 4))
    grid = write_grid(tmp_path / "g.json",
                      ["case1-main-x16-l1-s0-a", "case1-main-x16-l1-s1-gone"])
    assert len(from_grids([grid], tmp_path / "runs")) == 1


def test_a_grid_naming_nothing_on_disk_raises(tmp_path):
    from stitch.compare import from_grids
    grid = write_grid(tmp_path / "g.json", ["case1-main-x16-l1-s0-gone"])
    with pytest.raises(SystemExit, match="no runs found"):
        from_grids([grid], tmp_path / "runs")


def test_a_failed_point_without_a_run_name_is_skipped(tmp_path):
    """A point that raised is recorded with its error and no run."""
    import json
    from stitch.compare import from_grids
    write_run(tmp_path / "runs" / "case1-main-x16-l1-s0-a", unit(8, 4))
    grid = tmp_path / "g.json"
    grid.write_text(json.dumps([
        {"point": {"train.seed": 0}, "run_name": "case1-main-x16-l1-s0-a"},
        {"point": {"train.seed": 1}, "error": "RuntimeError: out of memory"},
    ]))
    assert len(from_grids([grid], tmp_path / "runs")) == 1

# ----------------------------------------------------------------------
# what the figures need
#
# The numbers average over every pair. A figure that draws features one at a
# time cannot: feature seventeen of one run is not feature seventeen of
# another, so there is nothing to average. These produce the per-feature
# material for one pair, and the shapes the rest of the figures read.
# ----------------------------------------------------------------------

def test_ranks_order_by_firing_rate_most_active_first():
    from stitch.compare import _ranks
    places = _ranks(torch.tensor([0.1, 0.9, 0.5]), 3)
    assert places.tolist() == [1.0, 0.0, 0.5]


def test_ranks_fall_back_to_index_order_without_frequencies():
    """A run that died before its evaluation still draws, saying less."""
    from stitch.compare import _ranks
    assert _ranks(None, 3).tolist() == [0.0, 0.5, 1.0]


def test_ranks_fall_back_when_the_counts_disagree():
    from stitch.compare import _ranks
    assert _ranks(torch.tensor([0.1, 0.2]), 3).tolist() == [0.0, 0.5, 1.0]


def test_ranks_of_one_feature_do_not_divide_by_zero():
    from stitch.compare import _ranks
    assert _ranks(torch.tensor([0.5]), 1).tolist() == [0.0]


def test_detail_returns_one_score_per_feature(tmp_path):
    from stitch.compare import detail
    a = write_run(tmp_path / "a", unit(16, 24, seed=0))
    b = write_run(tmp_path / "b", unit(16, 24, seed=1))
    out = detail(a, b, side="decoder")
    assert out["features"] == 24
    assert len(out["squares"]) == 24


def test_detail_scores_a_dictionary_against_itself_at_one(tmp_path):
    from stitch.compare import detail
    weight = unit(16, 20, seed=0)
    a = write_run(tmp_path / "a", weight)
    b = write_run(tmp_path / "b", weight.clone())
    out = detail(a, b, side="decoder")
    assert min(out["squares"]) == pytest.approx(1.0, abs=1e-5)


def test_detail_puts_the_pairing_on_the_diagonal(tmp_path):
    """What the matrix figure relies on: the block is ordered so that a pair
    sits at the same row and column, and a shuffle of one dictionary must not
    change that."""
    from stitch.compare import detail
    weight = unit(16, 20, seed=0)
    order = torch.randperm(20, generator=torch.Generator().manual_seed(3))
    a = write_run(tmp_path / "a", weight)
    b = write_run(tmp_path / "b", weight[:, order].clone())

    block = torch.tensor(detail(a, b, side="decoder", block=20)["block"])
    assert torch.allclose(block.diagonal(), torch.ones(20), atol=1e-5)


def test_detail_caps_what_it_returns(tmp_path):
    from stitch.compare import detail
    a = write_run(tmp_path / "a", unit(16, 40, seed=0))
    b = write_run(tmp_path / "b", unit(16, 40, seed=1))
    out = detail(a, b, side="decoder", squares=10, ribbons=5, block=6)
    assert len(out["squares"]) == 10
    assert len(out["pairs"]) == 5
    assert len(out["block"]) == 6 and len(out["block"][0]) == 6


def test_detail_asks_for_more_than_there_is(tmp_path):
    """A small dictionary gives what it has rather than raising."""
    from stitch.compare import detail
    a = write_run(tmp_path / "a", unit(8, 6, seed=0))
    b = write_run(tmp_path / "b", unit(8, 6, seed=1))
    out = detail(a, b, side="decoder", squares=500, ribbons=500, block=500)
    assert len(out["squares"]) == 6
    assert len(out["pairs"]) == 6
    assert len(out["block"]) == 6


def test_detail_pairs_carry_positions_and_a_score(tmp_path):
    from stitch.compare import detail
    a = write_run(tmp_path / "a", unit(16, 20, seed=0))
    b = write_run(tmp_path / "b", unit(16, 20, seed=1))
    for left, right, score in detail(a, b, side="decoder")["pairs"]:
        assert 0.0 <= left <= 1.0 and 0.0 <= right <= 1.0
        assert -1.0 <= score <= 1.0


def test_detail_places_pairs_by_firing_rate(tmp_path):
    """The feature that fires most sits at the left-hand end, whatever slot it
    occupies, because the slot numbering is arbitrary and the figure would
    otherwise show a perfect pairing as a tangle."""
    from stitch.compare import detail
    weight = unit(16, 4, seed=0)
    a = write_frequencies(write_run(tmp_path / "a", weight),
                          torch.tensor([0.1, 0.9, 0.2, 0.3]))
    b = write_frequencies(write_run(tmp_path / "b", weight.clone()),
                          torch.tensor([0.1, 0.9, 0.2, 0.3]))

    pairs = detail(a, b, side="decoder")["pairs"]
    assert pairs[1][0] == pytest.approx(0.0)
    assert pairs[1][1] == pytest.approx(0.0)


def test_detail_reads_the_side_it_is_asked_for(tmp_path):
    """The encoder reading is the only one that means anything across the two
    objectives, so a figure drawn from the wrong side would be drawn from
    nothing."""
    from stitch.compare import detail
    decoder = unit(16, 12, seed=0)
    a = write_run(tmp_path / "a", decoder, encoder=unit(12, 16, seed=5))
    b = write_run(tmp_path / "b", decoder.clone(), encoder=unit(12, 16, seed=6))

    assert min(detail(a, b, side="decoder")["squares"]) == pytest.approx(1.0, abs=1e-5)
    assert max(detail(a, b, side="encoder")["squares"]) < 0.99


# ----------------------------------------------------------------------
# three shares instead of a shape
# ----------------------------------------------------------------------

def spike(centre: float, bins: int = 200) -> list[float]:
    """A distribution with all of its weight in one bin."""
    out = [0.0] * bins
    out[min(int((centre + 1.0) / 2.0 * bins), bins - 1)] = 1.0
    return out


def test_everything_above_the_threshold_is_strong():
    from stitch.compare import shares
    assert shares(spike(0.95), floor=0.2, close=0.9)["strong"] == pytest.approx(1.0)


def test_everything_below_the_floor_is_absent():
    from stitch.compare import shares
    assert shares(spike(0.05), floor=0.2, close=0.9)["absent"] == pytest.approx(1.0)


def test_between_the_two_is_partial():
    from stitch.compare import shares
    assert shares(spike(0.5), floor=0.2, close=0.9)["partial"] == pytest.approx(1.0)


def test_the_three_shares_account_for_everything():
    from stitch.compare import histogram, shares
    values = torch.linspace(-0.5, 1.0, 400)
    out = shares(histogram(values), floor=0.2, close=0.9)
    assert sum(out.values()) == pytest.approx(1.0)


def test_an_empty_distribution_raises():
    from stitch.compare import shares
    with pytest.raises(ValueError, match="empty distribution"):
        shares([], floor=0.2, close=0.9)


def test_a_floor_above_the_threshold_raises():
    """The bands would overlap and a feature would be counted twice."""
    from stitch.compare import shares
    with pytest.raises(ValueError, match="not below"):
        shares(spike(0.5), floor=0.95, close=0.9)


# ----------------------------------------------------------------------
# averaging the bands, and choosing what to draw
# ----------------------------------------------------------------------

def banded(a: str, b: str, values: list[float]) -> dict:
    """A record carrying per-band means and the groups that place it."""
    return {"a": a, "b": b, "a_group": a[:-1], "b_group": b[:-1],
            "encoder_assigned_by_frequency": values}


def test_each_kind_gets_its_own_curve():
    from stitch.compare import by_kind
    records = [banded("c1s0", "c1s1", [0.7, 0.9]),
               banded("c1s0", "c2s0", [0.3, 0.5])]
    out = by_kind(records, "encoder_assigned_by_frequency")
    assert out["within"] == [0.7, 0.9]
    assert out["across"] == [0.3, 0.5]


def test_the_pairs_of_a_kind_are_averaged():
    from stitch.compare import by_kind
    records = [banded("c1s0", "c1s1", [0.6, 0.8]),
               banded("c1s0", "c1s2", [0.8, 1.0])]
    assert by_kind(records, "encoder_assigned_by_frequency")["within"] == [0.7, 0.9]


def test_a_kind_with_no_pairs_is_left_out():
    from stitch.compare import by_kind
    out = by_kind([banded("c1s0", "c1s1", [0.7])], "encoder_assigned_by_frequency")
    assert "across" not in out


def test_records_without_the_bands_are_left_out():
    from stitch.compare import by_kind
    records = [banded("c1s0", "c1s1", [0.6]),
               {"a": "c1s0", "b": "c1s3", "a_group": "c1s", "b_group": "c1s"}]
    assert by_kind(records, "encoder_assigned_by_frequency")["within"] == [0.6]


def test_no_records_give_no_curves():
    from stitch.compare import by_kind
    assert by_kind([], "encoder_assigned_by_frequency") == {}


def test_one_pair_of_each_kind_is_chosen():
    from stitch.compare import representatives
    pairs = [(Path("c1s0"), Path("c1s1")), (Path("c1s0"), Path("c2s0"))]
    chosen = representatives(pairs, [banded("c1s0", "c1s1", [0.7]),
                                     banded("c1s0", "c2s0", [0.3])])
    assert chosen["within"] == (Path("c1s0"), Path("c1s1"))
    assert chosen["across"] == (Path("c1s0"), Path("c2s0"))


def test_the_first_pair_of_a_kind_is_the_one_chosen():
    """Run order rather than a judgement about which pair looks better."""
    from stitch.compare import representatives
    pairs = [(Path("c1s0"), Path("c1s1")), (Path("c1s0"), Path("c1s2"))]
    chosen = representatives(pairs, [banded("c1s0", "c1s1", [0.7]),
                                     banded("c1s0", "c1s2", [0.9])])
    assert chosen["within"] == (Path("c1s0"), Path("c1s1"))


def test_a_kind_that_was_never_compared_is_left_out():
    from stitch.compare import representatives
    chosen = representatives([(Path("c1s0"), Path("c1s1"))],
                             [banded("c1s0", "c1s1", [0.7])])
    assert "across" not in chosen


def test_nothing_compared_chooses_nothing():
    from stitch.compare import representatives
    assert representatives([], []) == {}