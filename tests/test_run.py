"""Tests for the run entry point.

Most of what this module does is connect other modules, which is tested where
those modules are. What is tested here is the part that belongs to the entry
point itself: that the command line reaches the configuration, that a run can
be described without being performed, and that a request for hardware that is
not present fails before any work starts.

The end to end test is skipped unless the dataset is present and the graph
library is installed. It is the only test in the suite that runs a real
backbone, and it still runs on CPU.

    python -m pytest tests/test_run.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")

from stitch.config import Config, ModelConfig, dump_config             # noqa: E402
from stitch.run import (                                               # noqa: E402
    build_gaussian_streams, describe, load_models, main, resolve_baseline,
    resolve_linear_fit,
)
from stitch.sweep import (                                             # noqa: E402
    CrossMoments, moments_path, save_moments, write_ranking,
)

DATA = Path("data/imagenet-100")

try:
    import nn_lib                                                      # noqa: F401
    HAVE_NN_LIB = True
except ImportError:
    HAVE_NN_LIB = False

needs_everything = pytest.mark.skipif(
    not (HAVE_NN_LIB and DATA.is_dir()),
    reason="needs the graph library and the dataset")


@pytest.fixture
def cfg():
    c = Config()
    c.dataset.root = str(DATA)
    c.train.device = "cpu"
    return c


@pytest.fixture
def config_file(cfg, tmp_path):
    path = tmp_path / "run.yaml"
    dump_config(cfg, path)
    return path


# ----------------------------------------------------------------------
# describing a run
# ----------------------------------------------------------------------

def test_describe_names_what_is_read_and_reconstructed(cfg):
    text = describe(cfg)
    assert cfg.activations.layers[0] in text
    assert "reads" in text and "reconstructs" in text


def test_describe_reports_the_step_count(cfg):
    cfg.train.tokens = 1000
    cfg.train.token_batch = 100
    assert "10 steps" in describe(cfg)


def test_describe_rounds_a_partial_step_up(cfg):
    """A budget that is not a multiple of the batch still needs the last step."""
    cfg.train.tokens = 1001
    cfg.train.token_batch = 100
    assert "11 steps" in describe(cfg)


def test_describe_carries_the_hash(cfg):
    assert cfg.hash() in describe(cfg)


def test_describe_lists_pending_decisions(cfg):
    text = describe(cfg)
    assert "pending decisions:" in text
    for note in cfg.pending_decisions():
        assert note in text


def test_describe_says_when_the_dataset_is_unset(cfg):
    cfg.dataset.root = ""
    assert "(unset)" in describe(cfg)


# ----------------------------------------------------------------------
# the command line
# ----------------------------------------------------------------------

def test_a_dry_run_does_nothing(config_file, capsys):
    assert main(["--config", str(config_file), "--dry-run"]) == 0
    assert "budget" in capsys.readouterr().out


def test_an_override_reaches_the_configuration(config_file, capsys):
    main(["--config", str(config_file), "--set", "train.tokens=4096",
          "--set", "train.token_batch=1024", "--dry-run"])
    assert "4 steps" in capsys.readouterr().out


def test_several_overrides_are_all_applied(config_file, capsys):
    main(["--config", str(config_file), "--set", "coder.expansion=32",
          "--set", "train.seed=7", "--dry-run"])
    out = capsys.readouterr().out
    assert "32x" in out


def test_a_typo_in_an_override_fails_loudly(config_file):
    """Silently having no effect is the worst outcome for a swept setting."""
    with pytest.raises(ValueError, match="unknown key"):
        main(["--config", str(config_file), "--set", "train.sed=1", "--dry-run"])


def test_no_config_file_uses_the_defaults(capsys):
    assert main(["--dry-run"]) == 0
    assert "pending decisions:" in capsys.readouterr().out


def test_asking_for_a_missing_gpu_stops_before_any_work(config_file):
    if torch.cuda.is_available():
        pytest.skip("this machine has a GPU")
    with pytest.raises(SystemExit, match="cuda"):
        main(["--config", str(config_file), "--set", "train.device=cuda"])


# ----------------------------------------------------------------------
# what the reconstruction gate judges this run against
#
# The gate asks for a share of what a linear map reaches at the same depth, so
# every run has to arrive at the number for the pair it actually trains at. No
# backbone is involved: the pair comes from the configuration and the number
# from the sweep's ranking file.
# ----------------------------------------------------------------------

def cross(cfg, ranking=None):
    """The same config, reading one model and reconstructing another."""
    cfg.objective = "case2"
    cfg.models.b = ModelConfig(label="B")
    cfg.activations.layers = ["A@add_14", "B@add_14"]
    if ranking is not None:
        cfg.layer_pairs.ranking_file = str(ranking)
    return cfg


def ranking_file(tmp_path, score, source="A@add_14", target="B@add_14"):
    path = tmp_path / "sweeps" / "ranking.json"
    write_ranking([{"source": source, "target": target,
                    "explained_variance": score, "rank": 0}], path)
    return path


def test_reproducing_its_own_input_is_judged_against_one(cfg):
    """The best linear map from a layer to itself is the identity."""
    assert resolve_linear_fit(cfg) == 1.0


def test_a_cross_model_run_takes_its_fit_from_the_ranking(cfg, tmp_path):
    assert resolve_linear_fit(
        cross(cfg, ranking_file(tmp_path, 0.4744))) == pytest.approx(0.4744)


def test_each_depth_gets_its_own_fit(cfg, tmp_path):
    """Which is the whole point: one threshold would mean different demands."""
    path = tmp_path / "sweeps" / "ranking.json"
    write_ranking([{"source": "A@add_2", "target": "B@add_2",
                    "explained_variance": 0.94, "rank": 0},
                   {"source": "A@add_24", "target": "B@add_24",
                    "explained_variance": 0.22, "rank": 1}], path)

    shallow = cross(Config(), path)
    shallow.activations.layers = ["A@add_2", "B@add_2"]
    deep = cross(Config(), path)
    deep.activations.layers = ["A@add_24", "B@add_24"]

    assert resolve_linear_fit(shallow) == pytest.approx(0.94)
    assert resolve_linear_fit(deep) == pytest.approx(0.22)


def test_a_depth_with_nothing_to_stitch_stops_the_run(cfg, tmp_path):
    """A share of nothing is not a threshold, and the dictionary would
    describe nothing."""
    with pytest.raises(ValueError, match="nothing to stitch"):
        resolve_linear_fit(cross(cfg, ranking_file(tmp_path, -0.02)))


def test_a_fit_of_exactly_zero_stops_the_run_too(cfg, tmp_path):
    with pytest.raises(ValueError, match="nothing to stitch"):
        resolve_linear_fit(cross(cfg, ranking_file(tmp_path, 0.0)))


def test_a_pair_missing_from_the_ranking_raises(cfg, tmp_path):
    path = ranking_file(tmp_path, 0.5, source="A@add_2", target="B@add_2")
    with pytest.raises(KeyError, match="A@add_14"):
        resolve_linear_fit(cross(cfg, path))


def test_without_a_ranking_file_the_configured_value_stands(cfg):
    """Smoke runs and tests have no sweep behind them."""
    cfg = cross(cfg)
    cfg.eval.linear_fit = 0.5
    assert resolve_linear_fit(cfg) == 0.5


# ----------------------------------------------------------------------
# what a fixed basis of the same size reaches
#
# No backbone: the sums the sweep saved are all a baseline needs, so this runs
# on CPU in milliseconds and a run pays a solve rather than a pass over data.
# ----------------------------------------------------------------------

WIDTH = 6


def swept(tmp_path, cfg, tokens=4000, width=WIDTH, source="A@add_14",
          target="B@add_14", seed=0):
    """A ranking and its sums, as a finished sweep leaves them."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(tokens, width, generator=generator)
    y = x @ torch.randn(width, 2, generator=generator) @ \
        torch.randn(2, width, generator=generator)

    ranking = tmp_path / "sweeps" / "ranking.json"
    write_ranking([{"source": source, "target": target,
                    "explained_variance": 0.5, "rank": 0}], ranking)

    for split in (cfg.dataset.split, cfg.dataset.eval_split):
        moments = CrossMoments(pairs=[(source, target)],
                               widths={source: width, target: width})
        moments.update({source: x, target: y})
        save_moments(moments, moments_path(ranking, split))

    cfg.layer_pairs.ranking_file = str(ranking)
    return cfg


def test_a_run_is_judged_against_a_basis_of_its_own_size(cfg, tmp_path):
    cross(cfg, None)
    result = resolve_baseline(swept(tmp_path, cfg), "A@add_14", "B@add_14", l0=2.0)

    assert result["baseline_rank"] == 2.0
    assert result["baseline_explained_variance"] == pytest.approx(1.0, abs=1e-2)


def test_a_coder_reproducing_its_own_input_gets_one_too(cfg, tmp_path):
    """The sweep holds no self pair, so this is the case that has to be derived."""
    result = resolve_baseline(swept(tmp_path, cfg), "A@add_14", "A@add_14", l0=3.0)
    assert result["baseline_rank"] == 3.0
    assert 0.0 <= result["baseline_explained_variance"] <= 1.0


def test_without_a_sweep_there_is_no_baseline(cfg):
    assert resolve_baseline(cfg, "A@add_14", "A@add_14", l0=3.0) == {}


def test_a_ranking_written_before_the_sums_were_kept_is_not_fatal(cfg, tmp_path):
    """Old sweeps have a ranking and no moments. The run is still valid."""
    ranking = tmp_path / "ranking.json"
    write_ranking([{"source": "A@add_14", "target": "B@add_14",
                    "explained_variance": 0.5, "rank": 0}], ranking)
    cfg.layer_pairs.ranking_file = str(ranking)

    assert resolve_baseline(cfg, "A@add_14", "B@add_14", l0=3.0) == {}


# ----------------------------------------------------------------------
# the control
#
# Same coder, same loop, same budget, tokens drawn from the real activations'
# statistics instead of from images.
# ----------------------------------------------------------------------

def test_a_control_needs_no_images_and_no_backbones(cfg, tmp_path):
    swept(tmp_path, cfg)
    cfg.activations.mode = "gaussian"
    cfg.activations.layers = ["A@add_14"]
    cfg.train.token_batch = 64
    cfg.activations.scale_tokens = 128
    cfg.dataset.root = ""                      # nothing on disk is read

    training, evaluation, loader = build_gaussian_streams(cfg)

    assert loader is None
    assert training.widths() == {"A@add_14": WIDTH}
    assert next(training.batches())["A@add_14"].shape == (64, WIDTH)


def test_both_control_streams_share_one_set_of_units(cfg, tmp_path):
    """The same reason the real streams do: two sets of scalars would put the
    training and evaluation numbers in different units."""
    swept(tmp_path, cfg)
    cfg.activations.mode = "gaussian"
    cfg.activations.layers = ["A@add_14"]
    cfg.train.token_batch = 64
    cfg.activations.scale_tokens = 128

    training, evaluation, _ = build_gaussian_streams(cfg)
    assert evaluation.scales() == training.scales()


def test_a_control_carries_both_layers_for_a_cross_model_coder(cfg, tmp_path):
    swept(tmp_path, cfg)
    cross(cfg, None)
    cfg.activations.mode = "gaussian"
    cfg.train.token_batch = 64
    cfg.activations.scale_tokens = 128

    training, _, _ = build_gaussian_streams(cfg)
    assert set(training.widths()) == {"A@add_14", "B@add_14"}


def test_a_dry_run_says_the_images_are_unused(cfg, tmp_path):
    swept(tmp_path, cfg)
    cfg.activations.mode = "gaussian"
    assert "tokens are generated" in describe(cfg)


# ----------------------------------------------------------------------
# loading backbones
# ----------------------------------------------------------------------

def test_a_layer_naming_an_unconfigured_model_raises(cfg):
    cfg.activations.layers = ["B@add_14"]
    with pytest.raises(ValueError, match="models.b"):
        load_models(cfg)


@needs_everything
def test_both_the_traced_and_untraced_models_come_back(cfg):
    """Tracing drops the record of what preprocessing the model expects."""
    from stitch.data import build_transform

    traced, backbones = load_models(cfg)
    assert set(traced) == set(backbones)
    build_transform(next(iter(backbones.values())))


# ----------------------------------------------------------------------
# end to end
# ----------------------------------------------------------------------

@needs_everything
def test_a_tiny_run_produces_a_summary(cfg, tmp_path):
    """The only test that loads a real backbone. Still CPU, still seconds."""
    from stitch.run import run

    cfg.logging.run_root = str(tmp_path / "runs")
    cfg.tag = "endtoend"
    cfg.coder.expansion = 2
    cfg.activations.extraction_batch_size = 2
    cfg.activations.shuffle_buffer_tokens = 64
    cfg.activations.scale_tokens = 64
    cfg.train.tokens = 128
    cfg.train.token_batch = 64
    cfg.train.log_every = 1
    cfg.train.eval_every = 1000
    cfg.train.checkpoint_every = 1000
    cfg.dataset.num_workers = 0
    cfg.eval.batches = 1
    cfg.eval.images = 4
    cfg.eval.wider_tokens = 128

    summary = run(cfg)

    # The identity has to name the directory the run actually wrote into. It
    # used to be recomputed at the end, by which time opening the dataset had
    # written the class count back into the configuration and changed the hash,
    # so a run's summary named a run that did not exist.
    run_dir = tmp_path / "runs" / summary["run_name"]
    assert run_dir.is_dir()
    assert run_dir.name.endswith(summary["hash"][:6])

    for key in ("mse", "eval_explained_variance", "gate_passed",
                "eval_dead_or_dormant", "eval_dead_or_dormant_wider",
                "eval_wider_tokens",
                "plain_top1", "substituted_top1", "agreement"):
        assert key in summary

    written = json.loads((run_dir / "summary.json").read_text())
    assert written["hash"] == summary["hash"]
    assert written["run_name"] == summary["run_name"]


@needs_everything
def test_the_scales_are_recorded(cfg, tmp_path):
    """Evaluation later has to use the numbers training used."""
    from stitch.run import run

    cfg.logging.run_root = str(tmp_path / "runs")
    cfg.coder.expansion = 2
    cfg.activations.extraction_batch_size = 2
    cfg.activations.shuffle_buffer_tokens = 64
    cfg.activations.scale_tokens = 64
    cfg.train.tokens = 64
    cfg.train.token_batch = 64
    cfg.train.checkpoint_every = 1000
    cfg.dataset.num_workers = 0
    cfg.eval.batches = 1
    cfg.eval.images = 4
    cfg.eval.wider_tokens = 128

    summary = run(cfg)
    assert set(summary["scales"]) == set(cfg.activations.layers)
    assert all(value > 0 for value in summary["scales"].values())