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

from stitch.config import Config, dump_config                          # noqa: E402
from stitch.run import describe, load_models, main                     # noqa: E402

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

    summary = run(cfg)

    assert summary["hash"] == cfg.hash()
    for key in ("mse", "eval_explained_variance", "gate_passed",
                "plain_top1", "substituted_top1", "agreement"):
        assert key in summary

    written = json.loads(
        (next((tmp_path / "runs").iterdir()) / "summary.json").read_text())
    assert written["hash"] == summary["hash"]


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

    summary = run(cfg)
    assert set(summary["scales"]) == set(cfg.activations.layers)
    assert all(value > 0 for value in summary["scales"].values())