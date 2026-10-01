"""Tests for configuration and logging.

Runs on CPU with no model, no dataset and no tracking server. Everything these
exercise is code the training loop will use unchanged, so a pass here means the
plumbing is sound even though nothing has trained yet.

    python -m pytest tests/ -q
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stitch.config import Config, ModelConfig, load_config, dump_config  # noqa: E402
from stitch.logging import JsonlLogger, get_logger, setup_logging        # noqa: E402


# ----------------------------------------------------------------------
# config
# ----------------------------------------------------------------------

def test_defaults_are_valid():
    """The zero-argument config must be a runnable case 1 configuration."""
    cfg = Config()
    cfg.validate()
    assert cfg.objective == "case1"
    assert len(cfg.activations.layers) == 1


def test_case2_requires_model_b():
    cfg = Config(objective="case2")
    cfg.activations.layers = ["A@blocks.6", "B@blocks.6"]
    with pytest.raises(ValueError, match="models.b"):
        cfg.validate()


def test_case2_requires_two_layers():
    cfg = Config(objective="case2")
    cfg.models.b = ModelConfig(label="B")
    with pytest.raises(ValueError, match="input layer"):
        cfg.validate()


def test_case2_valid_when_both_present():
    """Case 2 differs from case 1 by a target layer and a second model, nothing else."""
    cfg = Config(objective="case2")
    cfg.models.b = ModelConfig(label="B")
    cfg.activations.layers = ["A@blocks.6", "B@blocks.6"]
    cfg.validate()


def test_cached_mode_requires_cache_dir():
    cfg = Config()
    cfg.activations.mode = "cached"
    with pytest.raises(ValueError, match="cache_dir"):
        cfg.validate()


def test_layer_key_format_is_enforced():
    cfg = Config()
    cfg.activations.layers = ["blocks.6"]
    with pytest.raises(ValueError, match="model label"):
        cfg.validate()


def test_hash_is_stable_and_seed_sensitive():
    """Identical settings hash identically; a different seed must not."""
    a, b = Config(), Config()
    assert a.hash() == b.hash()

    c = Config()
    c.train.seed = 1
    assert c.hash() != a.hash()


def test_hash_tracks_nested_changes():
    a = Config()
    b = Config()
    b.coder.sparsity.coeff = 5e-4
    assert a.hash() != b.hash()


def test_flat_uses_dotted_keys():
    flat = Config().flat()
    assert flat["train.seed"] == 0
    assert flat["coder.sparsity.mode"] == "l1"
    assert "coder" not in flat


def test_run_name_is_readable():
    cfg = Config(tag="smoke")
    name = cfg.run_name()
    assert name.startswith("case1-smoke-x16-l1-s0-")


def test_yaml_roundtrip(tmp_path):
    cfg = Config(tag="rt")
    cfg.train.seed = 7
    cfg.coder.expansion = 32
    path = tmp_path / "c.yaml"
    dump_config(cfg, path)

    back = load_config(path)
    assert back.hash() == cfg.hash()


def test_overrides_are_typed(tmp_path):
    """Overrides parse as YAML scalars, so types arrive correctly."""
    cfg = load_config(None, ["train.seed=3",
                             "coder.expansion=8",
                             "coder.sparsity.coeff=5e-4",
                             "logging.csv_mirror=false"])
    assert cfg.train.seed == 3 and isinstance(cfg.train.seed, int)
    assert cfg.coder.expansion == 8
    assert cfg.coder.sparsity.coeff == pytest.approx(5e-4)
    assert cfg.logging.csv_mirror is False


def test_scientific_notation_is_coerced(tmp_path):
    """YAML 1.1 reads 1e-4 as a string. Learning rates must still be floats."""
    p = tmp_path / "sci.yaml"
    p.write_text("train:\n  lr: 1e-4\ncoder:\n  sparsity:\n    coeff: 5e-4\n")
    cfg = load_config(p)
    assert isinstance(cfg.train.lr, float) and cfg.train.lr == pytest.approx(1e-4)
    assert isinstance(cfg.coder.sparsity.coeff, float)


def test_unknown_key_in_override_raises():
    """A typo must fail loudly rather than silently having no effect."""
    with pytest.raises(ValueError, match="unknown key"):
        load_config(None, ["train.sed=3"])


def test_unknown_key_in_yaml_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("train:\n  lerning_rate: 0.1\n")
    with pytest.raises(ValueError, match="unknown key"):
        load_config(p)


def test_pending_decisions_are_reported():
    """Placeholder values must be visible, not silently baked into a result."""
    pending = Config().pending_decisions()
    assert any("train.tokens" in p for p in pending)
    assert any("dataset.root" in p for p in pending)


# ----------------------------------------------------------------------
# logging
# ----------------------------------------------------------------------

def test_jsonl_logger_writes_expected_files(tmp_path):
    cfg = Config(tag="log")
    log = JsonlLogger(run_root=tmp_path)
    run_id = log.start_run(cfg.run_name())
    log.log_params(cfg.flat())
    for step in range(3):
        log.log_metrics({"loss": 1.0 / (step + 1), "l0": 30.0}, step=step)
    log.log_dict({"note": "hello"}, "extra.json")
    log.end_run()

    d = tmp_path / run_id
    assert (d / "params.json").exists()
    assert (d / "metrics.csv").exists()
    assert (d / "artifacts" / "extra.json").exists()

    rows = [json.loads(l) for l in (d / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    assert rows[0]["loss"] == 1.0
    assert rows[2]["step"] == 2
    assert all("wall" in r for r in rows)

    status = json.loads((d / "status.json").read_text())
    assert status["status"] == "FINISHED"


def test_metrics_csv_widens_when_a_new_metric_appears(tmp_path):
    """A metric that starts partway through must not corrupt earlier rows."""
    log = JsonlLogger(run_root=tmp_path)
    run_id = log.start_run("widen")
    log.log_metrics({"loss": 1.0}, step=0)
    log.log_metrics({"loss": 0.5, "explained_variance": 0.9}, step=1)
    log.end_run()

    text = (tmp_path / run_id / "metrics.csv").read_text().splitlines()
    assert "explained_variance" in text[0]
    assert len(text) == 3            # header plus two rows


def test_end_run_is_idempotent(tmp_path):
    """The crash handler may close a run the normal path has already closed."""
    log = JsonlLogger(run_root=tmp_path)
    log.start_run("idem")
    log.end_run("FINISHED")
    log.end_run("FAILED")            # must not raise or overwrite
    status = json.loads((tmp_path / "idem" / "status.json").read_text())
    assert status["status"] == "FINISHED"


def test_logging_before_start_raises(tmp_path):
    log = JsonlLogger(run_root=tmp_path)
    with pytest.raises(RuntimeError, match="start_run"):
        log.log_metrics({"loss": 1.0}, step=0)


def test_context_manager_marks_failure(tmp_path):
    log = JsonlLogger(run_root=tmp_path)
    log.start_run("ctx")
    with pytest.raises(ValueError):
        with log:
            raise ValueError("boom")
    status = json.loads((tmp_path / "ctx" / "status.json").read_text())
    assert status["status"] == "FAILED"


def test_missing_artifact_does_not_kill_the_run(tmp_path):
    log = JsonlLogger(run_root=tmp_path)
    log.start_run("missing")
    log.log_artifact(tmp_path / "nope.txt")     # must not raise
    log.end_run()


def test_get_logger_rejects_unknown_backend():
    cfg = Config()
    cfg.logging.backend = "wandb"
    with pytest.raises(ValueError, match="unknown logging backend"):
        get_logger(cfg)


def test_setup_logging_writes_run_log(tmp_path):
    log = setup_logging(tmp_path / "r", console=False)
    log.info("a line")
    log.warning("another")
    text = (tmp_path / "r" / "run.log").read_text()
    assert "a line" in text and "WARNING" in text


# ----------------------------------------------------------------------
# crash capture
#
# Run in a subprocess, because installing an excepthook and letting the
# interpreter die is not something that can be done inside the test process.
# ----------------------------------------------------------------------

CRASH_SCRIPT = """
import sys
sys.path.insert(0, {root!r})
from stitch.config import Config
from stitch.logging import JsonlLogger, setup_logging, install_crash_handler

cfg = Config(tag="crash")
log = JsonlLogger(run_root={runs!r})
run_id = log.start_run("crashrun")
setup_logging(log.run_dir, console=False)
install_crash_handler(log.run_dir, crash_dir={crashes!r}, logger=log,
                      context={{"hash": cfg.hash()}})
log.log_metrics({{"loss": 1.0}}, step=0)
raise RuntimeError("deliberate failure")
"""


def test_crash_handler_records_and_marks_failed(tmp_path):
    root = str(Path(__file__).resolve().parents[1])
    runs = str(tmp_path / "runs")
    crashes = str(tmp_path / "crashes")
    script = tmp_path / "crash.py"
    script.write_text(CRASH_SCRIPT.format(root=root, runs=runs, crashes=crashes))

    proc = subprocess.run([sys.executable, str(script)],
                          capture_output=True, text=True)
    assert proc.returncode != 0, "the script must actually fail"

    # the run is marked FAILED rather than left looking alive
    status = json.loads((Path(runs) / "crashrun" / "status.json").read_text())
    assert status["status"] == "FAILED"

    # the traceback is captured in both places
    report = json.loads((Path(runs) / "crashrun" / "crash.json").read_text())
    assert report["exception"] == "RuntimeError"
    assert "deliberate failure" in report["message"]
    assert "RuntimeError" in report["traceback"]
    assert report["context"]["hash"]

    shared = list(Path(crashes).glob("*.json"))
    assert len(shared) == 1

    # and the failure is in the process log
    assert "deliberate failure" in (Path(runs) / "crashrun" / "run.log").read_text()
