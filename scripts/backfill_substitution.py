#!/usr/bin/env python3
"""Recompute substitution accuracy for runs that already finished.

Accuracy was measured by comparing a subset loader's class numbering against a
model's, which are different numberings, so every accuracy recorded before the
label translation existed is wrong. Agreement was never affected.

Nothing is retrained. The coder is loaded from the run's last checkpoint and
the measurement is redone, which is a few thousand images rather than a few
hundred million tokens.

    python scripts/backfill_substitution.py --runs runs/case1-control-x16-l1-s0-2eb9b2
    python scripts/backfill_substitution.py --from-grid grids/noise_control_200M_c5e-3_x16.json
    python scripts/backfill_substitution.py --from-grid grids/*.json --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch                                                          # noqa: E402

from stitch.coder import build_coder                                  # noqa: E402
from stitch.compare import from_grids, latest_checkpoint              # noqa: E402
from stitch.config import Config, ModelConfig, load_config            # noqa: E402
from stitch.data import build_dataset, build_loader, build_transform  # noqa: E402
from stitch.evaluate import substitution_accuracy                     # noqa: E402
from stitch.labels import label_map_for                               # noqa: E402
from stitch.run import build_halves, load_models                      # noqa: E402
from stitch.train import resolve_layers                               # noqa: E402

log = logging.getLogger("backfill")


def _as_override(key: str, value) -> str:
    """One flattened parameter as a ``key=value`` override.

    JSON rather than ``str``: the override parser runs the right-hand side
    through ``yaml.safe_load``, where ``None`` is the string "None" while
    ``null`` is nothing, and a list printed by ``str`` is not reliably YAML.
    JSON is valid YAML for every scalar and list that appears here.

    A list that reached ``params.json`` as its printed form rather than as a
    list, which some metric stores do, is read back before being re-encoded.
    """
    if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
        try:
            import yaml
            value = yaml.safe_load(value)
        except Exception:                               # noqa: BLE001 - leave it as text
            pass
    return f"{key}={json.dumps(value)}"


def current_keys() -> set[str]:
    """Every dotted key the configuration defines today.

    Built from a Config that has both models populated, since a one-model
    default records ``models.b`` as a single empty key rather than the nested
    ones a case 2 run was logged with.
    """
    reference = Config()
    reference.models.b = ModelConfig(label="B")
    return set(reference.flat())


def config_of(run_dir: Path):
    """The configuration a run was produced by, rebuilt from what it recorded.

    ``params.json`` holds the whole resolved configuration flattened to dotted
    keys, so replaying it as overrides onto the defaults reproduces it exactly,
    without depending on which YAML file was originally passed.

    Settings that have since been renamed or removed are dropped rather than
    replayed. The loader rejects a key the configuration no longer defines,
    which is the right behaviour for a config file somebody is writing and the
    wrong one for a historical record of a run that happened. What is dropped
    is named in the log, because a dropped key means the rebuilt configuration
    differs from the one the run used.
    """
    params = json.loads((Path(run_dir) / "params.json").read_text())
    known = current_keys()

    kept = {key: value for key, value in params.items() if key in known}
    dropped = sorted(set(params) - set(kept))
    if dropped:
        log.info("%s: settings no longer in the configuration, ignored: %s",
                 Path(run_dir).name, ", ".join(dropped))

    overrides = [_as_override(key, value) for key, value in sorted(kept.items())]
    return load_config(None, overrides)


def eval_loader_for(cfg, backbones: dict[str, object]):
    """The evaluation loader a run used.

    Built directly rather than through ``build_streams``, which would also
    construct an activation source and spend a pass measuring scales. The
    scales are already in the checkpoint, and nothing here needs a token
    stream. The loader itself is assembled exactly as ``build_streams``
    assembles it, so it sees the same images in the same order.
    """
    reference = next(iter(backbones.values()))
    transform = build_transform(reference)
    dataset = build_dataset(cfg, cfg.dataset.eval_split, transform)
    return build_loader(cfg, dataset, cfg.activations.extraction_batch_size,
                        shuffle=False)


def redo(run_dir: Path, reference: str | None = None, dry_run: bool = False) -> dict | None:
    """Measure substitution again for one run, and update its summary.

    Returns:
        The new numbers, or None if nothing was changed.
    """
    run_dir = Path(run_dir)
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        log.warning("%s has no summary, skipping", run_dir.name)
        return None

    summary = json.loads(summary_path.read_text())
    cfg = config_of(run_dir)
    if reference:
        cfg.dataset.label_reference = reference

    checkpoint = latest_checkpoint(run_dir)

    if dry_run:
        log.info("%s: would redo from %s, was top1 %.4f agreement %.4f",
                 run_dir.name, checkpoint.name,
                 summary.get("substituted_top1", float("nan")),
                 summary.get("agreement", float("nan")))
        return None

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    scales = state.get("scales") or summary.get("scales") or {}

    source_key, target_key = resolve_layers(cfg)
    if source_key not in scales or target_key not in scales:
        raise KeyError(
            f"the checkpoint has scales for {sorted(scales)}, which does not "
            f"cover {source_key} and {target_key}")

    # Widths from the saved weights rather than from the models, so the coder
    # is rebuilt at exactly the shape the state dict expects.
    d_in = state["coder"]["encoder.weight"].shape[1]
    d_out = state["coder"]["decoder.weight"].shape[0]
    coder, _ = build_coder(cfg, d_in, d_out)
    coder.load_state_dict(state["coder"])

    models, backbones = load_models(cfg)
    loader = eval_loader_for(cfg, backbones)
    upper, lower = build_halves(cfg, models)

    result = substitution_accuracy(
        coder, upper, lower, loader,
        scale_in=scales[source_key], scale_out=scales[target_key],
        max_images=cfg.eval.images, device=cfg.train.device,
        label_map=label_map_for(cfg, loader.dataset))

    before = summary.get("substituted_top1")
    summary.update(result)
    summary["substitution_backfilled"] = True
    summary_path.write_text(json.dumps(summary, indent=2))

    log.info("%s: top1 %.4f -> %.4f, plain %.4f, agreement %.4f",
             run_dir.name, before if before is not None else float("nan"),
             result["substituted_top1"], result["plain_top1"], result["agreement"])
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", type=Path, default=None,
                        help="run directories")
    parser.add_argument("--from-grid", nargs="+", type=Path, default=None,
                        dest="from_grid", help="take the runs from these grid files")
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--label-reference", default=None,
                        help="split directory of the dataset the models were trained "
                             "on, when the run's own configuration does not name one")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.runs and not args.from_grid:
        raise SystemExit("give either --runs or --from-grid")

    directories = (from_grids(args.from_grid, args.run_root) if args.from_grid
                   else [Path(path) for path in args.runs])

    changed = 0
    for directory in directories:
        try:
            if redo(directory, args.label_reference, args.dry_run) is not None:
                changed += 1
        except Exception as error:                 # noqa: BLE001 - one bad run must not stop the rest
            log.error("%s failed: %s: %s", Path(directory).name,
                      type(error).__name__, error)

    print(f"\n{changed} of {len(directories)} runs updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())