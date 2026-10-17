"""Assembling and running one experiment.

Everything below this module is a piece that does one thing and is handed what
it needs. This is where the pieces are connected, in one place, so that reading
it tells you what a run actually consists of:

    configuration -> backbones -> images -> activations -> coder -> training
                                                               -> evaluation

Nothing here makes a decision. Every number comes from the configuration, and
anything the configuration leaves at a placeholder is printed at startup rather
than quietly used.

A run is identified by the configuration that produced it. The full settings,
their hash and the activation scales are written into the run directory, so a
result can be traced back to the exact thing that produced it without relying
on anyone's memory of what they ran.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from stitch.activations import build_source
from stitch.coder import build_coder
from stitch.config import load_config
from stitch.data import build_loader, build_transform, build_dataset
from stitch.evaluate import evaluate, substitution_accuracy
from stitch.labels import label_map_for
from stitch.models import parse_layer_key, trace
from stitch.train import resolve_layers, set_seed, train

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# assembling the pieces
# ----------------------------------------------------------------------

def load_models(cfg) -> tuple[dict[str, object], dict[str, object]]:
    """Load and trace every backbone the configured layers refer to.

    Only the models actually named are loaded, so case one does not pay for a
    second backbone it never reads.

    Returns:
        ``(traced, backbones)``. Both are needed. Graph surgery works on the
        traced form, but tracing drops the model's own record of the
        preprocessing it expects, which only the untraced backbone still
        carries.
    """
    from stitch.models import load_backbone

    wanted = {label for label, _ in (parse_layer_key(key) for key in cfg.activations.layers)}
    traced, backbones = {}, {}

    for label in sorted(wanted):
        spec = getattr(cfg.models, label.lower(), None)
        if spec is None:
            raise ValueError(
                f"activations.layers names model {label!r}, but models.{label.lower()} is unset")
        backbone = load_backbone(spec.name, pretrained=spec.pretrained,
                                 checkpoint=spec.checkpoint, source=spec.source)
        backbones[label] = backbone
        traced[label] = trace(backbone)
        log.info("loaded %s as %s", spec.name, label)

    return traced, backbones


def build_streams(cfg, models: dict[str, object], backbones: dict[str, object]):
    """Activation sources over the training and evaluation splits.

    The evaluation source adopts the training source's scales rather than
    measuring its own. Two sets of scalars would mean the two were measured in
    different units, and every evaluation number would be off by a factor
    nobody recorded.

    Args:
        cfg: the run configuration.
        models: traced backbones, which is what activations are extracted from.
        backbones: the same models untraced, which is where the preprocessing
            comes from, since tracing does not carry it.
    """
    reference = next(iter(backbones.values()))
    transform = build_transform(reference)
    batch_size = cfg.activations.extraction_batch_size

    train_loader = build_loader(cfg, build_dataset(cfg, cfg.dataset.split, transform),
                                batch_size, shuffle=True)
    eval_loader = build_loader(cfg, build_dataset(cfg, cfg.dataset.eval_split, transform),
                               batch_size, shuffle=False)

    training = build_source(cfg, models, train_loader)
    evaluation = build_source(cfg, models, eval_loader)
    evaluation.set_scales(training.scales())

    return training, evaluation, eval_loader


def build_halves(cfg, models: dict[str, object]):
    """The parts of the model above and below the cut, for substitution.

    The upper half comes from the model the coder reads. The lower half comes
    from the model it reconstructs, which is the same one when the coder
    reproduces its own input and a different one otherwise.

    Returns:
        ``(upper, lower, reference_upper)``. The third is the target model's own
        upper half, and is None when the two halves already come from one model.
        It exists so that the unmodified prediction is the target model cut and
        rejoined, rather than the source model's activation handed to the target
        model's layers, which is not a model and predicts at chance.
    """
    from stitch.models import downstream, upstream

    source_key, target_key = resolve_layers(cfg)
    source_label, source_node = parse_layer_key(source_key)
    target_label, target_node = parse_layer_key(target_key)

    upper = upstream(models[source_label], source_node)
    lower = downstream(models[target_label], target_node)
    reference_upper = (None if source_label == target_label
                       else upstream(models[target_label], target_node))

    return upper, lower, reference_upper


# ----------------------------------------------------------------------
# the run
# ----------------------------------------------------------------------

def run(cfg, resume: Path | None = None) -> dict:
    """One experiment, start to finish.

    Returns:
        A summary of what the run produced, also written to the run directory.
    """
    from stitch.logging import get_logger, install_crash_handler, setup_logging

    set_seed(cfg.train.seed)

    # The run's identity is fixed here, before anything can change the
    # configuration. Opening the dataset writes the class count it found back
    # into cfg, and the hash covers every field, so a name computed later in
    # this function names a different run than the directory being written to.
    name, config_hash = cfg.run_name(), cfg.hash()

    logger = get_logger(cfg)
    logger.start_run(name)
    run_dir = Path(logger.run_dir)

    setup_logging(run_dir, console=True)
    install_crash_handler(run_dir, crash_dir=Path(cfg.logging.run_root) / "crashes",
                          logger=logger, context={"hash": config_hash})

    for note in cfg.pending_decisions():
        log.warning("pending: %s", note)

    logger.log_params(cfg.flat())
    logger.log_dict({"hash": config_hash, "pending": cfg.pending_decisions()}, "config.json")

    models, backbones = load_models(cfg)
    training, evaluation, eval_loader = build_streams(cfg, models, backbones)

    source_key, target_key = resolve_layers(cfg)
    widths = training.widths()
    coder, tracker = build_coder(cfg, widths[source_key], widths[target_key])

    scales = training.scales()
    logger.log_dict(scales, "scales.json")

    def hook(current, step):
        return evaluate(current, evaluation.batches(), source_key, target_key,
                        cfg, cfg.train.device)

    final = train(cfg, training, coder, tracker, logger, run_dir,
                  evaluate=hook, resume=resume)

    # Held-out metrics and the gates, on a coder that has finished training.
    held_out = evaluate(coder, evaluation.batches(), source_key, target_key,
                        cfg, cfg.train.device)
    logger.log_metrics(held_out, step=int(final.get("tokens", 0)))

    # The expensive one: does the model still work with the reconstruction in place.
    upper, lower, reference_upper = build_halves(cfg, models)
    substitution = substitution_accuracy(
        coder, upper, lower, eval_loader,
        scale_in=scales[source_key], scale_out=scales[target_key],
        max_images=cfg.eval.images, device=cfg.train.device,
        label_map=label_map_for(cfg, eval_loader.dataset),
        reference_upper=reference_upper)
    logger.log_metrics(substitution, step=int(final.get("tokens", 0)))

    summary = {"hash": config_hash, "run_name": name,
               "scales": scales, **final, **held_out, **substitution}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.end_run()

    log.info("run complete: %s", run_dir)
    return summary


def describe(cfg) -> str:
    """What a run would do, without doing any of it."""
    source_key, target_key = resolve_layers(cfg)
    steps = -(-cfg.train.tokens // cfg.train.token_batch)

    lines = [
        f"objective         {cfg.objective}",
        f"run name          {cfg.run_name()}",
        f"hash              {cfg.hash()}",
        f"dataset           {cfg.dataset.name} at {cfg.dataset.root or '(unset)'}",
        f"reads             {source_key}",
        f"reconstructs      {target_key}",
        f"activations       {cfg.activations.mode}, scaling {cfg.activations.scaling}",
        f"dictionary        {cfg.coder.expansion}x, {cfg.coder.sparsity.mode} sparsity",
        f"budget            {cfg.train.tokens:,} tokens, {steps:,} steps "
        f"of {cfg.train.token_batch:,}",
        f"device            {cfg.train.device}, {cfg.train.dtype}",
    ]
    pending = cfg.pending_decisions()
    if pending:
        lines.append("")
        lines.append("pending decisions:")
        lines.extend(f"  {note}" for note in pending)
    return "\n".join(lines)


# ----------------------------------------------------------------------
# command line
# ----------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Usage:
        python scripts/run.py --config configs/case1_smoke.yaml
        python scripts/run.py --config configs/case1_smoke.yaml --set train.seed=1
        python scripts/run.py --config configs/case1_smoke.yaml --dry-run
    """
    parser = argparse.ArgumentParser(description="Train one sparse coder.")
    parser.add_argument("--config", type=Path, default=None,
                        help="YAML file. Omitted, the defaults are used")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="override one setting, repeatable")
    parser.add_argument("--resume", type=Path, default=None,
                        help="checkpoint to continue from")
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would happen and stop")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config, args.overrides)
    print(describe(cfg))

    if args.dry_run:
        return 0

    if not torch.cuda.is_available() and cfg.train.device.startswith("cuda"):
        raise SystemExit("train.device asks for cuda, which is not available here")

    run(cfg, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())