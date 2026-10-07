"""Configuration for the stitching experiments.

Every value that distinguishes one run from another lives here, and nothing
else does. The resolved configuration is hashed to produce the run identifier
and logged in full as run parameters, so a result can always be traced back to
the exact settings that produced it.

Layering, lowest priority first:

  1. the dataclass defaults in this file
  2. a YAML file passed with --config
  3. dotted overrides passed on the command line, e.g. train.seed=3

Later layers win. Only keys that already exist may be overridden, so a typo in
an override raises rather than silently creating a new field that nothing reads.

PENDING markers
---------------
Several values are placeholders awaiting a decision. They are marked
``PENDING()`` in the field comments and collected by
:func:`pending_decisions`, which the CLI prints at startup so an unresolved
default is never mistaken for a considered one. Grep for ``PENDING()``
to find them all.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields, is_dataclass, asdict
from pathlib import Path
from typing import Any, Literal, get_args, get_origin, get_type_hints, Union


# ----------------------------------------------------------------------
# dataset
# ----------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """Where the images come from.

    ``name`` and ``root`` are kept separate from ``source_url`` on purpose. The
    first two are what the code uses; the third is provenance, recorded so that
    a result can be reproduced by someone who does not have the same filesystem.
    """

    name: str = "imagenet-100"
    root: str = ""                    # absolute path on the machine that runs this
    source_url: str = ""              # where the data was obtained from
    split: str = "train"
    eval_split: str = "val"
    num_classes: int | None = None    # filled from the dataset when left unset
    notes: str = ""                   # e.g. which published ImageNet-100 subset
    num_workers: int = 8
    pin_memory: bool = True

    # PENDING(): full ImageNet or ImageNet-100 for the main grid. This
    # decides whether caching activations is viable at all, 361 GiB against 37.


# ----------------------------------------------------------------------
# models
# ----------------------------------------------------------------------

@dataclass
class ModelConfig:
    """One backbone.

    ``name`` is a timm identifier. ``checkpoint`` overrides timm's own weights
    when the model was trained locally, which is the likely situation for a pair
    that differs only by seed.
    """

    name: str = "vit_base_patch16_224"
    source: Literal["timm", "torchvision"] = "timm" 
    pretrained: bool = True
    checkpoint: str | None = None     # path to a local state dict, if any
    label: str = ""                   # short tag used in layer keys and run names


@dataclass
class ModelsConfig:
    """The model pair.

    Case 1 reads a single model and leaves ``b`` unset. Cases 2 onwards require
    both. Validation enforces this against the chosen objective rather than
    failing later inside the training loop.
    """

    a: ModelConfig = field(default_factory=lambda: ModelConfig(label="A"))
    b: ModelConfig | None = None

    # PENDING(): two independently trained ViT-B/16 differing only by
    # seed, or two different architectures. Different architectures make the
    # comparison more interesting and complicate layer correspondence.


# ----------------------------------------------------------------------
# layer pairs
# ----------------------------------------------------------------------

@dataclass
class LayerPairConfig:
    """Which depths to stitch at.

    The sweep (stage 3) fits a linear stitch at many depths and writes a ranked
    list to ``ranking_file``. Training then takes the top ``count`` entries from
    that file. ``explicit`` bypasses the ranking entirely, which is the escape
    hatch for when the viability curve suggests something other than the top N.
    """

    count: int = 3                    # how many of the ranked pairs to train on
    ranking_file: str | None = None   # written by the sweep, read by the grid
    matched_depth_only: bool = True   # block i of A against block i of B
    explicit: list[str] | None = None # e.g. ["A@blocks.6->B@blocks.6"]; overrides the ranking

    # PENDING(): how many pairs to keep. Three at five seeds is 15 runs
    # per case, six is 30. Drives the whole downstream budget.
    #
    # PENDING(): whether off-diagonal pairs are in scope. Matched depth
    # is the default because the case table says "B's activation at matching
    # layer". Sweeping off-diagonal is cheap; training on it is not.


# ----------------------------------------------------------------------
# the sweep itself
# ----------------------------------------------------------------------

@dataclass
class SweepConfig:
    """Stage 3: locate depths where a stitch works at all.

    A linear stitch is fitted in closed form at each candidate depth, so no
    training is involved and the whole sweep costs minutes. Its purpose is to
    exclude depths where any coder would fail, since a dictionary learned at a
    depth that cannot be stitched describes nothing.
    """

    depths: list[int] | None = None   # None means every transformer block
    metric: Literal["explained_variance", "accuracy"] = "explained_variance"
    fit_tokens: int = 500_000         # tokens used for the least-squares fit
    eval_images: int = 10_000         # images used when metric == "accuracy"
    out_file: str = "sweeps/layer_sweep.json"


# ----------------------------------------------------------------------
# the coder
# ----------------------------------------------------------------------

@dataclass
class SparsityConfig:
    """How sparsity is imposed.

    Both mechanisms are supported because they answer different needs. TopK sets
    the active count exactly, which makes a benchmark reproducible and removes
    one free parameter. L1 lets the active count emerge from a penalty, which is
    the regime the training runs use.

    Under ``topk``, ``k`` wins if set, otherwise ``k_frac`` times the dictionary
    width is used. Under ``l1``, ``coeff`` is the penalty weight and the
    resulting L0 is an observed quantity rather than a setting.
    """

    mode: Literal["l1", "topk"] = "l1"
    coeff: float = 1e-3               # L1 penalty weight; ignored under topk
    k: int | None = None              # exact active count; ignored under l1
    k_frac: float = 0.01              # fallback for k, as a fraction of width
    normalize_decoder: bool = True    # unit-norm decoder columns, standard for SAEs


@dataclass
class DeadFeatureConfig:
    """What to do about dictionary slots that stop firing.

    A slot that never activates contributes nothing and inflates the apparent
    dictionary size, which corrupts any comparison between dictionaries. Both
    remedies below are standard; resampling is the more aggressive one.
    """

    track_window: int = 10_000        # steps over which "never fired" is judged
    resample: bool = False            # reinitialise dead slots from high-error inputs
    resample_every: int = 25_000
    aux_loss_coeff: float = 0.0       # auxiliary reconstruction from dead slots only


@dataclass
class CoderConfig:
    """The sparse coder placed at the stitch point.

    ``expansion`` multiplies the input width to give the dictionary width. The
    input and output widths are not set here; they are read from the models, so
    that a cross-coder writing into a differently shaped space needs no extra
    configuration.
    """

    expansion: int = 16
    sparsity: SparsityConfig = field(default_factory=SparsityConfig)
    dead_features: DeadFeatureConfig = field(default_factory=DeadFeatureConfig)
    tie_decoder_init: bool = True     # initialise decoder as the encoder's transpose


# ----------------------------------------------------------------------
# activations
# ----------------------------------------------------------------------

@dataclass
class ActivationsConfig:
    """Where the coder's inputs and targets come from.

    ``mode`` selects between two implementations of the same interface, so the
    training loop is identical either way:

      on_the_fly  the backbones stay resident and run every step. No storage,
                  but extraction becomes the throughput bottleneck.
      cached      activations are written once and streamed from disk. Fast if
                  the volume sustains the required read rate, slower than
                  on_the_fly if it does not. See scripts/benchmark_case2.py for
                  the thresholds on a given machine.

    ``layers`` names what to extract, keyed as ``<model label>@<node name>``.
    Case 1 lists one entry, case 2 lists two. Nothing else changes between them.
    """

    mode: Literal["on_the_fly", "cached"] = "on_the_fly"
    cache_dir: str | None = None      # required when mode == "cached"
    layers: list[str] = field(default_factory=lambda: ["A@add_14"])
    extraction_batch_size: int = 128  # measured optimum for ViT-B/16; see benchmarks
    shuffle_buffer_tokens: int = 2_000_000
    cache_dtype: Literal["float16", "float32"] = "float16"

    # PENDING(): shuffle buffer size. Token-level shuffling decorrelates
    # the batch, and is also what turns a sequential read into a strided one.
    # The disk benchmark measures both patterns; the gap between them is the
    # cost of shuffling.


# ----------------------------------------------------------------------
# training
# ----------------------------------------------------------------------

@dataclass
class TrainConfig:
    """The optimisation itself.

    ``tokens`` rather than epochs, because the coder consumes tokens and the
    number of tokens per image depends on the backbone. Seeds are not optional:
    the comparison between objectives is only interpretable against the spread
    between runs that differ by initialisation alone.
    """

    tokens: int = 1_000_000_000       # total tokens seen by the coder
    token_batch: int = 4096
    lr: float = 1e-4
    optimizer: Literal["adam", "adamw"] = "adam"
    seed: int = 0
    dtype: Literal["float16", "float32"] = "float16"
    device: str = "cuda"
    log_every: int = 100              # steps
    eval_every: int = 5_000
    checkpoint_every: int = 25_000
    grad_clip: float | None = 1.0

    # PENDING(): token budget. 1 billion is roughly four passes over
    # ImageNet and is an estimate, never confirmed. It is the single largest
    # lever on total GPU-hours.


# ----------------------------------------------------------------------
# logging
# ----------------------------------------------------------------------

@dataclass
class LoggingConfig:
    """Experiment tracking and process logging.

    Two separate concerns that share a directory. ``backend`` selects where
    metrics go; the process log and crash capture are always written to the run
    directory regardless of backend, so a crash is recoverable even if the
    tracking backend is what failed.
    """

    backend: Literal["jsonl", "mlflow"] = "jsonl"
    tracking_uri: str = "file:./mlruns"   # local store; a server URL also works
    experiment: str = "stitch"
    run_root: str = "runs"                # per-run directory, logs and artifacts
    level: str = "INFO"
    csv_mirror: bool = True               # metrics also as CSV, for plotting


# ----------------------------------------------------------------------
# top level
# ----------------------------------------------------------------------

@dataclass
class Config:
    """The whole configuration for one run.

    ``objective`` selects what the coder is trained to reproduce. It is the only
    field that distinguishes case 1 from case 2; everything else follows from it
    through validation.
    """

    objective: Literal["case1", "case2"] = "case1"
    tag: str = ""                     # free-text label, appears in the run name
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    layer_pairs: LayerPairConfig = field(default_factory=LayerPairConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    coder: CoderConfig = field(default_factory=CoderConfig)
    activations: ActivationsConfig = field(default_factory=ActivationsConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Reject configurations that cannot run, before any work starts.

        Checks only what is knowable without touching the filesystem or the GPU.
        Anything requiring those is checked at the point of use.
        """
        if self.objective == "case2" and self.models.b is None:
            raise ValueError(
                "objective 'case2' reads model A and reproduces model B's "
                "activation, so models.b must be set")

        if self.objective == "case1" and len(self.activations.layers) != 1:
            raise ValueError(
                f"objective 'case1' uses exactly one layer, got "
                f"{self.activations.layers}")

        if self.objective == "case2" and len(self.activations.layers) != 2:
            raise ValueError(
                f"objective 'case2' needs an input layer from A and a target "
                f"layer from B, got {self.activations.layers}")

        if self.activations.mode == "cached" and not self.activations.cache_dir:
            raise ValueError("activations.mode is 'cached' but no cache_dir is set")

        if self.coder.sparsity.mode == "topk":
            if self.coder.sparsity.k is None and self.coder.sparsity.k_frac <= 0:
                raise ValueError("topk sparsity needs either k or a positive k_frac")
        elif self.coder.sparsity.coeff <= 0:
            raise ValueError("l1 sparsity needs a positive coeff")

        if self.train.token_batch <= 0 or self.train.tokens <= 0:
            raise ValueError("train.tokens and train.token_batch must be positive")

        for key in self.activations.layers:
            if "@" not in key:
                raise ValueError(
                    f"layer key {key!r} must be '<model label>@<node name>', "
                    f"for example 'A@blocks.6'")

    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """The configuration as plain data, for logging and hashing."""
        return asdict(self)

    def flat(self) -> dict[str, Any]:
        """Dotted-key view, which is the shape experiment trackers expect."""
        return _flatten(self.to_dict())

    def hash(self, length: int = 12) -> str:
        """A stable identifier for this configuration.

        Two runs with identical settings hash identically, which is what makes
        it possible to notice that a configuration has already been run. The
        seed is part of the configuration, so runs differing only by seed hash
        differently, as they must.
        """
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:length]

    def run_name(self) -> str:
        """A human-readable run name. Uniqueness comes from the hash suffix."""
        parts = [self.objective]
        if self.tag:
            parts.append(self.tag)
        parts.append(f"x{self.coder.expansion}")
        parts.append(self.coder.sparsity.mode)
        parts.append(f"s{self.train.seed}")
        parts.append(self.hash(6))
        return "-".join(parts)

    def pending_decisions(self) -> list[str]:
        """Settings still carrying a placeholder value.

        Printed at startup so that an unresolved default is visible in the run
        log rather than silently baked into a result.
        """
        out = []
        if not self.dataset.root:
            out.append("dataset.root is empty")
        if self.models.b is None and self.objective != "case1":
            out.append("models.b is unset")
        if self.train.tokens == 1_000_000_000:
            out.append("train.tokens is the unconfirmed 1B estimate")
        if self.activations.shuffle_buffer_tokens == 2_000_000:
            out.append("activations.shuffle_buffer_tokens is a placeholder")
        if self.layer_pairs.count == 3 and self.layer_pairs.ranking_file is None:
            out.append("layer_pairs.count is a placeholder and no sweep ranking exists yet")
        return out


# ----------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------

def _flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    """Nested dict to dotted keys. Lists are left as-is and stringified later."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


def _unwrap_optional(tp):
    """Return the non-None type from Optional[X], or tp unchanged."""
    if get_origin(tp) is Union:
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return tp


def _coerce(value, tp):
    """Coerce a loaded value to its declared type.

    YAML 1.1, which is what ``yaml.safe_load`` implements, does not recognise
    ``1e-4`` as a float: the spec requires a decimal point and a signed
    exponent, so ``1e-4`` arrives as the string "1e-4" while ``1.0e-4`` arrives
    as a float. Learning rates and L1 coefficients are exactly the values people
    write in the first form, and a silent string would surface much later as a
    comparison failing on a type error.

    Coercion is therefore driven by the declared field type. A value that cannot
    be coerced is returned unchanged so that validation reports it, rather than
    being masked by an exception here.
    """
    tp = _unwrap_optional(tp)
    if value is None or tp is Any:
        return value

    # Literal["a", "b"] constrains the value, it does not convert it.
    if get_origin(tp) is Literal:
        return value

    try:
        if tp is float and isinstance(value, (str, int)):
            return float(value)
        if tp is int and isinstance(value, str):
            return int(value)
        if tp is str and not isinstance(value, str):
            return str(value)
        if tp is bool and isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("true", "yes", "on", "1"):
                return True
            if lowered in ("false", "no", "off", "0"):
                return False
    except (TypeError, ValueError):
        pass
    return value


def _build(cls, data: dict | None):
    """Recursively construct a dataclass from plain data.

    Unknown keys raise rather than being ignored, so a misspelled key in a YAML
    file is caught at load time instead of quietly having no effect.
    """
    if data is None:
        return None
    if not isinstance(data, dict):
        raise TypeError(f"expected a mapping for {cls.__name__}, got {type(data).__name__}")

    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(
            f"unknown key(s) for {cls.__name__}: {sorted(unknown)}. "
            f"Valid keys: {sorted(known)}")

    # Field annotations are strings under `from __future__ import annotations`,
    # so the real types are resolved once here and used for coercion below.
    hints = get_type_hints(cls)

    kwargs = {}
    for name in known:
        if name not in data:
            continue
        nested = _NESTED.get(cls, {}).get(name)
        if nested is not None:
            kwargs[name] = _build(nested, data[name])
        else:
            kwargs[name] = _coerce(data[name], hints.get(name, Any))
    return cls(**kwargs)


# Nested dataclasses that have no usable default instance to infer from, most
# often because the default is None. Keyed by parent, then field name.
_NESTED = {
    ModelsConfig: {"a": ModelConfig, "b": ModelConfig},
    CoderConfig: {"sparsity": SparsityConfig, "dead_features": DeadFeatureConfig},
    Config: {
        "dataset": DatasetConfig,
        "models": ModelsConfig,
        "layer_pairs": LayerPairConfig,
        "sweep": SweepConfig,
        "coder": CoderConfig,
        "activations": ActivationsConfig,
        "train": TrainConfig,
        "logging": LoggingConfig,
    },
}


def load_config(path: str | Path | None = None,
                overrides: list[str] | None = None) -> Config:
    """Build a Config from a YAML file and dotted command-line overrides.

    Args:
        path: YAML file, or None for pure defaults.
        overrides: strings of the form ``train.seed=3``. Values are parsed as
            YAML scalars, so ``true``, ``3``, ``1e-4`` and ``null`` all arrive
            with the right type, and quoting forces a string.

    Raises:
        ValueError: on an unknown key in either the file or an override, so a
            typo cannot silently have no effect.
    """
    data: dict = {}
    if path is not None:
        import yaml
        text = Path(path).read_text()
        data = yaml.safe_load(text) or {}

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override {item!r} is not of the form key=value")
        key, raw = item.split("=", 1)
        import yaml
        value = yaml.safe_load(raw)
        _assign(data, key.strip().split("."), value)

    cfg = _build(Config, data)
    cfg.validate()
    return cfg


def _assign(d: dict, path: list[str], value) -> None:
    """Set a nested key, creating intermediate dicts. Validated later by _build."""
    for part in path[:-1]:
        d = d.setdefault(part, {})
        if not isinstance(d, dict):
            raise ValueError(f"cannot descend into {'.'.join(path)}")
    d[path[-1]] = value


def dump_config(cfg: Config, path: str | Path) -> None:
    """Write the resolved configuration next to the run's outputs.

    The resolved form, not the source file, because overrides and defaults are
    both invisible in the source and both affect the result.
    """
    import yaml
    Path(path).write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
