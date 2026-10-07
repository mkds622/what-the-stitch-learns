# Configuration reference

Every setting that distinguishes one run from another, what it does, and which
ones are still placeholders.

Updated whenever `stitch/config.py` changes. 
---

## How configuration is resolved

Three layers, lowest priority first:

1. dataclass defaults in `stitch/config.py`
2. a YAML file passed with `--config`
3. dotted overrides on the command line, `train.seed=3`

Later layers win. Only keys that already exist may be set, so a misspelled key
raises at load time rather than silently doing nothing.

```bash
python scripts/smoke_plumbing.py --config configs/case1_smoke.yaml train.seed=3 coder.expansion=32
```

The resolved configuration, not the source file, is written to
`runs/<run>/config.yaml` and logged as run parameters. Defaults and overrides
are both invisible in the source file and both affect the result.

### Scientific notation

`yaml.safe_load` implements YAML 1.1, which does not read `1e-4` as a number.
Values are therefore coerced to their declared field type on load, so `lr: 1e-4`
and `lr: 1.0e-4` both give a float. Without that coercion the first form would
arrive as the string `"1e-4"` and fail much later.

---

## Run identity

`Config.hash()` is a SHA-256 of the resolved configuration, truncated to 12
characters. Two runs with identical settings hash identically. The seed is part
of the configuration, so runs differing only by seed hash differently.

`Config.run_name()` produces `case1-smoke-x16-l1-s0-8af469`: objective, tag,
expansion, sparsity mode, seed, hash prefix.

---

## Sections

### `objective`

`case1` or `case2`. The only field that distinguishes the two cases; everything
else follows from it through validation.

| Value | Coder reads | Coder reproduces |
|---|---|---|
| `case1` | model A at layer ℓ | model A's activation at the same layer |
| `case2` | model A at layer ℓ | model B's activation at the matched layer |

Cases 3 and 4 train through a downstream stack and will need a third value plus
a merge path. Not present yet.

### `dataset`

| Key | Default | Meaning |
|---|---|---|
| `name` | `imagenet-100` | Which dataset |
| `root` | `""` | Absolute path on the machine that runs this |
| `source_url` | `""` | Where the data came from. Provenance, not used by code |
| `split` / `eval_split` | `train` / `val` | Split names |
| `num_classes` | `null` | Read from the dataset when unset |
| `notes` | `""` | Free text, e.g. which published ImageNet-100 subset |
| `num_workers` | `8` | Loader processes. 0 runs in the main process |
| `pin_memory` | `true` | Ignored when the device is CPU |

`root` and `source_url` are separate on purpose. The first is what the code
opens; the second is what lets someone else reproduce the run on a different
filesystem.

### `models`

`models.a` is always required. `models.b` is `null` for case 1 and required from
case 2 on.

| Key | Default | Meaning |
|---|---|---|
| `name` | `vit_base_patch16_224` | timm identifier |
| `pretrained` | `true` | Use timm's weights |
| `checkpoint` | `null` | Path to a local state dict, overriding timm |
| `label` | `A` / `B` | Short tag used in layer keys and run names |

`checkpoint` matters for a pair trained locally that differs only by seed, which
timm cannot supply.

### `layer_pairs`

| Key | Default | Meaning |
|---|---|---|
| `count` | `3` | How many of the ranked pairs to train on |
| `ranking_file` | `null` | Written by the sweep, read by the training grid |
| `matched_depth_only` | `true` | Block *i* of A against block *i* of B |
| `explicit` | `null` | Bypasses the ranking entirely |

`count` takes the top N from the sweep ranking. The full ranking is still
written, so `explicit` is the escape hatch when the viability curve suggests
something other than the top N.

`count` drives the whole downstream budget: three pairs at five seeds is 15 runs
per case, six is 30.

### `sweep`

Stage 3. Fits a linear stitch in closed form at each candidate depth and ranks
them, so that depths where no coder could succeed are excluded before any
training happens. A dictionary learned at a depth that cannot be stitched
describes nothing.

| Key | Default | Meaning |
|---|---|---|
| `depths` | `null` | `null` means every transformer block |
| `metric` | `explained_variance` | Or `accuracy`, which needs a merged model |
| `fit_tokens` | `500000` | Tokens used for the least-squares fit |
| `eval_images` | `10000` | Images used when `metric` is `accuracy` |
| `out_file` | `sweeps/layer_sweep.json` | Where the ranking goes |

Breadth is cheap here. Sweeping 144 pairs instead of 12 costs minutes under
`explained_variance` and about an hour under `accuracy`. What costs is `count`.

### `coder`

| Key | Default | Meaning |
|---|---|---|
| `expansion` | `16` | Dictionary width as a multiple of input width |
| `tie_decoder_init` | `true` | Initialise decoder as the encoder's transpose |

Input and output widths are not configured. They are read from the models, so a
cross-coder writing into a differently shaped space needs no extra setting.

#### `coder.sparsity`

| Key | Default | Meaning |
|---|---|---|
| `mode` | `l1` | `l1` or `topk` |
| `coeff` | `1e-3` | L1 penalty weight. Ignored under `topk` |
| `k` | `null` | Exact active count. Ignored under `l1` |
| `k_frac` | `0.01` | Fallback for `k`, as a fraction of width |
| `normalize_decoder` | `true` | Unit-norm decoder columns |

Both mechanisms are supported because they answer different needs. TopK fixes
the active count exactly, which is what the benchmarks use and what makes them
reproducible. L1 lets the active count emerge from a penalty, which is the
regime the training runs use; under L1 the resulting L0 is a measurement, not a
setting.

#### `coder.dead_features`

| Key | Default | Meaning |
|---|---|---|
| `track_window` | `10000` | Steps over which "never fired" is judged |
| `resample` | `false` | Reinitialise dead slots from high-error inputs |
| `resample_every` | `25000` | Cadence when resampling |
| `aux_loss_coeff` | `0.0` | Auxiliary reconstruction from dead slots only |

A slot that never activates contributes nothing and inflates the apparent
dictionary size, which corrupts any comparison between dictionaries.

### `activations`

| Key | Default | Meaning |
|---|---|---|
| `mode` | `on_the_fly` | Or `cached` |
| `cache_dir` | `null` | Required when `mode` is `cached` |
| `layers` | `["A@blocks.6"]` | What to extract, `<label>@<node name>` |
| `extraction_batch_size` | `128` | Measured optimum for ViT-B/16 |
| `shuffle_buffer_tokens` | `2000000` | Token-level shuffle buffer |
| `cache_dtype` | `float16` | Two bytes per value; `float32` doubles storage |

`mode` selects between two implementations of one interface, so the training
loop is identical either way:

- **`on_the_fly`** keeps the backbones resident and runs them every step. No
  storage, but extraction becomes the throughput bottleneck.
- **`cached`** writes activations once and streams them from disk. Faster if the
  volume sustains the required read rate, slower if it does not.

`scripts/benchmark_case2.py` measures the thresholds on a given machine. On the
reference figures, the cached path saturates at 1.12 GB/s for case 1 and
2.24 GB/s for case 2, and falls below the on-the-fly path entirely under about
0.66 GB/s.

`layers` is where case 2 enters. Case 1 lists one entry, case 2 lists two.
Nothing else in the extraction machinery changes between them.

### `train`

| Key | Default | Meaning |
|---|---|---|
| `tokens` | `1000000000` | Total tokens seen by the coder |
| `token_batch` | `4096` | Tokens per optimiser step |
| `lr` | `1e-4` | Learning rate |
| `optimizer` | `adam` | Or `adamw` |
| `seed` | `0` | Initialisation and data order |
| `dtype` | `float16` | Or `float32` |
| `device` | `cuda` | `cpu` for smoke runs |
| `log_every` | `100` | Steps between metric writes |
| `eval_every` | `5000` | Steps between Tier 0 evaluations |
| `checkpoint_every` | `25000` | Steps between checkpoints |
| `grad_clip` | `1.0` | Gradient norm clip, `null` to disable |

Tokens rather than epochs, because the coder consumes tokens and the tokens per
image depend on the backbone.

Seeds are not optional. The comparison between objectives is only interpretable
against the spread between runs that differ by initialisation alone.

### `logging`

| Key | Default | Meaning |
|---|---|---|
| `backend` | `jsonl` | Or `mlflow` |
| `tracking_uri` | `file:./mlruns` | Local store, or a server URL |
| `experiment` | `stitch` | MLflow experiment name |
| `run_root` | `runs` | Per-run directory for logs and artifacts |
| `level` | `INFO` | Log threshold |
| `csv_mirror` | `true` | Also write `metrics.csv`, for plotting |

The process log and crash capture are written to the run directory regardless of
backend, so a crash stays diagnosable even when the tracking backend is what
failed.

---

## Run directory layout

```
runs/<run-name>/
  config.yaml      the resolved configuration
  params.json      flattened parameters, as logged
  metrics.jsonl    one object per log_metrics call
  metrics.csv      the same as a table
  status.json      RUNNING, FINISHED, FAILED or KILLED
  run.log          the process log
  faults.log       C-level faults, from faulthandler
  crash.json       written only on failure
  artifacts/       checkpoints, the config, the process log

crashes/
  <timestamp>_<run>.json    every failure across every run, in one place
```

A process killed outright, by the OOM killer or SIGKILL, writes nothing. That
case shows as a `status.json` still saying RUNNING with no process behind the
recorded pid.

---

## Pending decisions

Placeholder values, printed as warnings at startup so a result is never quietly
produced under a value nobody chose. 

| Setting | Placeholder | What decides it |
|---|---|---|
| `models.b` | unset | Two ViT-B/16 differing by seed, or two architectures |
| `layer_pairs.count` | `3` | How many pairs are defensible; drives the whole budget |
| `layer_pairs.matched_depth_only` | `true` | Whether off-diagonal pairs are in scope |
| `activations.shuffle_buffer_tokens` | `2000000` | Shuffling convention, and the disk read pattern it implies |
| `train.tokens` | `1e9` | Largest single lever on total GPU-hours |
| `dataset.root`, `dataset.name` | empty, `imagenet-100` | Full ImageNet or ImageNet-100; decides whether caching is viable |
| `logging.tracking_uri` | local | tracking server uri |

Grep `PENDING()` in `stitch/config.py` for the same list in context.
