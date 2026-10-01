#!/usr/bin/env python3
"""
Resource benchmark for sparse cross-coder stitching on vision transformers.

Measures, on the current GPU:
  1. Forward-pass memory and throughput for the backbone at several batch sizes
  2. Activation extraction throughput and per-image activation size
  3. Sparse coder training memory and step time at several dictionary widths
  4. Extrapolated storage and GPU-hour requirements for the full experiment grid

Usage:
    python benchmark.py
    python benchmark.py --model vit_base_patch16_224 --out results/
    python benchmark.py --quick
"""

import argparse
import json
import platform
import subprocess
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn


# ----------------------------------------------------------------------
# environment
# ----------------------------------------------------------------------

def gpu_info() -> dict:
    if not torch.cuda.is_available():
        return {"available": False}
    i = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(i)
    info = {
        "available": True,
        "name": props.name,
        "total_memory_gb": round(props.total_memory / 1024**3, 2),
        "capability": f"{props.major}.{props.minor}",
        "multiprocessors": props.multi_processor_count,
        "count": torch.cuda.device_count(),
    }
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True, timeout=10,
        )
        info["driver"] = out.strip().splitlines()[0]
    except Exception:
        pass
    return info


def env_info() -> dict:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": gpu_info(),
    }


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def reset_peak():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def peak_gb() -> float:
    return torch.cuda.max_memory_allocated() / 1024**3


def timed(fn, warmup: int = 3, iters: int = 10) -> float:
    """Mean seconds per call."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


# ----------------------------------------------------------------------
# the sparse coder under test
# ----------------------------------------------------------------------

class SparseCoder(nn.Module):
    """TopK sparse coder. Encoder reads d_in, decoder writes d_out."""

    def __init__(self, d_in: int, d_out: int, width: int, k: int):
        super().__init__()
        self.enc = nn.Linear(d_in, width)
        self.dec = nn.Linear(width, d_out, bias=False)
        self.k = k

    def forward(self, x):
        z = self.enc(x)
        vals, idx = torch.topk(z, self.k, dim=-1)
        z = torch.zeros_like(z).scatter_(-1, idx, torch.relu(vals))
        return self.dec(z), z


# ----------------------------------------------------------------------
# benchmarks
# ----------------------------------------------------------------------

@dataclass
class ForwardResult:
    batch_size: int
    peak_gb: float
    sec_per_batch: float
    images_per_sec: float


def bench_forward(model, batch_sizes, res, dtype, device) -> list:
    out = []
    for bs in batch_sizes:
        try:
            x = torch.randn(bs, 3, res, res, device=device, dtype=dtype)
            reset_peak()
            with torch.no_grad():
                t = timed(lambda: model(x))
            out.append(ForwardResult(bs, round(peak_gb(), 3), round(t, 4),
                                     round(bs / t, 1)))
            del x
        except torch.cuda.OutOfMemoryError:
            print(f"    batch {bs}: out of memory")
            torch.cuda.empty_cache()
            break
    return out


@dataclass
class CoderResult:
    expansion: int
    width: int
    k: int
    params_m: float
    peak_gb: float
    sec_per_step: float
    tokens_per_sec: float


def bench_coder(d_model, expansions, k_frac, token_batch, dtype, device) -> list:
    out = []
    for e in expansions:
        width = d_model * e
        k = max(1, int(width * k_frac))
        try:
            coder = SparseCoder(d_model, d_model, width, k).to(device=device, dtype=dtype)
            opt = torch.optim.Adam(coder.parameters(), lr=1e-4)
            x = torch.randn(token_batch, d_model, device=device, dtype=dtype)
            tgt = torch.randn(token_batch, d_model, device=device, dtype=dtype)

            def step():
                opt.zero_grad(set_to_none=True)
                recon, _ = coder(x)
                loss = ((recon - tgt) ** 2).mean()
                loss.backward()
                opt.step()

            reset_peak()
            t = timed(step, warmup=3, iters=10)
            params = sum(p.numel() for p in coder.parameters())
            out.append(CoderResult(e, width, k, round(params / 1e6, 2),
                                   round(peak_gb(), 3), round(t, 5),
                                   round(token_batch / t, 0)))
            del coder, opt, x, tgt
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print(f"    expansion {e}x: out of memory")
            torch.cuda.empty_cache()
            break
    return out


# ----------------------------------------------------------------------
# extrapolation
# ----------------------------------------------------------------------

def extrapolate(cfg, fwd, coder, n_tokens, d_model) -> dict:
    """Project storage and GPU-hours for the full grid."""
    best_fwd = max(fwd, key=lambda r: r.images_per_sec) if fwd else None
    ref = coder[len(coder) // 2] if coder else None

    bytes_per_el = 2 if cfg["dtype"] == "float16" else 4
    per_image_mb = n_tokens * d_model * bytes_per_el / 1024**2

    res = {
        "assumptions": {
            "images_in_dataset": cfg["n_images"],
            "tokens_per_image": n_tokens,
            "d_model": d_model,
            "cases": cfg["n_cases"],
            "layer_pairs": cfg["n_layer_pairs"],
            "seeds": cfg["n_seeds"],
            "coder_train_tokens": cfg["train_tokens"],
            "storage_dtype": cfg["dtype"],
        },
        "activation_storage": {
            "per_image_mb": round(per_image_mb, 3),
            "per_layer_gb": round(per_image_mb * cfg["n_images"] / 1024, 1),
            "note": "one cached layer, one model. Multiply by layers and by models cached.",
        },
    }

    if best_fwd:
        secs = cfg["n_images"] / best_fwd.images_per_sec
        res["activation_extraction"] = {
            "best_images_per_sec": best_fwd.images_per_sec,
            "at_batch_size": best_fwd.batch_size,
            "hours_per_full_pass": round(secs / 3600, 2),
        }

    if ref:
        steps = cfg["train_tokens"] / cfg["token_batch"]
        secs_per_run = steps * ref.sec_per_step
        n_runs = cfg["n_cases"] * cfg["n_layer_pairs"] * cfg["n_seeds"]
        res["coder_training"] = {
            "reference_expansion": ref.expansion,
            "sec_per_step": ref.sec_per_step,
            "steps_per_run": int(steps),
            "hours_per_run": round(secs_per_run / 3600, 2),
            "total_runs": n_runs,
            "total_gpu_hours": round(secs_per_run * n_runs / 3600, 1),
            "note": "runs = cases x layer pairs x seeds. Seeds are required for the noise floor.",
        }

    return res


# ----------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------

def render(env, fwd, coder, extra, model_name) -> str:
    L = []
    a = L.append
    a(f"# GPU benchmark: {model_name}\n")
    a(f"Run {env['timestamp']} on `{env['host']}`\n")

    g = env["gpu"]
    if g.get("available"):
        a(f"**GPU** {g['name']}, {g['total_memory_gb']} GB, "
          f"capability {g['capability']}, {g['count']} visible\n")
    else:
        a("**No CUDA device available.**\n")
    a(f"**Environment** torch {env['torch']}, CUDA {env['cuda']}, "
      f"python {env['python']}\n")

    a("\n## Backbone forward pass\n")
    a("| Batch | Peak GB | s/batch | images/s |")
    a("|---|---|---|---|")
    for r in fwd:
        a(f"| {r.batch_size} | {r.peak_gb} | {r.sec_per_batch} | {r.images_per_sec} |")

    a("\n## Sparse coder training step\n")
    a("| Expansion | Width | k | Params (M) | Peak GB | s/step | tokens/s |")
    a("|---|---|---|---|---|---|---|")
    for r in coder:
        a(f"| {r.expansion}x | {r.width} | {r.k} | {r.params_m} | "
          f"{r.peak_gb} | {r.sec_per_step} | {int(r.tokens_per_sec)} |")

    a("\n## Projected requirements\n")
    asm = extra["assumptions"]
    a(f"Grid: {asm['cases']} cases x {asm['layer_pairs']} layer pairs "
      f"x {asm['seeds']} seeds.\n")

    st = extra["activation_storage"]
    a(f"**Activation storage** {st['per_image_mb']} MB per image, "
      f"{st['per_layer_gb']} GB per cached layer at {asm['storage_dtype']}. "
      f"{st['note']}\n")

    if "activation_extraction" in extra:
        ex = extra["activation_extraction"]
        a(f"**Extraction** {ex['best_images_per_sec']} images/s at batch "
          f"{ex['at_batch_size']}, {ex['hours_per_full_pass']} h per full pass "
          f"over the dataset.\n")

    if "coder_training" in extra:
        ct = extra["coder_training"]
        a(f"**Coder training** {ct['hours_per_run']} h per run at "
          f"{ct['reference_expansion']}x expansion, {ct['total_runs']} runs, "
          f"**{ct['total_gpu_hours']} GPU-hours total**. {ct['note']}\n")

    return "\n".join(L)


# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="vit_base_patch16_224")
    p.add_argument("--batch-sizes", type=int, nargs="+",
                   default=[8, 16, 32, 64, 128, 256])
    p.add_argument("--expansions", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    p.add_argument("--k-frac", type=float, default=0.01,
                   help="active fraction of dictionary, sets TopK")
    p.add_argument("--token-batch", type=int, default=4096)
    p.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--n-images", type=int, default=1_281_167,
                   help="dataset size for extrapolation")
    p.add_argument("--n-cases", type=int, default=4)
    p.add_argument("--n-layer-pairs", type=int, default=6)
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--train-tokens", type=int, default=100_000_000)
    p.add_argument("--out", default="benchmarks")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.batch_sizes = [8, 32]
        args.expansions = [8, 16]

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device. Run this on a GPU node.")

    device = "cuda"
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    env = env_info()
    print(f"GPU: {env['gpu']['name']}, {env['gpu']['total_memory_gb']} GB\n")

    import timm
    print(f"Loading {args.model} ...")
    model = timm.create_model(args.model, pretrained=False).to(device=device, dtype=dtype).eval()
    cfg = model.default_cfg
    res = cfg["input_size"][-1]
    d_model = model.embed_dim
    patch = getattr(model.patch_embed, "patch_size", (16, 16))[0]
    n_tokens = (res // patch) ** 2 + 1
    print(f"  d_model={d_model}  tokens={n_tokens}  input={res}\n")

    print("Benchmarking forward pass ...")
    fwd = bench_forward(model, args.batch_sizes, res, dtype, device)
    for r in fwd:
        print(f"    batch {r.batch_size:>4}  {r.peak_gb:>6.2f} GB  "
              f"{r.images_per_sec:>8.1f} img/s")

    del model
    torch.cuda.empty_cache()

    print("\nBenchmarking sparse coder ...")
    coder = bench_coder(d_model, args.expansions, args.k_frac,
                        args.token_batch, dtype, device)
    for r in coder:
        print(f"    {r.expansion:>3}x  width {r.width:>6}  k={r.k:>4}  "
              f"{r.peak_gb:>6.2f} GB  {r.sec_per_step*1000:>7.2f} ms/step")

    extra = extrapolate(
        {"n_images": args.n_images, "n_cases": args.n_cases,
         "n_layer_pairs": args.n_layer_pairs, "n_seeds": args.n_seeds,
         "train_tokens": args.train_tokens, "token_batch": args.token_batch,
         "dtype": args.dtype},
        fwd, coder, n_tokens, d_model)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    payload = {
        "env": env, "model": args.model, "d_model": d_model,
        "n_tokens": n_tokens, "args": vars(args),
        "forward": [asdict(r) for r in fwd],
        "coder": [asdict(r) for r in coder],
        "projected": extra,
    }

    if args.quick:
        stamp = stamp + "_quick"
    (out / f"{stamp}.json").write_text(json.dumps(payload, indent=2))
    report = render(env, fwd, coder, extra, args.model)
    (out / f"{stamp}.md").write_text(report)

    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)
    print(f"\nWritten to {out}/{stamp}.json and {out}/{stamp}.md")


if __name__ == "__main__":
    main()