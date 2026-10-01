#!/usr/bin/env python3
"""
Resource benchmark for case 2: cross-coding from model A to model B.

Case 2 trains the coder to reproduce model B's activation at the matched layer
while still reading model A only. Compared with case 1 the coder is unchanged
in compute, so what actually differs is data supply:

  - two backbones resident during extraction rather than one
  - two cached tensors per token rather than one, input and target
  - twice the sustained disk read to keep the coder fed

Measures, on the current GPU and the current filesystem:
  1. Forward-pass memory and throughput with both backbones resident
  2. Sparse coder training step with a cross target, d_in from A, d_out from B
  3. Sustained disk write and read throughput on the cache volume
  4. Where the cached path saturates, and whether this machine reaches it

Usage:
    python benchmark_case2.py
    python benchmark_case2.py --quick
    python benchmark_case2.py --disk-only --scratch /scratch/kumar
    python benchmark_case2.py --model-b vit_base_patch16_224 --file-gb 32


IMPLEMENTATION NOTES
--------------------

Scope. This script covers only case 2. The projected grid is
layer pairs times seeds. 

Paired extraction. Case 2 must push every image through both models, A to
produce the coder's input and B to produce its target. bench_forward_pair
therefore times one backbone and then both at each batch size, and reports
the slowdown between them. The paired rate is the one that applies; the
single rate is kept only as the reference point.

Cross target. The coder reads model A and writes into model B's space, so
the encoder takes d_in from A and the decoder writes d_out from B. The
target is a separate tensor rather than the input, which is the only thing
that changes inside the training loop between case 1 and case 2. When the
two widths match, the arithmetic is identical to case 1 and this measurement
confirms it. When they differ, it gives the real figures. --model-b sets the second model.

Disk. Caching activations is what makes the cached path fast, so the volume
that would hold them is measured directly. Three read patterns, because the
training loop does not read the way a file copy does:

  sequential  the ceiling, and what writing the cache once will get
  strided     shuffled block order, which is how a token stream is actually
              consumed once the ordering is randomised for training
  paired      two interleaved streams, which is what case 2 does when it
              pulls A's activation and B's target for the same tokens

Each pattern is repeated and reported as min, median and max. On a shared
volume the spread matters as much as the median, since a run competes with
whatever else is using the filesystem.

Page cache. Read figures are meaningless if the file is still in RAM, so
each file is evicted with posix_fadvise(POSIX_FADV_DONTNEED) before reading.
That call is unreliable on networked filesystems and is refused outright on
some. When it fails the script records it and warns in the report, and the
test file must then exceed host RAM for the numbers to mean anything. Raise
--file-gb above the reported RAM size in that situation.

Saturation. The coder consumes tokens at a fixed rate set by the GPU.
Multiplying that rate by the bytes read per token gives the disk throughput
required to keep it fed: one tensor per token for case 1, two for case 2. At
or above that rate the run is GPU-bound and further bandwidth changes
nothing. Below it, the run time is set by the disk instead. A second
threshold, the crossover, is the rate below which reading the cache is
slower than recomputing activations during training, at which point caching
is not worth doing at all. Both thresholds are computed from measurements
taken in the same run rather than assumed.

Token correspondence. A token-wise stitch needs A and B to produce the same
number of tokens per image. The script warns when they differ, since the
storage and throughput figures then assume A's count and the correspondence
between the two models has to be defined before any of this is meaningful.

Output. Results are written as timestamped JSON and Markdown under --out,
tagged _case2, and further tagged _quick when --quick was passed, so reduced
runs are never mistaken for full ones. --disk-only skips every GPU
measurement and runs without torch, which allows the filesystem to be
measured from a node with no GPU.

Not measured. Dataloading and image decoding are excluded, as in the first
benchmark, so every throughput figure is a ceiling. Training through a
downstream stack is absent by design.
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


# ----------------------------------------------------------------------
# environment
# ----------------------------------------------------------------------

def gpu_info() -> dict:
    if not HAVE_TORCH or not torch.cuda.is_available():
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


def ram_gb() -> float:
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 1)
    except Exception:
        return 0.0


def fs_info(path: Path) -> dict:
    """Filesystem type, device and free space for the cache volume."""
    info = {"path": str(path)}
    try:
        st = os.statvfs(path)
        info["free_gb"] = round(st.f_bavail * st.f_frsize / 1024**3, 1)
        info["total_gb"] = round(st.f_blocks * st.f_frsize / 1024**3, 1)
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["df", "-PT", str(path)], text=True, timeout=10).splitlines()
        if len(out) > 1:
            f = out[1].split()
            info["device"] = f[0]
            info["fstype"] = f[1]
            info["mount"] = f[-1]
    except Exception:
        pass
    info["networked"] = info.get("fstype", "") in {
        "nfs", "nfs4", "cifs", "smb3", "lustre", "gpfs", "beegfs", "ceph", "fuse.sshfs"}
    return info


def env_info(scratch: Path) -> dict:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__ if HAVE_TORCH else None,
        "cuda": torch.version.cuda if HAVE_TORCH else None,
        "ram_gb": ram_gb(),
        "gpu": gpu_info(),
        "filesystem": fs_info(scratch),
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

class SparseCoder(nn.Module if HAVE_TORCH else object):
    """TopK sparse coder. Encoder reads d_in, decoder writes d_out.

    Case 1 sets d_out = d_in and the target is the input tensor.
    Case 2 sets d_out to model B's width and the target is B's activation.
    """

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
# GPU benchmarks
# ----------------------------------------------------------------------

@dataclass
class PairForwardResult:
    batch_size: int
    peak_gb_one: float
    peak_gb_pair: float
    sec_per_batch_one: float
    sec_per_batch_pair: float
    images_per_sec_one: float
    images_per_sec_pair: float


def bench_forward_pair(model_a, model_b, batch_sizes, res, dtype, device) -> list:
    """Extraction throughput with one backbone, then with both resident.

    Case 2 must run both models over the same image to produce the input and
    the target, so the relevant extraction rate is the paired one.
    """
    out = []
    for bs in batch_sizes:
        try:
            x = torch.randn(bs, 3, res, res, device=device, dtype=dtype)

            reset_peak()
            with torch.no_grad():
                t1 = timed(lambda: model_a(x))
            p1 = peak_gb()

            reset_peak()
            with torch.no_grad():
                def pair():
                    model_a(x)
                    model_b(x)
                t2 = timed(pair)
            p2 = peak_gb()

            out.append(PairForwardResult(
                bs, round(p1, 3), round(p2, 3),
                round(t1, 4), round(t2, 4),
                round(bs / t1, 1), round(bs / t2, 1)))
            del x
            torch.cuda.empty_cache()
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
    d_in: int
    d_out: int
    params_m: float
    peak_gb: float
    sec_per_step: float
    tokens_per_sec: float


def bench_coder_cross(d_a, d_b, expansions, k_frac, token_batch, dtype, device) -> list:
    """Coder training step with a cross target.

    The target is a separate tensor rather than the input, which is the only
    thing that changes between case 1 and case 2 inside the training loop.
    When d_a == d_b the arithmetic is identical to case 1 and this confirms it.
    """
    out = []
    for e in expansions:
        width = d_a * e
        k = max(1, int(width * k_frac))
        try:
            coder = SparseCoder(d_a, d_b, width, k).to(device=device, dtype=dtype)
            opt = torch.optim.Adam(coder.parameters(), lr=1e-4)
            x = torch.randn(token_batch, d_a, device=device, dtype=dtype)
            tgt = torch.randn(token_batch, d_b, device=device, dtype=dtype)

            def step():
                opt.zero_grad(set_to_none=True)
                recon, _ = coder(x)
                loss = ((recon - tgt) ** 2).mean()
                loss.backward()
                opt.step()

            reset_peak()
            t = timed(step, warmup=3, iters=10)
            params = sum(p.numel() for p in coder.parameters())
            out.append(CoderResult(e, width, k, d_a, d_b, round(params / 1e6, 2),
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
# disk benchmark
# ----------------------------------------------------------------------

def drop_from_cache(fd, size: int) -> bool:
    """Evict a file from the page cache without root.

    Returns False when the call is unavailable or refused, which is common on
    networked filesystems. A False here means the read figures may be measuring
    RAM rather than the device, so the file must exceed RAM to be trusted.
    """
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, size, os.POSIX_FADV_DONTNEED)
        return True
    except (AttributeError, OSError):
        return False


@dataclass
class DiskResult:
    pattern: str
    gb_per_sec: float
    seconds: float
    block_mb: float


def bench_disk(scratch: Path, file_gb: float, block_mb: float, repeats: int,
               token_bytes: int) -> dict:
    """Sustained write and read throughput on the cache volume.

    Three patterns are measured because the training loop does not read the way
    a copy does:
      sequential  the ceiling, and what caching the activations once will get
      strided     a shuffled-buffer read, which is how a token stream is
                  actually consumed once the order is randomised
      paired      two sequential streams interleaved, which is what case 2 does
                  when it pulls A's activation and B's target together
    """
    scratch.mkdir(parents=True, exist_ok=True)
    nbytes = int(file_gb * 1024**3)
    block = int(block_mb * 1024**2)
    path_a = scratch / ".bench_case2_a.bin"
    path_b = scratch / ".bench_case2_b.bin"
    res = {"file_gb": file_gb, "block_mb": block_mb, "repeats": repeats,
           "write": [], "read": [], "cache_dropped": True}

    chunk = os.urandom(block)

    try:
        # ---- write ----
        for path in (path_a, path_b):
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
            try:
                t0 = time.perf_counter()
                written = 0
                while written < nbytes:
                    n = min(block, nbytes - written)
                    written += os.write(fd, chunk[:n])
                os.fsync(fd)
                dt = time.perf_counter() - t0
            finally:
                os.close(fd)
            res["write"].append(asdict(DiskResult(
                "sequential", round(written / dt / 1e9, 3), round(dt, 2), block_mb)))

        # ---- sequential read ----
        for _ in range(repeats):
            fd = os.open(path_a, os.O_RDONLY)
            try:
                if not drop_from_cache(fd, nbytes):
                    res["cache_dropped"] = False
                t0 = time.perf_counter()
                read = 0
                while True:
                    b = os.read(fd, block)
                    if not b:
                        break
                    read += len(b)
                dt = time.perf_counter() - t0
            finally:
                os.close(fd)
            res["read"].append(asdict(DiskResult(
                "sequential", round(read / dt / 1e9, 3), round(dt, 2), block_mb)))

        # ---- strided read, shuffled block order ----
        import random
        nblocks = nbytes // block
        order = list(range(nblocks))
        random.Random(0).shuffle(order)
        for _ in range(repeats):
            fd = os.open(path_a, os.O_RDONLY)
            try:
                if not drop_from_cache(fd, nbytes):
                    res["cache_dropped"] = False
                t0 = time.perf_counter()
                read = 0
                for i in order:
                    os.lseek(fd, i * block, os.SEEK_SET)
                    read += len(os.read(fd, block))
                dt = time.perf_counter() - t0
            finally:
                os.close(fd)
            res["read"].append(asdict(DiskResult(
                "strided", round(read / dt / 1e9, 3), round(dt, 2), block_mb)))

        # ---- paired read, two interleaved streams, the case 2 pattern ----
        for _ in range(repeats):
            fa = os.open(path_a, os.O_RDONLY)
            fb = os.open(path_b, os.O_RDONLY)
            try:
                if not (drop_from_cache(fa, nbytes) and drop_from_cache(fb, nbytes)):
                    res["cache_dropped"] = False
                t0 = time.perf_counter()
                read = 0
                while True:
                    ba = os.read(fa, block)
                    bb = os.read(fb, block)
                    if not ba and not bb:
                        break
                    read += len(ba) + len(bb)
                dt = time.perf_counter() - t0
            finally:
                os.close(fa)
                os.close(fb)
            res["read"].append(asdict(DiskResult(
                "paired", round(read / dt / 1e9, 3), round(dt, 2), block_mb)))

    finally:
        for p in (path_a, path_b):
            try:
                p.unlink()
            except OSError:
                pass

    def summarise(rows, pattern):
        v = sorted(r["gb_per_sec"] for r in rows if r["pattern"] == pattern)
        if not v:
            return None
        return {"min": v[0], "median": v[len(v) // 2], "max": v[-1], "n": len(v)}

    res["summary"] = {
        "write_sequential": summarise(res["write"], "sequential"),
        "read_sequential": summarise(res["read"], "sequential"),
        "read_strided": summarise(res["read"], "strided"),
        "read_paired": summarise(res["read"], "paired"),
    }
    res["tokens_per_sec_at_measured_rate"] = {
        k: (None if s is None else int(s["median"] * 1e9 / token_bytes))
        for k, s in res["summary"].items() if k.startswith("read")
    }
    return res


# ----------------------------------------------------------------------
# saturation
# ----------------------------------------------------------------------

def saturation(coder_tokens_per_sec: float, d_a: int, d_b: int, bytes_per_el: int,
               disk: dict, extract_one: float, extract_pair: float,
               n_tokens_per_image: int, train_tokens: int) -> dict:
    """Where the cached path stops being disk-bound, and where this machine sits.

    The coder consumes tokens at a fixed rate. Case 1 reads one tensor per
    token, case 2 reads two. Dividing one by the other gives the read rate
    required to keep the coder fed. Above it the run is GPU-bound and more
    bandwidth buys nothing. Below it the run time is set by the disk.
    """
    b1 = d_a * bytes_per_el
    b2 = (d_a + d_b) * bytes_per_el

    sat1 = coder_tokens_per_sec * b1 / 1e9
    sat2 = coder_tokens_per_sec * b2 / 1e9

    # on-the-fly alternatives, for the crossover
    ext1 = extract_one * n_tokens_per_image if extract_one else None
    ext2 = extract_pair * n_tokens_per_image if extract_pair else None

    def run_min(rate):
        return None if not rate else round(train_tokens / rate / 60, 1)

    def cached_rate(gbs, bpt):
        return min(gbs * 1e9 / bpt, coder_tokens_per_sec)

    out = {
        "bytes_per_token": {"case_1": b1, "case_2": b2},
        "coder_tokens_per_sec": int(coder_tokens_per_sec),
        "saturation_gb_per_sec": {"case_1": round(sat1, 2), "case_2": round(sat2, 2)},
        "gpu_bound_run_min": run_min(coder_tokens_per_sec),
        "on_the_fly": {
            "extract_tokens_per_sec_one": None if ext1 is None else int(ext1),
            "extract_tokens_per_sec_pair": None if ext2 is None else int(ext2),
            "case_1_overlapped_min": run_min(min(ext1, coder_tokens_per_sec)) if ext1 else None,
            "case_2_overlapped_min": run_min(min(ext2, coder_tokens_per_sec)) if ext2 else None,
            "case_1_serial_min": None if not ext1 else round(
                (train_tokens / ext1 + train_tokens / coder_tokens_per_sec) / 60, 1),
            "case_2_serial_min": None if not ext2 else round(
                (train_tokens / ext2 + train_tokens / coder_tokens_per_sec) / 60, 1),
        },
    }

    # crossover: the read rate below which caching is slower than on-the-fly
    if ext1:
        out["crossover_gb_per_sec"] = {
            "case_1": round(min(ext1, coder_tokens_per_sec) * b1 / 1e9, 2),
            "case_2": round(min(ext2, coder_tokens_per_sec) * b2 / 1e9, 2) if ext2 else None,
        }

    # where this machine actually lands
    measured = {}
    for key, s in (disk.get("summary") or {}).items():
        if not key.startswith("read") or s is None:
            continue
        gbs = s["median"]
        measured[key] = {
            "gb_per_sec": gbs,
            "case_1_cached_min": run_min(cached_rate(gbs, b1)),
            "case_2_cached_min": run_min(cached_rate(gbs, b2)),
            "case_1_saturated": gbs >= sat1,
            "case_2_saturated": gbs >= sat2,
        }
    out["measured"] = measured
    return out


# ----------------------------------------------------------------------
# extrapolation
# ----------------------------------------------------------------------

def extrapolate(cfg, coder, pair_fwd, d_a, d_b, n_tokens) -> dict:
    """Project storage and GPU-hours for case 2 alone.

    Scoped deliberately. This covers case 2 and nothing else, so runs are
    layer pairs times seeds. Cases 3 and 4 add a downstream stack to the
    training loop and are not represented here.
    """
    ref = coder[len(coder) // 2] if coder else None
    bpe = 2 if cfg["dtype"] == "float16" else 4

    per_image_in = n_tokens * d_a * bpe / 1024**2
    per_image_tg = n_tokens * d_b * bpe / 1024**2

    res = {
        "scope": "case 2 only. Cases 3 and 4 train through a downstream stack "
                 "and are not covered.",
        "assumptions": {
            "images_in_dataset": cfg["n_images"],
            "tokens_per_image": n_tokens,
            "d_model_a": d_a,
            "d_model_b": d_b,
            "layer_pairs": cfg["n_layer_pairs"],
            "seeds": cfg["n_seeds"],
            "coder_train_tokens": cfg["train_tokens"],
            "storage_dtype": cfg["dtype"],
        },
        "activation_storage": {
            "input_per_layer_gib": round(per_image_in * cfg["n_images"] / 1024, 1),
            "target_per_layer_gib": round(per_image_tg * cfg["n_images"] / 1024, 1),
            "total_per_layer_pair_gib": round(
                (per_image_in + per_image_tg) * cfg["n_images"] / 1024, 1),
            "note": "case 2 caches A's activation as input and B's as target, "
                    "so one layer pair costs both.",
        },
    }

    best = max(pair_fwd, key=lambda r: r.images_per_sec_pair) if pair_fwd else None
    if best:
        res["extraction"] = {
            "best_images_per_sec_pair": best.images_per_sec_pair,
            "best_images_per_sec_one": best.images_per_sec_one,
            "at_batch_size": best.batch_size,
            "pair_slowdown": round(
                best.images_per_sec_one / best.images_per_sec_pair, 2),
            "minutes_per_full_pass": round(
                cfg["n_images"] / best.images_per_sec_pair / 60, 1),
            "peak_gb_pair": best.peak_gb_pair,
            "note": "one pass captures every chosen layer of both models at once.",
        }

    if ref:
        steps = cfg["train_tokens"] / cfg["token_batch"]
        secs = steps * ref.sec_per_step
        n_runs = cfg["n_layer_pairs"] * cfg["n_seeds"]
        res["coder_training"] = {
            "reference_expansion": ref.expansion,
            "sec_per_step": ref.sec_per_step,
            "steps_per_run": int(steps),
            "minutes_per_run_gpu_bound": round(secs / 60, 1),
            "total_runs": n_runs,
            "total_gpu_hours": round(secs * n_runs / 3600, 1),
            "note": "runs = layer pairs x seeds, case 2 only. Assumes the disk "
                    "keeps the coder fed; see the saturation section.",
        }
    return res


# ----------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------

def render(env, pair_fwd, coder, disk, sat, extra, name_a, name_b) -> str:
    L = []
    a = L.append
    a(f"# Case 2 benchmark: {name_a} to {name_b}\n")
    a(f"Run {env['timestamp']} on `{env['host']}`\n")

    g = env["gpu"]
    if g.get("available"):
        a(f"**GPU** {g['name']}, {g['total_memory_gb']} GB, "
          f"capability {g['capability']}, {g['count']} visible\n")
    else:
        a("**No CUDA device. Disk figures only.**\n")
    if env.get("torch"):
        a(f"**Environment** torch {env['torch']}, CUDA {env['cuda']}, "
          f"python {env['python']}, {env['ram_gb']} GB RAM\n")

    fsx = env["filesystem"]
    a(f"**Cache volume** `{fsx.get('path')}` on {fsx.get('device','?')} "
      f"({fsx.get('fstype','?')}), {fsx.get('free_gb','?')} GB free"
      + (", networked" if fsx.get("networked") else "") + "\n")

    if pair_fwd:
        a("\n## Extraction, one backbone against two\n")
        a("Case 2 runs both models over the same image, so the paired rate is "
          "the one that applies.\n")
        a("| Batch | Peak GB one | Peak GB pair | img/s one | img/s pair | slowdown |")
        a("|---|---|---|---|---|---|")
        for r in pair_fwd:
            sd = r.images_per_sec_one / r.images_per_sec_pair if r.images_per_sec_pair else 0
            a(f"| {r.batch_size} | {r.peak_gb_one} | {r.peak_gb_pair} | "
              f"{r.images_per_sec_one} | {r.images_per_sec_pair} | {sd:.2f}x |")

    if coder:
        a("\n## Sparse coder training step, cross target\n")
        a("| Expansion | Width | k | d_in | d_out | Params (M) | Peak GB | s/step | tokens/s |")
        a("|---|---|---|---|---|---|---|---|---|")
        for r in coder:
            a(f"| {r.expansion}x | {r.width} | {r.k} | {r.d_in} | {r.d_out} | "
              f"{r.params_m} | {r.peak_gb} | {r.sec_per_step} | {int(r.tokens_per_sec)} |")

    if disk:
        a("\n## Disk throughput\n")
        if not disk.get("cache_dropped", True):
            a("> Page cache could not be evicted. If the test file is smaller "
              "than RAM these figures measure memory, not the device. Rerun "
              "with `--file-gb` above the host RAM size.\n")
        a(f"Test file {disk['file_gb']} GB per stream, {disk['block_mb']} MB "
          f"blocks, {disk['repeats']} repeats.\n")
        a("| Pattern | Min GB/s | Median GB/s | Max GB/s |")
        a("|---|---|---|---|")
        for key, s in (disk.get("summary") or {}).items():
            if s:
                a(f"| {key.replace('_',' ')} | {s['min']} | {s['median']} | {s['max']} |")
        a("\nSpread between min and max matters as much as the median on shared "
          "storage, since a run competes with whatever else is on the volume.\n")

    if sat:
        a("\n## Saturation\n")
        s1 = sat["saturation_gb_per_sec"]["case_1"]
        s2 = sat["saturation_gb_per_sec"]["case_2"]
        bt = sat["bytes_per_token"]
        a(f"The coder consumes {sat['coder_tokens_per_sec']:,} tokens/s. "
          f"Case 1 reads {bt['case_1']} bytes per token and case 2 reads "
          f"{bt['case_2']}.\n")
        a(f"**Saturation** {s1} GB/s for case 1, {s2} GB/s for case 2. "
          f"At or above, the run is GPU-bound at "
          f"{sat['gpu_bound_run_min']} min and more bandwidth buys nothing.\n")
        if "crossover_gb_per_sec" in sat:
            c = sat["crossover_gb_per_sec"]
            a(f"**Crossover** {c['case_1']} GB/s for case 1, {c['case_2']} GB/s "
              f"for case 2. Below it, caching is slower than recomputing "
              f"activations during training and there is no reason to store them.\n")
        if sat.get("measured"):
            a("\n| Read pattern | GB/s | Case 1 run | Case 2 run | Case 1 fed | Case 2 fed |")
            a("|---|---|---|---|---|---|")
            for key, m in sat["measured"].items():
                a(f"| {key.replace('read_','')} | {m['gb_per_sec']} | "
                  f"{m['case_1_cached_min']} min | {m['case_2_cached_min']} min | "
                  f"{'yes' if m['case_1_saturated'] else 'no'} | "
                  f"{'yes' if m['case_2_saturated'] else 'no'} |")
        otf = sat["on_the_fly"]
        if otf.get("case_2_overlapped_min"):
            a(f"\nFor comparison, computing activations during training: case 1 "
              f"{otf['case_1_overlapped_min']} to {otf['case_1_serial_min']} min, "
              f"case 2 {otf['case_2_overlapped_min']} to "
              f"{otf['case_2_serial_min']} min, depending on whether extraction "
              f"and training overlap.\n")

    if extra:
        a("\n## Projected requirements, case 2 only\n")
        asm = extra["assumptions"]
        a(f"Grid: {asm['layer_pairs']} layer pairs x {asm['seeds']} seeds. "
          f"{extra['scope']}\n")
        st = extra["activation_storage"]
        a(f"**Storage** {st['input_per_layer_gib']} GiB input plus "
          f"{st['target_per_layer_gib']} GiB target, "
          f"**{st['total_per_layer_pair_gib']} GiB per layer pair** at "
          f"{asm['storage_dtype']}.\n")
        if "extraction" in extra:
            ex = extra["extraction"]
            a(f"**Extraction** {ex['best_images_per_sec_pair']} images/s paired "
              f"at batch {ex['at_batch_size']}, {ex['pair_slowdown']}x slower "
              f"than a single backbone, {ex['minutes_per_full_pass']} min per "
              f"full pass, {ex['peak_gb_pair']} GB peak.\n")
        if "coder_training" in extra:
            ct = extra["coder_training"]
            a(f"**Coder training** {ct['minutes_per_run_gpu_bound']} min per run "
              f"at {ct['reference_expansion']}x, {ct['total_runs']} runs, "
              f"**{ct['total_gpu_hours']} GPU-hours**. {ct['note']}\n")

    return "\n".join(L)


# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-a", default="vit_base_patch16_224")
    p.add_argument("--model-b", default="vit_base_patch16_224")
    p.add_argument("--batch-sizes", type=int, nargs="+",
                   default=[8, 16, 32, 64, 128, 256])
    p.add_argument("--expansions", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    p.add_argument("--k-frac", type=float, default=0.01,
                   help="active fraction of dictionary, sets TopK")
    p.add_argument("--token-batch", type=int, default=4096)
    p.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--n-images", type=int, default=1_281_167,
                   help="dataset size for extrapolation")
    p.add_argument("--n-layer-pairs", type=int, default=6)
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--train-tokens", type=int, default=1_000_000_000)
    p.add_argument("--scratch", default=None,
                   help="directory on the volume that would hold cached "
                        "activations. Defaults to --out.")
    p.add_argument("--file-gb", type=float, default=8.0,
                   help="size of each disk test file. Should exceed host RAM "
                        "when the page cache cannot be evicted.")
    p.add_argument("--block-mb", type=float, default=8.0)
    p.add_argument("--disk-repeats", type=int, default=3)
    p.add_argument("--skip-disk", action="store_true")
    p.add_argument("--disk-only", action="store_true",
                   help="skip every GPU measurement. Runs without torch.")
    p.add_argument("--out", default="benchmarks")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.batch_sizes = [8, 32]
        args.expansions = [8, 16]
        args.file_gb = 1.0
        args.disk_repeats = 2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scratch = Path(args.scratch) if args.scratch else out
    env = env_info(scratch)

    pair_fwd, coder, disk, sat, extra = [], [], None, None, None
    d_a = d_b = None
    n_tokens = None

    if not args.disk_only:
        if not HAVE_TORCH:
            raise SystemExit("torch not importable. Use --disk-only to run the "
                             "filesystem measurements alone.")
        if not torch.cuda.is_available():
            raise SystemExit("No CUDA device. Run this on a GPU node, or pass "
                             "--disk-only.")

        device = "cuda"
        dtype = torch.float16 if args.dtype == "float16" else torch.float32
        print(f"GPU: {env['gpu']['name']}, {env['gpu']['total_memory_gb']} GB\n")

        import timm
        print(f"Loading {args.model_a} and {args.model_b} ...")
        model_a = timm.create_model(args.model_a, pretrained=False).to(
            device=device, dtype=dtype).eval()
        model_b = timm.create_model(args.model_b, pretrained=False).to(
            device=device, dtype=dtype).eval()

        cfg_a = model_a.default_cfg
        res = cfg_a["input_size"][-1]
        d_a = model_a.embed_dim
        d_b = model_b.embed_dim
        patch = getattr(model_a.patch_embed, "patch_size", (16, 16))[0]
        n_tokens = (res // patch) ** 2 + 1
        tb = getattr(model_b.patch_embed, "patch_size", (16, 16))[0]
        nb = (model_b.default_cfg["input_size"][-1] // tb) ** 2 + 1
        print(f"  A: d={d_a} tokens={n_tokens}   B: d={d_b} tokens={nb}\n")
        if nb != n_tokens:
            print("  WARNING: token counts differ between A and B. A token-wise "
                  "stitch needs a defined correspondence; storage and "
                  "throughput figures below assume A's count.\n")

        print("Benchmarking extraction, one backbone then two ...")
        pair_fwd = bench_forward_pair(model_a, model_b, args.batch_sizes,
                                      res, dtype, device)
        for r in pair_fwd:
            print(f"    batch {r.batch_size:>4}  one {r.images_per_sec_one:>8.1f} "
                  f"img/s  pair {r.images_per_sec_pair:>8.1f} img/s  "
                  f"{r.peak_gb_pair:>5.2f} GB")

        del model_a, model_b
        torch.cuda.empty_cache()

        print("\nBenchmarking sparse coder with cross target ...")
        coder = bench_coder_cross(d_a, d_b, args.expansions, args.k_frac,
                                  args.token_batch, dtype, device)
        for r in coder:
            print(f"    {r.expansion:>3}x  width {r.width:>6}  k={r.k:>4}  "
                  f"{r.peak_gb:>6.2f} GB  {r.sec_per_step*1000:>7.2f} ms/step")

    bpe = 2 if args.dtype == "float16" else 4

    if not args.skip_disk:
        print(f"\nBenchmarking disk on {scratch} "
              f"({args.file_gb} GB x 2 files) ...")
        tok_bytes = (d_a or 768) * bpe
        disk = bench_disk(scratch, args.file_gb, args.block_mb,
                          args.disk_repeats, tok_bytes)
        for key, s in (disk.get("summary") or {}).items():
            if s:
                print(f"    {key:<18} {s['median']:>6.2f} GB/s median "
                      f"({s['min']:.2f} to {s['max']:.2f})")
        if not disk.get("cache_dropped", True):
            print("    WARNING: page cache not evicted. Raise --file-gb above "
                  f"{env['ram_gb']} GB RAM to trust these numbers.")

    if coder and disk:
        ref = coder[len(coder) // 2]
        best = max(pair_fwd, key=lambda r: r.images_per_sec_pair) if pair_fwd else None
        sat = saturation(
            ref.tokens_per_sec, d_a, d_b, bpe, disk,
            best.images_per_sec_one if best else None,
            best.images_per_sec_pair if best else None,
            n_tokens, args.train_tokens)

    if coder:
        extra = extrapolate(
            {"n_images": args.n_images, "n_layer_pairs": args.n_layer_pairs,
             "n_seeds": args.n_seeds, "train_tokens": args.train_tokens,
             "token_batch": args.token_batch, "dtype": args.dtype},
            coder, pair_fwd, d_a, d_b, n_tokens)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{stamp}_case2" + ("_quick" if args.quick else "")

    payload = {
        "env": env, "model_a": args.model_a, "model_b": args.model_b,
        "d_model_a": d_a, "d_model_b": d_b, "n_tokens": n_tokens,
        "args": vars(args),
        "forward_pair": [asdict(r) for r in pair_fwd],
        "coder": [asdict(r) for r in coder],
        "disk": disk,
        "saturation": sat,
        "projected": extra,
    }
    (out / f"{tag}.json").write_text(json.dumps(payload, indent=2))
    report = render(env, pair_fwd, coder, disk, sat, extra,
                    args.model_a, args.model_b)
    (out / f"{tag}.md").write_text(report)

    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)
    print(f"\nWritten to {out}/{tag}.json and {out}/{tag}.md")


if __name__ == "__main__":
    main()