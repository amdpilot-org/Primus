#!/usr/bin/env python3
"""
Self-contained GEMM benchmark for the AMD Instinct MI355X.

WHAT THIS MEASURES
------------------
The single dense forward GEMM, C = A @ B, executed as ``torch.matmul(A, B,
out=C)``.  This is the headline operator measured by Primus's own
``primus-cli direct -- benchmark gemm`` command, implemented in
``primus/tools/benchmark/gemm_bench.py::profile_gemm``, and the first operator
class named in ``benchmark/kernel/README.md``.

FAITHFULNESS
------------
This script imports and calls the REAL upstream ``profile_gemm`` from
``primus.tools.benchmark.gemm_bench``.  (In this environment the ``primus``
package imports without a build, so there is no need to reimplement.)  The
upstream function returns only a single mean per call, so this script calls it
``--rounds`` times per shape and reports the SPREAD (min/max/mean/median/std/
slow-tail-p10) of the achieved TFLOPS, as the task requires.

If the import ever fails, the script falls back to a faithful local
reimplementation of the same methodology (rotating ~2 GB buffer, CUDA-event
timing, duration-based sampling) and prints a warning so the substitution is
explicit.

FP8 NOTE (a named gap)
----------------------
Primus's ``profile_gemm_fp8`` path uses ``torchao``, which is NOT installed in
this environment, so the FP8 GEMM path is not measured here.  Only the BF16 path
(the upstream default and the MegaMoE runtime target "EP-only + bf16") is
measured.

USAGE
-----
    python benchmark_gemm.py --duration 3 --rounds 10
    python benchmark_gemm.py --shapes 4096:4096:4096:bf16 --duration 3 --rounds 10
"""

import argparse
import math
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch

CACHE_ROTATING_BUFFER_BYTES = 2 * 1024 * 1024 * 1024  # matches upstream 2 GB

# ---------------------------------------------------------------------------
# Try to import the REAL upstream profile_gemm from the repo.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]  # reports/<job>/ -> repo root
USE_REAL = False
try:
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from primus.tools.benchmark.gemm_bench import profile_gemm as _upstream_profile_gemm

    USE_REAL = True
except Exception as _import_err:  # pragma: no cover - fallback path
    _IMPORT_ERROR = _import_err


# ---------------------------------------------------------------------------
# Environment capture (read from the machine, not assumed).
# ---------------------------------------------------------------------------
def gpu_env() -> dict:
    env = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "rocm": getattr(torch.version, "rocm", None),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        env.update(
            {
                "device_name": p.name,
                "gcn_arch": getattr(p, "gcnArchName", "n/a"),
                "total_mem_gb": round(p.total_memory / 1e9, 2),
                "multiprocessor_count": p.multi_processor_count,
            }
        )
    try:
        with open("/opt/rocm/.info/version") as fh:
            env["rocm_filesystem"] = fh.read().strip()
    except OSError:
        env["rocm_filesystem"] = "n/a"
    try:
        out = subprocess.run(
            ["hipcc", "--version"], capture_output=True, text=True, check=True
        ).stdout
        env["hipcc"] = out.splitlines()[0].strip() if out.strip() else "n/a"
    except (OSError, subprocess.CalledProcessError):
        env["hipcc"] = "n/a"
    env["measurement_backend"] = (
        "real upstream primus.tools.benchmark.gemm_bench.profile_gemm"
        if USE_REAL
        else "local reimplementation (import failed)"
    )
    return env


# ---------------------------------------------------------------------------
# Fallback reimplementation (only used if the real import failed).
# ---------------------------------------------------------------------------
def _maybe_transpose(tensor, transpose):
    return tensor.t() if transpose else tensor


def _fallback_profile_gemm(m, n, k, dtype, trans_a, trans_b, duration_s):
    device = torch.cuda.current_device()
    dtype_size = torch.tensor([], dtype=dtype).element_size()
    mem_size_bytes = (m * k + k * n + m * n) * dtype_size
    num_rotations = max(2, math.ceil(CACHE_ROTATING_BUFFER_BYTES / max(1, mem_size_bytes)) + 1)
    a_shape = (k, m) if trans_a else (m, k)
    b_shape = (n, k) if trans_b else (k, n)
    a_list = [torch.randn(a_shape, device=device, dtype=dtype) for _ in range(num_rotations)]
    b_list = [torch.randn(b_shape, device=device, dtype=dtype) for _ in range(num_rotations)]
    c_list = [torch.empty((m, n), device=device, dtype=dtype) for _ in range(num_rotations)]
    for i in range(num_rotations):
        torch.matmul(
            _maybe_transpose(a_list[i], trans_a),
            _maybe_transpose(b_list[i], trans_b),
            out=c_list[i],
        )
    torch.cuda.synchronize()
    target_ms = max(100.0, duration_s * 1000.0)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    total_calls = 0
    start.record()
    while True:
        for i in range(num_rotations):
            torch.matmul(
                _maybe_transpose(a_list[i], trans_a),
                _maybe_transpose(b_list[i], trans_b),
                out=c_list[i],
            )
        end.record()
        torch.cuda.synchronize()
        total_calls += num_rotations
        if start.elapsed_time(end) >= target_ms:
            break
    avg_time_ms = start.elapsed_time(end) / total_calls
    tflop = 2.0 * m * n * k / 1e12
    return {
        "m": m, "n": n, "k": k,
        "trans_a": trans_a, "trans_b": trans_b,
        "avg_time_ms": avg_time_ms,
        "tflop": tflop,
        "tflops": tflop / (avg_time_ms / 1000.0),
        "bandwidth_gbps": mem_size_bytes / 1e9 / (avg_time_ms / 1000.0),
        "arith_intensity": (2.0 * m * n * k) / mem_size_bytes,
    }


def measure_once(m, n, k, dtype, trans_a, trans_b, duration):
    """Call whichever profiler is available once, return its result dict."""
    if USE_REAL:
        return _upstream_profile_gemm(m, n, k, dtype, trans_a, trans_b, duration)
    return _fallback_profile_gemm(m, n, k, dtype, trans_a, trans_b, duration)


# ---------------------------------------------------------------------------
# Representative forward-GEMM shapes.
# The 4096^3 is the `benchmark gemm` default (no transposes).
# The others are the exact forward shapes Primus's `dense_gemm_bench` derives
# for Llama-3.1-8B (seqlen 8192, hidden 4096, intermediate 14336), with
# trans_b=True matching _profile_fwd -> profile_gemm(m,n,k,dtype,False,True,...).
# ---------------------------------------------------------------------------
DEFAULT_SHAPES = [
    # (label, M, N, K, dtype-name, trans_a, trans_b)
    ("gemm_4096^3_bf16", 4096, 4096, 4096, "bf16", False, False),
    ("llama3.1_8B_attn_out_fwd_bf16", 8192, 4096, 4096, "bf16", False, True),
    ("llama3.1_8B_mlp_down_fwd_bf16", 8192, 4096, 14336, "bf16", False, True),
    ("llama3.1_8B_mlp_up_fwd_bf16", 8192, 28672, 4096, "bf16", False, True),
]


def pct(seq, q):
    s = sorted(seq)
    idx = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return s[idx]


@torch.inference_mode()
def profile_with_spread(m, n, k, dtype, trans_a, trans_b, duration, rounds):
    """Call the profiler `rounds` times and return a spread distribution."""
    tflops_samples = []
    times_ms = []
    last = None
    for _ in range(rounds):
        r = measure_once(m, n, k, dtype, trans_a, trans_b, duration)
        last = r
        tflops_samples.append(r["tflops"])
        times_ms.append(r["avg_time_ms"])

    mem_size_bytes = (m * k + k * n + m * n) * torch.tensor([], dtype=dtype).element_size()
    tflop = 2.0 * m * n * k / 1e12
    return {
        "m": m, "n": n, "k": k,
        "trans_a": trans_a, "trans_b": trans_b,
        "dtype": str(dtype).split(".")[-1],
        "tflop": tflop,
        "arith_intensity": (2.0 * m * n * k) / mem_size_bytes,
        "rounds": rounds,
        "time_min_ms": min(times_ms),
        "time_max_ms": max(times_ms),
        "time_mean_ms": statistics.mean(times_ms),
        "time_median_ms": statistics.median(times_ms),
        "time_std_ms": statistics.pstdev(times_ms) if len(times_ms) > 1 else 0.0,
        "tflops_min": min(tflops_samples),
        "tflops_max": max(tflops_samples),
        "tflops_mean": statistics.mean(tflops_samples),
        "tflops_median": statistics.median(tflops_samples),
        "tflops_std": statistics.pstdev(tflops_samples) if len(tflops_samples) > 1 else 0.0,
        "tflops_p10": pct(tflops_samples, 0.10),
        "bandwidth_gbps": mem_size_bytes / 1e9 / (statistics.median(times_ms) / 1000.0),
    }


def run(shapes, duration, rounds):
    torch.cuda.init()
    env = gpu_env()
    print("=" * 70)
    print("ENVIRONMENT (read from the machine)")
    print("=" * 70)
    for k, v in env.items():
        print(f"  {k}: {v}")
    if not USE_REAL:
        print(f"  [WARNING] upstream import failed: {_IMPORT_ERROR}")
        print("  [WARNING] using local reimplementation fallback")
    print()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    results = []
    for entry in shapes:
        label, m, n, k, dname, trans_a, trans_b = entry
        dtype = dtype_map[dname]
        print("-" * 70)
        print(f"BENCHMARK  {label}   M={m} N={n} K={k} dtype={dname} trans_a={trans_a} trans_b={trans_b}")
        print("-" * 70)
        r = profile_with_spread(m, n, k, dtype, trans_a, trans_b, duration, rounds)
        r["label"] = label
        results.append(r)
        print(
            f"  tflops: median={r['tflops_median']:.2f}  mean={r['tflops_mean']:.2f}  "
            f"min={r['tflops_min']:.2f}  max={r['tflops_max']:.2f}  "
            f"std={r['tflops_std']:.2f}  slow_p10={r['tflops_p10']:.2f}"
        )
        print(
            f"  time(ms)/call: median={r['time_median_ms']:.4f}  "
            f"min={r['time_min_ms']:.4f}  max={r['time_max_ms']:.4f}  "
            f"spread={r['time_max_ms']-r['time_min_ms']:.4f}"
        )
        print(f"  arith_intensity={r['arith_intensity']:.2f}  bw~{r['bandwidth_gbps']:.1f} GB/s")
        print()

    print("=" * 70)
    print("SUMMARY TABLE (TFLOPS, sorted by label)")
    print("=" * 70)
    hdr = f"{'shape':<34} {'dtype':<6} {'median':>8} {'mean':>8} {'min':>8} {'max':>8} {'std':>7} {'slow_p10':>9}"
    print(hdr)
    for r in results:
        print(
            f"{r['label']:<34} {r['dtype']:<6} "
            f"{r['tflops_median']:>8.2f} {r['tflops_mean']:>8.2f} "
            f"{r['tflops_min']:>8.2f} {r['tflops_max']:>8.2f} "
            f"{r['tflops_std']:>7.2f} {r['tflops_p10']:>9.2f}"
        )
    return env, results


def main():
    ap = argparse.ArgumentParser(description="Primus GEMM benchmark with spread reporting")
    ap.add_argument("--duration", type=float, default=3.0, help="seconds per timed round (min 0.1s)")
    ap.add_argument("--rounds", type=int, default=10, help="number of timed rounds per shape")
    ap.add_argument("--shapes", type=str, default="", help="comma list M:N:K:dtype[:trans_b]")
    args = ap.parse_args()

    shapes = list(DEFAULT_SHAPES)
    if args.shapes:
        shapes = []
        for tok in args.shapes.split(","):
            parts = tok.split(":")
            m, n, k, d = parts[0], parts[1], parts[2], parts[3]
            trans_b = len(parts) > 4 and parts[4].lower() in ("1", "true", "t")
            shapes.append((f"gemm_{m}x{n}x{k}_{d}", int(m), int(n), int(k), d, False, trans_b))

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA/HIP device available; this benchmark needs a GPU.")
    duration = max(0.1, args.duration)
    run(shapes, duration, args.rounds)


if __name__ == "__main__":
    main()
