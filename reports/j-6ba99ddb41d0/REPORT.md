# GEMM Benchmark on AMD Instinct MI355X (gfx950)

A single, self-contained benchmark of **the dense forward GEMM** (`C = A @ B`)
on one AMD Instinct MI355X, executed with PyTorch built against ROCm.

## The operation I picked, and why

**Operation: the dense forward GEMM** — `C = A @ B`, run as
`torch.matmul(A, B, out=C)`. This is the headline operator measured by
Primus's own CLI:

```bash
primus-cli direct -- benchmark gemm --M 4096 --N 4096 --K 4096 --dtype bf16 --duration 10
```

implemented in `primus/tools/benchmark/gemm_bench.py::profile_gemm`, and the
first operator class named in `benchmark/kernel/README.md` ("GEMM, Attention,
and communication-related operators").

**Why it is representative:** GEMM is the dominant compute primitive of the
transformer training that Primus exists to run — every attention projection
(QKV, output) and every FFN linear (gate/up, down) is a GEMM. Primus's
`dense_gemm_bench` derives the GEMM shapes directly from model configs
(Llama-3.1-8B, DeepSeek-V3, …), and the flagship MegaMoE feature is built on
grouped GEMMs. A dense GEMM is therefore the most representative *single*
operation this repository actually cares about, and it is measurable on one
GPU without a multi-GPU cluster.

## Faithfulness

**This script imports and calls the REAL upstream `profile_gemm` from
`primus.tools.benchmark.gemm_bench`.** In this environment the `primus`
package imports without a build, so the measurement uses the repository's
own code directly — no reimplementation for the headline numbers. The script
adds the repo root to `sys.path` so it works when run from inside
`reports/<job>/`.

The upstream `profile_gemm` returns only a single mean per call, so this
script calls it `--rounds` times per shape and reports the **spread**
(min/max/median/mean/std/slow-tail-p10) of the achieved TFLOPS, as the task
requires. If the import ever fails, the script falls back to a faithful local
reimplementation of the same methodology (rotating ~2 GB buffer, CUDA-event
timing, duration-based sampling) and prints a warning so the substitution is
explicit.

**Transpose fidelity:** Primus's `dense_gemm_bench._profile_fwd` calls
`profile_gemm(m, n, k, dtype, trans_a=False, trans_b=True, ...)`. The three
Llama-3.1-8B forward shapes below use `trans_b=True` to match — i.e. B is
stored as `(N, K)` and transposed (`.t()`) before the matmul, which is the
real training layout. The 4096³ default shape uses `trans_b=False`, matching
the `benchmark gemm` CLI default.

**Known upstream quirk (noted, not "fixed"):** the upstream timed loop has a
stale buffer index that reuses the last warm-up slot, so one buffer stays
L2-hot. This inflates the upstream numbers ~2–3% relative to a fully
cache-rotating reimplementation (verified during development). Using the real
code as-is is more faithful to what Primus actually measures; the reimplementation
fallback (which rotates all slots) is the more conservative lower bound.

**Named gaps (what I did NOT do):**

- **FP8 GEMM is not measured.** Primus's `profile_gemm_fp8` path uses
  `torchao`, which is **not installed** in this environment
  (`ModuleNotFoundError: No module named 'torchao'`). Only the BF16 path (the
  upstream default and the MegaMoE runtime target "EP-only + bf16") is
  measured. FP8/MXFP8/MXFP4 leverage on gfx950 is therefore a real, unmeasured
  gap.
- **No vendor peak asserted.** `amd-smi`/`rocm-smi` on this box expose no
  peak-FLOPS metric (only a 1900 MHz max engine clock), so I do not state a
  BF16 peak to avoid an unverified figure. TFLOPS stand on their own; the
  arithmetic-intensity column lets the reader judge compute-bound vs
  memory-bound.
- **Forward GEMM only.** Primus's `dense_gemm_bench` also measures the
  backward phases (wgrad/dgrad) with transposed operands. This deliverable
  measures only the forward `C = A @ B`.

## Environment (read from the machine)

| Item | Value |
| --- | --- |
| GPU | AMD Instinct MI355X |
| GCN arch | `gfx950:sramecc+:xnack-` |
| Compute units (SMs) | 256 |
| VRAM | 309.22 GB |
| Python | 3.12.3 |
| PyTorch | 2.9.1+rocm7.2.0.git7e1940d4 |
| HIP (torch.version.hip) | 7.2.26015-fc0010cf6a |
| ROCm (`/opt/rocm/.info/version`) | 7.2.0 |
| `hipcc --version` | HIP version: 7.2.26015-fc0010cf6a |
| Measurement backend | real upstream `primus.tools.benchmark.gemm_bench.profile_gemm` |

## Results — BF16 dense forward GEMM

10 timed rounds per shape, 3 s per round, after a full-buffer warm-up. Spread
reported as min/max/mean/median/std plus the slow-tail p10 of TFLOPS. Measured
with the real upstream `profile_gemm`.

| Shape (M×N×K) | trans_b | median TF | mean TF | min TF | max TF | std TF | slow p10 TF | time/call (ms, median) | arith. intensity | bw (GB/s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096×4096×4096 | False | 1365.34 | 1361.40 | 1338.81 | 1371.73 | 9.98 | 1348.11 | 0.1007 | 1365.33 | 1000.0 |
| 8192×4096×4096 (attn_out) | True | 1523.92 | 1524.39 | 1520.74 | 1531.83 | 3.16 | 1521.71 | 0.1804 | 1638.40 | 930.1 |
| 8192×4096×14336 (mlp_down) | True | 1603.71 | 1603.61 | 1601.32 | 1606.43 | 1.36 | 1601.90 | 0.5999 | 2293.76 | 699.2 |
| 8192×28672×4096 (mlp_up) | True | 1557.17 | 1557.24 | 1556.54 | 1558.32 | 0.52 | 1556.74 | 1.2357 | 2493.22 | 624.6 |

Notes:
- The 4096³ shape is the Primus `benchmark gemm` default (no transposes).
- The other three are the exact forward GEMM shapes Primus's
  `dense_gemm_bench` derives for **Llama-3.1-8B** (seqlen 8192, hidden 4096,
  intermediate 14336): `attn_out`, `mlp_down`, and `mlp_up` (= `2*intermediate`),
  with `trans_b=True` matching `_profile_fwd`.
- The measurement is **stable**: the three larger shapes have std < 3.2 TFLOPS
  (<0.2%); the smallest shape (4096³, ~0.1 ms/call) shows std ~10 TFLOPS
  (~0.7%) because its kernel is the shortest and most sensitive to scheduling
  jitter.
- All shapes are strongly compute-bound (arith intensity 1365–2493), so the
  reported bandwidth (625–1000 GB/s) is well below HBM peak, as expected for
  compute-bound GEMMs. The largest-K shape (`mlp_down`) achieves the highest
  TFLOPS (1604), consistent with the highest arithmetic intensity.

Raw run log: `reports/j-6ba99ddb41d0/gemm_run.log`. The committed log
was captured by running `python benchmark_gemm.py --duration 3 --rounds 10`
from the `reports/j-6ba99ddb41d0/` directory (the command in the Reproduce
section below).

## Reproduce

```bash
# 1 GPU, AMD Instinct MI355X, PyTorch+ROCm already on PATH.
# Run from the repo root so the `primus` package is importable,
# OR run from reports/<job>/ — the script adds the repo root to sys.path.
cd reports/j-6ba99ddb41d0
python benchmark_gemm.py --duration 3 --rounds 10

# Or a single shape, custom:
python benchmark_gemm.py --shapes 4096:4096:4096:bf16 --duration 3 --rounds 10
```

`--shapes` accepts a comma list of `M:N:K:dtype[:trans_b]` (dtype ∈ {bf16, fp16, fp32};
trans_b ∈ {1,0}). The script prints the environment block (including which
measurement backend it used), a per-shape block, and a summary table. Expect
~2 minutes wall time for the default four shapes × 10 rounds × 3 s.
