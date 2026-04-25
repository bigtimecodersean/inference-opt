# inference-opt

LLM inference optimization on a single GPU. One of three projects in a portfolio
targeting inference performance engineering roles.

![Baseline decode roofline](results/baseline/roofline.png)

## Project overview

Take Qwen2.5-3B and measurably improve its serving performance on a single
A100 80GB through:

1. **Profiling + roofline analysis** to identify bottlenecks (Week 1 — complete)
2. **Batching analysis** to measure arithmetic-intensity shift (Week 2 — next)
3. **KV cache quantization** (FP16 → FP8) with quality tracking (Week 3)
4. **Custom Triton kernel** for a fused attention variant (Week 4)

## Week 1 — Baseline and roofline

**Hardware:** NVIDIA A100-SXM4-80GB (400W cap), bf16, torch 2.5.1+cu121.

Measured ceilings via `scripts/00_microbench.py`:
- HBM bandwidth: **1.76 TB/s** (86% of 2.04 TB/s theoretical peak)
- Peak FP16 tensor core: **240 TFLOPS** (77% of 312 TFLOPS peak)
- Ridge point: **136 FLOPs/byte**

### Baseline sweep

| Config | TTFT (ms) | TPOT (ms) | Throughput (tok/s) |
|---|---|---|---|
| batch=1, prompt=128  | 27  | 26.4 | 38   |
| batch=1, prompt=2048 | 81  | 26.4 | 37   |
| batch=1, prompt=8192 | 331 | 26.4 | 35   |
| batch=4, prompt=512  | 78  | 26.8 | 147  |
| batch=16, prompt=512 | 280 | 26.7 | 556  |
| batch=64, prompt=512 | 1095 | 26.3 | 1843 |

Raw data: [`results/baseline/summary.csv`](results/baseline/summary.csv).

### Key findings

**Decode is memory-bound.** TPOT is nearly flat across batch 1 → 64
(26.4 → 26.3 ms). The GPU spends most of its decode time waiting for
HBM reads, not doing arithmetic. Adding concurrent users doesn't affect
per-token latency because the bottleneck is weight loading, not
compute.

**Batching amortizes weight loads.** Throughput scales 48× when batch
scales 64×, because weights are read from HBM once and reused across
the batch. This is the fundamental economic argument for continuous
batching in production serving.

**Prefill scales linearly until O(N²) attention dominates.** TTFT grows
linearly with prompt length up to ~2K tokens, superlinearly after that
(QKᵀ cost kicks in).

### Roofline analysis

All four linear projection categories sit at **AI ≈ 1 FLOP/byte** — the
mathematical signature of matrix-vector products at batch=1. They
achieve ~1 TFLOPS, roughly 56% of the memory-bound ceiling at that AI.

**Attention** sits at AI ≈ 8 but achieves only 2.5% of its ceiling,
suggesting significant kernel launch overhead at these tiny decode
shapes — attention kernels are complex but process very few tokens, so
most of their time isn't spent doing the work they're shaped for.

The roofline plot makes the optimization roadmap visual. The entire
compute-bound region of the plot (right of AI=136) is unused. Every
Week 2–4 intervention is a strategy to move workload points either **up**
(use available bandwidth more efficiently) or **right** (do more
arithmetic per byte via batching, fusion, or quantization).

## Repository layout

```
inference-opt/
├── src/inference_opt/          # library: model loading, benchmark harness, utils
├── scripts/
│   ├── 00_microbench.py        # measure this pod's HBM + compute ceilings
│   ├── 01_sanity_check.py      # verify the model runs
│   ├── 02_benchmark_baseline.py  # TTFT/TPOT/throughput sweep
│   ├── 03_profile_target.py    # NVTX-wrapped minimal decode target (for ncu)
│   ├── 04_profile_pytorch.py   # torch.profiler based profiling
│   └── 05_roofline_plot.py     # analytical roofline construction + plot
├── results/baseline/
│   ├── microbench.json         # measured hardware ceilings (per pod)
│   ├── summary.csv             # baseline sweep results
│   ├── profile/                # per-op PyTorch Profiler output
│   └── roofline.png            # headline plot
├── NOTES.md                    # engineering log, observations, gotchas
├── RUNBOOK.md                  # pod setup and reproduction instructions
└── pyproject.toml
```

## Reproducing

Pod setup is non-trivial due to RunPod-specific behaviors — see
[`RUNBOOK.md`](RUNBOOK.md) for the verified startup sequence.

Once the pod is set up:

```bash
python scripts/00_microbench.py       # measures this pod's ceilings (~30s)
python scripts/02_benchmark_baseline.py   # baseline sweep (~10 min)
python scripts/04_profile_pytorch.py  # per-op profile (~30s)
python scripts/05_roofline_plot.py    # render plot (~2s)
```

## Methodology notes

See [`NOTES.md`](NOTES.md) for engineering observations including:
- Decode-time variance across nominally-identical A100 pods (launch
  overhead dominates at batch=1, so host-level variance matters)
- Why we use analytical FLOPs/bytes instead of Nsight Compute
  measurements (RunPod Community Cloud blocks GPU performance counter
  access)
- Why all roofline ceilings come from per-pod microbench, not datasheet
  peaks
