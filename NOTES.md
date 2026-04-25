# Engineering notes

Running log of observations, gotchas, and learnings. Raw material for
end-of-project README distillation.

## 2026-04-24 — Host variability on nominally-identical GPUs

Observation: Two independent RunPod Community Cloud A100-SXM4-80GB pods
produced different baseline decode latencies despite running the same
code, same PyTorch version (2.5.1+cu121), and identical GPU model.

A 500W-capped pod yielded ~13 ms per decode step; a 400W-capped pod
yielded ~26 ms. GPU-side ceilings on the 400W pod measured via
microbench (1.76 TB/s HBM, 240 TFLOPS FP16) are within the expected
range for A100 under a 400W cap — memory and compute are not the
bottleneck.

Root cause: kernel launch overhead. A Qwen-3B decode step launches
~400–500 CUDA kernels; each step spends meaningful time on CPU → GPU
submission. Cloud hosts with slower CPU paths or higher PCIe contention
amplify this cost, which disproportionately hurts decode (per-kernel
overhead is a large fraction of each short kernel's runtime) relative
to prefill (per-kernel overhead is negligible relative to large matmuls).

Implications for this project:
- Baselines are pod-specific. Rolling forward with the 400W pod as our
  reference; see `results/baseline/` for its numbers.
- Roofline ceilings come from `microbench.json` (measured, this pod),
  not datasheet peaks. The shape of the analysis is invariant to which
  pod we run on; absolute numbers are not.
- Launch overhead could be reclaimed via CUDA graphs (`torch.compile`
  with cudagraph mode). Candidate for Week 4 if scope allows.
