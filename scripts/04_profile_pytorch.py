"""
Alternative profiling path using torch.profiler.

Why this path:
  Nsight Compute (ncu) requires GPU hardware performance counter
  access, which is disabled on most shared cloud hosts (RunPod
  Community Cloud, some Lambda instances, etc.) for security reasons.
  torch.profiler is framework-level, uses CUDA events for timing,
  and works without any special permissions.

What it gives us:
  - Per-operator wall-clock time (GPU side, via CUDA events)
  - Per-op CUDA kernel launches
  - Memory allocated per op
  - NVTX markers respected, so we can isolate one decode step

What it does NOT give us:
  - Direct measurement of bytes moved to HBM (we'll compute this
    analytically from op shapes)
  - Direct FLOP counts (same — computed from shapes)

For transformer decode, the analytical approach is very accurate:
  - We know every op's shape exactly from the model config
  - We know every op's algorithmic cost exactly (matmul FLOPs, etc.)
  - Bytes moved = weight bytes + activation I/O

Output:
  A table of ops sorted by self-time, and a raw JSON trace we'll
  post-process to build the roofline plot.
"""
import json
from pathlib import Path

import torch
from torch.profiler import profile, ProfilerActivity, record_function

from inference_opt.models import load_model_and_tokenizer
from inference_opt.workloads import make_random_prompts
from inference_opt.utils import cuda_sync, log_gpu_info


BATCH_SIZE = 1
PROMPT_LENGTH = 512
NUM_WARMUP_STEPS = 3

OUTPUT_DIR = Path("results/baseline/profile")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@torch.inference_mode()
def main():
    log_gpu_info()
    print()

    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer("Qwen/Qwen2.5-3B")
    vocab_size = model.config.vocab_size

    prompts = make_random_prompts(
        batch_size=BATCH_SIZE,
        prompt_length=PROMPT_LENGTH,
        vocab_size=vocab_size,
        seed=0,
    )

    # Prefill -- outside profiler so we don't capture it
    print("Running prefill (not profiled)...")
    outputs = model(prompts, use_cache=True)
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    past_kv = outputs.past_key_values
    cuda_sync()

    # Warmup decode steps -- outside profiler
    print("Warmup decode steps (not profiled)...")
    for _ in range(NUM_WARMUP_STEPS):
        outputs = model(next_token, past_key_values=past_kv, use_cache=True)
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        past_kv = outputs.past_key_values
    cuda_sync()

    # THE measured decode step, wrapped in profiler
    print("Profiling one decode step...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        with record_function("decode_step_measured"):
            outputs = model(next_token, past_key_values=past_kv, use_cache=True)
            _ = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            cuda_sync()

    # --- Print a human-readable summary ---
    print()
    print("Top 20 GPU operations by self-CUDA-time:")
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total",
        row_limit=20,
    ))

    # --- Save the raw trace so we can post-process it later ---
    trace_path = OUTPUT_DIR / "decode_trace.json"
    prof.export_chrome_trace(str(trace_path))
    print(f"\nChrome trace saved to: {trace_path}")
    print("  Open in chrome://tracing/ or https://ui.perfetto.dev/ to visualize.")

    # --- Save a structured summary for the roofline script to consume ---
    # Note: attribute names differ across PyTorch versions. Use getattr with
    # fallback to handle both old (self_cuda_time_total) and new
    # (self_device_time_total) naming conventions.
    summary = []
    for evt in prof.key_averages():
        self_device_us = getattr(
            evt, "self_device_time_total",
            getattr(evt, "self_cuda_time_total", 0),
        )
        device_us = getattr(
            evt, "device_time_total",
            getattr(evt, "cuda_time_total", 0),
        )
        summary.append({
            "name": evt.key,
            "self_device_time_us": self_device_us,
            "device_time_us": device_us,
            "cpu_time_us": evt.cpu_time_total,
            "count": evt.count,
            "input_shapes": str(evt.input_shapes) if evt.input_shapes else "",
        })

    summary_path = OUTPUT_DIR / "op_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Op summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
