"""
Week 2 — Batching sweep.

Goal: measure how decode performance scales with batch size while holding
prompt and output length fixed. The output is the data we need to plot
each batch size as a separate point on the roofline.

Why this is a separate script from 02_benchmark_baseline.py:
  - The Week 1 baseline is frozen — re-running it produces matched-pod
    numbers but the file is intended as a fixed reference.
  - The batching sweep has a different shape: many batch sizes, fixed
    prompt/output, much wider memory footprint.
  - We want to capture per-batch GPU memory and AI alongside timing,
    which the older script doesn't track.

OOM handling:
  Very large batches will run out of HBM (KV cache + activations exceed
  available memory). That's expected and *part of the finding* — finding
  the maximum batch we can serve at this prompt length is a real result.
  We catch torch.cuda.OutOfMemoryError, log it, and continue.

What gets logged per batch size:
  - TTFT, TPOT (median + p99), throughput
  - Peak GPU memory used during the run
  - Analytical AI for an FFN matmul at that batch (the simplest
    representative op for visualizing the AI shift)
"""
import json
import statistics
from pathlib import Path

import pandas as pd
import torch

from inference_opt.benchmark import benchmark
from inference_opt.models import load_model_and_tokenizer
from inference_opt.utils import log_gpu_info


# --- Sweep config ---
# Logarithmic-ish progression. We expect interesting things to happen around
# batch=64-256 (approaching the ridge) and OOM somewhere in that range.
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
PROMPT_LENGTH = 512
OUTPUT_LENGTH = 128

# --- Model architecture (Qwen2.5-3B) — used for analytical AI ---
HIDDEN = 2048
INTERMEDIATE = 11008
DTYPE_BYTES = 2  # bfloat16


# --- Output paths ---
OUTPUT_DIR = Path("results/batching")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def analytical_ai_ffn_up(batch_size: int) -> float:
    """
    Analytical arithmetic intensity for the FFN up-projection at given batch.

    This is the cleanest op to track because it's the largest matmul in
    the model (hidden -> intermediate = 2048 -> 11008). Its AI shift with
    batch size is the canonical "weight amortization" calculation.

    Shape: (batch, hidden) @ (hidden, intermediate) -> (batch, intermediate)

    FLOPs: 2 * batch * hidden * intermediate
    Bytes:
      - Weight read: hidden * intermediate * dtype_bytes  (loaded once)
      - Activation in:  batch * hidden * dtype_bytes
      - Activation out: batch * intermediate * dtype_bytes

    At batch=1, weight bytes dominate completely (AI = 1).
    At large batch, activation bytes start to matter slightly.
    """
    flops = 2 * batch_size * HIDDEN * INTERMEDIATE
    weight_bytes = HIDDEN * INTERMEDIATE * DTYPE_BYTES
    act_in_bytes = batch_size * HIDDEN * DTYPE_BYTES
    act_out_bytes = batch_size * INTERMEDIATE * DTYPE_BYTES
    total_bytes = weight_bytes + act_in_bytes + act_out_bytes
    return flops / total_bytes


def main():
    log_gpu_info()
    print()

    print("Loading model...")
    model, _ = load_model_and_tokenizer("Qwen/Qwen2.5-3B")
    vocab_size = model.config.vocab_size
    print()

    rows = []
    for batch_size in BATCH_SIZES:
        print(f"\n=== batch={batch_size}  prompt={PROMPT_LENGTH}  out={OUTPUT_LENGTH} ===")

        # Reset memory stats so peak-mem reflects only this run
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        try:
            result = benchmark(
                model=model,
                batch_size=batch_size,
                prompt_length=PROMPT_LENGTH,
                output_length=OUTPUT_LENGTH,
                vocab_size=vocab_size,
                num_warmup=2,
                num_repeats=3,   # fewer repeats than baseline — large batches are slow
            )
        except torch.cuda.OutOfMemoryError as e:
            print(f"  OOM at batch={batch_size}. Stopping sweep.")
            torch.cuda.empty_cache()
            rows.append({
                "batch_size": batch_size,
                "prompt_length": PROMPT_LENGTH,
                "output_length": OUTPUT_LENGTH,
                "ttft_ms": None,
                "tpot_ms": None,
                "tpot_p99_ms": None,
                "throughput_tok_per_s": None,
                "peak_mem_gb": None,
                "analytical_ai_ffn_up": analytical_ai_ffn_up(batch_size),
                "oom": True,
            })
            break

        peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
        ai = analytical_ai_ffn_up(batch_size)

        row = {
            "batch_size": batch_size,
            "prompt_length": PROMPT_LENGTH,
            "output_length": OUTPUT_LENGTH,
            "ttft_ms": result.ttft_ms,
            "tpot_ms": result.tpot_ms,
            "tpot_p99_ms": result.tpot_p99_ms,
            "throughput_tok_per_s": result.throughput_tok_per_s,
            "peak_mem_gb": peak_mem_gb,
            "analytical_ai_ffn_up": ai,
            "oom": False,
        }
        rows.append(row)

        print(f"  TTFT     : {row['ttft_ms']:.1f} ms")
        print(f"  TPOT med : {row['tpot_ms']:.2f} ms")
        print(f"  TPOT p99 : {row['tpot_p99_ms']:.2f} ms")
        print(f"  Thruput  : {row['throughput_tok_per_s']:.1f} tok/s")
        print(f"  Peak mem : {peak_mem_gb:.2f} GB")
        print(f"  AI (FFN) : {ai:.1f} FLOPs/byte")

    df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {len(rows)} rows -> {csv_path}")
    print()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()