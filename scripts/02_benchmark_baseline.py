"""
Week 1 baseline sweep.

Two sweeps:
  (A) Prompt length ∈ {128, 512, 2048, 8192} at batch=1, output=128.
      Shows how TTFT scales with prompt length (prefill cost).

  (B) Batch size ∈ {1, 4, 16, 64} at prompt_length=512, output=128.
      Shows how throughput scales with batch size. This is the figure
      that motivates Week 2's batching analysis.

Writes:
  results/baseline/summary.csv  — one row per config, high-level metrics
  results/baseline/raw.json     — full per-token latency arrays
"""

import json
from pathlib import Path

import pandas as pd
import torch

from inference_opt.benchmark import benchmark
from inference_opt.models import load_model_and_tokenizer
from inference_opt.utils import log_gpu_info


OUTPUT_DIR = Path("results/baseline")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Sweeps as (batch_size, prompt_length) pairs.
PROMPT_LENGTH_SWEEP = [(1, n) for n in [128, 512, 2048, 8192]]
BATCH_SIZE_SWEEP = [(b, 512) for b in [1, 4, 16, 64]]
OUTPUT_LENGTH = 128


def main():
    log_gpu_info()
    print()

    model, tokenizer = load_model_and_tokenizer("Qwen/Qwen2.5-3B")
    vocab_size = model.config.vocab_size

    configs = PROMPT_LENGTH_SWEEP + BATCH_SIZE_SWEEP

    summary_rows = []
    raw_records = []

    for i, (batch_size, prompt_length) in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] "
              f"batch={batch_size}  prompt={prompt_length}  out={OUTPUT_LENGTH}")

        try:
            result = benchmark(
                model=model,
                batch_size=batch_size,
                prompt_length=prompt_length,
                output_length=OUTPUT_LENGTH,
                vocab_size=vocab_size,
            )
        except torch.cuda.OutOfMemoryError:
            print("  OOM — skipping.")
            torch.cuda.empty_cache()
            continue

        row = result.summary_dict()
        summary_rows.append(row)
        raw_records.append({
            "config": row,
            "per_token_latencies_ms": result.per_token_latencies_ms,
        })

        print(f"  TTFT      : {row['ttft_ms']:.1f} ms")
        print(f"  TPOT med  : {row['tpot_ms']:.2f} ms")
        print(f"  TPOT p99  : {row['tpot_p99_ms']:.2f} ms")
        print(f"  Thruput   : {row['throughput_tok_per_s']:.1f} tok/s")

    df = pd.DataFrame(summary_rows)
    df.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    with open(OUTPUT_DIR / "raw.json", "w") as f:
        json.dump(raw_records, f)

    print(f"\nWrote {len(summary_rows)} rows → {OUTPUT_DIR/'summary.csv'}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()