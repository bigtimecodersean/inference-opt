"""
Minimal profiling target for Nsight Compute.

Structure:
  prefill (NVTX range "prefill") -> populates KV cache
  3 warmup decode steps (NVTX range "decode_warmup") -> not measured
  1 decode step (NVTX range "decode_step_measured") -> this is what we profile

NVTX ranges are markers in the CUDA stream. ncu can filter its profiling
to only kernels launched inside a specific range, which is how we isolate
the one decode step we care about.

Why only one measured step:
  ncu is slow. It replays each kernel to collect metrics. Profiling
  10 decode steps would take 10x as long for no more useful data —
  each decode step launches the same set of kernels, so one run is
  enough to characterize the workload.
"""
import torch

from inference_opt.models import load_model_and_tokenizer
from inference_opt.workloads import make_random_prompts
from inference_opt.utils import cuda_sync, log_gpu_info


BATCH_SIZE = 1
PROMPT_LENGTH = 512
NUM_WARMUP_STEPS = 3


@torch.inference_mode()
def main():
    log_gpu_info()
    print()

    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer("Qwen/Qwen2.5-3B")
    vocab_size = model.config.vocab_size

    print(f"Generating prompt (batch={BATCH_SIZE}, len={PROMPT_LENGTH})...")
    prompts = make_random_prompts(
        batch_size=BATCH_SIZE,
        prompt_length=PROMPT_LENGTH,
        vocab_size=vocab_size,
        seed=0,
    )

    # --- Prefill ---
    torch.cuda.nvtx.range_push("prefill")
    outputs = model(prompts, use_cache=True)
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    past_kv = outputs.past_key_values
    cuda_sync()
    torch.cuda.nvtx.range_pop()

    # --- Warmup decode (NOT profiled) ---
    torch.cuda.nvtx.range_push("decode_warmup")
    for _ in range(NUM_WARMUP_STEPS):
        outputs = model(next_token, past_key_values=past_kv, use_cache=True)
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        past_kv = outputs.past_key_values
    cuda_sync()
    torch.cuda.nvtx.range_pop()

    # --- THE measured decode step ---
    torch.cuda.nvtx.range_push("decode_step_measured")
    outputs = model(next_token, past_key_values=past_kv, use_cache=True)
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    cuda_sync()
    torch.cuda.nvtx.range_pop()

    print("Done.")


if __name__ == "__main__":
    main()
