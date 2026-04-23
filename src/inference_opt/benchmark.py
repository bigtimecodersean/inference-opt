"""
Core benchmark harness.

Design principles, all of which matter:

1. MANUAL generation loop, not model.generate(). We need to time prefill
   independently from each decode step. generate() hides this.

2. Greedy decoding (argmax). No sampling randomness. Output lengths are
   deterministic; compute paths are identical across repeats.

3. WARMUP matters a lot. First calls hit: cuBLAS/cuDNN workspace
   allocation, kernel autotune passes, CUDA memory pool expansion, KV
   cache tensor allocation. Discard these — they're not steady state.

4. CUDA events + synchronize everywhere, not wall-clock.

5. MEDIAN across repeats, not mean. GPU workloads have occasional
   stragglers (kernel launch jitter, driver housekeeping, OS noise)
   that pull means around without telling you anything useful.
"""

import statistics
from typing import List, Tuple

import torch

from .metrics import BenchmarkResult
from .utils import CudaTimer, cuda_sync
from .workloads import make_random_prompts


@torch.inference_mode()
def run_single_generation(
    model,
    input_ids: torch.Tensor,
    output_length: int,
) -> Tuple[float, List[float]]:
    """
    Run one generation with fine-grained timing.

    Returns:
        ttft_ms: prefill-pass time (produces the first output token).
        per_token_ms: list of (output_length - 1) inter-token latencies
                      from the decode phase.

    Two phases:

    PREFILL. We pass the full prompt through the model in one forward
    call. This builds the KV cache (one entry per layer per position)
    and produces logits at the last position, which we argmax to get
    the first generated token. Cost scales with prompt_length; at
    reasonable sizes the big matmuls dominate and we're compute-bound.

    DECODE. We feed one token at a time, extending the KV cache each
    step. Each decode step reads *all past K and V* from HBM (memory
    traffic grows linearly with sequence length) but does only O(1)
    tokens of compute. This is why decode is the memory-bound phase
    we'll spend the rest of the project optimizing.
    """

    # --- Prefill ---
    with CudaTimer() as ttft:
        outputs = model(input_ids, use_cache=True)
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        past_kv = outputs.past_key_values

    # --- Decode ---
    per_token_ms: List[float] = []
    for _ in range(output_length - 1):
        with CudaTimer() as step:
            outputs = model(
                next_token,
                past_key_values=past_kv,
                use_cache=True,
            )
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            past_kv = outputs.past_key_values
        per_token_ms.append(step.elapsed_ms)

    return ttft.elapsed_ms, per_token_ms


def benchmark(
    model,
    batch_size: int,
    prompt_length: int,
    output_length: int,
    vocab_size: int,
    num_warmup: int = 2,
    num_repeats: int = 5,
    dtype_str: str = "bfloat16",
    device: str = "cuda",
) -> BenchmarkResult:
    """
    Benchmark one (batch_size, prompt_length, output_length) configuration.

    num_warmup: runs whose timings we discard. Absorbs first-call overheads.
    num_repeats: measured runs. We report per-run averages / medians.
    """

    # Warmup.
    for _ in range(num_warmup):
        prompts = make_random_prompts(
            batch_size, prompt_length, vocab_size, device=device, seed=0,
        )
        _ = run_single_generation(model, prompts, output_length)

    # Return freed blocks to the driver so we start measurement from a
    # clean allocator state. (PyTorch's caching allocator otherwise holds
    # onto memory it's reserved.)
    cuda_sync()
    torch.cuda.empty_cache()

    # Measured runs.
    all_ttft_ms: List[float] = []
    all_per_token_ms: List[float] = []  # flattened across repeats

    with CudaTimer() as total:
        for i in range(num_repeats):
            # Different seed per repeat so we don't repeatedly hit the same
            # pathological cache-line / branch pattern.
            prompts = make_random_prompts(
                batch_size, prompt_length, vocab_size,
                device=device, seed=i + 1,
            )
            ttft_ms, per_token_ms = run_single_generation(
                model, prompts, output_length
            )
            all_ttft_ms.append(ttft_ms)
            all_per_token_ms.extend(per_token_ms)

    return BenchmarkResult(
        batch_size=batch_size,
        prompt_length=prompt_length,
        output_length=output_length,
        dtype=dtype_str,
        ttft_ms=statistics.median(all_ttft_ms),
        per_token_latencies_ms=all_per_token_ms,
        # total_time_ms represents one run, for throughput accounting.
        total_time_ms=total.elapsed_ms / num_repeats,
    )