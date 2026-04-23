"""
Workload generation.

Why random token IDs and not real text? We need to control prompt length
*exactly* in tokens. Real text is variable: "the quick brown fox" might
be 4 tokens or 6 depending on the tokenizer, and that variance pollutes
measurements. The GPU performs identical work regardless of which tokens
you feed it — the compute graph depends on shape, not content. (Token
identity *would* matter if we were evaluating quality, but benchmarking
throughput/latency is a separate concern.)
"""

import torch


def make_random_prompts(
    batch_size: int,
    prompt_length: int,
    vocab_size: int,
    device: str = "cuda",
    seed: int = 0,
) -> torch.Tensor:
    """
    Generate a tensor of shape (batch_size, prompt_length) with random
    token IDs in [10, vocab_size - 10).

    We avoid the low end of the vocab because tokens 0-9 often include
    special tokens (pad, bos, eos) that can trigger fast-paths or early
    exits in some model implementations — not what we want to profile.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randint(
        low=10,
        high=vocab_size - 10,
        size=(batch_size, prompt_length),
        dtype=torch.int64,
        device=device,
        generator=g,
    )