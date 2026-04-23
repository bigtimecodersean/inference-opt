"""
Model loading.

Why bfloat16 by default? Same exponent range as fp32 but fewer mantissa
bits. Rarely overflows during inference (unlike fp16, which can produce
NaNs with aggressive attention scores). Same speed as fp16 on A100/H100.
fp16 only makes sense on Volta-era GPUs that don't have bf16 tensor cores.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(
    model_name: str = "Qwen/Qwen2.5-3B",
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """
    Load model in eval mode with no gradient tracking.

    attn_implementation="sdpa" uses PyTorch's scaled-dot-product-attention,
    which dispatches to FlashAttention-2 when the shapes/dtypes align. This
    is our baseline — we'll compare custom Triton kernels against it in
    Week 4.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
        attn_implementation="sdpa",
    )

    # eval() disables dropout etc. Disabling grad on every parameter also
    # prevents any accidental autograd graph buildup from sneaking in.
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model, tokenizer