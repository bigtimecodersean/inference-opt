"""
Sanity check. Run this first. If it fails, nothing else will work.
If it succeeds, you have the right GPU, CUDA stack, and enough memory
to proceed.

Verifies:
- Model loads in the expected dtype
- Memory footprint is sane (~6GB for Qwen2.5-3B in bf16)
- End-to-end generation produces plausible text
- Rough tokens/sec is in the expected ballpark
"""

import torch

from inference_opt.models import load_model_and_tokenizer
from inference_opt.utils import CudaTimer, log_gpu_info


def main():
    log_gpu_info()
    print()

    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer("Qwen/Qwen2.5-3B")

    # Parameter count + weight memory footprint.
    n_params = sum(p.numel() for p in model.parameters())
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"Model: {n_params/1e9:.2f}B parameters, "
          f"{weight_bytes/1e9:.2f} GB of weights in {next(model.parameters()).dtype}")
    print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print()

    # End-to-end generation test using the model's built-in generate().
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with CudaTimer() as t:
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,   # greedy — deterministic
            )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Prompt : {prompt!r}")
    print(f"Output : {output_text!r}")
    print(f"Time   : {t.elapsed_ms:.1f} ms for 20 new tokens")
    print(f"         ≈ {20 / (t.elapsed_ms/1000):.1f} tok/s "
          f"(includes prefill, so not a pure decode number)")


if __name__ == "__main__":
    main()