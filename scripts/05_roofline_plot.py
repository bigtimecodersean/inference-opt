"""
Roofline analysis for Qwen2.5-3B decode, using:
  - Measured hardware ceilings from results/baseline/microbench.json
  - Measured per-op wall-clock time from results/baseline/profile/op_summary.json
  - Analytically-derived FLOPs and bytes for each operator category

The roofline plot answers: for each major operator in the decode path,
(a) how much arithmetic is it doing per byte of memory traffic (AI),
(b) what throughput is it achieving, and (c) where does it sit relative
to the hardware ceilings?

Why analytical FLOPs/bytes instead of ncu-measured ones:
  Nsight Compute (ncu) is the standard way to measure actual bytes and
  FLOPs per kernel. It requires GPU hardware performance counter access,
  which is disabled on RunPod Community Cloud. Since transformer decode
  has a fully-defined algorithmic structure, we can compute the
  theoretical bytes and FLOPs from the model shapes — and for these
  workloads (matmul/gemv + flash-attention + elementwise), the
  analytical numbers match ncu measurements to within a few percent.

Operator categories (see the roofline plot):
  qkv_proj        : Q, K, V linear projections (per layer × 36 layers)
  o_proj          : Attention output projection
  ffn_gate_up     : FFN gate + up projections (SwiGLU has two)
  ffn_down        : FFN down projection
  attention       : Flash attention forward pass
  elementwise     : RMSNorm, RoPE, residual adds, multiplies (lumped)
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# --- Qwen2.5-3B architecture constants ---
# From the model's config.json on HuggingFace.
HIDDEN = 2048
INTERMEDIATE = 11008           # FFN up-projection dim
NUM_LAYERS = 36
NUM_Q_HEADS = 16
NUM_KV_HEADS = 2               # GQA: 16 query heads share 2 KV heads
HEAD_DIM = HIDDEN // NUM_Q_HEADS  # 128
VOCAB = 151936
DTYPE_BYTES = 2                # bfloat16

# Decode-time shapes. At decode with batch=1, each forward pass
# processes exactly ONE new token. The KV cache holds all prior positions.
BATCH = 1
NEW_TOKENS = 1                 # one new token per decode step
# After prefill of 512 + 3 warmup decodes + 1 to the measured step,
# the KV cache contains roughly 516 positions.
KV_CACHE_LEN = 516


# --- Paths ---
RESULTS = Path("results/baseline")
MICROBENCH = RESULTS / "microbench.json"
OP_SUMMARY = RESULTS / "profile" / "op_summary.json"
OUTPUT_PNG = RESULTS / "roofline.png"


# --- Derived dimensions ---
Q_OUT = NUM_Q_HEADS * HEAD_DIM          # 2048 (same as hidden, since full heads)
KV_OUT = NUM_KV_HEADS * HEAD_DIM        # 256  (GQA saves 8x on KV dims)
QKV_OUT = Q_OUT + 2 * KV_OUT            # 2560


# -----------------------------------------------------------------------
#  Per-operator analytical math
# -----------------------------------------------------------------------
# For each category we compute FLOPs and bytes for ONE full decode step
# (summed across all 36 layers where applicable).
#
# FLOPs convention: one multiply-add = 2 FLOPs (the "MAC = 2 FLOPs" rule).
# A matmul of shape (M, K) @ (K, N) does M*N dot products of length K,
# each being K multiplies + K adds = 2*K FLOPs. Total: 2 * M * K * N.
#
# Bytes convention: for each op, bytes = weights_loaded + inputs_loaded
# + outputs_written. For decode at batch=1, weight bytes dominate — the
# weight matrix is loaded once from HBM per step, the activation vector
# is tiny. That's the whole reason decode is memory-bound.


def qkv_proj():
    """
    Per layer: (1, hidden) @ (hidden, q_out + 2*kv_out)
              = (1, 2048) @ (2048, 2560)
    """
    weight_params = HIDDEN * QKV_OUT
    flops_per_layer = 2 * HIDDEN * QKV_OUT        # M=1, K=hidden, N=qkv_out
    weight_bytes = weight_params * DTYPE_BYTES
    act_bytes = (HIDDEN + QKV_OUT) * DTYPE_BYTES  # input + output
    bytes_per_layer = weight_bytes + act_bytes
    return {
        "flops": flops_per_layer * NUM_LAYERS,
        "bytes": bytes_per_layer * NUM_LAYERS,
    }


def o_proj():
    """
    Attention output projection, per layer: (1, hidden) @ (hidden, hidden)
    """
    weight_params = HIDDEN * HIDDEN
    flops_per_layer = 2 * HIDDEN * HIDDEN
    weight_bytes = weight_params * DTYPE_BYTES
    act_bytes = (HIDDEN + HIDDEN) * DTYPE_BYTES
    bytes_per_layer = weight_bytes + act_bytes
    return {
        "flops": flops_per_layer * NUM_LAYERS,
        "bytes": bytes_per_layer * NUM_LAYERS,
    }


def ffn_gate_up():
    """
    SwiGLU has a gate and an up projection, each (hidden, intermediate).
    Per layer: two (1, hidden) @ (hidden, intermediate) matmuls.
    """
    # Two separate matmuls, each of the same size
    weight_params = 2 * HIDDEN * INTERMEDIATE
    flops_per_layer = 2 * (2 * HIDDEN * INTERMEDIATE)
    weight_bytes = weight_params * DTYPE_BYTES
    # Each produces one intermediate-sized activation; one input shared
    act_bytes = (HIDDEN + 2 * INTERMEDIATE) * DTYPE_BYTES
    bytes_per_layer = weight_bytes + act_bytes
    return {
        "flops": flops_per_layer * NUM_LAYERS,
        "bytes": bytes_per_layer * NUM_LAYERS,
    }


def ffn_down():
    """
    Per layer: (1, intermediate) @ (intermediate, hidden)
    """
    weight_params = INTERMEDIATE * HIDDEN
    flops_per_layer = 2 * INTERMEDIATE * HIDDEN
    weight_bytes = weight_params * DTYPE_BYTES
    act_bytes = (INTERMEDIATE + HIDDEN) * DTYPE_BYTES
    bytes_per_layer = weight_bytes + act_bytes
    return {
        "flops": flops_per_layer * NUM_LAYERS,
        "bytes": bytes_per_layer * NUM_LAYERS,
    }


def attention():
    """
    FlashAttention for one decode step:
      - Q has shape (batch, num_q_heads, 1, head_dim) — for new token
      - K, V have shape (batch, num_kv_heads, kv_cache_len, head_dim)
    
    FLOPs (dominant term):
      QK^T:    2 * num_q_heads * 1 * kv_cache_len * head_dim
      SV:      2 * num_q_heads * 1 * kv_cache_len * head_dim
      Total:   4 * num_q_heads * kv_cache_len * head_dim per layer
    
    Bytes (dominant term is KV cache read):
      K read:  num_kv_heads * kv_cache_len * head_dim * DTYPE_BYTES
      V read:  same
      Q read:  tiny (num_q_heads * head_dim * DTYPE_BYTES)
      Output:  tiny
    """
    flops_per_layer = 4 * NUM_Q_HEADS * KV_CACHE_LEN * HEAD_DIM
    kv_bytes_per_layer = 2 * NUM_KV_HEADS * KV_CACHE_LEN * HEAD_DIM * DTYPE_BYTES
    q_bytes = NUM_Q_HEADS * HEAD_DIM * DTYPE_BYTES
    out_bytes = NUM_Q_HEADS * HEAD_DIM * DTYPE_BYTES
    bytes_per_layer = kv_bytes_per_layer + q_bytes + out_bytes
    return {
        "flops": flops_per_layer * NUM_LAYERS,
        "bytes": bytes_per_layer * NUM_LAYERS,
    }


# -----------------------------------------------------------------------
#  Match measured times from op_summary.json to each category
# -----------------------------------------------------------------------
# PyTorch Profiler reports many kernel variants. We bucket them by matching
# op names to each category. These patterns are based on actually reading
# the top 20 ops from our profile run.

CATEGORY_PATTERNS = {
    # All aten::mm calls — matmul dispatches. In Qwen-3B decode:
    #   aten::mm dispatches 4 matmuls per layer (QKV, O, up, gate) + ffn_down,
    #   so 5 per layer × 36 = 180 calls total. But we saw 145 calls of
    #   aten::mm, suggesting some calls use addmm (bias-fused) variants
    #   instead. We lump all the cutlass/gemv matmul kernels by category
    #   via a separate bucketing below.
    #
    # Rather than trying to parse which aten::mm call is which projection,
    # we take a simpler approach: lump ALL matmul-ish time into a single
    # "linear_layers" measurement, and use the analytical FLOPs/bytes to
    # break it down by category for the plot.
    #
    # This is imprecise per-op but honest: the profiler can't cleanly
    # separate which matmul kernel came from which layer, and the
    # analytical math is deterministic.

    "linear_total": [
        "aten::mm", "aten::addmm",
    ],
    "attention": [
        "aten::_flash_attention_forward",
    ],
    # Elementwise bucket: RMSNorm parts (mul, rsqrt, add), RoPE (mul, add,
    # cat, copy), residual adds, activation functions.
    "elementwise": [
        "aten::mul", "aten::add", "aten::cat", "aten::copy_",
        "aten::rsqrt", "aten::silu",
    ],
}


def load_op_times(op_summary_path):
    """
    Load op_summary.json and return a dict of category_name -> total CUDA us.
    """
    with open(op_summary_path) as f:
        ops = json.load(f)

    # Build a lookup: op_name -> self_device_time_us
    name_to_time = {}
    for op in ops:
        name = op["name"]
        us = op.get("self_device_time_us", 0)
        # Some ops can appear multiple times with slightly different shape
        # signatures — sum them all.
        name_to_time[name] = name_to_time.get(name, 0) + us

    # Aggregate into categories
    category_times_us = {}
    for cat, patterns in CATEGORY_PATTERNS.items():
        total = sum(name_to_time.get(p, 0) for p in patterns)
        category_times_us[cat] = total

    return category_times_us, name_to_time


# -----------------------------------------------------------------------
#  Main
# -----------------------------------------------------------------------
def main():
    # Load measured ceilings
    with open(MICROBENCH) as f:
        mb = json.load(f)
    bw_tbs = mb["hbm_bandwidth_measured_tbs"]
    peak_tflops = mb["compute_fp16_measured_tflops"]
    ridge_point = mb["ridge_point_measured_flops_per_byte"]

    print(f"Hardware ceilings (measured):")
    print(f"  HBM bandwidth:  {bw_tbs:.2f} TB/s")
    print(f"  Peak FP16:      {peak_tflops:.1f} TFLOPS")
    print(f"  Ridge point:    {ridge_point:.1f} FLOPs/byte")
    print()

    # Compute per-category FLOPs and bytes
    cats = {
        "qkv_proj":    qkv_proj(),
        "o_proj":      o_proj(),
        "ffn_gate_up": ffn_gate_up(),
        "ffn_down":    ffn_down(),
        "attention":   attention(),
    }

    # Load measured op times
    cat_times_us, name_to_time = load_op_times(OP_SUMMARY)

    # Time allocation strategy:
    #   - attention: directly measured (aten::_flash_attention_forward)
    #   - qkv_proj, o_proj, ffn_gate_up, ffn_down: share the total
    #     linear-op time, apportioned by their analytical FLOPs share
    linear_total_us = cat_times_us["linear_total"]
    attn_us = cat_times_us["attention"]
    elementwise_us = cat_times_us["elementwise"]

    # Apportion linear time by FLOPs share (good approximation for matmuls
    # at the same precision — they run at similar TFLOPS)
    linear_cats = ["qkv_proj", "o_proj", "ffn_gate_up", "ffn_down"]
    total_linear_flops = sum(cats[c]["flops"] for c in linear_cats)
    for c in linear_cats:
        cats[c]["time_us"] = linear_total_us * (cats[c]["flops"] / total_linear_flops)
    cats["attention"]["time_us"] = attn_us

    # Compute AI and achieved TFLOPS for each category
    for c, info in cats.items():
        info["ai"] = info["flops"] / info["bytes"]
        if info["time_us"] > 0:
            info["tflops"] = info["flops"] / (info["time_us"] * 1e-6) / 1e12
        else:
            info["tflops"] = 0

    # Print summary table
    print(f"{'Category':<14} {'FLOPs (G)':>11} {'Bytes (MB)':>12} {'AI':>6} "
          f"{'Time (us)':>10} {'TFLOPS':>8} {'% of HBM@AI':>12}")
    print("-" * 78)
    for c, info in cats.items():
        # HBM ceiling at this AI
        mem_ceiling = min(bw_tbs * info["ai"], peak_tflops)
        pct_ceiling = (info["tflops"] / mem_ceiling * 100) if mem_ceiling > 0 else 0
        print(f"{c:<14} "
              f"{info['flops']/1e9:>11.2f} "
              f"{info['bytes']/1e6:>12.2f} "
              f"{info['ai']:>6.2f} "
              f"{info['time_us']:>10.0f} "
              f"{info['tflops']:>8.2f} "
              f"{pct_ceiling:>11.1f}%")
    print()
    print(f"(Elementwise ops not plotted: {elementwise_us:.0f} us total, "
          f"fragmented across {sum(1 for p in CATEGORY_PATTERNS['elementwise'] if name_to_time.get(p, 0) > 0)} kernel types)")

    # -------- Plot --------
    fig, ax = plt.subplots(figsize=(10, 7))

    # Axis range
    ai_min, ai_max = 0.3, 300
    ai_line = np.logspace(np.log10(ai_min), np.log10(ai_max), 200)

    # Sloped (memory-bound) ceiling: y = bw * x
    mem_line = bw_tbs * 1e12 * ai_line
    # Flat (compute-bound) ceiling: y = peak
    compute_line = np.full_like(ai_line, peak_tflops * 1e12)
    # Envelope: take min of the two
    envelope = np.minimum(mem_line, compute_line)

    ax.plot(ai_line, envelope / 1e12, "k-", linewidth=2.5,
            label=f"Roofline (this pod)")
    ax.axhline(peak_tflops, linestyle=":", color="gray", alpha=0.5,
               label=f"Peak compute: {peak_tflops:.0f} TFLOPS")

    # Ridge point marker
    ax.axvline(ridge_point, linestyle=":", color="gray", alpha=0.5)
    ax.text(ridge_point * 1.1, 1.5, f"ridge @ AI={ridge_point:.0f}",
            fontsize=9, color="gray")

    # Plot each operator
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, (c, info) in enumerate(cats.items()):
        if info["tflops"] <= 0:
            continue
        ax.scatter(info["ai"], info["tflops"],
                   s=150, color=colors[i], edgecolor="black", linewidth=1.2,
                   zorder=5, label=f"{c} ({info['time_us']:.0f}μs)")
        # Label
        ax.annotate(c, (info["ai"], info["tflops"]),
                    textcoords="offset points", xytext=(8, 5),
                    fontsize=9)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(ai_min, ai_max)
    ax.set_ylim(0.5, peak_tflops * 2)
    ax.set_xlabel("Arithmetic intensity (FLOPs / byte)", fontsize=11)
    ax.set_ylabel("Achieved throughput (TFLOPS)", fontsize=11)
    ax.set_title(
        "Qwen2.5-3B decode roofline\n"
        f"batch=1, A100-SXM4-80GB @ 400W, bf16, torch 2.5.1",
        fontsize=12,
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=150)
    print(f"\nSaved plot to: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()