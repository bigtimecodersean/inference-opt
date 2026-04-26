"""
Week 2 — Batching roofline.

Plot one point per batch size from results/batching/summary.csv on the
roofline. The headline story: as batch grows, the workload slides right
along the memory-bound ceiling, then crosses the ridge and starts being
limited by compute instead.

What we plot:
  - Same hardware ceilings as Week 1 (HBM, peak compute, ridge point)
  - One point per batch size at the FFN up-projection's analytical AI
    and achieved TFLOPS
  - Connecting line so the trajectory is visible
  - Color graded from cool (small batch) to warm (large batch)
  - Annotation at the ridge crossing

Why FFN up-projection specifically:
  See scripts/06_batching_sweep.py — it's the largest matmul in the
  model and a representative example of how all linear ops behave with
  batch. Attention has different scaling (its AI doesn't grow with
  batch because each sequence has its own KV cache), so it's plotted
  separately if at all.

Achieved TFLOPS for the FFN up-projection at each batch:
  We don't have a per-op profile for every batch size — running
  04_profile_pytorch.py at every batch would be expensive. Instead we
  estimate it from the measured TPOT.

  Per decode step, the model runs all 36 layers × 4 linear projections.
  For each layer the FFN up-projection is responsible for some fraction
  of the wall-clock time. We use the same proportional apportionment
  as Week 1's roofline:

      time_per_op_per_layer = TPOT * (op_flops / total_decode_flops)

  This isn't perfect (assumes all linear ops achieve similar efficiency)
  but for a roofline trajectory plot it captures the trend correctly:
  the linear ops migrate as a group as batch grows.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# --- Paths ---
RESULTS = Path("results")
MICROBENCH = RESULTS / "baseline" / "microbench.json"
BATCH_CSV = RESULTS / "batching" / "summary.csv"
OUTPUT_PNG = RESULTS / "batching" / "roofline.png"


# --- Qwen2.5-3B architecture (same as Week 1 roofline script) ---
HIDDEN = 2048
INTERMEDIATE = 11008
NUM_LAYERS = 36
NUM_Q_HEADS = 16
NUM_KV_HEADS = 2
HEAD_DIM = HIDDEN // NUM_Q_HEADS
DTYPE_BYTES = 2

Q_OUT = NUM_Q_HEADS * HEAD_DIM
KV_OUT = NUM_KV_HEADS * HEAD_DIM
QKV_OUT = Q_OUT + 2 * KV_OUT


def ffn_up_flops_and_bytes(batch_size: int):
    """
    Per-decode-step FLOPs and bytes for ALL FFN up-projections across
    all layers. (One per layer × NUM_LAYERS layers.)
    """
    flops_per_layer = 2 * batch_size * HIDDEN * INTERMEDIATE
    weight_bytes_per_layer = HIDDEN * INTERMEDIATE * DTYPE_BYTES
    act_in_bytes = batch_size * HIDDEN * DTYPE_BYTES
    act_out_bytes = batch_size * INTERMEDIATE * DTYPE_BYTES
    bytes_per_layer = weight_bytes_per_layer + act_in_bytes + act_out_bytes
    return {
        "flops": flops_per_layer * NUM_LAYERS,
        "bytes": bytes_per_layer * NUM_LAYERS,
    }


def all_linear_flops_per_step(batch_size: int):
    """
    Total FLOPs for all four linear projections (QKV, O, FFN gate+up,
    FFN down) across all layers in one decode step. We use this to
    apportion TPOT into a "linear ops only" time budget so the FFN
    up-projection's TFLOPS estimate doesn't include attention or
    elementwise time.

    Per-layer breakdown:
      qkv:       2 * B * hidden * (q_out + 2*kv_out)
      o:         2 * B * hidden * hidden
      gate+up:   2 * (2 * B * hidden * intermediate)
      down:      2 * B * intermediate * hidden
    """
    qkv = 2 * batch_size * HIDDEN * QKV_OUT
    o = 2 * batch_size * HIDDEN * HIDDEN
    gate_up = 2 * (2 * batch_size * HIDDEN * INTERMEDIATE)
    down = 2 * batch_size * INTERMEDIATE * HIDDEN
    return (qkv + o + gate_up + down) * NUM_LAYERS


def main():
    # --- Load measured ceilings (per-pod) ---
    with open(MICROBENCH) as f:
        mb = json.load(f)
    bw_tbs = mb["hbm_bandwidth_measured_tbs"]
    peak_tflops = mb["compute_fp16_measured_tflops"]
    ridge = mb["ridge_point_measured_flops_per_byte"]

    # --- Load sweep data ---
    df = pd.read_csv(BATCH_CSV)
    df = df[~df["oom"]].copy()  # drop the OOM row(s) if any

    # For each batch size, compute the analytical AI and achieved TFLOPS
    # for the FFN up-projection.
    rows = []
    for _, r in df.iterrows():
        B = int(r["batch_size"])
        ffn = ffn_up_flops_and_bytes(B)
        ai = ffn["flops"] / ffn["bytes"]

        # Apportion TPOT to FFN up-projection only.
        # FFN up's share of total linear FLOPs at this batch:
        ffn_share_of_linear = ffn["flops"] / all_linear_flops_per_step(B)

        # Estimate the time that the FFN up-projection takes per step.
        # We assume linear ops dominate decode time; this is roughly
        # true at batch=1 (50% of GPU time) and even more so at larger
        # batches (linear ops scale; elementwise stays fixed-ish).
        # We DO NOT subtract attention or elementwise time here — for
        # the trajectory the relative position is what matters.
        tpot_s = r["tpot_ms"] / 1000.0
        # Roughly, ~50% of decode time is spent in linear ops at small
        # batch, growing toward ~75% at large batches as elementwise
        # time stays fixed while matmuls grow. We use a rough constant
        # 0.6 — the trajectory shape is robust to this estimate.
        linear_time_fraction = 0.6
        ffn_time_s = tpot_s * linear_time_fraction * ffn_share_of_linear

        achieved_tflops = ffn["flops"] / ffn_time_s / 1e12

        rows.append({
            "batch_size": B,
            "ai": ai,
            "achieved_tflops": achieved_tflops,
            "ceiling_at_ai": min(bw_tbs * ai, peak_tflops),
        })
        rows[-1]["pct_of_ceiling"] = (
            rows[-1]["achieved_tflops"] / rows[-1]["ceiling_at_ai"] * 100
        )

    plot_df = pd.DataFrame(rows)
    print("Batching trajectory (FFN up-projection):")
    print(plot_df.to_string(index=False))
    print()

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 7))

    # Ceilings
    ai_min, ai_max = 0.3, 1000
    ai_line = np.logspace(np.log10(ai_min), np.log10(ai_max), 300)
    mem_line = bw_tbs * 1e12 * ai_line
    compute_line = np.full_like(ai_line, peak_tflops * 1e12)
    envelope = np.minimum(mem_line, compute_line)

    ax.plot(ai_line, envelope / 1e12, "k-", linewidth=2.5,
            label=f"Roofline (HBM={bw_tbs:.2f} TB/s, peak={peak_tflops:.0f} TFLOPS)",
            zorder=3)
    ax.axvline(ridge, linestyle=":", color="gray", alpha=0.5, zorder=2)
    ax.text(ridge * 1.1, peak_tflops * 0.05,
            f"ridge @ AI={ridge:.0f}",
            fontsize=10, color="gray")

    # Trajectory: connect points in batch order.
    plot_df = plot_df.sort_values("batch_size").reset_index(drop=True)
    ax.plot(plot_df["ai"], plot_df["achieved_tflops"],
            "-", color="lightgray", linewidth=1.5, zorder=4)

    # Color graded by batch size (cool → warm)
    norm = plt.Normalize(vmin=plot_df["batch_size"].min(),
                         vmax=plot_df["batch_size"].max())
    cmap = plt.cm.viridis

    for _, r in plot_df.iterrows():
        color = cmap(norm(r["batch_size"]))
        ax.scatter(r["ai"], r["achieved_tflops"],
                   s=180, color=color, edgecolor="black", linewidth=1.4,
                   zorder=10,
                   label=f"batch={int(r['batch_size'])}  (AI={r['ai']:.1f}, "
                         f"{r['achieved_tflops']:.1f} TFLOPS)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(ai_min, ai_max)
    ax.set_ylim(0.3, peak_tflops * 2)
    ax.set_xlabel("Arithmetic intensity (FLOPs / byte)", fontsize=11)
    ax.set_ylabel("Achieved throughput (TFLOPS)", fontsize=11)
    ax.set_title(
        "Qwen2.5-3B FFN up-projection: batch-size trajectory on roofline\n"
        f"prompt=512, A100-SXM4-80GB @ 400W, bf16",
        fontsize=12,
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=150)
    print(f"Saved plot to: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()