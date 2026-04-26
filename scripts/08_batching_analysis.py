"""
Week 2 — Latency and throughput analysis plots.

Two charts beyond the AI-trajectory roofline:

  1. TPOT vs batch size — shows that per-token latency is invariant
     through the memory-bound regime, then jumps at the ridge crossing.
     This is the clearest visualization of "decode is memory-bound."

  2. Throughput cost-efficiency — aggregate throughput and per-user
     effective throughput, both as a function of batch. The knee in
     these curves is the economically optimal operating point.

Both plots read from results/batching/summary.csv.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# --- Paths ---
BATCH_CSV = Path("results/batching/summary.csv")
OUTPUT_TPOT = Path("results/batching/tpot_vs_batch.png")
OUTPUT_THROUGHPUT = Path("results/batching/throughput_vs_batch.png")


def main():
    df = pd.read_csv(BATCH_CSV)
    # Drop OOM rows in case we ever produce them
    df = df[~df["oom"]].copy()
    # Defensive sort
    df = df.sort_values("batch_size").reset_index(drop=True)

    print("Loaded batching sweep data:")
    print(df[["batch_size", "ttft_ms", "tpot_ms", "throughput_tok_per_s"]].to_string(index=False))
    print()

    # =================================================================
    # Plot 1 — TPOT vs batch size
    # =================================================================
    # Story: the line stays flat from batch=1 to batch=128, then jumps
    # at batch=256 where AI crosses the ridge point. We annotate this.
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df["batch_size"], df["tpot_ms"],
            "o-", color="#1f77b4", linewidth=2, markersize=10,
            label="Median TPOT")
    ax.plot(df["batch_size"], df["tpot_p99_ms"],
            "s--", color="#ff7f0e", linewidth=1.5, markersize=8,
            alpha=0.7, label="p99 TPOT")

    # Annotate the regime transition. The TPOT jump between batch=128
    # and batch=256 is the visible ridge crossing.
    ax.axvspan(160, 200, alpha=0.15, color="gray")
    ax.text(180, df["tpot_ms"].max() * 0.95,
            "ridge crossing\n(AI ~ 144)", ha="center", fontsize=10,
            color="gray", style="italic")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Batch size", fontsize=11)
    ax.set_ylabel("TPOT (ms)", fontsize=11)
    ax.set_title(
        "Per-token decode latency vs batch size\n"
        "Qwen2.5-3B, prompt=512, A100-SXM4-80GB",
        fontsize=12,
    )
    ax.set_xticks(df["batch_size"])
    ax.set_xticklabels(df["batch_size"])
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=10)

    # Y-axis from 0 so the "flat line" is visually convincing.
    # If TPOT bounces around 26-28, plotting from 0 makes that range
    # look like a perfect plateau, which is the point.
    ax.set_ylim(0, df["tpot_p99_ms"].max() * 1.1)

    fig.tight_layout()
    fig.savefig(OUTPUT_TPOT, dpi=150)
    print(f"Saved: {OUTPUT_TPOT}")

    # =================================================================
    # Plot 2 — Throughput and per-user effective rate
    # =================================================================
    # Two lines: aggregate throughput (operator's view) and per-user
    # effective throughput (user's view). Their divergence shows the
    # economic story: aggregate keeps rising, per-user falls.
    fig, ax = plt.subplots(figsize=(10, 6))

    df["per_user_tok_per_s"] = df["throughput_tok_per_s"] / df["batch_size"]

    color_agg = "#2ca02c"  # green
    color_user = "#d62728"  # red

    # Aggregate throughput — left y-axis
    ax.plot(df["batch_size"], df["throughput_tok_per_s"],
            "o-", color=color_agg, linewidth=2, markersize=10,
            label="Aggregate throughput")
    ax.set_xlabel("Batch size", fontsize=11)
    ax.set_ylabel("Aggregate throughput (tokens/sec)",
                  fontsize=11, color=color_agg)
    ax.tick_params(axis="y", labelcolor=color_agg)
    ax.set_xscale("log", base=2)
    ax.set_xticks(df["batch_size"])
    ax.set_xticklabels(df["batch_size"])
    ax.grid(True, which="both", alpha=0.3)

    # Per-user effective throughput — right y-axis
    ax2 = ax.twinx()
    ax2.plot(df["batch_size"], df["per_user_tok_per_s"],
             "s--", color=color_user, linewidth=2, markersize=10,
             label="Per-user effective rate")
    ax2.set_ylabel("Per-user effective throughput (tokens/sec/user)",
                   fontsize=11, color=color_user)
    ax2.tick_params(axis="y", labelcolor=color_user)

    # Combined legend (both axes)
    lines_a, labels_a = ax.get_legend_handles_labels()
    lines_b, labels_b = ax2.get_legend_handles_labels()
    ax.legend(lines_a + lines_b, labels_a + labels_b,
              loc="center left", fontsize=10)

    ax.set_title(
        "Throughput economics: operator vs per-user view\n"
        "Qwen2.5-3B, prompt=512, A100-SXM4-80GB",
        fontsize=12,
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_THROUGHPUT, dpi=150)
    print(f"Saved: {OUTPUT_THROUGHPUT}")


if __name__ == "__main__":
    main()