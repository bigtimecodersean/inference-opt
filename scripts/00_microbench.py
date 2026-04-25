"""
Micro-benchmark: measure this pod's achievable HBM bandwidth and
tensor-core compute. Use these as roofline ceilings.

Writes results to results/baseline/microbench.json so the roofline plot
can pick up per-pod ceilings automatically.
"""
import json
from pathlib import Path

import torch

assert torch.cuda.is_available()
device = "cuda"

OUTPUT = Path("results/baseline/microbench.json")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)


def measure_hbm_bandwidth():
    """Copy a 1 GB tensor to another 1 GB tensor, repeatedly. Read+write."""
    N = 1 << 28   # 256M elements
    x = torch.randn(N, dtype=torch.float32, device=device)
    y = torch.empty_like(x)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(5):
        y.copy_(x)
    torch.cuda.synchronize()

    iters = 30
    start.record()
    for _ in range(iters):
        y.copy_(x)
    end.record()
    torch.cuda.synchronize()

    elapsed_s = start.elapsed_time(end) / 1000 / iters
    bytes_per_copy = 2 * N * 4
    tbs = bytes_per_copy / elapsed_s / 1e12

    del x, y
    torch.cuda.empty_cache()
    return tbs


def measure_fp16_tensor_core():
    """Big square matmul — should saturate tensor cores."""
    M = 8192
    a = torch.randn(M, M, dtype=torch.float16, device=device)
    b = torch.randn(M, M, dtype=torch.float16, device=device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(5):
        _ = a @ b
    torch.cuda.synchronize()

    iters = 30
    start.record()
    for _ in range(iters):
        c = a @ b
    end.record()
    torch.cuda.synchronize()

    elapsed_s = start.elapsed_time(end) / 1000 / iters
    flops_per_matmul = 2 * M * M * M
    tflops = flops_per_matmul / elapsed_s / 1e12

    del a, b
    torch.cuda.empty_cache()
    return tflops


def measure_power_cap_w():
    """Read the GPU's configured power cap via nvidia-smi."""
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=power.max_limit",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def main():
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print()

    print("Measuring HBM bandwidth...")
    bw = measure_hbm_bandwidth()
    print(f"  {bw:.2f} TB/s  ({bw/2.04*100:.1f}% of 2.04 TB/s peak)")

    print("Measuring FP16 tensor core throughput...")
    tflops = measure_fp16_tensor_core()
    print(f"  {tflops:.1f} TFLOPS  ({tflops/312*100:.1f}% of 312 TFLOPS peak)")

    power = measure_power_cap_w()
    if power:
        print(f"Power cap: {power:.0f} W")

    ridge = tflops / bw
    print(f"Measured ridge point: {ridge:.1f} FLOPs/byte")

    # Persist to JSON for downstream consumers (roofline script)
    result = {
        "gpu": gpu_name,
        "power_cap_w": power,
        "hbm_bandwidth_measured_tbs": round(bw, 3),
        "hbm_bandwidth_peak_tbs": 2.04,
        "compute_fp16_measured_tflops": round(tflops, 2),
        "compute_fp16_peak_tflops": 312,
        "ridge_point_measured_flops_per_byte": round(ridge, 2),
    }
    with open(OUTPUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote: {OUTPUT}")


if __name__ == "__main__":
    main()