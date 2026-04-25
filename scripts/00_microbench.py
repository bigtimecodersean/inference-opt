"""
Micro-benchmark: measure this pod's achievable HBM bandwidth and
tensor-core compute. Use these as roofline ceilings.
"""
import torch

assert torch.cuda.is_available()
device = "cuda"

print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

# --- HBM bandwidth ---
N = 1 << 28
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
bw_tbs = bytes_per_copy / elapsed_s / 1e12
print(f"HBM bandwidth (FP32 copy): {bw_tbs:.2f} TB/s")
print(f"  Theoretical A100 peak:     2.04 TB/s")
print(f"  Achieved fraction of peak: {bw_tbs/2.04*100:.1f}%")
print()

del x, y
torch.cuda.empty_cache()

# --- FP16 tensor core compute ---
M = 8192
a = torch.randn(M, M, dtype=torch.float16, device=device)
b = torch.randn(M, M, dtype=torch.float16, device=device)

for _ in range(5):
    _ = a @ b
torch.cuda.synchronize()

start.record()
for _ in range(iters):
    c = a @ b
end.record()
torch.cuda.synchronize()

elapsed_s = start.elapsed_time(end) / 1000 / iters
flops_per_matmul = 2 * M * M * M
tflops = flops_per_matmul / elapsed_s / 1e12
print(f"FP16 tensor core (matmul): {tflops:.1f} TFLOPS")
print(f"  Theoretical A100 peak:     312 TFLOPS")
print(f"  Achieved fraction of peak: {tflops/312*100:.1f}%")
print()

ridge = tflops / bw_tbs
print(f"Measured ridge point: {ridge:.1f} FLOPs/byte")
print(f"  Below this AI -> memory-bound")
print(f"  Above this AI -> compute-bound")
