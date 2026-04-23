"""
Timing and GPU utilities.

Core insight driving this file: CPU-side timing (time.time, perf_counter) is
*wrong* for GPU work. CUDA kernels launch asynchronously — the CPU returns
immediately after enqueueing kernels onto the GPU stream. A CPU timer would
therefore measure launch overhead, not actual work.

The correct primitive is a CUDA Event: a marker recorded on the GPU stream
itself. Paired with torch.cuda.synchronize() (block CPU until stream drains),
you get real GPU execution time.
"""

import subprocess
import torch


def cuda_sync():
    """Block CPU until all queued GPU work completes."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class CudaTimer:
    """
    Context manager for accurate GPU timing using CUDA events.

    Usage:
        with CudaTimer() as t:
            # ... GPU work ...
        print(t.elapsed_ms)

    Why events + sync both sides:
    - Sync on entry: ensures previously-queued work has finished, so we
      don't count it in this measurement.
    - start.record()/end.record(): these are the actual timestamps on the
      GPU stream.
    - Sync on exit: elapsed_time() requires the end event to have
      completed; without sync, you'd race with the GPU.
    """

    def __init__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.elapsed_ms = None

    def __enter__(self):
        torch.cuda.synchronize()
        self.start.record()
        return self

    def __exit__(self, *_):
        self.end.record()
        torch.cuda.synchronize()
        self.elapsed_ms = self.start.elapsed_time(self.end)


def log_gpu_info():
    """Print GPU specs. Include this in every run for reproducibility."""
    if not torch.cuda.is_available():
        print("No CUDA device found.")
        return

    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}")
    print(f"  Compute capability: {props.major}.{props.minor}")
    print(f"  Total memory: {props.total_memory / 1e9:.1f} GB")
    print(f"  SMs: {props.multi_processor_count}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA runtime: {torch.version.cuda}")

    # GPUs throttle under sustained load, which poisons benchmarks. Before
    # serious measurement runs:
    #   sudo nvidia-smi -pm 1                 # persistence mode on
    #   sudo nvidia-smi -lgc <min>,<max>      # lock graphics clock
    # For an A100 you might lock to 1410 MHz (its boost clock). Check the
    # supported range with: nvidia-smi --query-supported-clocks=gr --format=csv
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=clocks.current.graphics,clocks.max.graphics",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        current, maximum = result.stdout.strip().split(", ")
        print(f"  Graphics clock: {current} / {maximum} MHz (current/max)")
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass