"""
Benchmark result container.

Terminology:

TTFT (Time To First Token): wall time from the start of prefill to the
moment the first generated token is available. Dominated by the cost of
processing the full prompt through the model. Scales with prompt length.

TPOT (Time Per Output Token): inter-token latency during decode. The time
between successive generated tokens once generation is underway. We report
the *median* because the first decode step after prefill can be anomalous
(cache handoff, allocator behavior), and we want the steady-state number.

Throughput: (batch_size × output_tokens) / wall_time. The aggregate
tokens/sec across the batch — this is the number that determines serving
cost. Goes up with batch size until you hit the ridge point of the
roofline (or run out of memory).
"""

import statistics
from dataclasses import dataclass
from typing import List


@dataclass
class BenchmarkResult:
    # Configuration
    batch_size: int
    prompt_length: int
    output_length: int
    dtype: str

    # Raw timing data
    ttft_ms: float                        # time to first token
    per_token_latencies_ms: List[float]   # one entry per decode step after the first
    total_time_ms: float                  # full end-to-end per run (one run = one generation)

    @property
    def tpot_ms(self) -> float:
        """Median per-token decode latency."""
        if not self.per_token_latencies_ms:
            return float("nan")
        return statistics.median(self.per_token_latencies_ms)

    @property
    def tpot_p99_ms(self) -> float:
        """99th-percentile per-token latency — reveals tail behavior."""
        if len(self.per_token_latencies_ms) < 10:
            return float("nan")
        return statistics.quantiles(self.per_token_latencies_ms, n=100)[98]

    @property
    def throughput_tok_per_s(self) -> float:
        """Aggregate decode throughput across the batch, tokens/sec."""
        total_tokens = self.batch_size * self.output_length
        return total_tokens / (self.total_time_ms / 1000.0)

    def summary_dict(self) -> dict:
        """Flat dict suitable for CSV / DataFrame."""
        return {
            "batch_size": self.batch_size,
            "prompt_length": self.prompt_length,
            "output_length": self.output_length,
            "dtype": self.dtype,
            "ttft_ms": self.ttft_ms,
            "tpot_ms": self.tpot_ms,
            "tpot_p99_ms": self.tpot_p99_ms,
            "throughput_tok_per_s": self.throughput_tok_per_s,
            "total_time_ms": self.total_time_ms,
        }