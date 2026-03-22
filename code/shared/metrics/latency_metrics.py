"""
Latency metrics for LoraForge benchmarking.
Tracks Time to First Token (TTFT) and Time Per Output Token (TPOT).
"""

import statistics
import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RequestLatency:
    """Latency measurements for a single inference request."""

    request_id: str
    prompt_tokens: int
    output_tokens: int
    ttft_ms: float          # Time to First Token (ms)
    tpot_ms: float          # Time Per Output Token (ms), average across output tokens
    total_latency_ms: float # End-to-end latency (ms)
    batch_size: int = 1


@dataclass
class LatencyStats:
    """Aggregated latency statistics across multiple requests."""

    num_requests: int
    batch_size: int
    prompt_length: int

    # TTFT statistics (ms)
    ttft_p50_ms: float
    ttft_p90_ms: float
    ttft_p99_ms: float
    ttft_mean_ms: float

    # TPOT statistics (ms)
    tpot_p50_ms: float
    tpot_p90_ms: float
    tpot_p99_ms: float

    # End-to-end latency (ms)
    e2e_p50_ms: float
    e2e_p90_ms: float
    e2e_p99_ms: float

    # Derived
    throughput_tokens_per_sec: float
    total_output_tokens: int

    raw_requests: List[RequestLatency] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict:
        return {
            "num_requests": self.num_requests,
            "batch_size": self.batch_size,
            "prompt_length": self.prompt_length,
            "ttft": {
                "p50_ms": self.ttft_p50_ms,
                "p90_ms": self.ttft_p90_ms,
                "p99_ms": self.ttft_p99_ms,
                "mean_ms": self.ttft_mean_ms,
            },
            "tpot": {
                "p50_ms": self.tpot_p50_ms,
                "p90_ms": self.tpot_p90_ms,
                "p99_ms": self.tpot_p99_ms,
            },
            "e2e": {
                "p50_ms": self.e2e_p50_ms,
                "p90_ms": self.e2e_p90_ms,
                "p99_ms": self.e2e_p99_ms,
            },
            "throughput_tokens_per_sec": self.throughput_tokens_per_sec,
        }

    def summary_line(self) -> str:
        return (
            f"BS={self.batch_size} | "
            f"TTFT P50={self.ttft_p50_ms:.0f}ms P99={self.ttft_p99_ms:.0f}ms | "
            f"Throughput={self.throughput_tokens_per_sec:.0f} tok/s"
        )


def _percentile(data: List[float], p: float) -> float:
    """Compute percentile p (0-100) of a list of values."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(sorted_data) - 1)
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (k - lo)


def compute_latency_stats(
    requests: List[RequestLatency],
    batch_size: int,
    prompt_length: int,
    wall_clock_seconds: Optional[float] = None,
) -> LatencyStats:
    """
    Compute aggregated latency statistics from a list of RequestLatency objects.

    Args:
        requests: List of per-request latency measurements.
        batch_size: Concurrency level used during the benchmark.
        prompt_length: Input prompt length (tokens) for this benchmark run.
        wall_clock_seconds: Total wall clock time for all requests.
            If None, computed from individual latencies.

    Returns:
        LatencyStats with all percentile metrics.
    """
    if not requests:
        raise ValueError("Cannot compute stats on empty request list.")

    ttft_values = [r.ttft_ms for r in requests]
    tpot_values = [r.tpot_ms for r in requests]
    e2e_values = [r.total_latency_ms for r in requests]

    total_output_tokens = sum(r.output_tokens for r in requests)

    if wall_clock_seconds is None:
        wall_clock_seconds = sum(r.total_latency_ms for r in requests) / 1000.0 / batch_size

    throughput = total_output_tokens / wall_clock_seconds if wall_clock_seconds > 0 else 0.0

    return LatencyStats(
        num_requests=len(requests),
        batch_size=batch_size,
        prompt_length=prompt_length,
        ttft_p50_ms=_percentile(ttft_values, 50),
        ttft_p90_ms=_percentile(ttft_values, 90),
        ttft_p99_ms=_percentile(ttft_values, 99),
        ttft_mean_ms=statistics.mean(ttft_values),
        tpot_p50_ms=_percentile(tpot_values, 50),
        tpot_p90_ms=_percentile(tpot_values, 90),
        tpot_p99_ms=_percentile(tpot_values, 99),
        e2e_p50_ms=_percentile(e2e_values, 50),
        e2e_p90_ms=_percentile(e2e_values, 90),
        e2e_p99_ms=_percentile(e2e_values, 99),
        throughput_tokens_per_sec=throughput,
        total_output_tokens=total_output_tokens,
        raw_requests=requests,
    )


class LatencyTimer:
    """
    Helper for measuring TTFT and total latency in streaming inference.

    Usage:
        timer = LatencyTimer()
        timer.start()
        async for token in model.stream(prompt):
            if timer.first_token_time is None:
                timer.record_first_token()
            # process token
        result = timer.finish(num_output_tokens=len(tokens))
    """

    def __init__(self) -> None:
        self._start: float = 0.0
        self._first_token: Optional[float] = None
        self._end: Optional[float] = None

    def start(self) -> None:
        self._start = time.perf_counter()
        self._first_token = None
        self._end = None

    def record_first_token(self) -> None:
        if self._first_token is None:
            self._first_token = time.perf_counter()

    def finish(self, num_output_tokens: int) -> tuple[float, float, float]:
        """
        Finish timing and return (ttft_ms, tpot_ms, total_ms).

        Args:
            num_output_tokens: Number of tokens generated.

        Returns:
            Tuple of (ttft_ms, tpot_ms, total_ms).
        """
        self._end = time.perf_counter()
        total_ms = (self._end - self._start) * 1000.0
        ttft_ms = ((self._first_token or self._end) - self._start) * 1000.0

        if num_output_tokens > 1 and self._first_token is not None:
            decode_time_ms = (self._end - self._first_token) * 1000.0
            tpot_ms = decode_time_ms / (num_output_tokens - 1)
        else:
            tpot_ms = 0.0

        return ttft_ms, tpot_ms, total_ms
