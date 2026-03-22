"""
GPU profiler for LoraForge.
Tracks VRAM usage, GPU utilization, and memory bandwidth as context managers.
"""

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator, List, Optional

import torch

try:
    import pynvml

    PYNVML_AVAILABLE = True
    pynvml.nvmlInit()
except ImportError:
    PYNVML_AVAILABLE = False


@dataclass
class MemorySnapshot:
    """Point-in-time GPU memory measurement."""

    timestamp: float
    allocated_gb: float
    reserved_gb: float
    free_gb: float
    utilization_pct: float = 0.0


@dataclass
class GPUProfileResult:
    """Aggregated GPU profiling result for a code block."""

    peak_allocated_gb: float
    peak_reserved_gb: float
    avg_utilization_pct: float
    duration_seconds: float
    snapshots: List[MemorySnapshot] = field(default_factory=list)

    @property
    def model_footprint_estimate_gb(self) -> float:
        """Estimate model weight footprint (min allocated during run)."""
        if not self.snapshots:
            return 0.0
        return min(s.allocated_gb for s in self.snapshots)

    def summary(self) -> str:
        return (
            f"Peak VRAM: {self.peak_allocated_gb:.2f}GB | "
            f"Avg GPU util: {self.avg_utilization_pct:.1f}% | "
            f"Duration: {self.duration_seconds:.1f}s"
        )


class GPUProfiler:
    """
    Context manager for profiling GPU memory and utilization.

    Usage:
        with GPUProfiler(poll_interval=0.5) as prof:
            run_inference(model, inputs)
        print(prof.result.summary())
    """

    def __init__(self, device: int = 0, poll_interval: float = 1.0) -> None:
        """
        Args:
            device: CUDA device index to profile.
            poll_interval: Seconds between memory/utilization polls.
        """
        self.device = device
        self.poll_interval = poll_interval
        self.result: Optional[GPUProfileResult] = None
        self._snapshots: List[MemorySnapshot] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_time: float = 0.0

    def _poll_loop(self) -> None:
        """Background thread: polls GPU stats every poll_interval seconds."""
        while not self._stop_event.is_set():
            snapshot = self._take_snapshot()
            self._snapshots.append(snapshot)
            time.sleep(self.poll_interval)

    def _take_snapshot(self) -> MemorySnapshot:
        allocated = torch.cuda.memory_allocated(self.device) / 1024**3
        reserved = torch.cuda.memory_reserved(self.device) / 1024**3
        total = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
        free = total - reserved

        utilization = 0.0
        if PYNVML_AVAILABLE:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(self.device)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                utilization = float(util.gpu)
            except Exception:
                pass

        return MemorySnapshot(
            timestamp=time.monotonic(),
            allocated_gb=allocated,
            reserved_gb=reserved,
            free_gb=free,
            utilization_pct=utilization,
        )

    def __enter__(self) -> "GPUProfiler":
        if not torch.cuda.is_available():
            return self
        torch.cuda.reset_peak_memory_stats(self.device)
        self._snapshots = []
        self._stop_event.clear()
        self._start_time = time.monotonic()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

        duration = time.monotonic() - self._start_time

        if not torch.cuda.is_available() or not self._snapshots:
            self.result = GPUProfileResult(
                peak_allocated_gb=0.0,
                peak_reserved_gb=0.0,
                avg_utilization_pct=0.0,
                duration_seconds=duration,
                snapshots=[],
            )
            return

        peak_allocated = torch.cuda.max_memory_allocated(self.device) / 1024**3
        peak_reserved = torch.cuda.max_memory_reserved(self.device) / 1024**3
        avg_util = (
            sum(s.utilization_pct for s in self._snapshots) / len(self._snapshots)
            if self._snapshots
            else 0.0
        )

        self.result = GPUProfileResult(
            peak_allocated_gb=peak_allocated,
            peak_reserved_gb=peak_reserved,
            avg_utilization_pct=avg_util,
            duration_seconds=duration,
            snapshots=self._snapshots,
        )


@contextmanager
def profile_gpu(device: int = 0, poll_interval: float = 1.0) -> Generator[GPUProfiler, None, None]:
    """
    Convenience context manager for GPU profiling.

    Example:
        with profile_gpu() as prof:
            run_benchmark()
        print(f"Peak VRAM: {prof.result.peak_allocated_gb:.2f} GB")
    """
    profiler = GPUProfiler(device=device, poll_interval=poll_interval)
    with profiler:
        yield profiler


def get_gpu_info(device: int = 0) -> dict:
    """Return basic GPU info as a dict."""
    if not torch.cuda.is_available():
        return {"available": False}
    props = torch.cuda.get_device_properties(device)
    return {
        "available": True,
        "name": props.name,
        "total_memory_gb": props.total_memory / 1024**3,
        "compute_capability": f"{props.major}.{props.minor}",
        "multi_processor_count": props.multi_processor_count,
    }
