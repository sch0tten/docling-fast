"""Timing, throughput, and GPU utilization metrics collection."""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class DocMetrics:
    """Metrics for a single document extraction."""
    source: str = ""
    pages: int = 0
    text_length: int = 0
    tables: int = 0
    wall_time_s: float = 0.0
    success: bool = True
    error: str | None = None


@dataclass
class BatchMetrics:
    """Aggregate metrics for a batch run."""
    total_docs: int = 0
    successful_docs: int = 0
    failed_docs: int = 0
    total_pages: int = 0
    total_time_s: float = 0.0
    pages_per_second: float = 0.0
    docs_per_minute: float = 0.0
    peak_vram_mb: int = 0
    avg_gpu_util_pct: float = 0.0
    per_doc: list[dict] = field(default_factory=list)

    def finalize(self):
        if self.total_time_s > 0:
            self.pages_per_second = round(self.total_pages / self.total_time_s, 2)
            self.docs_per_minute = round((self.successful_docs / self.total_time_s) * 60, 2)

    def to_dict(self) -> dict:
        return {
            "total_docs": self.total_docs,
            "successful_docs": self.successful_docs,
            "failed_docs": self.failed_docs,
            "total_pages": self.total_pages,
            "total_time_s": round(self.total_time_s, 2),
            "pages_per_second": self.pages_per_second,
            "docs_per_minute": self.docs_per_minute,
            "peak_vram_mb": self.peak_vram_mb,
            "avg_gpu_util_pct": self.avg_gpu_util_pct,
            "per_doc": self.per_doc,
        }


@contextmanager
def timer():
    """Simple wall-clock timer context manager. Yields a dict with 'elapsed' key."""
    result = {"elapsed": 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed"] = time.perf_counter() - start


def get_peak_vram_mb() -> int:
    """Get peak GPU memory allocated by PyTorch (MB)."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() // (1024 * 1024)
    except ImportError:
        pass
    return 0


def reset_vram_stats():
    """Reset PyTorch CUDA memory tracking."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
    except ImportError:
        pass


def get_gpu_utilization() -> float:
    """Get current GPU utilization percentage via NVML."""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        pynvml.nvmlShutdown()
        return float(util.gpu)
    except Exception:
        return 0.0


def get_system_info() -> dict:
    """Collect system info for benchmark reports."""
    info = {"gpu": "unknown", "vram_gb": 0, "driver": "unknown", "torch": "unknown"}
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = props.name
            info["vram_gb"] = round(props.total_memory / (1024**3), 1)
            info["cuda_version"] = torch.version.cuda
    except ImportError:
        pass

    try:
        import pynvml

        pynvml.nvmlInit()
        info["driver"] = pynvml.nvmlSystemGetDriverVersion()
        pynvml.nvmlShutdown()
    except Exception:
        pass

    return info
