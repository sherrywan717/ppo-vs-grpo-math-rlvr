"""PyTorch allocator evidence with an injectable CUDA backend."""

from __future__ import annotations

from typing import Any


def _memory_value(bytes_value: int | None) -> dict[str, int | float | None]:
    if bytes_value is None:
        return {"bytes": None, "mib": None}
    return {"bytes": int(bytes_value), "mib": float(bytes_value) / (1024 * 1024)}


class CudaAllocatorEvidence:
    def __init__(self, cuda_backend, device: str = "cuda:0"):
        self.cuda = cuda_backend
        self.device = device
        self.available = False
        self.unavailable_reason: str | None = None
        self.started = False

    def start(self) -> None:
        if not bool(self.cuda.is_available()):
            self.unavailable_reason = "CUDA unavailable; allocator metrics not collected"
            return
        self.cuda.reset_peak_memory_stats(self.device)
        self.available = True
        self.started = True

    def snapshot(self) -> dict[str, Any]:
        if not self.available:
            return {
                "available": False,
                "device": None,
                "max_memory_allocated": _memory_value(None),
                "max_memory_reserved": _memory_value(None),
                "memory_allocated": _memory_value(None),
                "memory_reserved": _memory_value(None),
                "unavailable_reason": self.unavailable_reason
                or "allocator collection was not started",
            }
        return {
            "available": True,
            "device": self.device,
            "max_memory_allocated": _memory_value(
                self.cuda.max_memory_allocated(self.device)
            ),
            "max_memory_reserved": _memory_value(
                self.cuda.max_memory_reserved(self.device)
            ),
            "memory_allocated": _memory_value(self.cuda.memory_allocated(self.device)),
            "memory_reserved": _memory_value(self.cuda.memory_reserved(self.device)),
            "unavailable_reason": None,
        }
