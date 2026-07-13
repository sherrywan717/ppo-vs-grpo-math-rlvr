"""PyTorch allocator evidence with strict CUDA device normalization."""

from __future__ import annotations

import inspect
import re
from typing import Any

import torch

_CUDA_DEVICE = re.compile(r"cuda(?::(?:0|[1-9][0-9]*))?\Z")


def _memory_value(bytes_value: int | None) -> dict[str, int | float | None]:
    if bytes_value is None:
        return {"bytes": None, "mib": None}
    return {"bytes": int(bytes_value), "mib": float(bytes_value) / (1024 * 1024)}


def _safe_device_info(value: object) -> dict[str, object]:
    try:
        representation = repr(value)
    except Exception:
        representation = "<repr unavailable>"
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "safe_repr": representation[:160],
        "callable": callable(value),
        "is_function": inspect.isfunction(value),
        "is_bool": isinstance(value, bool),
        "is_int": isinstance(value, int) and not isinstance(value, bool),
        "is_string": isinstance(value, str),
        "is_torch_device": isinstance(value, torch.device),
    }


def normalize_cuda_device_index(device: object, cuda_backend) -> int:
    """Return a validated integer CUDA index for allocator APIs."""
    if callable(device):
        raise TypeError("CUDA device must be a value, not a callable")
    if isinstance(device, bool):
        raise TypeError("boolean is not a valid CUDA device index")
    if isinstance(device, int):
        index = device
    elif isinstance(device, torch.device):
        if device.type != "cuda":
            raise ValueError("only CUDA torch.device values are allowed")
        index = device.index
    elif isinstance(device, str):
        if not _CUDA_DEVICE.fullmatch(device):
            raise ValueError("CUDA device string must be exactly 'cuda' or 'cuda:<index>'")
        parsed = torch.device(device)
        index = parsed.index
    elif device is None:
        index = None
    else:
        raise TypeError("unsupported CUDA device value")

    if index is None:
        index = cuda_backend.current_device()
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("resolved CUDA device index must be an integer")
    if index < 0:
        raise ValueError("CUDA device index must be non-negative")
    count = cuda_backend.device_count()
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise RuntimeError("CUDA device_count must be a positive integer")
    if index >= count:
        raise ValueError(f"CUDA device index {index} is out of range for {count} devices")
    return index


class CudaAllocatorEvidence:
    def __init__(self, cuda_backend, device: object = "cuda:0"):
        self.cuda = cuda_backend
        self.requested_device = device
        self.device_index: int | None = None
        self.device_label: str | None = None
        self.device_name: str | None = None
        self.state = "not_started"
        self.failure_phase: str | None = None
        self.exception_type: str | None = None
        self.exception_message: str | None = None
        self.unavailable_reason: str | None = None

    @property
    def available(self) -> bool:
        return self.state in {"active", "finalized"}

    @property
    def started(self) -> bool:
        return self.state in {"active", "finalized"}

    def _record_failure(self, phase: str, exc: Exception) -> None:
        self.state = "failed"
        self.failure_phase = phase
        self.exception_type = type(exc).__name__
        self.exception_message = str(exc)[:240]
        self.unavailable_reason = f"{phase}: {type(exc).__name__}: {exc}"[:320]

    def start(self) -> None:
        if self.state != "not_started":
            raise RuntimeError(f"allocator evidence cannot start from state {self.state}")
        if not bool(self.cuda.is_available()):
            self.state = "unavailable"
            self.unavailable_reason = "CUDA unavailable; allocator metrics not collected"
            return
        try:
            index = normalize_cuda_device_index(self.requested_device, self.cuda)
        except Exception as exc:
            self._record_failure("device_normalization", exc)
            raise
        self.device_index = index
        self.device_label = f"cuda:{index}"
        try:
            self.device_name = str(self.cuda.get_device_name(index))
            self.cuda.reset_peak_memory_stats(index)
        except Exception as exc:
            self._record_failure("reset_peak_memory_stats", exc)
            raise
        self.state = "active"

    def _unavailable_snapshot(self) -> dict[str, Any]:
        return {
            "available": False,
            "state": self.state,
            "device_index": self.device_index,
            "device_label": self.device_label,
            "device_name": self.device_name,
            "requested_device": _safe_device_info(self.requested_device),
            "max_memory_allocated": _memory_value(None),
            "max_memory_reserved": _memory_value(None),
            "memory_allocated": _memory_value(None),
            "memory_reserved": _memory_value(None),
            "failure_phase": self.failure_phase,
            "exception_type": self.exception_type,
            "exception_message": self.exception_message,
            "unavailable_reason": self.unavailable_reason
            or "allocator collection was not started",
        }

    def collect(self) -> dict[str, Any]:
        if not self.available or self.device_index is None:
            return self._unavailable_snapshot()
        index = self.device_index
        try:
            values = {
                "max_memory_allocated": self.cuda.max_memory_allocated(index),
                "max_memory_reserved": self.cuda.max_memory_reserved(index),
                "memory_allocated": self.cuda.memory_allocated(index),
                "memory_reserved": self.cuda.memory_reserved(index),
            }
        except Exception as exc:
            self._record_failure("allocator_collection", exc)
            raise
        return {
            "available": True,
            "state": self.state,
            "device_index": index,
            "device_label": self.device_label,
            "device_name": self.device_name,
            "requested_device": _safe_device_info(self.requested_device),
            **{name: _memory_value(value) for name, value in values.items()},
            "failure_phase": None,
            "exception_type": None,
            "exception_message": None,
            "unavailable_reason": None,
        }

    def snapshot(self) -> dict[str, Any]:
        return self.collect()

    def finalize(self) -> dict[str, Any]:
        payload = self.collect()
        if self.state == "active":
            self.state = "finalized"
            payload["state"] = "finalized"
        return payload
