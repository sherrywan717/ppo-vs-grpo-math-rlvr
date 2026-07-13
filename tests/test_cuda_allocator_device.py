import json

import pytest
import torch

from math_rlvr.training.resource_evidence import (
    CudaAllocatorEvidence,
    normalize_cuda_device_index,
)


class FakeCuda:
    def __init__(self, *, count=1, current=0, reset_error=None):
        self.count = count
        self.current = current
        self.reset_error = reset_error
        self.calls = []
        self.current_calls = 0

    def is_available(self):
        return True

    def device_count(self):
        self.calls.append(("device_count", None))
        return self.count

    def current_device(self):
        self.current_calls += 1
        self.calls.append(("current_device", None))
        return self.current

    def get_device_name(self, index):
        self.calls.append(("get_device_name", index))
        return "NVIDIA H800 PCIe"

    def reset_peak_memory_stats(self, index):
        self.calls.append(("reset", index))
        if self.reset_error is not None:
            raise self.reset_error

    def memory_allocated(self, index):
        self.calls.append(("allocated", index))
        return 1_048_576

    def memory_reserved(self, index):
        self.calls.append(("reserved", index))
        return 2_097_152

    def max_memory_allocated(self, index):
        self.calls.append(("max_allocated", index))
        return 1_048_576

    def max_memory_reserved(self, index):
        self.calls.append(("max_reserved", index))
        return 2_097_152


@pytest.mark.parametrize(
    "device",
    [0, torch.device("cuda:0"), "cuda:0"],
)
def test_explicit_cuda_devices_normalize_to_integer_zero(device):
    cuda = FakeCuda()
    assert normalize_cuda_device_index(device, cuda) == 0
    assert cuda.current_calls == 0


@pytest.mark.parametrize("device", [torch.device("cuda"), "cuda", None])
def test_implicit_cuda_devices_call_current_device(device):
    cuda = FakeCuda()
    assert normalize_cuda_device_index(device, cuda) == 0
    assert cuda.current_calls == 1


def test_current_device_is_called_instead_of_returning_function():
    cuda = FakeCuda()
    assert normalize_cuda_device_index(None, cuda) == 0
    assert cuda.current_calls == 1


@pytest.mark.parametrize(
    "device",
    [
        lambda: 0,
        FakeCuda().current_device,
        True,
        False,
        0.0,
        torch.device("cpu"),
        "NVIDIA H800 PCIe",
        "gpu0",
        "0",
        "cpu",
        -1,
        object(),
    ],
)
def test_invalid_cuda_device_values_fail_closed(device):
    with pytest.raises((TypeError, ValueError)):
        normalize_cuda_device_index(device, FakeCuda())


def test_out_of_range_index_is_rejected():
    with pytest.raises(ValueError, match="out of range"):
        normalize_cuda_device_index(1, FakeCuda(count=1))


def test_allocator_lifecycle_uses_one_integer_index():
    cuda = FakeCuda()
    evidence = CudaAllocatorEvidence(cuda, "cuda:0")
    assert evidence.state == "not_started"
    evidence.start()
    assert evidence.state == "active"
    collected = evidence.collect()
    finalized = evidence.finalize()
    assert finalized["state"] == "finalized"
    assert collected["device_index"] == finalized["device_index"] == 0
    api_calls = [value for name, value in cuda.calls if name not in {"device_count"}]
    assert all(value == 0 for value in api_calls)
    assert finalized["device_label"] == "cuda:0"
    assert finalized["device_name"] == "NVIDIA H800 PCIe"


def test_reset_failure_is_primitive_only_and_not_swallowed():
    cuda = FakeCuda(reset_error=RuntimeError("Invalid device argument"))
    evidence = CudaAllocatorEvidence(cuda, "cuda:0")
    with pytest.raises(RuntimeError, match="Invalid device argument"):
        evidence.start()
    payload = evidence.finalize()
    json.dumps(payload)
    assert payload["state"] == "failed"
    assert payload["failure_phase"] == "reset_peak_memory_stats"
    assert payload["exception_type"] == "RuntimeError"
    assert payload["requested_device"]["safe_repr"] == "'cuda:0'"
    assert payload["available"] is False


def test_third_failure_exact_string_regression_now_passes_integer():
    cuda = FakeCuda()
    evidence = CudaAllocatorEvidence(cuda, "cuda:0")
    evidence.start()
    assert ("reset", 0) in cuda.calls
    assert ("reset", "cuda:0") not in cuda.calls


def test_cpu_unavailable_path_does_not_initialize_real_cuda():
    class UnavailableCuda:
        def is_available(self):
            return False

    before = torch.cuda.is_initialized()
    evidence = CudaAllocatorEvidence(UnavailableCuda(), None)
    evidence.start()
    payload = evidence.finalize()
    assert payload["state"] == "unavailable"
    assert torch.cuda.is_initialized() is before is False
