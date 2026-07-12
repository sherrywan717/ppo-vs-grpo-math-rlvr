import pytest

from code_rlvr.execution.capabilities import CapabilityReport
from code_rlvr.execution.interface import ExecutionRequest, SafeExecutor, UnsafeExecutionError


def test_executor_fails_closed_without_verified_backend() -> None:
    report = CapabilityReport(False, False, False, False, False, None)
    with pytest.raises(UnsafeExecutionError, match="Refusing untrusted code"):
        SafeExecutor(report).execute_untrusted(ExecutionRequest("print('unsafe')", ()))

