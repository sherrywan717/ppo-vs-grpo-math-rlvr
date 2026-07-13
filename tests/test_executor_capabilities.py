import pytest

from math_rlvr.execution.capabilities import CapabilityReport, ToolCapability
from math_rlvr.execution.interface import ExecutionRequest, SafeExecutor, UnsafeExecutionError


def test_executor_fails_closed_without_verified_backend() -> None:
    missing = ToolCapability(False, False, "missing")
    report = CapabilityReport(missing, missing, missing, missing, None)
    with pytest.raises(UnsafeExecutionError, match="Refusing untrusted code"):
        SafeExecutor(report).execute_untrusted(ExecutionRequest("print('unsafe')", ()))


def test_tool_presence_never_implicitly_enables_backend() -> None:
    usable = ToolCapability(True, True, "trusted no-op passed")
    report = CapabilityReport(usable, usable, usable, usable, None)
    assert not report.can_execute_untrusted
