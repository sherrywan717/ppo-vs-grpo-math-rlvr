"""LEGACY/OUT-OF-SCOPE: code execution is retained for history and must not be used."""

from dataclasses import dataclass

from math_rlvr.execution.capabilities import CapabilityReport, detect_capabilities


class UnsafeExecutionError(RuntimeError):
    """Raised when no verified isolation backend is available."""


@dataclass(frozen=True)
class ExecutionRequest:
    source: str
    tests: tuple[str, ...]
    timeout_seconds: int = 5


@dataclass(frozen=True)
class ExecutionResult:
    passed: bool
    stdout: str
    stderr: str


class SafeExecutor:
    def __init__(self, capabilities: CapabilityReport | None = None) -> None:
        self.capabilities = capabilities or detect_capabilities()

    def execute_untrusted(self, request: ExecutionRequest) -> ExecutionResult:
        del request
        if not self.capabilities.can_execute_untrusted:
            raise UnsafeExecutionError(
                "Refusing untrusted code: no verified isolation backend is available"
            )
        raise NotImplementedError("A verified isolation adapter must be implemented first")
