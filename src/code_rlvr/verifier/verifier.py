"""Shared syntax and execution verifier."""

import ast
from dataclasses import dataclass

from code_rlvr.execution.interface import ExecutionRequest, SafeExecutor


@dataclass(frozen=True)
class VerificationResult:
    syntax_valid: bool
    tests_passed: bool
    detail: str


def verify(source: str, tests: tuple[str, ...], executor: SafeExecutor) -> VerificationResult:
    try:
        ast.parse(source)
    except SyntaxError as error:
        return VerificationResult(False, False, str(error))
    result = executor.execute_untrusted(ExecutionRequest(source=source, tests=tests))
    return VerificationResult(True, result.passed, result.stderr)

