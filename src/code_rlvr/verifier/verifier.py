"""Shared syntax and execution verifier."""

import ast

from code_rlvr.execution.interface import ExecutionRequest, SafeExecutor, UnsafeExecutionError
from code_rlvr.rewards.result import RewardResult, RewardStatus


def verify(source: str, tests: tuple[str, ...], executor: SafeExecutor) -> RewardResult:
    try:
        ast.parse(source)
    except SyntaxError as error:
        return RewardResult(RewardStatus.FORMAT_ERROR, str(error))
    try:
        result = executor.execute_untrusted(ExecutionRequest(source=source, tests=tests))
    except UnsafeExecutionError as error:
        return RewardResult(RewardStatus.SANDBOX_UNAVAILABLE, str(error))
    status = RewardStatus.VERIFIED_PASS if result.passed else RewardStatus.VERIFIED_FAIL
    return RewardResult(status, result.stderr)
