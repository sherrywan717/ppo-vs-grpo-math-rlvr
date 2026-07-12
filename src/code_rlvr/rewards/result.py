"""Canonical verifier outcomes and the single scalar reward policy."""

from dataclasses import dataclass
from enum import StrEnum


class RewardStatus(StrEnum):
    VERIFIED_PASS = "verified_pass"
    VERIFIED_FAIL = "verified_fail"
    FORMAT_ERROR = "format_error"
    TIMEOUT = "timeout"
    RESOURCE_LIMIT = "resource_limit"
    SANDBOX_UNAVAILABLE = "sandbox_unavailable"
    INFRA_ERROR = "infra_error"


@dataclass(frozen=True)
class RewardResult:
    status: RewardStatus
    detail: str = ""


class RewardInfrastructureError(RuntimeError):
    """A run-stopping verifier infrastructure failure."""

    def __init__(self, result: RewardResult) -> None:
        super().__init__(f"{result.status.value}: {result.detail}")
        self.result = result


@dataclass(frozen=True)
class RewardPolicy:
    verified_pass: float = 1.0
    verified_fail: float = 0.0
    format_error: float = -0.2
    timeout: float = -0.5
    resource_limit: float = -0.5

    def to_scalar(self, result: RewardResult) -> float:
        if result.status in {
            RewardStatus.SANDBOX_UNAVAILABLE,
            RewardStatus.INFRA_ERROR,
        }:
            raise RewardInfrastructureError(result)
        scores = {
            RewardStatus.VERIFIED_PASS: self.verified_pass,
            RewardStatus.VERIFIED_FAIL: self.verified_fail,
            RewardStatus.FORMAT_ERROR: self.format_error,
            RewardStatus.TIMEOUT: self.timeout,
            RewardStatus.RESOURCE_LIMIT: self.resource_limit,
        }
        return scores[result.status]


DEFAULT_REWARD_POLICY = RewardPolicy()
