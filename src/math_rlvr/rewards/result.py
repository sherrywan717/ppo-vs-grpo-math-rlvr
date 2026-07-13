from dataclasses import dataclass
from enum import StrEnum


class RewardStatus(StrEnum):
    VERIFIED_PASS = "verified_pass"
    WRONG_ANSWER = "wrong_answer"
    FORMAT_ERROR = "format_error"
    PARSE_ERROR = "parse_error"
    INVALID_EXPRESSION = "invalid_expression"
    INVALID_NUMBER_USAGE = "invalid_number_usage"
    RESOURCE_LIMIT = "resource_limit"
    INFRA_ERROR = "infra_error"


@dataclass(frozen=True)
class RewardResult:
    status: RewardStatus
    detail: str = ""


class RewardInfrastructureError(RuntimeError):
    def __init__(self, result):
        super().__init__(f"{result.status.value}: {result.detail}")
        self.result = result


@dataclass(frozen=True)
class RewardPolicy:
    format_reward: float = 0.1
    parse_reward: float = 0.1
    correct_reward: float = 0.8

    def to_scalar(self, result):
        if result.status == RewardStatus.INFRA_ERROR:
            raise RewardInfrastructureError(result)
        if result.status == RewardStatus.VERIFIED_PASS:
            return 1.0
        if result.status == RewardStatus.WRONG_ANSWER:
            return 0.2
        if result.status in {
            RewardStatus.PARSE_ERROR,
            RewardStatus.INVALID_EXPRESSION,
            RewardStatus.INVALID_NUMBER_USAGE,
            RewardStatus.RESOURCE_LIMIT,
        }:
            return 0.1
        return 0.0


DEFAULT_REWARD_POLICY = RewardPolicy()
