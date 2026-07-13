import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


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


class RewardPolicyError(RuntimeError):
    pass


def policy_contract_sha256(descriptor: dict[str, Any]) -> str:
    encoded = json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class RewardEvaluation:
    canonical_result: RewardResult
    scalar_reward: float
    reward_policy_version: str
    reward_policy_sha256: str
    reward_component_weights: dict[str, float]
    components: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_status": self.canonical_result.status.value,
            "scalar_reward": float(self.scalar_reward),
            "reward_policy_version": self.reward_policy_version,
            "reward_policy_sha256": self.reward_policy_sha256,
            "reward_component_weights": dict(self.reward_component_weights),
            "answer_block_component": float(self.components.get("answer_block", 0.0)),
            "strict_protocol_component": float(self.components.get("strict_protocol", 0.0)),
            "valid_expression_component": float(self.components.get("valid_expression", 0.0)),
            "exact_number_usage_component": float(
                self.components.get("exact_number_usage", 0.0)
            ),
            "correctness_component": float(self.components.get("correctness", 0.0)),
            "verifier_detail": str(self.canonical_result.detail),
        }


LEGACY_SHAPED_DESCRIPTOR = {
    "version": "shaped_v1_legacy",
    "weights": {"format": 0.1, "parse_or_semantic": 0.1, "correctness": 0.8},
    "status_mapping": {
        "verified_pass": 1.0,
        "wrong_answer": 0.2,
        "parse_or_semantic_error": 0.1,
        "format_error": 0.0,
    },
}
LEGACY_SHAPED_SHA256 = policy_contract_sha256(LEGACY_SHAPED_DESCRIPTOR)


@dataclass(frozen=True)
class RewardPolicy:
    """Historical shaped-v1 scalar mapping retained for immutable old runs."""

    format_reward: float = 0.1
    parse_reward: float = 0.1
    correct_reward: float = 0.8
    version: str = "shaped_v1_legacy"
    policy_sha256: str = LEGACY_SHAPED_SHA256

    @property
    def component_weights(self) -> dict[str, float]:
        return {
            "format": self.format_reward,
            "parse_or_semantic": self.parse_reward,
            "correctness": self.correct_reward,
        }

    def to_scalar(self, result: RewardResult) -> float:
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

    def evaluate(self, completion: str, verifier) -> RewardEvaluation:
        result = verifier(completion)
        return RewardEvaluation(
            canonical_result=result,
            scalar_reward=self.to_scalar(result),
            reward_policy_version=self.version,
            reward_policy_sha256=self.policy_sha256,
            reward_component_weights=self.component_weights,
            components={},
        )


SPARSE_DESCRIPTOR = {
    "version": "sparse_v1",
    "weights": {"verified_pass": 1.0},
    "status_mapping": {"verified_pass": 1.0, "all_other_canonical_statuses": 0.0},
}
SPARSE_REWARD_SHA256 = policy_contract_sha256(SPARSE_DESCRIPTOR)


@dataclass(frozen=True)
class SparseRewardPolicy:
    version: str = "sparse_v1"
    policy_sha256: str = SPARSE_REWARD_SHA256

    @property
    def component_weights(self) -> dict[str, float]:
        return {"verified_pass": 1.0}

    def to_scalar(self, result: RewardResult) -> float:
        if result.status == RewardStatus.INFRA_ERROR:
            raise RewardInfrastructureError(result)
        return float(result.status == RewardStatus.VERIFIED_PASS)

    def evaluate(self, completion: str, verifier) -> RewardEvaluation:
        result = verifier(completion)
        scalar = self.to_scalar(result)
        return RewardEvaluation(
            canonical_result=result,
            scalar_reward=scalar,
            reward_policy_version=self.version,
            reward_policy_sha256=self.policy_sha256,
            reward_component_weights=self.component_weights,
            components={"correctness": scalar},
        )


DEFAULT_REWARD_POLICY = RewardPolicy()
SPARSE_REWARD_POLICY = SparseRewardPolicy()
