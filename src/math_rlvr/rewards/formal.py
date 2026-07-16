"""Domain-aware shaped reward for formal GSM8K and MATH training."""

from __future__ import annotations

from dataclasses import dataclass

from math_rlvr.parser import ParsedCompletion, parse_completion
from math_rlvr.rewards.result import (
    RewardEvaluation,
    RewardInfrastructureError,
    RewardPolicyError,
    RewardResult,
    RewardStatus,
    policy_contract_sha256,
)
from math_rlvr.rewards.staged import extract_unique_answer_block

FORMAL_REWARD_VERSION = "shaped_v3_domain"
FORMAL_COMPONENT_WEIGHTS = {
    "answer_block": 0.05,
    "strict_protocol": 0.05,
    "valid_answer": 0.10,
    "correctness": 0.80,
}
FORMAL_COMPONENT_WEIGHTS_SHA256 = policy_contract_sha256(FORMAL_COMPONENT_WEIGHTS)
FORMAL_REWARD_DESCRIPTOR = {
    "version": FORMAL_REWARD_VERSION,
    "domains": ["gsm8k", "math"],
    "component_weights": FORMAL_COMPONENT_WEIGHTS,
    "format_total": "answer_block + strict_protocol = 0.10",
    "answer_block": "exactly one non-empty correctly ordered and closed answer block",
    "strict_protocol": "unchanged strict completion parser succeeds",
    "valid_answer": (
        "the domain canonical verifier parses the extracted answer and returns "
        "wrong_answer or verified_pass"
    ),
    "correctness": "the domain canonical verifier returns verified_pass",
    "countdown_exact_number_usage": "not_applicable_and_never_rewarded",
    "formal_correctness_metric": "canonical verified_pass only",
    "resource_limit": "zero all components",
    "infra_error": "raise and fail closed",
    "range": [0.0, 1.0],
}
FORMAL_REWARD_SHA256 = policy_contract_sha256(FORMAL_REWARD_DESCRIPTOR)


def _strict_probe(answer: str) -> str:
    return f"<reasoning>formal structural validation</reasoning><answer>{answer}</answer>"


@dataclass(frozen=True)
class FormalDomainRewardPolicy:
    """Shared formal reward whose validity semantics come from the domain verifier."""

    version: str = FORMAL_REWARD_VERSION
    policy_sha256: str = FORMAL_REWARD_SHA256

    @property
    def component_weights(self) -> dict[str, float]:
        return dict(FORMAL_COMPONENT_WEIGHTS)

    def metadata(self) -> dict[str, object]:
        return {
            "reward_policy_version": self.version,
            "reward_component_weights": self.component_weights,
            "reward_component_weights_sha256": FORMAL_COMPONENT_WEIGHTS_SHA256,
            "reward_policy_sha256": self.policy_sha256,
        }

    def _evaluation(
        self,
        result: RewardResult,
        components: dict[str, float],
    ) -> RewardEvaluation:
        scalar = round(sum(components.values()), 10)
        if not 0.0 <= scalar <= 1.0:
            raise RewardPolicyError(f"formal reward outside [0, 1]: {scalar}")
        return RewardEvaluation(
            canonical_result=result,
            scalar_reward=scalar,
            reward_policy_version=self.version,
            reward_policy_sha256=self.policy_sha256,
            reward_component_weights=self.component_weights,
            components=components,
        )

    def evaluate(self, completion: str, verifier) -> RewardEvaluation:
        canonical = verifier(completion)
        if not isinstance(canonical, RewardResult):
            raise RewardPolicyError("canonical verifier must return RewardResult")
        if canonical.status == RewardStatus.INFRA_ERROR:
            raise RewardInfrastructureError(canonical)

        zero = {name: 0.0 for name in FORMAL_COMPONENT_WEIGHTS}
        if canonical.status == RewardStatus.RESOURCE_LIMIT:
            return self._evaluation(canonical, zero)

        answer = extract_unique_answer_block(completion)
        if answer is None:
            return self._evaluation(canonical, zero)

        strict_protocol = isinstance(parse_completion(completion), ParsedCompletion)
        probe = verifier(_strict_probe(answer))
        if not isinstance(probe, RewardResult):
            raise RewardPolicyError("answer probe verifier must return RewardResult")
        if probe.status == RewardStatus.INFRA_ERROR:
            raise RewardInfrastructureError(probe)
        if probe.status == RewardStatus.RESOURCE_LIMIT:
            return self._evaluation(canonical, zero)

        valid_answer = probe.status in {
            RewardStatus.WRONG_ANSWER,
            RewardStatus.VERIFIED_PASS,
        }
        correct = canonical.status == RewardStatus.VERIFIED_PASS
        if correct and not (strict_protocol and valid_answer):
            raise RewardPolicyError("VERIFIED_PASS contradicted formal reward components")

        components = {
            "answer_block": FORMAL_COMPONENT_WEIGHTS["answer_block"],
            "strict_protocol": (
                FORMAL_COMPONENT_WEIGHTS["strict_protocol"] if strict_protocol else 0.0
            ),
            "valid_answer": FORMAL_COMPONENT_WEIGHTS["valid_answer"] if valid_answer else 0.0,
            "correctness": FORMAL_COMPONENT_WEIGHTS["correctness"] if correct else 0.0,
        }
        return self._evaluation(canonical, components)


FORMAL_REWARD_POLICY = FormalDomainRewardPolicy()
