"""Versioned staged reward for smoke training; strict evaluation remains canonical."""

from __future__ import annotations

import re
from dataclasses import dataclass

from math_rlvr.parser import ParsedCompletion, parse_completion
from math_rlvr.rewards.result import (
    DEFAULT_REWARD_POLICY,
    RewardEvaluation,
    RewardInfrastructureError,
    RewardPolicyError,
    RewardResult,
    RewardStatus,
    policy_contract_sha256,
)

STAGED_REWARD_VERSION = "shaped_v2_staged"
STAGED_COMPONENT_WEIGHTS = {
    "answer_block": 0.05,
    "strict_protocol": 0.05,
    "valid_expression": 0.05,
    "exact_number_usage": 0.05,
    "correctness": 0.80,
}
STAGED_REWARD_DESCRIPTOR = {
    "version": STAGED_REWARD_VERSION,
    "component_weights": STAGED_COMPONENT_WEIGHTS,
    "answer_block": "exactly one non-empty correctly ordered and closed answer block",
    "strict_protocol": "unchanged strict completion parser succeeds",
    "valid_expression": "canonical safe expression verifier accepts the extracted answer AST",
    "exact_number_usage": "canonical verifier confirms exact Countdown number multiset",
    "correctness": "original strict canonical verifier returns VERIFIED_PASS",
    "resource_limit": "zero all components",
    "infra_error": "raise and fail closed",
    "range": [0.0, 1.0],
}
STAGED_REWARD_SHA256 = policy_contract_sha256(STAGED_REWARD_DESCRIPTOR)
_ANSWER_BLOCK = re.compile(r"<answer>(?P<answer>.*?)</answer>", re.DOTALL)
_MAX_ANSWER_CHARS = 512


def extract_unique_answer_block(completion: str) -> str | None:
    """Extract one usable answer block without changing the strict parser."""
    if completion.count("<answer>") != 1 or completion.count("</answer>") != 1:
        return None
    opening = completion.find("<answer>")
    closing = completion.find("</answer>")
    if opening < 0 or closing < opening + len("<answer>"):
        return None
    match = _ANSWER_BLOCK.search(completion)
    if match is None:
        return None
    answer = match["answer"].strip()
    return answer or None


def _strict_probe(answer: str) -> str:
    return f"<reasoning>staged structural validation</reasoning><answer>{answer}</answer>"


@dataclass(frozen=True)
class StagedRewardPolicy:
    version: str = STAGED_REWARD_VERSION
    policy_sha256: str = STAGED_REWARD_SHA256

    @property
    def component_weights(self) -> dict[str, float]:
        return dict(STAGED_COMPONENT_WEIGHTS)

    def metadata(self) -> dict[str, object]:
        return {
            "reward_policy_version": self.version,
            "reward_component_weights": self.component_weights,
            "reward_policy_sha256": self.policy_sha256,
        }

    def _evaluation(
        self,
        result: RewardResult,
        components: dict[str, float],
    ) -> RewardEvaluation:
        scalar = round(sum(components.values()), 10)
        if not 0.0 <= scalar <= 1.0:
            raise RewardPolicyError(f"staged reward outside [0, 1]: {scalar}")
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

        zero = {name: 0.0 for name in STAGED_COMPONENT_WEIGHTS}
        if canonical.status == RewardStatus.RESOURCE_LIMIT:
            return self._evaluation(canonical, zero)

        answer = extract_unique_answer_block(completion)
        if answer is None:
            return self._evaluation(canonical, zero)
        if len(answer) > _MAX_ANSWER_CHARS:
            return self._evaluation(canonical, zero)

        strict_protocol = isinstance(parse_completion(completion), ParsedCompletion)
        probe = verifier(_strict_probe(answer))
        if not isinstance(probe, RewardResult):
            raise RewardPolicyError("answer probe verifier must return RewardResult")
        if probe.status == RewardStatus.INFRA_ERROR:
            raise RewardInfrastructureError(probe)
        if probe.status == RewardStatus.RESOURCE_LIMIT:
            return self._evaluation(canonical, zero)

        valid_expression = probe.status in {
            RewardStatus.INVALID_NUMBER_USAGE,
            RewardStatus.WRONG_ANSWER,
            RewardStatus.VERIFIED_PASS,
        }
        exact_number_usage = probe.status in {
            RewardStatus.WRONG_ANSWER,
            RewardStatus.VERIFIED_PASS,
        }
        correct = canonical.status == RewardStatus.VERIFIED_PASS
        if correct and not (strict_protocol and valid_expression and exact_number_usage):
            raise RewardPolicyError("VERIFIED_PASS contradicted staged protocol components")

        components = {
            "answer_block": STAGED_COMPONENT_WEIGHTS["answer_block"],
            "strict_protocol": (
                STAGED_COMPONENT_WEIGHTS["strict_protocol"] if strict_protocol else 0.0
            ),
            "valid_expression": (
                STAGED_COMPONENT_WEIGHTS["valid_expression"] if valid_expression else 0.0
            ),
            "exact_number_usage": (
                STAGED_COMPONENT_WEIGHTS["exact_number_usage"]
                if exact_number_usage
                else 0.0
            ),
            "correctness": STAGED_COMPONENT_WEIGHTS["correctness"] if correct else 0.0,
        }
        return self._evaluation(canonical, components)


STAGED_REWARD_POLICY = StagedRewardPolicy()


def reward_policy_from_selector(selector: str):
    if selector == STAGED_REWARD_VERSION:
        return STAGED_REWARD_POLICY
    if selector == "shaped_v3_domain":
        from math_rlvr.rewards.formal import FORMAL_REWARD_POLICY

        return FORMAL_REWARD_POLICY
    if selector in {"shaped_v1_legacy", "shaped"}:
        return DEFAULT_REWARD_POLICY
    raise RewardPolicyError(f"unknown reward policy selector: {selector}")


def reward_policy_from_config(config: dict):
    selector = config.get("reward", {}).get("policy", "shaped_v1_legacy")
    policy = reward_policy_from_selector(selector)
    for key, value in {
        "reward_policy_version": policy.version,
        "reward_component_weights": policy.component_weights,
        "reward_policy_sha256": policy.policy_sha256,
    }.items():
        if key in config and config[key] != value:
            raise RewardPolicyError(f"resolved reward metadata mismatch: {key}")
    return policy


def reward_metadata_from_config(config: dict) -> dict[str, object]:
    policy = reward_policy_from_config(config)
    return {
        "reward_policy_version": policy.version,
        "reward_component_weights": policy.component_weights,
        "reward_policy_sha256": policy.policy_sha256,
    }
