import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from trl.trainer.utils import get_reward

from math_rlvr.contracts import (
    FORMAL_VERIFIER_BUNDLE_CONTRACT_DESCRIPTOR,
    FORMAL_VERIFIER_BUNDLE_CONTRACT_SHA256,
    GSM8K_VERIFIER_CONTRACT_DESCRIPTOR,
    GSM8K_VERIFIER_CONTRACT_SHA256,
    MATH_VERIFIER_CONTRACT_DESCRIPTOR,
    MATH_VERIFIER_CONTRACT_SHA256,
    contract_sha256,
    formal_parser_verifier_metadata,
)
from math_rlvr.dataset import load_manifest
from math_rlvr.rewards.adapters import GRPOVerifierRewardAdapter, PPOVerifierRewardModel
from math_rlvr.rewards.formal import (
    FORMAL_COMPONENT_WEIGHTS,
    FORMAL_COMPONENT_WEIGHTS_SHA256,
    FORMAL_REWARD_DESCRIPTOR,
    FORMAL_REWARD_POLICY,
    FORMAL_REWARD_SHA256,
    FORMAL_REWARD_VERSION,
)
from math_rlvr.rewards.result import RewardInfrastructureError, RewardResult, RewardStatus
from math_rlvr.rewards.staged import STAGED_REWARD_SHA256, reward_policy_from_selector
from math_rlvr.verifier import GSM8KVerifier, MathExpressionVerifier, MathVerifier


def envelope(answer: str) -> str:
    return f"<reasoning>brief</reasoning><answer>{answer}</answer>"


@pytest.mark.parametrize(
    ("verifier", "wrong", "correct"),
    [
        (GSM8KVerifier("2"), "3", "2"),
        (MathExpressionVerifier("$x^2+2*x+1$"), "$x^2+1$", "$(x+1)^2$"),
    ],
)
def test_formal_reward_uses_domain_validity_without_countdown_number_usage(
    verifier, wrong, correct
):
    wrong_evaluation = FORMAL_REWARD_POLICY.evaluate(envelope(wrong), verifier)
    assert wrong_evaluation.canonical_result.status == RewardStatus.WRONG_ANSWER
    assert wrong_evaluation.scalar_reward == pytest.approx(0.20)
    assert wrong_evaluation.components == {
        "answer_block": 0.05,
        "strict_protocol": 0.05,
        "valid_answer": 0.10,
        "correctness": 0.0,
    }
    assert "exact_number_usage" not in wrong_evaluation.components
    assert wrong_evaluation.to_dict()["valid_answer_component"] == pytest.approx(0.10)
    assert wrong_evaluation.to_dict()["exact_number_usage_component"] == 0.0

    correct_evaluation = FORMAL_REWARD_POLICY.evaluate(envelope(correct), verifier)
    assert correct_evaluation.canonical_result.status == RewardStatus.VERIFIED_PASS
    assert correct_evaluation.scalar_reward == 1.0
    assert correct_evaluation.components["correctness"] == 0.80


def test_formal_reward_keeps_answer_block_and_strict_protocol_separate():
    verifier = GSM8KVerifier("2")
    answer_only = FORMAL_REWARD_POLICY.evaluate("<answer>3</answer>", verifier)
    assert answer_only.canonical_result.status == RewardStatus.FORMAT_ERROR
    assert answer_only.scalar_reward == pytest.approx(0.15)
    assert answer_only.components["answer_block"] == 0.05
    assert answer_only.components["strict_protocol"] == 0.0
    assert answer_only.components["valid_answer"] == 0.10

    invalid = FORMAL_REWARD_POLICY.evaluate(envelope("2 or 3"), verifier)
    assert invalid.canonical_result.status == RewardStatus.PARSE_ERROR
    assert invalid.scalar_reward == pytest.approx(0.10)
    assert invalid.components["strict_protocol"] == 0.05
    assert invalid.components["valid_answer"] == 0.0


def test_formal_reward_infrastructure_error_fails_closed():
    def broken(_completion):
        return RewardResult(RewardStatus.INFRA_ERROR, "verifier unavailable")

    with pytest.raises(RewardInfrastructureError, match="infra_error"):
        FORMAL_REWARD_POLICY.evaluate(envelope("2"), broken)


class FixedTokenizer:
    def __init__(self, text):
        self.text = text

    def decode(self, _token_ids, skip_special_tokens=False):
        assert skip_special_tokens is False
        return self.text


def test_ppo_and_grpo_share_formal_reward_identity_and_scalar():
    completion = envelope("3")
    grpo = GRPOVerifierRewardAdapter(
        GSM8KVerifier("2"), FORMAL_REWARD_POLICY
    )([completion])[0]
    ppo_model = PPOVerifierRewardModel(
        FixedTokenizer(completion),
        GSM8KVerifier("2"),
        lambda decoded: decoded,
        FORMAL_REWARD_POLICY,
    )
    query_response = torch.tensor([[1, 0]])
    _, ppo, _ = get_reward(ppo_model, query_response, pad_token_id=0, context_length=0)
    assert ppo.tolist() == pytest.approx([grpo])
    assert FORMAL_REWARD_POLICY.metadata() == {
        "reward_policy_version": FORMAL_REWARD_VERSION,
        "reward_component_weights": FORMAL_COMPONENT_WEIGHTS,
        "reward_component_weights_sha256": FORMAL_COMPONENT_WEIGHTS_SHA256,
        "reward_policy_sha256": FORMAL_REWARD_SHA256,
    }
    assert reward_policy_from_selector(FORMAL_REWARD_VERSION) is FORMAL_REWARD_POLICY


def test_formal_reward_and_verifier_hashes_are_canonical_and_domain_specific():
    assert FORMAL_REWARD_SHA256 == contract_sha256(FORMAL_REWARD_DESCRIPTOR)
    assert FORMAL_REWARD_SHA256 == (
        "b9eda9520bb0271e28f6c209db85a408cdc0a65c2d403871b2b0fcc06e06a463"
    )
    assert FORMAL_COMPONENT_WEIGHTS_SHA256 == contract_sha256(FORMAL_COMPONENT_WEIGHTS)
    assert FORMAL_COMPONENT_WEIGHTS_SHA256 == (
        "a5e7d4a5f2db49121501b3d19c25b1fe3dd68b257fb60833c93f03c474402128"
    )
    assert GSM8K_VERIFIER_CONTRACT_SHA256 == (
        "91f9de474df89f63cd208a5621fbb7a678dadbe73e4a3f3426afb5f59fbe4b50"
    )
    assert MATH_VERIFIER_CONTRACT_SHA256 == (
        "0a4fb547959d1edc3c157392fb49f209a140981757f39a0c25c454868e8aefa7"
    )
    assert FORMAL_VERIFIER_BUNDLE_CONTRACT_SHA256 == (
        "ac3603158e31c8603c21e5d33445745bb56f3ccf946b055db9544a3dbc5886fd"
    )
    assert GSM8K_VERIFIER_CONTRACT_SHA256 == contract_sha256(
        GSM8K_VERIFIER_CONTRACT_DESCRIPTOR
    )
    assert MATH_VERIFIER_CONTRACT_SHA256 == contract_sha256(
        MATH_VERIFIER_CONTRACT_DESCRIPTOR
    )
    assert FORMAL_VERIFIER_BUNDLE_CONTRACT_SHA256 == contract_sha256(
        FORMAL_VERIFIER_BUNDLE_CONTRACT_DESCRIPTOR
    )
    metadata = formal_parser_verifier_metadata()
    assert metadata["verifier_contract"]["contract_sha256"] == (
        FORMAL_VERIFIER_BUNDLE_CONTRACT_SHA256
    )
    assert set(metadata["domain_verifier_contracts"]) == {"gsm8k", "math"}
    assert metadata["domain_verifier_contracts"]["gsm8k"]["contract_sha256"] == (
        GSM8K_VERIFIER_CONTRACT_SHA256
    )
    assert metadata["domain_verifier_contracts"]["math"]["contract_sha256"] == (
        MATH_VERIFIER_CONTRACT_SHA256
    )


def test_shaped_v2_identity_is_unchanged():
    assert STAGED_REWARD_SHA256 == (
        "90af0614676279eb8a47636acfdbeaded6d92237d3b16f027d79557057ca0e14"
    )


def test_math_verifier_unknown_source_and_bad_gold_fail_closed():
    verifier = MathVerifier()
    unknown = SimpleNamespace(source="spoof", gold_answer="2", metadata={})
    assert verifier(unknown, envelope("2")).status == RewardStatus.INFRA_ERROR
    bad_gold = SimpleNamespace(source="gsm8k", gold_answer="not numeric", metadata={})
    assert verifier(bad_gold, envelope("2")).status == RewardStatus.INFRA_ERROR


def test_all_frozen_formal_gold_answers_construct_canonical_verifiers():
    root = Path("/root/autodl-tmp/datasets/math_rlvr/manifests")
    names = (
        "train_core_128.json",
        "validation_64.json",
        "gsm8k_test_200.json",
        "math500_test_200.json",
    )
    seen = {"gsm8k": 0, "math": 0}
    for name in names:
        assert hashlib.sha256((root / name).read_bytes()).hexdigest()
        for problem in load_manifest(root / name):
            if problem.source == "gsm8k":
                GSM8KVerifier(problem.gold_answer)
            else:
                MathExpressionVerifier(problem.gold_answer)
            seen[problem.source] += 1
    assert seen == {"gsm8k": 296, "math": 296}
