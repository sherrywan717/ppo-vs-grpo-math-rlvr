import hashlib
import inspect
from pathlib import Path

import pytest
import torch
from trl.trainer.utils import get_reward

from math_rlvr.config import load_config, resolve_training_config
from math_rlvr.prompt import PROMPT_V0_SHA256, PROMPT_V1_SHA256
from math_rlvr.rewards.adapters import GRPOVerifierRewardAdapter, PPOVerifierRewardModel
from math_rlvr.rewards.result import (
    SPARSE_REWARD_POLICY,
    RewardInfrastructureError,
    RewardResult,
    RewardStatus,
)
from math_rlvr.rewards.staged import (
    STAGED_COMPONENT_WEIGHTS,
    STAGED_REWARD_POLICY,
    STAGED_REWARD_SHA256,
    STAGED_REWARD_VERSION,
)
from math_rlvr.verifier import CountdownVerifier

HISTORICAL_COMPLETIONS = Path(
    "reports/runs/grpo_single_update_qwen25_05b_20260713T112100Z/completions.jsonl"
)
HISTORICAL_SUMMARY = Path(
    "reports/runs/grpo_single_update_qwen25_05b_20260713T112100Z/summary.json"
)


def envelope(answer: str, reasoning: str = "brief") -> str:
    return f"<reasoning>{reasoning}</reasoning><answer>{answer}</answer>"


def verifier():
    return CountdownVerifier([1, 2, 3, 4], 10)


@pytest.mark.parametrize(
    "status",
    [
        RewardStatus.WRONG_ANSWER,
        RewardStatus.FORMAT_ERROR,
        RewardStatus.PARSE_ERROR,
        RewardStatus.INVALID_EXPRESSION,
        RewardStatus.INVALID_NUMBER_USAGE,
        RewardStatus.RESOURCE_LIMIT,
    ],
)
def test_sparse_reward_remains_zero_for_every_non_pass_status(status):
    assert SPARSE_REWARD_POLICY.to_scalar(RewardResult(status)) == 0.0


def test_sparse_reward_verified_pass_is_exactly_one():
    assert SPARSE_REWARD_POLICY.to_scalar(RewardResult(RewardStatus.VERIFIED_PASS)) == 1.0


@pytest.mark.parametrize(
    ("completion", "expected", "status"),
    [
        ("prose only", 0.0, RewardStatus.FORMAT_ERROR),
        ("<answer>1+2+3</answer>", 0.10, RewardStatus.FORMAT_ERROR),
        ("<answer>1+2+3+4</answer>", 0.15, RewardStatus.FORMAT_ERROR),
        ("<answer>1+2=3</answer>", 0.05, RewardStatus.FORMAT_ERROR),
        (envelope("1*2+3+4"), 0.20, RewardStatus.WRONG_ANSWER),
        (envelope("1+2+3+4"), 1.0, RewardStatus.VERIFIED_PASS),
    ],
)
def test_staged_reward_is_monotonic_bounded_and_preserves_canonical_status(
    completion, expected, status
):
    evaluation = STAGED_REWARD_POLICY.evaluate(completion, verifier())
    assert evaluation.canonical_result.status == status
    assert evaluation.scalar_reward == pytest.approx(expected)
    assert 0.0 <= evaluation.scalar_reward <= 1.0
    assert sum(evaluation.components.values()) == pytest.approx(expected)


@pytest.mark.parametrize(
    "answer",
    [
        "1+2=3",
        "1+2;3+4",
        "x+1",
        "f(1)",
        "obj.value+1",
        "1**2+3+4",
    ],
)
def test_illegal_ast_never_gets_valid_expression_component(answer):
    evaluation = STAGED_REWARD_POLICY.evaluate(f"<answer>{answer}</answer>", verifier())
    assert evaluation.scalar_reward == pytest.approx(0.05)
    assert evaluation.components["valid_expression"] == 0.0


@pytest.mark.parametrize("answer", ["1+2+3", "1+1+2+3+4", "1+2+3+5"])
def test_missing_duplicate_or_new_numbers_never_get_number_usage_component(answer):
    evaluation = STAGED_REWARD_POLICY.evaluate(f"<answer>{answer}</answer>", verifier())
    assert evaluation.components["valid_expression"] == 0.05
    assert evaluation.components["exact_number_usage"] == 0.0


@pytest.mark.parametrize(
    "completion",
    [
        "<answer>1+2+3+4",
        "</answer><answer>1+2+3+4",
        "<answer>1</answer><answer>2</answer>",
        "<answer></answer>",
    ],
)
def test_missing_misordered_duplicate_or_empty_answer_block_gets_zero(completion):
    evaluation = STAGED_REWARD_POLICY.evaluate(completion, verifier())
    assert evaluation.scalar_reward == 0.0


def test_resource_limit_gets_no_partial_reward():
    many = "+".join(["1"] * 70)
    limited = CountdownVerifier([1] * 70, 70, max_nodes=64)
    evaluation = STAGED_REWARD_POLICY.evaluate(f"<answer>{many}</answer>", limited)
    assert evaluation.canonical_result.status == RewardStatus.FORMAT_ERROR
    assert evaluation.scalar_reward == 0.0


def test_infrastructure_error_fails_closed():
    def broken(_completion):
        return RewardResult(RewardStatus.INFRA_ERROR, "offline")

    with pytest.raises(RewardInfrastructureError, match="infra_error"):
        STAGED_REWARD_POLICY.evaluate("<answer>1</answer>", broken)


class FixedTokenizer:
    def __init__(self, text):
        self.text = text

    def decode(self, token_ids, skip_special_tokens=False):
        assert skip_special_tokens is False
        return self.text


@pytest.mark.parametrize(
    "completion",
    [
        "<answer>1+2+3</answer>",
        "<answer>1+2+3+4</answer>",
        envelope("1*2+3+4"),
        envelope("1+2+3+4"),
    ],
)
def test_ppo_and_grpo_use_identical_staged_scalar(completion):
    canonical = verifier()
    grpo = GRPOVerifierRewardAdapter(canonical, STAGED_REWARD_POLICY)([completion])[0]
    ppo_model = PPOVerifierRewardModel(
        FixedTokenizer(completion),
        verifier(),
        lambda decoded: decoded,
        STAGED_REWARD_POLICY,
    )
    query_response = torch.tensor([[1, 0]])
    _, ppo, _ = get_reward(ppo_model, query_response, pad_token_id=0, context_length=0)
    assert ppo.tolist() == pytest.approx([grpo])


def test_reward_policy_identity_is_shared_in_both_resolved_smoke_configs():
    grpo = resolve_training_config(load_config("configs/smoke/grpo.yaml"))
    ppo = resolve_training_config(load_config("configs/smoke/ppo.yaml"))
    for config in (grpo, ppo):
        assert config["reward"]["policy"] == STAGED_REWARD_VERSION
        assert config["reward_policy_version"] == STAGED_REWARD_VERSION
        assert config["reward_component_weights"] == STAGED_COMPONENT_WEIGHTS
        assert config["reward_policy_sha256"] == STAGED_REWARD_SHA256
    assert grpo["reward_policy_sha256"] == ppo["reward_policy_sha256"]


def test_main_configs_do_not_activate_staged_reward():
    for path in ("configs/main/grpo.yaml", "configs/main/ppo.yaml"):
        assert "reward" not in load_config(path)


def test_staged_implementation_contains_no_saved_sample_special_case():
    source = inspect.getsource(type(STAGED_REWARD_POLICY))
    for forbidden_api in ("eval(", "exec(", "subprocess", "os.system"):
        assert forbidden_api not in source
    for forbidden in (
        "countdown:train:0",
        "countdown:train:1",
        "13 + 13 / 6",
        "grpo_single_update_qwen25_05b_20260713T112100Z",
    ):
        assert forbidden not in source


def test_historical_run_evidence_hashes_are_immutable():
    assert hashlib.sha256(HISTORICAL_COMPLETIONS.read_bytes()).hexdigest() == (
        "1b4e213df6d69aa7cc1663deeb92ca27850346a8bde552360a78448ce9ff8d02"
    )
    assert hashlib.sha256(HISTORICAL_SUMMARY.read_bytes()).hexdigest() == (
        "23aa81fcfe900146e62585a6389d3503bb0edaba901eb3f391e5b42f964004ec"
    )


def test_protected_prompt_and_main_config_hashes_are_unchanged():
    assert PROMPT_V0_SHA256 == "20b54a2ae00ebc762a1a90a3221f5c2409c7e64d2b35fcf2c6dfaaff48a9ef4f"
    assert PROMPT_V1_SHA256 == "6842002e4591630a4105a4ca8fdf4cab91676b3902708ec0cb7f7b458864ecd7"
    expected = {
        "configs/main/grpo.yaml": (
            "fc1b0c73de431d81e9e827107d8491aba4d54b92f7e04fd4678b6fd828b6f675"
        ),
        "configs/main/ppo.yaml": "1ced44a672fa3a5dcf9871bd8c1893a3bdad641d756dcf9de226b20440d1ad74",
        "src/math_rlvr/prompt.py": (
            "e7cb7cef5cdba403cd05763b48bf53817bafab70a424580b200254a88f6a4562"
        ),
    }
    for path, digest in expected.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
