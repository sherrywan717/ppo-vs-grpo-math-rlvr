from collections.abc import Sequence

import pytest
import torch
import trl
from trl.trainer.utils import get_reward

from math_rlvr.rewards.adapters import GRPOVerifierRewardAdapter, PPOVerifierRewardModel
from math_rlvr.rewards.result import (
    RewardInfrastructureError,
    RewardResult,
    RewardStatus,
)


class FakeTokenizer:
    vocabulary = {
        1: "PROMPT",
        2: "|",
        3: "pa",
        4: "fail",
        5: "timeout",
        6: "infra",
        7: "ss",
    }

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        return "".join(self.vocabulary[token_id] for token_id in token_ids)


class FakeVerifier:
    outcomes = {
        "pass": RewardStatus.VERIFIED_PASS,
        "fail": RewardStatus.WRONG_ANSWER,
        "timeout": RewardStatus.RESOURCE_LIMIT,
        "infra": RewardStatus.INFRA_ERROR,
    }

    def __init__(self) -> None:
        self.seen: list[str] = []

    def __call__(self, completion: str) -> RewardResult:
        self.seen.append(completion)
        return RewardResult(self.outcomes[completion], f"fake:{completion}")


def extract_completion(decoded: str) -> str:
    return decoded.split("|", maxsplit=1)[1]


def test_trl_024_get_reward_contract_padding_last_token_and_no_grad() -> None:
    assert trl.__version__ == "0.24.0"
    verifier = FakeVerifier()
    model = PPOVerifierRewardModel(FakeTokenizer(), verifier, extract_completion)
    query_responses = torch.tensor([[1, 2, 3, 7, 0], [1, 2, 5, 0, 0]])

    logits, scores, lengths = get_reward(model, query_responses, pad_token_id=0, context_length=2)

    assert model.base_model_prefix == "backbone"
    assert verifier.seen == ["pass", "timeout"]
    assert lengths.tolist() == [3, 2]
    assert scores.tolist() == pytest.approx([1.0, 0.1])
    expected_logits = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.1, 0.0, 0.0]])
    assert torch.allclose(logits[:, :, 0], expected_logits)
    assert not logits.requires_grad
    assert list(model.parameters()) == []


@pytest.mark.parametrize("completion", ["pass", "fail", "timeout"])
def test_ppo_and_grpo_share_identical_scalar_policy(completion: str) -> None:
    token_ids = {"pass": [3, 7], "fail": [4], "timeout": [5]}[completion]
    ppo_model = PPOVerifierRewardModel(FakeTokenizer(), FakeVerifier(), extract_completion)
    query_response = torch.tensor([[1, 2, *token_ids, 0]])
    _, ppo_score, _ = get_reward(ppo_model, query_response, pad_token_id=0, context_length=2)
    grpo_score = GRPOVerifierRewardAdapter(FakeVerifier())([completion])
    assert ppo_score.tolist() == pytest.approx(grpo_score)


@pytest.mark.parametrize("adapter", ["ppo", "grpo"])
def test_infrastructure_error_aborts_batch_without_executing_text(adapter: str) -> None:
    verifier = FakeVerifier()
    with pytest.raises(RewardInfrastructureError, match="infra_error"):
        if adapter == "ppo":
            model = PPOVerifierRewardModel(FakeTokenizer(), verifier, extract_completion)
            get_reward(model, torch.tensor([[1, 2, 6, 0]]), pad_token_id=0, context_length=2)
        else:
            GRPOVerifierRewardAdapter(verifier)(["infra"])
    assert verifier.seen == ["infra"]


def test_generated_text_is_never_executed_by_fake_verifier() -> None:
    generated = "raise RuntimeError('must never execute')"
    verifier = FakeVerifier()
    verifier.outcomes = verifier.outcomes | {generated: RewardStatus.WRONG_ANSWER}
    assert GRPOVerifierRewardAdapter(verifier)([generated]) == [0.2]
    assert verifier.seen == [generated]


def test_ppo_reward_dispatches_by_exact_prompt_tokens() -> None:
    seen = []

    def prompt_verifier(prompt_ids, completion):
        seen.append((prompt_ids, completion))
        return RewardResult(RewardStatus.VERIFIED_PASS, "prompt-bound")

    model = PPOVerifierRewardModel(
        FakeTokenizer(),
        FakeVerifier(),
        extract_completion,
        prompt_verifier=prompt_verifier,
    )
    model.set_context_length(2)
    get_reward(model, torch.tensor([[1, 2, 3, 7, 0]]), pad_token_id=0, context_length=2)
    assert seen == [((1, 2), "pass")]
