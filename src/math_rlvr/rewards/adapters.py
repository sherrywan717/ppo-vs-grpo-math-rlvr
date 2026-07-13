"""TRL 0.24.0 adapters backed by the canonical verifier result policy."""

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Any, Protocol

import torch
from torch import nn

from math_rlvr.rewards.result import DEFAULT_REWARD_POLICY, RewardPolicy, RewardResult


class Verifier(Protocol):
    def __call__(self, completion: str) -> RewardResult: ...


def completion_text(completion: str | Sequence[dict[str, str]]) -> str:
    if isinstance(completion, str):
        return completion
    if not completion:
        return ""
    return completion[-1].get("content", "")


class GRPOVerifierRewardAdapter:
    """Callable matching TRL 0.24.0 GRPO reward_funcs semantics."""

    def __init__(
        self,
        verifier: Verifier,
        policy: RewardPolicy = DEFAULT_REWARD_POLICY,
    ) -> None:
        self.verifier = verifier
        self.policy = policy

    def __call__(
        self,
        completions: Sequence[str | Sequence[dict[str, str]]],
        **_: Any,
    ) -> list[float]:
        return [
            self.policy.evaluate(completion_text(item), self.verifier).scalar_reward
            for item in completions
        ]


class _VerifierBackbone(nn.Module):
    def __init__(
        self,
        tokenizer: Any,
        verifier: Verifier,
        extract_completion: Callable[[str], str],
        policy: RewardPolicy,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.verifier = verifier
        self.extract_completion = extract_completion
        self.policy = policy

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **_: Any,
    ) -> SimpleNamespace:
        batch_size, sequence_width = input_ids.shape
        reward_hidden = torch.zeros(
            (batch_size, sequence_width, 1),
            device=input_ids.device,
            dtype=torch.float32,
        )
        for row in range(batch_size):
            valid_ids = input_ids[row][attention_mask[row].bool()].detach().cpu().tolist()
            decoded = self.tokenizer.decode(valid_ids, skip_special_tokens=False)
            completion = self.extract_completion(decoded)
            scalar = self.policy.evaluate(completion, self.verifier).scalar_reward
            last_valid = torch.nonzero(attention_mask[row], as_tuple=False)[-1, 0]
            reward_hidden[row, last_valid, 0] = scalar
        return SimpleNamespace(hidden_states=(reward_hidden.detach(),))


class PPOVerifierRewardModel(nn.Module):
    """Reward model contract consumed by TRL 0.24.0 trainer.utils.get_reward.

    This deliberately parameter-free module decodes each completed sequence in its
    backbone, calls the non-differentiable verifier, and places the scalar only at
    the final non-padding token. ``score`` is an identity projection, as required
    by ``get_reward``. Infrastructure failures propagate and abort the batch.
    """

    base_model_prefix = "backbone"

    def __init__(
        self,
        tokenizer: Any,
        verifier: Verifier,
        extract_completion: Callable[[str], str],
        policy: RewardPolicy = DEFAULT_REWARD_POLICY,
    ) -> None:
        super().__init__()
        self.backbone = _VerifierBackbone(tokenizer, verifier, extract_completion, policy)
        self.score = nn.Identity()
        self.requires_grad_(False)
