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
        evidence_callback: Callable[[str, Any], None] | None = None,
        prompt_verifier: Callable[[tuple[int, ...], str], RewardResult] | None = None,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.verifier = verifier
        self.extract_completion = extract_completion
        self.policy = policy
        self.evidence_callback = evidence_callback
        self.prompt_verifier = prompt_verifier
        self.context_length: int | None = None

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
            if self.context_length is None:
                completion = self.extract_completion(decoded)
            else:
                response_ids = (
                    input_ids[row, self.context_length :][
                        attention_mask[row, self.context_length :].bool()
                    ]
                    .detach()
                    .cpu()
                    .tolist()
                )
                completion = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            if self.prompt_verifier is None:
                verifier = self.verifier
            else:
                prompt_ids = tuple(
                    int(value)
                    for value in input_ids[row, : self.context_length][
                        attention_mask[row, : self.context_length].bool()
                    ]
                    .detach()
                    .cpu()
                    .tolist()
                )

                def verifier(candidate, prompt_ids=prompt_ids):
                    return self.prompt_verifier(prompt_ids, candidate)

            evaluation = self.policy.evaluate(completion, verifier)
            if self.evidence_callback is not None:
                self.evidence_callback(completion, evaluation)
            scalar = evaluation.scalar_reward
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
        evidence_callback: Callable[[str, Any], None] | None = None,
        prompt_verifier: Callable[[tuple[int, ...], str], RewardResult] | None = None,
    ) -> None:
        super().__init__()
        self.backbone = _VerifierBackbone(
            tokenizer,
            verifier,
            extract_completion,
            policy,
            evidence_callback,
            prompt_verifier,
        )
        self.score = nn.Identity()
        self._math_rlvr_parameter_free_reward = True
        self.requires_grad_(False)

    def set_context_length(self, context_length: int) -> None:
        if not isinstance(context_length, int) or context_length <= 0:
            raise ValueError("invalid PPO context length")
        self.backbone.context_length = context_length
