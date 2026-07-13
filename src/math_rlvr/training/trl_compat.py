"""Narrow TRL 0.24.0 compatibility hooks for exact rollout evidence."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import trl

TRL_VERSION = "0.24.0"
KL_KEY_ALIASES = ("kl", "train/kl", "objective/kl")


class TRLContractError(RuntimeError):
    pass


def assert_trl_version() -> None:
    if trl.__version__ != TRL_VERSION:
        raise TRLContractError(f"requires trl=={TRL_VERSION}, found {trl.__version__}")


def exact_completion_counts(payload: dict[str, Any]) -> tuple[int, int]:
    """Validate TRL's private rollout fields and count non-padding tokens exactly."""
    assert_trl_version()
    if not {"completion_ids", "completion_mask"} <= payload.keys():
        raise TRLContractError("missing completion_ids/completion_mask")
    ids, mask = payload["completion_ids"], payload["completion_mask"]
    if getattr(ids, "shape", None) != getattr(mask, "shape", None) or len(ids.shape) != 2:
        raise TRLContractError("invalid completion tensor shapes")
    if not bool(((mask == 0) | (mask == 1)).all().item()):
        raise TRLContractError("completion mask must be binary")
    return int(ids.shape[0]), int(mask.sum().item())


class CompletionEvidenceRecorder:
    """Bind TRL token tensors to the exact ordered reward callback inputs."""

    def __init__(self, expected_completions: int = 8):
        self.expected_completions = expected_completions
        self._reward_records: list[dict[str, Any]] = []
        self._completion_records: list[dict[str, Any]] | None = None

    def record_reward(
        self,
        problem_id: str,
        completion_text: str,
        reward_result,
        scalar_reward: float,
        reward_evidence: dict[str, Any] | None = None,
    ) -> None:
        if self._completion_records is not None:
            raise TRLContractError("reward arrived after completion evidence finalization")
        if len(self._reward_records) >= self.expected_completions:
            raise TRLContractError("too many ordered reward records")
        if not isinstance(completion_text, str):
            raise TRLContractError("verifier input must be text")
        if not math.isfinite(float(scalar_reward)):
            raise TRLContractError("non-finite scalar reward")
        row = {
            "problem_id": str(problem_id),
            "raw_completion": completion_text,
            "verifier_input": completion_text,
            "reward_status": reward_result.status.value,
            "scalar_reward": float(scalar_reward),
            "verifier_detail": str(reward_result.detail),
        }
        if reward_evidence is not None:
            if (
                reward_evidence.get("canonical_status") != reward_result.status.value
                or reward_evidence.get("scalar_reward") != float(scalar_reward)
            ):
                raise TRLContractError("reward component evidence mismatch")
            row.update(reward_evidence)
        self._reward_records.append(row)

    def capture_generation(self, inputs, payload, tokenizer) -> None:
        if self._completion_records is not None:
            raise TRLContractError("completion evidence captured more than once")
        completion_count, token_count = exact_completion_counts(payload)
        if completion_count != self.expected_completions:
            raise TRLContractError("unexpected completion evidence count")
        if len(inputs) != completion_count or len(self._reward_records) != completion_count:
            raise TRLContractError("generation/reward evidence count mismatch")

        ids_tensor = payload["completion_ids"].detach().cpu()
        mask_tensor = payload["completion_mask"].detach().cpu()
        ids_rows = [[int(value) for value in row] for row in ids_tensor.tolist()]
        mask_rows = [[int(value) for value in row] for row in mask_tensor.tolist()]
        decoded = tokenizer.batch_decode(ids_tensor, skip_special_tokens=True)
        if len(decoded) != completion_count:
            raise TRLContractError("tokenizer decode count mismatch")

        per_problem = Counter()
        records = []
        exact_total = 0
        for index, (item, ids, mask, text, reward) in enumerate(
            zip(inputs, ids_rows, mask_rows, decoded, self._reward_records, strict=True)
        ):
            problem_id = str(item.get("problem_id"))
            prompt_hash = item.get("prompt_hash")
            if not problem_id or problem_id == "None" or not isinstance(prompt_hash, str):
                raise TRLContractError("missing problem_id or prompt_hash")
            if reward["problem_id"] != problem_id:
                raise TRLContractError("completion/reward problem order mismatch")
            if text != reward["raw_completion"] or text != reward["verifier_input"]:
                raise TRLContractError("decoded completion differs from verifier input")
            if len(ids) != len(mask) or any(value not in (0, 1) for value in mask):
                raise TRLContractError("invalid serialized completion mask")
            exact_count = sum(mask)
            exact_total += exact_count
            generation_index = per_problem[problem_id]
            per_problem[problem_id] += 1
            records.append(
                {
                    "problem_id": problem_id,
                    "prompt_hash": prompt_hash,
                    "generation_index": generation_index,
                    "completion_index": index,
                    "completion_ids": ids,
                    "completion_mask": mask,
                    "exact_token_count": exact_count,
                    "decoded_completion": text,
                    **reward,
                }
            )
        if exact_total != token_count:
            raise TRLContractError("serialized completion token total mismatch")
        if sorted(per_problem.values()) != [4, 4]:
            raise TRLContractError("expected four ordered generations for each prompt")
        self._completion_records = records

    def records(self) -> list[dict[str, Any]]:
        if self._completion_records is None:
            raise TRLContractError("completion evidence was not captured")
        if len(self._completion_records) != self.expected_completions:
            raise TRLContractError("completion evidence is incomplete")
        return [dict(record) for record in self._completion_records]


def extract_kl_metric(log_history: list[dict[str, Any]], beta: float) -> dict[str, Any]:
    """Normalize reviewed TRL 0.24.0 KL aliases without inventing missing values."""
    if beta == 0:
        return {
            "beta": 0.0,
            "kl_available": False,
            "kl": None,
            "kl_raw_key": None,
            "kl_unavailable_reason": (
                "GRPO beta=0.0; TRL 0.24.0 does not compute reference-model KL"
            ),
        }
    for row in reversed(log_history):
        for key in KL_KEY_ALIASES:
            if key in row:
                value = row[key]
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise TRLContractError(f"non-finite KL metric at {key}")
                return {
                    "beta": float(beta),
                    "kl_available": True,
                    "kl": float(value),
                    "kl_raw_key": key,
                    "kl_unavailable_reason": None,
                }
    return {
        "beta": float(beta),
        "kl_available": False,
        "kl": None,
        "kl_raw_key": None,
        "kl_unavailable_reason": (
            "beta>0 but reviewed TRL 0.24.0 KL keys were absent from trainer log history"
        ),
    }


def guarded_trainer_class(guard, evidence_recorder: CompletionEvidenceRecorder):
    """Create the sole private-API subclass; importing this does not construct a model."""
    assert_trl_version()
    from trl import GRPOTrainer

    class GuardedGRPOTrainer(GRPOTrainer):
        def _generate_and_score_completions(self, inputs):
            payload = super()._generate_and_score_completions(inputs)
            completions, tokens = exact_completion_counts(payload)
            evidence_recorder.capture_generation(inputs, payload, self.processing_class)
            guard.record_generation(completions, tokens)
            return payload

        def training_step(self, model, inputs, num_items_in_batch=None):
            guard.record_microstep()
            return super().training_step(model, inputs, num_items_in_batch)

    return GuardedGRPOTrainer


def optimizer_guard_callback(guard):
    assert_trl_version()
    from transformers import TrainerCallback

    class GuardCallback(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            guard.record_optimizer_step()

        def on_step_end(self, args, state, control, **kwargs):
            guard.record_global_step(int(state.global_step))

    return GuardCallback()
