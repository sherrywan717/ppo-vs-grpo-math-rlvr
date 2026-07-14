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
            if reward_evidence.get(
                "canonical_status"
            ) != reward_result.status.value or reward_evidence.get("scalar_reward") != float(
                scalar_reward
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


class PPOCompletionEvidenceRecorder:
    """Join exact TRL PPO response tensors to ordered shaped-reward callbacks."""

    def __init__(self, expected_completions: int = 4):
        self.expected_completions = expected_completions
        self._generation: list[dict[str, Any]] | None = None
        self._rewards: list[dict[str, Any]] = []

    def capture_generation(self, queries, query_responses, tokenizer, prompt_lookup, pad_token_id):
        assert_trl_version()
        if self._generation is not None:
            raise TRLContractError("PPO generation evidence captured more than once")
        if queries.ndim != 2 or query_responses.ndim != 2:
            raise TRLContractError("PPO query/response tensors must be rank two")
        if (
            queries.shape[0] != self.expected_completions
            or query_responses.shape[0] != queries.shape[0]
        ):
            raise TRLContractError("unexpected PPO response count")
        context_length = int(queries.shape[1])
        response_ids = query_responses[:, context_length:].detach().cpu()
        rows = []
        for index in range(self.expected_completions):
            query_row = [int(v) for v in queries[index].detach().cpu().tolist()]
            prompt_ids = [v for v in query_row if v != pad_token_id]
            metadata = prompt_lookup.get(tuple(prompt_ids))
            if metadata is None:
                raise TRLContractError("PPO prompt tokens not found in fixed lookup")
            ids = [int(v) for v in response_ids[index].tolist()]
            mask = [int(v != pad_token_id) for v in ids]
            valid_ids = [v for v, keep in zip(ids, mask, strict=True) if keep]
            text = tokenizer.decode(valid_ids, skip_special_tokens=True)
            rows.append(
                {
                    "problem_id": metadata["problem_id"],
                    "prompt_hash": metadata["prompt_hash"],
                    "generation_index": 0,
                    "completion_index": index,
                    "prompt_token_ids": prompt_ids,
                    "response_token_ids": ids,
                    "response_mask": mask,
                    "exact_token_count": sum(mask),
                    "decoded_completion": text,
                }
            )
        self._generation = rows
        return len(rows), sum(row["exact_token_count"] for row in rows)

    def record_reward(self, completion: str, evaluation, guard) -> None:
        if len(self._rewards) >= self.expected_completions:
            raise TRLContractError("too many PPO reward records")
        result = evaluation.canonical_result
        evidence = evaluation.to_dict()
        guard.record_reward(result, evaluation.scalar_reward, evidence)
        self._rewards.append(
            {
                "reward_callback_text": completion,
                "verifier_input": completion,
                "reward_status": result.status.value,
                "scalar_reward": float(evaluation.scalar_reward),
                "verifier_detail": str(result.detail),
                **evidence,
            }
        )

    def records(self) -> list[dict[str, Any]]:
        if self._generation is None or len(self._rewards) != self.expected_completions:
            raise TRLContractError("incomplete PPO completion/reward evidence")
        records = []
        for generated, reward in zip(self._generation, self._rewards, strict=True):
            if generated["decoded_completion"] != reward["reward_callback_text"]:
                raise TRLContractError("PPO decoded completion differs from reward input")
            row = {**generated, **reward}
            if row["reward_callback_text"] != row["verifier_input"]:
                raise TRLContractError("PPO verifier input mismatch")
            records.append(row)
        return records


def validate_ppo_value_shape(values, batch_size: int, sequence_length: int) -> None:
    assert_trl_version()
    if tuple(values.shape) != (batch_size, sequence_length, 1):
        raise TRLContractError(
            f"PPO scalar-head shape {tuple(values.shape)} != {(batch_size, sequence_length, 1)}"
        )
    if not bool(values.is_floating_point()):
        raise TRLContractError("PPO value tensor must be floating point")


PPO_METRIC_ALIASES = {
    "policy_loss": ("loss/policy_avg",),
    "value_loss": ("loss/value_avg",),
    "objective_kl": ("objective/kl",),
    "approximate_kl": ("policy/approxkl_avg",),
    "entropy": ("policy/entropy_avg", "objective/entropy"),
    "clip_fraction": ("policy/clipfrac_avg",),
    "ratio": ("val/ratio",),
    "learning_rate": ("lr",),
    "reward_mean": ("objective/scores",),
}

PPO_NULLABLE_TELEMETRY = {
    "ratio_variance": {
        "raw_key": "val/ratio_var",
        "reason": (
            "TRL 0.24.0 may emit an undefined sample variance when only one "
            "ratio observation is available; this diagnostic is not used for "
            "rewards, losses, optimization, checkpoint counters, or budgets"
        ),
    }
}


def _non_finite_kind(value: int | float) -> str:
    numeric = float(value)
    if math.isnan(numeric):
        return "nan"
    return "positive_infinity" if numeric > 0 else "negative_infinity"


def _sanitize_ppo_log_history(
    log_history: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    allowed = {
        telemetry["raw_key"]: telemetry["reason"]
        for telemetry in PPO_NULLABLE_TELEMETRY.values()
    }
    sanitized, nullable = [], []
    for index, item in enumerate(log_history):
        row = {}
        for raw_key, value in item.items():
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                reason = allowed.get(raw_key)
                if reason is None:
                    raise TRLContractError(f"non-finite PPO metric at {raw_key}")
                evidence = {
                    "log_history_index": index,
                    "raw_key": raw_key,
                    "value": None,
                    "available": False,
                    "classification": "non_finite",
                    "non_finite_kind": _non_finite_kind(value),
                    "reason": reason,
                }
                row[raw_key] = None
                nullable.append(evidence)
                continue
            row[raw_key] = value
        sanitized.append(row)
    return sanitized, nullable


def extract_ppo_metrics(log_history: list[dict[str, Any]]) -> dict[str, Any]:
    """Normalize only reviewed TRL 0.24.0 metric keys; missing means unavailable."""
    assert_trl_version()
    sanitized_history, nullable = _sanitize_ppo_log_history(log_history)
    selected_index, row = next(
        (
            (index, entry)
            for index, entry in reversed(list(enumerate(log_history)))
            if "loss/policy_avg" in entry
        ),
        (None, {}),
    )
    normalized = {}
    for name, aliases in PPO_METRIC_ALIASES.items():
        raw_key = next((key for key in aliases if key in row), None)
        if raw_key is None:
            normalized[name] = {"available": False, "value": None, "raw_key": None}
            continue
        value = row[raw_key]
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise TRLContractError(f"non-finite PPO metric at {raw_key}")
        normalized[name] = {
            "available": True,
            "value": float(value),
            "raw_key": raw_key,
        }
    for name, telemetry in PPO_NULLABLE_TELEMETRY.items():
        raw_key = telemetry["raw_key"]
        if raw_key not in row:
            normalized[name] = {
                "available": False,
                "value": None,
                "raw_key": None,
                "classification": "missing",
                "non_finite_kind": None,
                "reason": f"TRL PPO log row did not expose {raw_key}",
            }
            continue
        value = row[raw_key]
        if not isinstance(value, (int, float)):
            raise TRLContractError(f"invalid PPO telemetry at {raw_key}")
        if math.isfinite(float(value)):
            normalized[name] = {
                "available": True,
                "value": float(value),
                "raw_key": raw_key,
                "classification": None,
                "non_finite_kind": None,
                "reason": None,
            }
            continue
        evidence = next(
            item
            for item in nullable
            if item["log_history_index"] == selected_index and item["raw_key"] == raw_key
        )
        normalized[name] = {
            key: evidence[key]
            for key in (
                "available",
                "value",
                "raw_key",
                "classification",
                "non_finite_kind",
                "reason",
            )
        }
    return {
        "normalized": normalized,
        "raw_log_history": sanitized_history,
        "nullable_telemetry": nullable,
        "warnings": [
            {"category": "nullable_nonessential_telemetry", **item}
            for item in nullable
        ],
    }


def enforce_ppo_generation_contract(generation_config, contract) -> None:
    """Validate TRL-owned fields and apply the YAML top-p omitted by PPOConfig."""
    assert_trl_version()
    expected = {
        "max_new_tokens": contract["max_new_tokens"],
        "temperature": contract["temperature"],
    }
    for name, value in expected.items():
        actual = getattr(generation_config, name)
        if not math.isclose(float(actual), float(value), rel_tol=0, abs_tol=1e-6):
            raise TRLContractError(f"unexpected TRL PPO generation {name}: {actual}")
    generation_config.top_p = contract["top_p"]


def ppo_guarded_trainer_class(guard, evidence_recorder, prompt_lookup, generation_contract):
    """The sole TRL 0.24.0 PPO private compatibility shim."""
    assert_trl_version()
    import trl.trainer.ppo_trainer as module
    from trl import PPOTrainer

    class GuardedPPOTrainer(PPOTrainer):
        def _save_checkpoint(self, model, trial):
            # The runner writes one role-separated authoritative checkpoint after success.
            return None

        def log(self, logs, start_time=None):
            if "loss/policy_avg" in logs:
                guard.record_update()
                guard.record_global_step(int(self.state.global_step))
            return super().log(logs, start_time)

        def train(self):
            original_generation = module.batch_generation
            original_get_reward = module.get_reward
            original_step = self.optimizer.step

            def guarded_generation(
                policy, queries, forward_batch_size, pad_token_id, generation_config
            ):
                enforce_ppo_generation_contract(generation_config, generation_contract)
                payload = original_generation(
                    policy, queries, forward_batch_size, pad_token_id, generation_config
                )
                query_responses, _ = payload
                completions, tokens = evidence_recorder.capture_generation(
                    queries, query_responses, self.processing_class, prompt_lookup, pad_token_id
                )
                guard.record_generation(completions, tokens)
                return payload

            def guarded_get_reward(model, query_responses, pad_token_id, context_length):
                if getattr(model, "_math_rlvr_parameter_free_reward", False):
                    model.set_context_length(int(context_length))
                payload = original_get_reward(model, query_responses, pad_token_id, context_length)
                logits, scores, _ = payload
                validate_ppo_value_shape(
                    logits, int(query_responses.shape[0]), int(query_responses.shape[1])
                )
                if tuple(scores.shape) != (int(query_responses.shape[0]),):
                    raise TRLContractError("PPO terminal scalar score shape mismatch")
                return payload

            def guarded_step(*args, **kwargs):
                guard.record_epoch_minibatch()
                guard.record_optimizer_step()
                return original_step(*args, **kwargs)

            module.batch_generation = guarded_generation
            module.get_reward = guarded_get_reward
            self.optimizer.step = guarded_step
            try:
                return super().train()
            finally:
                self.optimizer.step = original_step
                module.get_reward = original_get_reward
                module.batch_generation = original_generation

    return GuardedPPOTrainer
