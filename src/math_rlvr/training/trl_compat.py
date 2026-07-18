"""Narrow TRL 0.24.0 compatibility hooks for exact rollout evidence."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterator, Mapping, MutableMapping
from typing import Any

import torch
import trl

TRL_VERSION = "0.24.0"
KL_KEY_ALIASES = ("kl", "train/kl", "objective/kl")


class TRLContractError(RuntimeError):
    pass


ORDERED_EPISODE_FIELDS = (
    "episode_position",
    "problem_id",
    "generation_index",
    "pair_key",
    "problem_hash",
    "rendered_prompt_hash",
    "seed",
    "algorithm",
)


def _batch_values(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise TRLContractError("ordered PPO batch metadata must be a list or tensor")
    return value


def extract_ordered_episode_batch(batch: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract the metadata from the actual Accelerator-prepared rollout batch."""
    if not isinstance(batch, Mapping) or any(
        field not in batch for field in ORDERED_EPISODE_FIELDS
    ):
        raise TRLContractError("prepared PPO batch is missing ordered episode metadata")
    columns = {field: _batch_values(batch[field]) for field in ORDERED_EPISODE_FIELDS}
    lengths = {len(values) for values in columns.values()}
    if len(lengths) != 1:
        raise TRLContractError("prepared PPO batch metadata columns have different lengths")
    return [
        {field: columns[field][index] for field in ORDERED_EPISODE_FIELDS}
        for index in range(lengths.pop())
    ]


def validate_ordered_episode_batch(
    batch: Mapping[str, Any], expected_records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    actual = extract_ordered_episode_batch(batch)
    expected = [{field: row[field] for field in ORDERED_EPISODE_FIELDS} for row in expected_records]
    if actual != expected:
        raise TRLContractError("prepared PPO rollout batch order/identity mismatch")
    return actual


class OrderedMetadataCollator:
    """Keep comparison metadata out of token padding, then reattach it unchanged."""

    def __init__(self, base_collator):
        self.base_collator = base_collator

    def __call__(self, features):
        metadata = [
            {field: feature[field] for field in ORDERED_EPISODE_FIELDS} for feature in features
        ]
        model_features = [
            {key: value for key, value in feature.items() if key not in ORDERED_EPISODE_FIELDS}
            for feature in features
        ]
        batch = self.base_collator(model_features)
        if not isinstance(batch, MutableMapping):
            raise TRLContractError("PPO data collator must return a mapping")
        for field in ORDERED_EPISODE_FIELDS:
            batch[field] = [row[field] for row in metadata]
        return batch


class VerifiedSequentialDataLoader:
    """Proxy that revalidates the first prepared batch on every consumed iterator."""

    def __init__(self, prepared_loader, raw_loader, expected_records):
        self.prepared_loader = prepared_loader
        self.raw_loader = raw_loader
        self.expected_records = [dict(row) for row in expected_records]
        self.dataset = raw_loader.dataset
        self.sampler = raw_loader.sampler
        self.batch_size = raw_loader.batch_size
        self.drop_last = raw_loader.drop_last
        self.num_workers = raw_loader.num_workers

    def __len__(self):
        return len(self.prepared_loader)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        consumed = 0
        for batch in self.prepared_loader:
            size = len(_batch_values(batch["pair_key"]))
            expected = self.expected_records[consumed : consumed + size]
            if len(expected) != size:
                raise TRLContractError("prepared PPO loader produced excess episode rows")
            validate_ordered_episode_batch(batch, expected)
            consumed += size
            yield batch
        if consumed != len(self.expected_records):
            raise TRLContractError("prepared PPO loader did not consume every ordered episode")


def _sampler_name(loader) -> str:
    sampler = getattr(loader, "sampler", None)
    if sampler is None:
        batch_sampler = getattr(loader, "batch_sampler", None)
        sampler = getattr(batch_sampler, "sampler", None)
    return type(sampler).__name__ if sampler is not None else "unavailable"


def require_sequential_sampler(loader) -> str:
    from torch.utils.data import RandomSampler, SequentialSampler

    sampler = getattr(loader, "sampler", None)
    if isinstance(sampler, RandomSampler) or not isinstance(sampler, SequentialSampler):
        raise TRLContractError("PPO pilot loader must use SequentialSampler")
    return type(sampler).__name__


def install_sequential_ppo_dataloader(
    trainer, expected_records, expected_contract, *, completed_updates=0
):
    """Replace only TRL PPOs shuffled loader and prepare it with its Accelerator."""
    assert_trl_version()
    from torch.utils.data import DataLoader, SequentialSampler

    allowed_profiles = {"ppo_matched_pilot", "ppo_formal_1p5b"}
    if expected_contract.algorithm != "ppo" or expected_contract.profile not in allowed_profiles:
        raise TRLContractError("sequential loader requires a protected PPO pilot/formal profile")
    records = [dict(row) for row in expected_records]
    if len(records) != expected_contract.expected_completions:
        raise TRLContractError("ordered PPO record count differs from protected profile")
    if tuple(row.get("pair_key") for row in records) != expected_contract.pair_keys:
        raise TRLContractError("ordered PPO records differ from protected pair keys")
    if (
        not isinstance(completed_updates, int)
        or isinstance(completed_updates, bool)
        or completed_updates < 0
        or completed_updates >= expected_contract.expected_updates
    ):
        raise TRLContractError("PPO completed-update prefix is invalid")
    completions_per_update = (
        expected_contract.expected_completions // expected_contract.expected_updates
    )
    completed_rows = completed_updates * completions_per_update
    active_records = records[completed_rows:]
    if len(trainer.train_dataset) != expected_contract.expected_completions:
        raise TRLContractError("PPO pilot dataset must contain exactly sixteen records")
    accelerator = trainer.accelerator
    if int(getattr(accelerator, "num_processes", 1)) != 1:
        raise TRLContractError("ordered PPO pilot supports world_size=1 only")
    batch_size = int(trainer.local_dataloader_batch_size)
    expected_batch_size = (
        expected_contract.expected_completions
        if expected_contract.profile == "ppo_matched_pilot"
        else completions_per_update
    )
    if batch_size != expected_batch_size:
        raise TRLContractError("PPO rollout batch differs from the protected profile")
    original_sampler = _sampler_name(trainer.dataloader)
    from torch.utils.data import Subset

    active_dataset = Subset(
        trainer.train_dataset, range(completed_rows, len(trainer.train_dataset))
    )
    sampler = SequentialSampler(active_dataset)
    loader = DataLoader(
        active_dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=OrderedMetadataCollator(trainer.data_collator),
        drop_last=True,
        num_workers=0,
    )
    require_sequential_sampler(loader)
    prepared = accelerator.prepare_data_loader(loader)
    preview = next(iter(prepared), None)
    if preview is None:
        raise TRLContractError("prepared PPO rollout loader is empty")
    actual = validate_ordered_episode_batch(preview, active_records[:batch_size])
    guarded = VerifiedSequentialDataLoader(prepared, loader, active_records)
    trainer.dataloader = guarded
    return {
        "trl_original_sampler_type": original_sampler,
        "replacement_sampler_type": type(loader.sampler).__name__,
        "prepared_loader_type": type(prepared).__name__,
        "batch_size": batch_size,
        "drop_last": loader.drop_last,
        "num_workers": loader.num_workers,
        "world_size": int(getattr(accelerator, "num_processes", 1)),
        "prepared_first_batch_pair_keys": [row["pair_key"] for row in actual],
        "completed_updates": completed_updates,
        "completed_comparison_keys": completed_rows,
        "prepared_first_batch_records": actual,
    }


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
    """Bind TRL GRPO token tensors to ordered reward evidence, batch by batch."""

    def __init__(self, expected_contract):
        if expected_contract.algorithm != "grpo":
            raise TRLContractError("GRPO evidence requires a GRPO execution profile")
        self.contract = expected_contract
        self.expected_completions = expected_contract.expected_completions
        self.multi_batch = expected_contract.profile == "grpo_formal_1p5b"
        self._reward_records: list[dict[str, Any]] = []
        self._completion_records: list[dict[str, Any]] = []

    def record_reward(
        self,
        problem_id: str,
        completion_text: str,
        reward_result,
        scalar_reward: float,
        reward_evidence: dict[str, Any] | None = None,
    ) -> None:
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
        if self._completion_records and not self.multi_batch:
            raise TRLContractError("completion evidence captured more than once")
        completion_count, token_count = exact_completion_counts(payload)
        offset = len(self._completion_records)
        end = offset + completion_count
        if end > self.expected_completions:
            raise TRLContractError("unexpected completion evidence count")
        if len(inputs) != completion_count or len(self._reward_records) != end:
            raise TRLContractError("generation/reward evidence count mismatch")

        ids_tensor = payload["completion_ids"].detach().cpu()
        mask_tensor = payload["completion_mask"].detach().cpu()
        ids_rows = [[int(value) for value in row] for row in ids_tensor.tolist()]
        mask_rows = [[int(value) for value in row] for row in mask_tensor.tolist()]
        decoded = tokenizer.batch_decode(ids_tensor, skip_special_tokens=True)
        if len(decoded) != completion_count:
            raise TRLContractError("tokenizer decode count mismatch")

        per_problem = Counter(row["problem_id"] for row in self._completion_records)
        records = []
        exact_total = 0
        rewards = self._reward_records[offset:end]
        for local_index, (item, ids, mask, text, reward) in enumerate(
            zip(inputs, ids_rows, mask_rows, decoded, rewards, strict=True)
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
            valid_ids = [value for value, keep in zip(ids, mask, strict=True) if keep]
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
            eos_reached = eos_token_id is not None and eos_token_id in valid_ids
            completion_limit = getattr(self.contract, "max_completion_length", None)
            truncated = (
                completion_limit is not None
                and exact_count == completion_limit
                and not eos_reached
            )
            generation_index = per_problem[problem_id]
            per_problem[problem_id] += 1
            pair_key = f"{problem_id}::generation:{generation_index}"
            records.append(
                {
                    "problem_id": problem_id,
                    "prompt_hash": prompt_hash,
                    "generation_index": generation_index,
                    "pair_key": pair_key,
                    "completion_index": offset + local_index,
                    "completion_ids": ids,
                    "completion_mask": mask,
                    "exact_token_count": exact_count,
                    "eos_reached": eos_reached,
                    "truncated": truncated,
                    "decoded_completion": text,
                    **reward,
                }
            )
        if exact_total != token_count:
            raise TRLContractError("serialized completion token total mismatch")
        expected_keys = tuple(self.contract.pair_keys[offset:end])
        actual_keys = tuple(row["pair_key"] for row in records)
        if self.multi_batch:
            valid_keys = actual_keys == expected_keys
        else:
            valid_keys = len(set(actual_keys)) == len(actual_keys) and set(actual_keys) == set(
                expected_keys
            )
        if not valid_keys:
            raise TRLContractError("GRPO completions differ from protected pair keys")
        self._completion_records.extend(records)

    def restore_prefix(self, records: list[dict[str, Any]]) -> None:
        if self._completion_records or self._reward_records:
            raise TRLContractError("GRPO evidence prefix can be restored only once")
        expected = self.contract.pair_keys[: len(records)]
        if (
            len(records) >= self.expected_completions
            or tuple(row.get("pair_key") for row in records) != tuple(expected)
        ):
            raise TRLContractError("GRPO restored evidence is not a proper protected prefix")
        self._completion_records = [dict(row) for row in records]
        self._reward_records = [
            {
                "problem_id": row["problem_id"],
                "raw_completion": row["raw_completion"],
                "verifier_input": row["verifier_input"],
                "reward_status": row["reward_status"],
                "scalar_reward": row["scalar_reward"],
            }
            for row in records
        ]

    def partial_records(self) -> list[dict[str, Any]]:
        return [dict(record) for record in self._completion_records]

    def records(self) -> list[dict[str, Any]]:
        if (
            len(self._completion_records) != self.expected_completions
            or len(self._reward_records) != self.expected_completions
        ):
            raise TRLContractError("completion evidence is incomplete")
        return self.partial_records()


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


def guarded_trainer_class(
    guard,
    evidence_recorder: CompletionEvidenceRecorder,
    checkpoint_callback=None,
    *,
    step_offset=0,
):
    """Create the sole private-API subclass; importing this does not construct a model."""
    assert_trl_version()
    from trl import GRPOTrainer

    class GuardedGRPOTrainer(GRPOTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if step_offset:
                self.args.max_steps = evidence_recorder.contract.expected_updates - step_offset

        def train(self, *args, **kwargs):
            result = super().train(*args, **kwargs)
            self.state.global_step = step_offset + int(self.state.global_step)
            return result

        def _save_checkpoint(self, model, trial):
            if checkpoint_callback is None:
                return super()._save_checkpoint(model, trial)
            return checkpoint_callback(
                self, step_offset + int(self.state.global_step)
            )

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


def optimizer_guard_callback(guard, *, step_offset=0):
    assert_trl_version()
    from transformers import TrainerCallback

    class GuardCallback(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            guard.record_optimizer_step()

        def on_step_end(self, args, state, control, **kwargs):
            guard.record_global_step(step_offset + int(state.global_step))

            if hasattr(guard, "record_update"):
                guard.record_update()

    return GuardCallback()


class PPOCompletionEvidenceRecorder:
    """Join PPO response tensors to protected prompt-major episode identities."""

    def __init__(self, expected_contract, episode_records):
        if expected_contract.algorithm != "ppo":
            raise TRLContractError("PPO evidence requires a PPO execution profile")
        self.contract = expected_contract
        self.expected_completions = expected_contract.expected_completions
        self.multi_batch = expected_contract.profile == "ppo_formal_1p5b"
        self.episode_records = [dict(row) for row in episode_records]
        if (
            len(self.episode_records) != self.expected_completions
            or tuple(row.get("pair_key") for row in self.episode_records)
            != expected_contract.pair_keys
        ):
            raise TRLContractError("PPO episode records differ from protected pair keys")
        self._generation: list[dict[str, Any]] = []
        self._rewards: list[dict[str, Any]] = []

    def capture_generation(self, queries, query_responses, tokenizer, prompt_lookup, pad_token_id):
        assert_trl_version()
        if self._generation and not self.multi_batch:
            raise TRLContractError("PPO generation evidence captured more than once")
        if queries.ndim != 2 or query_responses.ndim != 2:
            raise TRLContractError("PPO query/response tensors must be rank two")
        batch_size = int(queries.shape[0])
        if query_responses.shape[0] != batch_size:
            raise TRLContractError("unexpected PPO response count")
        offset = len(self._generation)
        end = offset + batch_size
        if end > self.expected_completions:
            raise TRLContractError("unexpected PPO response count")
        context_length = int(queries.shape[1])
        response_ids = query_responses[:, context_length:].detach().cpu()
        rows = []
        for local_index, episode in enumerate(self.episode_records[offset:end]):
            query_row = [int(v) for v in queries[local_index].detach().cpu().tolist()]
            prompt_ids = [v for v in query_row if v != pad_token_id]
            metadata = prompt_lookup.get(tuple(prompt_ids))
            if metadata is None:
                raise TRLContractError("PPO prompt tokens not found in fixed lookup")
            if metadata["problem_id"] != episode["problem_id"]:
                raise TRLContractError("PPO generated query differs from ordered episode")
            ids = [int(v) for v in response_ids[local_index].tolist()]
            mask = [int(v != pad_token_id) for v in ids]
            valid_ids = [v for v, keep in zip(ids, mask, strict=True) if keep]
            text = tokenizer.decode(valid_ids, skip_special_tokens=True)
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
            eos_reached = eos_token_id is not None and eos_token_id in valid_ids
            completion_limit = getattr(self.contract, "max_completion_length", None)
            truncated = (
                completion_limit is not None
                and sum(mask) == completion_limit
                and not eos_reached
            )
            rows.append(
                {
                    **episode,
                    "prompt_hash": episode["rendered_prompt_hash"],
                    "completion_index": offset + local_index,
                    "prompt_token_ids": prompt_ids,
                    "response_token_ids": ids,
                    "response_mask": mask,
                    "exact_token_count": sum(mask),
                    "eos_reached": eos_reached,
                    "truncated": truncated,
                    "decoded_completion": text,
                }
            )
        self._generation.extend(rows)
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

    def restore_prefix(self, records: list[dict[str, Any]]) -> None:
        if self._generation or self._rewards:
            raise TRLContractError("PPO evidence prefix can be restored only once")
        expected = self.contract.pair_keys[: len(records)]
        if (
            len(records) >= self.expected_completions
            or tuple(row.get("pair_key") for row in records) != tuple(expected)
        ):
            raise TRLContractError("PPO restored evidence is not a proper protected prefix")
        generation = []
        rewards = []
        for raw in records:
            row = dict(raw)
            row["response_token_ids"] = list(row.pop("completion_ids"))
            row["response_mask"] = list(row.pop("completion_mask"))
            generation.append(row)
            rewards.append(
                {
                    "reward_callback_text": row["raw_completion"],
                    "verifier_input": row["verifier_input"],
                    "reward_status": row["reward_status"],
                    "scalar_reward": row["scalar_reward"],
                    "verifier_detail": row.get("verifier_detail", ""),
                    "canonical_status": row.get("canonical_status", row["reward_status"]),
                    "components": row.get("components", {}),
                }
            )
        self._generation = generation
        self._rewards = rewards

    def partial_records(self) -> list[dict[str, Any]]:
        records = []
        for generated, reward in zip(self._generation, self._rewards, strict=False):
            if generated["decoded_completion"] != reward["reward_callback_text"]:
                raise TRLContractError("PPO decoded completion differs from reward input")
            row = {**generated, **reward}
            if row["reward_callback_text"] != row["verifier_input"]:
                raise TRLContractError("PPO verifier input mismatch")
            records.append(row)
        expected_prefix = self.contract.pair_keys[: len(records)]
        if tuple(row["pair_key"] for row in records) != tuple(expected_prefix):
            raise TRLContractError("PPO completion order differs from protected pair keys")
        return records

    def records(self) -> list[dict[str, Any]]:
        if (
            len(self._generation) != self.expected_completions
            or len(self._rewards) != self.expected_completions
        ):
            raise TRLContractError("incomplete PPO completion/reward evidence")
        return self.partial_records()


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
        telemetry["raw_key"]: telemetry["reason"] for telemetry in PPO_NULLABLE_TELEMETRY.values()
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
        "warnings": [{"category": "nullable_nonessential_telemetry", **item} for item in nullable],
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


def ppo_train_loop_contract(
    args, expected_contract, *, world_size: int, completed_updates: int = 0
) -> dict[str, int]:
    """Validate the TRL 0.24.0 PPO loop derived from the frozen execution profile."""
    if world_size != 1:
        raise TRLContractError("protected PPO execution requires world_size=1")
    if (
        not isinstance(completed_updates, int)
        or isinstance(completed_updates, bool)
        or completed_updates < 0
        or completed_updates >= expected_contract.expected_updates
        or (completed_updates and expected_contract.profile != "ppo_formal_1p5b")
    ):
        raise TRLContractError("PPO completed-update loop prefix is invalid")
    remaining_updates = expected_contract.expected_updates - completed_updates
    local_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
    micro_batch_size = args.per_device_train_batch_size * world_size
    batch_size = local_batch_size * world_size
    local_mini_batch_size = local_batch_size // args.num_mini_batches
    mini_batch_size = batch_size // args.num_mini_batches
    if local_batch_size % args.num_mini_batches:
        raise TRLContractError("PPO local batch is not divisible by num_mini_batches")
    if local_mini_batch_size % args.per_device_train_batch_size:
        raise TRLContractError("PPO minibatch is not divisible into whole microbatches")
    num_total_batches = math.ceil(args.total_episodes / batch_size)
    contract = {
        "total_episodes": args.total_episodes,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "local_batch_size": local_batch_size,
        "micro_batch_size": micro_batch_size,
        "batch_size": batch_size,
        "local_mini_batch_size": local_mini_batch_size,
        "mini_batch_size": mini_batch_size,
        "local_rollout_forward_batch_size": args.local_rollout_forward_batch_size,
        "num_ppo_epochs": args.num_ppo_epochs,
        "num_mini_batches": args.num_mini_batches,
        "microbatches_per_minibatch": (local_mini_batch_size // args.per_device_train_batch_size),
        "num_total_batches": num_total_batches,
        "outer_updates": num_total_batches,
        "expected_optimizer_steps": (
            num_total_batches * args.num_ppo_epochs * args.num_mini_batches
        ),
        "expected_global_steps": num_total_batches,
    }
    completions_per_update = (
        expected_contract.expected_completions // expected_contract.expected_updates
    )
    expected = {
        "total_episodes": remaining_updates * completions_per_update,
        "num_ppo_epochs": expected_contract.expected_ppo_epochs,
        "num_mini_batches": expected_contract.expected_minibatches,
        "outer_updates": remaining_updates,
        "expected_optimizer_steps": remaining_updates,
        "expected_global_steps": remaining_updates,
    }
    for name, value in expected.items():
        if contract[name] != value:
            raise TRLContractError(
                f"PPO train loop contract mismatch for {name}: {contract[name]} != {value}"
            )
    for name in (
        "local_batch_size",
        "micro_batch_size",
        "batch_size",
        "local_mini_batch_size",
        "mini_batch_size",
        "num_total_batches",
    ):
        actual = getattr(args, name, None)
        if actual is not None and actual != contract[name]:
            raise TRLContractError(f"TRL-derived PPO {name} mismatch: {actual} != {contract[name]}")
    return contract


def ppo_loop_position(optimizer_step_index: int, args) -> tuple[int, int, int]:
    """Map a zero-based synchronized optimizer step to TRL's loop indices."""
    if (
        not isinstance(optimizer_step_index, int)
        or isinstance(optimizer_step_index, bool)
        or optimizer_step_index < 0
    ):
        raise TRLContractError("invalid PPO optimizer-step index")
    steps_per_update = args.num_ppo_epochs * args.num_mini_batches
    outer_update, within_update = divmod(optimizer_step_index, steps_per_update)
    epoch_index, minibatch_index = divmod(within_update, args.num_mini_batches)
    return outer_update, epoch_index, minibatch_index


def configure_ppo_gradient_accumulation(
    accelerator, loop_contract: Mapping[str, int]
) -> dict[str, Any]:
    """Prevent a consumed one-batch rollout loader from forcing every inner sync."""
    state = accelerator.gradient_state
    expected_steps = loop_contract["microbatches_per_minibatch"]
    if state.num_steps != expected_steps:
        raise TRLContractError(
            f"Accelerate gradient accumulation mismatch: {state.num_steps} != {expected_steps}"
        )
    before = bool(state.sync_with_dataloader)
    state.plugin_kwargs["sync_with_dataloader"] = False
    if state.sync_with_dataloader:
        raise TRLContractError("PPO requires sync_with_dataloader=false for inner microbatches")
    return {
        "num_steps": state.num_steps,
        "sync_with_dataloader_before": before,
        "sync_with_dataloader": state.sync_with_dataloader,
    }


class PPOBackwardEventGuard:
    """Authoritative PPO microbatch evidence from training forward/backward events."""

    def __init__(self, loop_contract: Mapping[str, int], accumulation_evidence: Mapping[str, Any]):
        self.loop_contract = dict(loop_contract)
        self.accumulation_evidence = dict(accumulation_evidence)
        self.events_per_optimizer = self.loop_contract["microbatches_per_minibatch"]
        self.expected_optimizer_steps = self.loop_contract["expected_optimizer_steps"]
        self.expected_events = self.events_per_optimizer * self.expected_optimizer_steps
        self.expected_batch_size = self.loop_contract["per_device_train_batch_size"]
        self.expected_samples_per_optimizer = self.loop_contract["local_mini_batch_size"]
        self.expected_samples = self.expected_samples_per_optimizer * self.expected_optimizer_steps
        self.events: list[dict[str, Any]] = []
        self.pending_batch_size: int | None = None
        self.optimizer_boundaries = 0

    def note_training_forward(self, batch_size: int) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise TRLContractError("invalid PPO training microbatch size")
        if self.pending_batch_size is not None:
            raise TRLContractError("multiple PPO training forwards before backward")
        self.pending_batch_size = batch_size

    def prepare_backward(self, sync_gradients: bool) -> dict[str, Any]:
        index = len(self.events)
        if index >= self.expected_events:
            raise TRLContractError("too many PPO backward microbatch events")
        if self.pending_batch_size is None:
            raise TRLContractError("PPO backward event has no matching training forward")
        if self.pending_batch_size != self.expected_batch_size:
            raise TRLContractError(
                "unexpected PPO backward microbatch size: "
                f"{self.pending_batch_size} != {self.expected_batch_size}"
            )
        expected_sync = (index + 1) % self.events_per_optimizer == 0
        if bool(sync_gradients) != expected_sync:
            raise TRLContractError(
                f"unexpected PPO sync_gradients at microbatch {index}: "
                f"{bool(sync_gradients)} != {expected_sync}"
            )
        return {
            "microbatch_index": index,
            "optimizer_step_index": index // self.events_per_optimizer,
            "batch_size": self.pending_batch_size,
            "sync_gradients": bool(sync_gradients),
        }

    def commit_backward(self, event: Mapping[str, Any]) -> None:
        if event.get("microbatch_index") != len(self.events):
            raise TRLContractError("PPO backward event commit order mismatch")
        self.events.append(dict(event))
        self.pending_batch_size = None

    def assert_ready_for_optimizer(self) -> None:
        expected_count = (self.optimizer_boundaries + 1) * self.events_per_optimizer
        if self.pending_batch_size is not None or len(self.events) != expected_count:
            raise TRLContractError("PPO optimizer step before all expected backward events")
        group = self.events[-self.events_per_optimizer :]
        if sum(event["batch_size"] for event in group) != self.expected_samples_per_optimizer:
            raise TRLContractError("PPO backward sample total mismatch")
        if not group[-1]["sync_gradients"]:
            raise TRLContractError("PPO optimizer step without final synchronized backward")

    def commit_optimizer_step(self) -> None:
        self.assert_ready_for_optimizer()
        self.optimizer_boundaries += 1
        if self.optimizer_boundaries > self.expected_optimizer_steps:
            raise TRLContractError("too many PPO optimizer boundaries")

    def assert_complete(self, underlying_optimizer_steps: int) -> dict[str, Any]:
        if self.pending_batch_size is not None or len(self.events) != self.expected_events:
            raise TRLContractError("PPO optimizer step before all expected backward events")
        if sum(event["batch_size"] for event in self.events) != self.expected_samples:
            raise TRLContractError("PPO backward sample total mismatch")
        expected_optimizer_steps = self.loop_contract["expected_optimizer_steps"]
        if underlying_optimizer_steps != expected_optimizer_steps:
            raise TRLContractError(
                "PPO underlying optimizer-step count mismatch: "
                f"{underlying_optimizer_steps} != {expected_optimizer_steps}"
            )
        return {
            "source": "accelerator.backward",
            **self.accumulation_evidence,
            "events_per_optimizer_step": self.events_per_optimizer,
            "expected_backward_events": self.expected_events,
            "backward_events": len(self.events),
            "microbatch_sizes": [event["batch_size"] for event in self.events],
            "processed_samples": sum(event["batch_size"] for event in self.events),
            "sync_gradients": [event["sync_gradients"] for event in self.events],
            "underlying_optimizer_steps": underlying_optimizer_steps,
        }


def ppo_guarded_trainer_class(
    guard,
    evidence_recorder,
    prompt_lookup,
    generation_contract,
    *,
    ordered_episode_records=None,
    expected_contract=None,
    checkpoint_callback=None,
    update_callback=None,
    completed_updates=0,
):
    """The sole TRL 0.24.0 PPO private compatibility shim."""
    assert_trl_version()
    import trl.trainer.ppo_trainer as module
    from trl import PPOTrainer

    class GuardedPPOTrainer(PPOTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if expected_contract is None:
                raise TRLContractError("guarded PPO trainer requires an execution profile")
            remaining_updates = expected_contract.expected_updates - completed_updates
            completions_per_update = (
                expected_contract.expected_completions // expected_contract.expected_updates
            )
            if completed_updates:
                self.args.total_episodes = remaining_updates * completions_per_update
                self.args.num_total_batches = remaining_updates
            self.ppo_loop_contract = ppo_train_loop_contract(
                self.args,
                expected_contract,
                world_size=self.accelerator.num_processes,
                completed_updates=completed_updates,
            )
            self.ppo_accumulation_evidence = configure_ppo_gradient_accumulation(
                self.accelerator, self.ppo_loop_contract
            )
            self.ppo_backward_evidence = None
            self.ordered_loader_evidence = None
            if ordered_episode_records is not None:
                self.ordered_loader_evidence = install_sequential_ppo_dataloader(
                    self,
                    ordered_episode_records,
                    expected_contract,
                    completed_updates=completed_updates,
                )

        def _save_checkpoint(self, model, trial):
            if checkpoint_callback is None:
                # Single-update runners write one role-separated checkpoint after success.
                return None
            return checkpoint_callback(
                self, completed_updates + int(self.state.global_step)
            )

        def log(self, logs, start_time=None):
            local_step = int(self.state.global_step)
            is_update = "loss/policy_avg" in logs
            absolute_step = completed_updates + local_step
            if is_update:
                guard.record_update()
                guard.record_global_step(absolute_step)
            self.state.global_step = absolute_step
            try:
                result = super().log(logs, start_time)
                if is_update and update_callback is not None:
                    update_callback(self, absolute_step)
                return result
            finally:
                self.state.global_step = local_step

        def train(self, *args, **kwargs):
            original_generation = module.batch_generation
            original_get_reward = module.get_reward
            original_forward = module.forward
            original_backward = self.accelerator.backward
            underlying_optimizer = getattr(self.optimizer, "optimizer", None)
            if underlying_optimizer is None:
                raise TRLContractError("guarded PPO requires an AcceleratedOptimizer")
            original_underlying_step = underlying_optimizer.step
            backward_guard = PPOBackwardEventGuard(
                self.ppo_loop_contract, self.ppo_accumulation_evidence
            )
            underlying_optimizer_steps = 0

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

            def guarded_forward(model, query_responses, pad_token_id):
                if torch.is_grad_enabled() and model is self.model:
                    backward_guard.note_training_forward(int(query_responses.shape[0]))
                return original_forward(model, query_responses, pad_token_id)

            def guarded_backward(*args, **kwargs):
                event = backward_guard.prepare_backward(self.accelerator.sync_gradients)
                result = original_backward(*args, **kwargs)
                backward_guard.commit_backward(event)
                return result

            def guarded_underlying_step(*args, **kwargs):
                nonlocal underlying_optimizer_steps
                backward_guard.assert_ready_for_optimizer()
                loop_key = ppo_loop_position(underlying_optimizer_steps, self.args)
                guard.record_loop_position(
                    loop_key[0] + completed_updates, loop_key[1], loop_key[2]
                )
                guard.record_optimizer_step()
                result = original_underlying_step(*args, **kwargs)
                underlying_optimizer_steps += 1
                backward_guard.commit_optimizer_step()
                return result

            module.batch_generation = guarded_generation
            module.get_reward = guarded_get_reward
            module.forward = guarded_forward
            self.accelerator.backward = guarded_backward
            underlying_optimizer.step = guarded_underlying_step
            try:
                result = super().train(*args, **kwargs)
                self.ppo_backward_evidence = backward_guard.assert_complete(
                    underlying_optimizer_steps
                )
                return result
            finally:
                underlying_optimizer.step = original_underlying_step
                self.state.global_step = (
                    completed_updates + int(self.state.global_step)
                )
                self.accelerator.backward = original_backward
                module.forward = original_forward
                module.get_reward = original_get_reward
                module.batch_generation = original_generation

    return GuardedPPOTrainer
