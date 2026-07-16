"""Formal 1.5B multi-update orchestration with no eager model or CUDA imports."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from math_rlvr.training.formal import (
    FORMAL_CHECKPOINT_STEPS,
    FORMAL_COMPLETIONS,
    FORMAL_MAX_COMPLETION_LENGTH,
    FORMAL_MODEL,
    FORMAL_RESPONSES_PER_PROMPT,
    FORMAL_REVISION,
    FORMAL_TOKEN_CAP,
    FORMAL_UPDATES,
    TRAIN_MANIFEST,
    file_sha256,
    formal_pair_keys,
    formal_training_schedule,
    validate_active_suite,
    validate_formal_config_file,
)

REQUIRED_TRAINING_FILES = (
    "resolved_config.json",
    "run_manifest.json",
    "metrics.csv",
    "metrics.jsonl",
    "completions.jsonl",
    "validation_metrics.csv",
    "checkpoint_inventory.json",
    "resource_metrics.csv",
    "report.md",
    "error_analysis.md",
)


class FormalRuntimeError(RuntimeError):
    """A frozen formal execution or evidence invariant was violated."""


@dataclass(frozen=True)
class FormalRunContract:
    algorithm: str
    seed: int
    config_path: str
    config_sha256: str
    active_suite_sha256: str
    updates: int = FORMAL_UPDATES
    prompts_per_update: int = 4
    responses_per_prompt: int = FORMAL_RESPONSES_PER_PROMPT
    completions_per_update: int = 16
    expected_completions: int = FORMAL_COMPLETIONS
    max_completion_length: int = FORMAL_MAX_COMPLETION_LENGTH
    token_cap: int = FORMAL_TOKEN_CAP
    optimizer_steps: int = FORMAL_UPDATES
    global_steps: int = FORMAL_UPDATES
    checkpoint_steps: tuple[int, ...] = FORMAL_CHECKPOINT_STEPS
    validation_steps: tuple[int, ...] = FORMAL_CHECKPOINT_STEPS

    @property
    def profile(self) -> str:
        return f"{self.algorithm}_formal_1p5b"

    @property
    def expected_prompt_count(self) -> int:
        return 128

    @property
    def expected_updates(self) -> int:
        return self.updates

    @property
    def expected_optimizer_steps(self) -> int:
        return self.optimizer_steps

    @property
    def expected_global_steps(self) -> int:
        return self.global_steps

    @property
    def expected_ppo_epochs(self) -> int:
        return 1 if self.algorithm == "ppo" else 0

    @property
    def expected_minibatches(self) -> int:
        return 1 if self.algorithm == "ppo" else 0

    @property
    def pair_keys(self) -> tuple[str, ...]:
        return tuple(formal_pair_keys())

    def pair_keys_for_update(self, update: int) -> tuple[str, ...]:
        if update < 1 or update > self.updates:
            raise FormalRuntimeError("formal update index outside frozen contract")
        start = (update - 1) * self.completions_per_update
        return self.pair_keys[start : start + self.completions_per_update]

    def as_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "seed": self.seed,
            "config_path": self.config_path,
            "config_sha256": self.config_sha256,
            "active_suite_sha256": self.active_suite_sha256,
            "updates": self.updates,
            "prompts_per_update": self.prompts_per_update,
            "responses_per_prompt": self.responses_per_prompt,
            "completions_per_update": self.completions_per_update,
            "expected_completions": self.expected_completions,
            "max_completion_length": self.max_completion_length,
            "token_cap": self.token_cap,
            "optimizer_steps": self.optimizer_steps,
            "global_steps": self.global_steps,
            "checkpoint_steps": list(self.checkpoint_steps),
            "validation_steps": list(self.validation_steps),
            "pair_keys_sha256": hashlib.sha256(
                json.dumps(self.pair_keys, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        }


def formal_training_problems() -> list[Any]:
    from math_rlvr.dataset import load_manifest

    problems = load_manifest(TRAIN_MANIFEST)
    by_id = {problem.problem_id: problem for problem in problems}
    ordered_ids = formal_training_schedule()["ordered_problem_ids"]
    ordered = [by_id[problem_id] for problem_id in ordered_ids]
    if len(ordered) != 128 or len({problem.problem_id for problem in ordered}) != 128:
        raise FormalRuntimeError("formal training problem order is incomplete")
    return ordered


def formal_episode_records(algorithm: str, seed: int) -> list[dict[str, Any]]:
    if algorithm not in {"ppo", "grpo"} or seed not in {42, 123}:
        raise FormalRuntimeError("formal episode records require an active algorithm/seed")
    from math_rlvr.training.pilot import (
        canonical_json_sha256,
        rendered_prompt_payload_sha256,
    )

    records = []
    for position, problem in enumerate(formal_training_problems()):
        generations = range(4) if algorithm == "ppo" else range(1)
        for generation_index in generations:
            records.append(
                {
                    "episode_position": len(records),
                    "problem_id": problem.problem_id,
                    "generation_index": generation_index,
                    "pair_key": f"{problem.problem_id}::generation:{generation_index}",
                    "problem_hash": canonical_json_sha256(
                        {
                            "problem_id": problem.problem_id,
                            "source": problem.source,
                            "split": problem.split,
                            "content_hash": problem.content_hash,
                            "category": problem.category,
                            "difficulty": problem.difficulty,
                            "dataset_id": problem.metadata.get("dataset_id"),
                            "revision": problem.metadata.get("revision"),
                            "source_split": problem.metadata.get("source_split"),
                            "source_index": problem.metadata.get("source_index"),
                        }
                    ),
                    "rendered_prompt_hash": rendered_prompt_payload_sha256(
                        problem, "prompt_v2_formal_math"
                    ),
                    "seed": seed,
                    "algorithm": algorithm,
                    "schedule_position": position,
                }
            )
    expected = 512 if algorithm == "ppo" else 128
    if len(records) != expected:
        raise FormalRuntimeError("formal episode row count mismatch")
    if algorithm == "ppo" and tuple(row["pair_key"] for row in records) != tuple(
        formal_pair_keys()
    ):
        raise FormalRuntimeError("formal PPO pair-key order mismatch")
    return records


def formal_run_contract(config: dict[str, Any]) -> FormalRunContract:
    algorithm = config.get("experiment", {}).get("algorithm")
    config_path = config.get("resolved_config_path")
    if algorithm not in {"ppo", "grpo"} or not isinstance(config_path, str):
        raise FormalRuntimeError("formal runtime requires a resolved PPO/GRPO config")
    frozen, derived = validate_formal_config_file(Path(config_path), algorithm)
    if config.get("resolved_config_sha256") != frozen["resolved_config_sha256"]:
        raise FormalRuntimeError("formal runtime resolved config identity mismatch")
    suite = validate_active_suite()
    accepted = {
        (row["algorithm"], row["seed"], row["config"], row["config_sha256"])
        for row in suite["active_training_runs"]
    }
    identity = (
        algorithm,
        config["experiment"]["seed"],
        config_path,
        config["resolved_config_sha256"],
    )
    if identity not in accepted:
        raise FormalRuntimeError("config is not in the four-run active suite")
    if any(
        derived[key] != expected
        for key, expected in {
            "outer_updates": 32,
            "optimizer_steps": 32,
            "global_steps": 32,
            "total_completions": 512,
            "total_generated_tokens": 131_072,
        }.items()
    ):
        raise FormalRuntimeError("formal derived training budget drift")
    return FormalRunContract(
        algorithm=algorithm,
        seed=config["experiment"]["seed"],
        config_path=config_path,
        config_sha256=config["resolved_config_sha256"],
        active_suite_sha256=suite["active_suite_sha256"],
    )


def _finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FormalRuntimeError(f"required metric {name} is not numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise FormalRuntimeError(f"required metric {name} is non-finite")
    return numeric


def _standard_json(value: Any, path: str = "root") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FormalRuntimeError(f"non-finite JSON value at {path}")
        return
    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _standard_json(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise FormalRuntimeError(f"non-string JSON key at {path}")
            _standard_json(item, f"{path}.{key}")
        return
    raise FormalRuntimeError(f"non-JSON value at {path}: {type(value).__name__}")


class FormalOnlineGuard:
    """Immediate caps for the delayed TRL loop; final evidence is checked separately."""

    def __init__(self, contract: FormalRunContract):
        self.contract = contract
        self.completions = 0
        self.generated_tokens = 0
        self.rewards = 0
        self.microsteps = 0
        self.optimizer_steps = 0
        self.global_steps = 0
        self.updates = 0
        self.loop_positions: set[tuple[int, int, int]] = set()

    def record_generation(self, completions: int, tokens: int) -> None:
        if completions != self.contract.completions_per_update or tokens < 0:
            raise FormalRuntimeError("formal TRL rollout batch differs from 16-completion contract")
        self.completions += completions
        self.generated_tokens += tokens
        if self.completions > self.contract.expected_completions:
            raise FormalRuntimeError("formal TRL completion cap exceeded")
        if self.generated_tokens > self.contract.token_cap:
            raise FormalRuntimeError("formal TRL generated-token cap exceeded")

    def record_reward(self, result: Any, scalar: float, evidence: dict[str, Any]) -> None:
        _finite("online_scalar_reward", scalar)
        _standard_json(evidence)
        self.rewards += 1
        if self.rewards > self.contract.expected_completions:
            raise FormalRuntimeError("formal TRL reward count exceeded")

    def record_microstep(self) -> None:
        self.microsteps += 1
        if self.microsteps > self.contract.updates * 4:
            raise FormalRuntimeError("formal GRPO microstep cap exceeded")

    def record_loop_position(self, outer: int, epoch: int, minibatch: int) -> None:
        key = (outer, epoch, minibatch)
        if epoch != 0 or minibatch != 0 or outer != len(self.loop_positions):
            raise FormalRuntimeError("formal PPO epoch/minibatch/update position drift")
        if key in self.loop_positions:
            raise FormalRuntimeError("formal PPO loop position repeated")
        self.loop_positions.add(key)

    def record_optimizer_step(self) -> None:
        self.optimizer_steps += 1
        if self.optimizer_steps > self.contract.optimizer_steps:
            raise FormalRuntimeError("formal optimizer-step cap exceeded")

    def record_global_step(self, step: int) -> None:
        if step != self.global_steps + 1:
            raise FormalRuntimeError("formal global-step continuity mismatch")
        self.global_steps = step
        if step > self.contract.global_steps:
            raise FormalRuntimeError("formal global-step cap exceeded")

    def record_update(self) -> None:
        self.updates += 1
        if self.updates > self.contract.updates:
            raise FormalRuntimeError("formal update cap exceeded")

    def assert_complete(self) -> dict[str, int]:
        expected_microsteps = self.contract.updates * 4 if self.contract.algorithm == "grpo" else 0
        if (
            self.completions != self.contract.expected_completions
            or self.rewards != self.contract.expected_completions
            or self.optimizer_steps != self.contract.optimizer_steps
            or self.global_steps != self.contract.global_steps
            or self.updates != self.contract.updates
            or (self.contract.algorithm == "grpo" and self.microsteps != expected_microsteps)
            or (
                self.contract.algorithm == "ppo"
                and len(self.loop_positions) != self.contract.updates
            )
        ):
            raise FormalRuntimeError("formal online TRL counters are incomplete")
        return {
            "completions": self.completions,
            "generated_tokens": self.generated_tokens,
            "rewards": self.rewards,
            "microsteps": self.microsteps,
            "optimizer_steps": self.optimizer_steps,
            "global_steps": self.global_steps,
            "updates": self.updates,
        }


class FormalProgressGuard:
    """Authoritative multi-update counters shared by PPO and GRPO formal backends."""

    def __init__(self, contract: FormalRunContract, *, run_id: str):
        self.contract = contract
        self.run_id = run_id
        self.updates = 0
        self.optimizer_steps = 0
        self.global_steps = 0
        self.completions = 0
        self.generated_tokens = 0
        self.seen_pair_keys: list[str] = []
        self.checkpoints: list[int] = []
        self.validations: list[int] = []

    def _validate_completion(self, row: dict[str, Any], expected_key: str, update: int) -> int:
        if row.get("pair_key") != expected_key or row.get("update") != update:
            raise FormalRuntimeError("formal completion key/update order mismatch")
        ids = row.get("completion_ids")
        mask = row.get("completion_mask")
        if not isinstance(ids, list) or not isinstance(mask, list) or len(ids) != len(mask):
            raise FormalRuntimeError("formal completion IDs/mask evidence missing")
        if any(not isinstance(value, int) or isinstance(value, bool) for value in ids):
            raise FormalRuntimeError("formal completion token IDs must be integers")
        if any(value not in (0, 1) for value in mask):
            raise FormalRuntimeError("formal completion mask must be binary")
        token_count = sum(mask)
        if row.get("exact_token_count") != token_count:
            raise FormalRuntimeError("formal exact token count is not mask-derived")
        if token_count > self.contract.max_completion_length:
            raise FormalRuntimeError("formal completion exceeds max completion length")
        _finite("scalar_reward", row.get("scalar_reward"))
        if not isinstance(row.get("raw_completion"), str):
            raise FormalRuntimeError("formal raw completion text missing")
        return token_count

    def record_update(
        self,
        *,
        update: int,
        completion_rows: list[dict[str, Any]],
        metrics: dict[str, Any],
        optimizer_step: int,
        global_step: int,
    ) -> None:
        if update != self.updates + 1 or optimizer_step != update or global_step != update:
            raise FormalRuntimeError("formal update/optimizer/global continuity mismatch")
        expected_keys = self.contract.pair_keys_for_update(update)
        if len(completion_rows) != self.contract.completions_per_update:
            raise FormalRuntimeError("formal update must contain exactly 16 completions")
        actual_keys = tuple(row.get("pair_key") for row in completion_rows)
        if actual_keys != expected_keys or len(set(actual_keys)) != len(actual_keys):
            raise FormalRuntimeError(
                "formal update comparison keys are missing, duplicated, or reordered"
            )
        update_tokens = sum(
            self._validate_completion(row, key, update)
            for row, key in zip(completion_rows, expected_keys, strict=True)
        )
        if self.generated_tokens + update_tokens > self.contract.token_cap:
            raise FormalRuntimeError("formal generated-token hard cap exceeded")
        required = {
            "reward_mean",
            "reward_std",
            "reward_variance",
            "loss",
            "grad_norm",
            "entropy",
            "learning_rate",
            "mean_completion_length",
            "zero_advantage_fraction",
            "format_accuracy",
            "valid_answer_rate",
            "canonical_pass_rate",
        }
        if self.contract.algorithm == "ppo":
            required |= {"policy_loss", "value_loss"}
        missing = required - metrics.keys()
        if missing:
            raise FormalRuntimeError(f"formal required metrics missing: {sorted(missing)}")
        for name in required:
            _finite(name, metrics[name])
        kl = metrics.get("kl")
        if kl is None:
            if not metrics.get("kl_unavailable_reason"):
                raise FormalRuntimeError("nullable KL requires an explicit reason")
        else:
            _finite("kl", kl)
        _standard_json(metrics)
        self.updates = update
        self.optimizer_steps = optimizer_step
        self.global_steps = global_step
        self.completions += len(completion_rows)
        self.generated_tokens += update_tokens
        self.seen_pair_keys.extend(actual_keys)

    def record_checkpoint(self, step: int) -> None:
        expected_index = len(self.checkpoints)
        if expected_index >= len(self.contract.checkpoint_steps):
            raise FormalRuntimeError("too many formal checkpoints")
        if step != self.contract.checkpoint_steps[expected_index] or step != self.updates:
            raise FormalRuntimeError("formal checkpoint cadence mismatch")
        self.checkpoints.append(step)

    def record_validation(self, step: int, rows: list[dict[str, Any]]) -> None:
        expected_index = len(self.validations)
        if expected_index >= len(self.contract.validation_steps):
            raise FormalRuntimeError("too many formal validations")
        if step != self.contract.validation_steps[expected_index] or step != self.updates:
            raise FormalRuntimeError("formal validation cadence mismatch")
        if len(rows) != 64 or any(row.get("checkpoint_step") != step for row in rows):
            raise FormalRuntimeError("formal validation must contain 64 rows for its checkpoint")
        self.validations.append(step)

    def assert_complete(self) -> dict[str, Any]:
        if (
            self.updates != self.contract.updates
            or self.optimizer_steps != self.contract.optimizer_steps
            or self.global_steps != self.contract.global_steps
            or self.completions != self.contract.expected_completions
            or tuple(self.seen_pair_keys) != self.contract.pair_keys
            or tuple(self.checkpoints) != self.contract.checkpoint_steps
            or tuple(self.validations) != self.contract.validation_steps
        ):
            raise FormalRuntimeError("formal run did not satisfy the exact final contract")
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "algorithm": self.contract.algorithm,
            "seed": self.contract.seed,
            "config_sha256": self.contract.config_sha256,
            "active_suite_sha256": self.contract.active_suite_sha256,
            "updates": self.updates,
            "optimizer_steps": self.optimizer_steps,
            "global_steps": self.global_steps,
            "completions": self.completions,
            "generated_tokens": self.generated_tokens,
            "seen_pair_keys": list(self.seen_pair_keys),
            "checkpoints": list(self.checkpoints),
            "validations": list(self.validations),
        }

    @classmethod
    def from_resume_manifest(
        cls, contract: FormalRunContract, manifest: dict[str, Any], *, run_id: str
    ) -> FormalProgressGuard:
        guard = cls(contract, run_id=run_id)
        for key, expected in {
            "run_id": run_id,
            "algorithm": contract.algorithm,
            "seed": contract.seed,
            "config_sha256": contract.config_sha256,
            "active_suite_sha256": contract.active_suite_sha256,
        }.items():
            if manifest.get(key) != expected:
                raise FormalRuntimeError(f"resume manifest {key} mismatch")
        step = manifest.get("updates")
        if step not in contract.checkpoint_steps[:-1]:
            raise FormalRuntimeError(
                "resume is allowed only from this run's step 8/16/24 checkpoint"
            )
        expected_count = step * contract.completions_per_update
        if (
            manifest.get("optimizer_steps") != step
            or manifest.get("global_steps") != step
            or manifest.get("completions") != expected_count
            or manifest.get("seen_pair_keys") != list(contract.pair_keys[:expected_count])
            or manifest.get("checkpoints") != [s for s in contract.checkpoint_steps if s <= step]
            or manifest.get("validations") != [s for s in contract.validation_steps if s <= step]
        ):
            raise FormalRuntimeError("resume counters are not a valid prefix of the frozen run")
        guard.updates = step
        guard.optimizer_steps = step
        guard.global_steps = step
        guard.completions = expected_count
        guard.generated_tokens = int(manifest.get("generated_tokens", -1))
        if guard.generated_tokens < 0 or guard.generated_tokens > contract.token_cap:
            raise FormalRuntimeError("resume generated-token counter invalid")
        guard.seen_pair_keys = list(manifest["seen_pair_keys"])
        guard.checkpoints = list(manifest["checkpoints"])
        guard.validations = list(manifest["validations"])
        return guard


_ALLOWED_COMMON = {
    "trainer_state.json",
    "resume_manifest.json",
}
_ALLOWED_PPO = _ALLOWED_COMMON | {
    "policy_adapter/adapter_model.safetensors",
    "policy_adapter/adapter_config.json",
    "value_adapter/adapter_model.safetensors",
    "value_adapter/adapter_config.json",
    "value_head/value_head.safetensors",
    "value_head/config.json",
}
_ALLOWED_GRPO = _ALLOWED_COMMON | {
    "policy_adapter/adapter_model.safetensors",
    "policy_adapter/adapter_config.json",
}


def formal_checkpoint_inventory(
    root: Path, contract: FormalRunContract, step: int
) -> dict[str, Any]:
    if (
        root.name != f"checkpoint-{step}"
        or root.is_symlink()
        or step not in contract.checkpoint_steps
    ):
        raise FormalRuntimeError("formal checkpoint root/step mismatch")
    resolved_root = root.resolve(strict=True)
    allowed = _ALLOWED_PPO if contract.algorithm == "ppo" else _ALLOWED_GRPO
    files: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted(root.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        if path.is_symlink() or not path.is_file():
            raise FormalRuntimeError("formal checkpoint symlink/non-file rejected")
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(resolved_root):
            raise FormalRuntimeError("formal checkpoint path escaped its root")
        relative = path.relative_to(root).as_posix()
        if relative not in allowed or relative in seen:
            raise FormalRuntimeError(f"unexpected formal checkpoint file: {relative}")
        size = path.stat().st_size
        if size >= 500_000_000 or path.name in {"model.safetensors", "pytorch_model.bin"}:
            raise FormalRuntimeError("full base-model weight detected in formal checkpoint")
        seen.add(relative)
        files.append(
            {
                "path": relative,
                "size_bytes": size,
                "sha256": file_sha256(path),
                "role": relative.split("/", 1)[0],
            }
        )
    if seen != allowed:
        raise FormalRuntimeError(f"formal checkpoint role files mismatch: {sorted(allowed - seen)}")
    resume = json.loads((root / "resume_manifest.json").read_text(encoding="utf-8"))
    if resume.get("updates") != step or resume.get("base_weights_included") is not False:
        raise FormalRuntimeError("formal checkpoint resume manifest is unsafe")
    return {
        "checkpoint_step": step,
        "checkpoint_root": root.name,
        "algorithm": contract.algorithm,
        "base_weights_included": False,
        "files": files,
        "total_size_bytes": sum(row["size_bytes"] for row in files),
    }


class FormalTrainingBackend(Protocol):
    def execute(
        self,
        contract: FormalRunContract,
        observer: FormalRuntimeObserver,
        *,
        start_update: int,
    ) -> None: ...


class CompletedTrainerBackend:
    """Minimal bridge from one real 32-step Trainer call to the formal observer.

    Model/tokenizer construction remains in the proven PPO/GRPO delayed builders. The
    bridge owns no model logic: it validates the trainer result, normalizes already-
    captured evidence, and replays each update into the same guard used by CPU fakes.
    """

    def __init__(
        self,
        *,
        trainer: Any,
        evidence_recorder: Any,
        completion_normalizer: Any,
        metric_normalizer: Any,
        validation_runner: Any,
        checkpoint_root: Path,
        online_guard: FormalOnlineGuard,
        resume_checkpoint: Path | None = None,
    ):
        self.trainer = trainer
        self.evidence_recorder = evidence_recorder
        self.completion_normalizer = completion_normalizer
        self.metric_normalizer = metric_normalizer
        self.validation_runner = validation_runner
        self.checkpoint_root = checkpoint_root
        self.online_guard = online_guard
        self.resume_checkpoint = resume_checkpoint

    def execute(
        self, contract: FormalRunContract, observer: FormalRuntimeObserver, *, start_update: int
    ) -> None:
        expected_start = 1 if self.resume_checkpoint is None else start_update
        if start_update != expected_start:
            raise FormalRuntimeError("formal Trainer start/update continuity mismatch")
        train_kwargs = {}
        if self.resume_checkpoint is not None:
            train_kwargs["resume_from_checkpoint"] = str(self.resume_checkpoint)
        self.trainer.train(**train_kwargs)
        absolute_global_step = int(self.trainer.state.global_step)
        if absolute_global_step != contract.global_steps:
            raise FormalRuntimeError("formal Trainer did not finish at global step 32")
        self.online_guard.assert_complete()
        raw_records = self.evidence_recorder.records()
        records = self.completion_normalizer(raw_records, contract)
        if len(records) != contract.expected_completions:
            raise FormalRuntimeError("formal Trainer completion evidence is incomplete")
        metrics = self.metric_normalizer(
            [dict(row) for row in self.trainer.state.log_history], records, contract
        )
        if len(metrics) != contract.updates:
            raise FormalRuntimeError("formal Trainer metrics do not cover 32 updates")
        for update in range(start_update, contract.updates + 1):
            start = (update - 1) * contract.completions_per_update
            end = start + contract.completions_per_update
            observer.update(
                update,
                records[start:end],
                metrics[update - 1],
                optimizer_step=update,
                global_step=update,
            )
            if update in contract.validation_steps:
                validation_rows = self.validation_runner(update)
                observer.validation(update, validation_rows)
                observer.checkpoint(update, self.checkpoint_root / f"checkpoint-{update}")


class FormalRuntimeObserver:
    """Backend-facing event sink; all evidence passes through the frozen guard."""

    def __init__(self, contract: FormalRunContract, run_dir: Path, run_id: str):
        self.contract = contract
        self.run_dir = run_dir
        self.guard = FormalProgressGuard(contract, run_id=run_id)
        self.metrics: list[dict[str, Any]] = []
        self.completions: list[dict[str, Any]] = []
        self.validation_metrics: list[dict[str, Any]] = []
        self.checkpoint_inventory: list[dict[str, Any]] = []
        self.resource_metrics: list[dict[str, Any]] = []

    def restore(self, checkpoint: Path) -> None:
        manifest = json.loads((checkpoint / "resume_manifest.json").read_text(encoding="utf-8"))
        self.guard = FormalProgressGuard.from_resume_manifest(
            self.contract, manifest, run_id=self.guard.run_id
        )
        self.metrics = _read_jsonl(self.run_dir / "metrics.jsonl")
        self.completions = _read_jsonl(self.run_dir / "completions.jsonl")
        self.validation_metrics = _read_csv(self.run_dir / "validation_metrics.csv")
        inventory_path = self.run_dir / "checkpoint_inventory.json"
        if inventory_path.is_file():
            self.checkpoint_inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        if (
            len(self.metrics) != self.guard.updates
            or len(self.completions) != self.guard.completions
        ):
            raise FormalRuntimeError("resume artifacts do not match checkpoint counters")

    def update(
        self,
        update: int,
        completion_rows: list[dict[str, Any]],
        metrics: dict[str, Any],
        *,
        optimizer_step: int,
        global_step: int,
    ) -> None:
        self.guard.record_update(
            update=update,
            completion_rows=completion_rows,
            metrics=metrics,
            optimizer_step=optimizer_step,
            global_step=global_step,
        )
        self.completions.extend(completion_rows)
        self.metrics.append({"update": update, **metrics})

    def checkpoint(self, step: int, root: Path) -> None:
        self.guard.record_checkpoint(step)
        self.checkpoint_inventory.append(formal_checkpoint_inventory(root, self.contract, step))

    def validation(self, step: int, rows: list[dict[str, Any]]) -> None:
        self.guard.record_validation(step, rows)
        self.validation_metrics.extend(rows)


def _write_json(path: Path, payload: Any) -> None:
    _standard_json(payload)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        _standard_json(row)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows)
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fieldnames:
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or not path.read_text(encoding="utf-8"):
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def finalize_formal_training(
    config: dict[str, Any],
    observer: FormalRuntimeObserver,
    *,
    status: str,
    reason: str | None = None,
) -> dict[str, Any]:
    run_dir = observer.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    counters = (
        observer.guard.assert_complete() if status == "success" else observer.guard.snapshot()
    )
    manifest = {
        "schema_version": 1,
        "run_id": observer.guard.run_id,
        "status": status,
        "reason": reason,
        "model": FORMAL_MODEL,
        "revision": FORMAL_REVISION,
        "local_files_only": True,
        "algorithm": observer.contract.algorithm,
        "seed": observer.contract.seed,
        "formal_contract": observer.contract.as_dict(),
        "prompt_version": config["prompt_version"],
        "prompt_sha256": config["prompt_sha256"],
        "reward_policy_version": config["reward_policy_version"],
        "reward_policy_sha256": config["reward_policy_sha256"],
        "parser_contract": config["parser_contract"],
        "verifier_contract": config["verifier_contract"],
        "counters": counters,
        "automatic_retries": 0,
    }
    _write_json(run_dir / "resolved_config.json", config)
    _write_json(run_dir / "run_manifest.json", manifest)
    _write_jsonl(run_dir / "metrics.jsonl", observer.metrics)
    _write_csv(run_dir / "metrics.csv", observer.metrics)
    _write_jsonl(run_dir / "completions.jsonl", observer.completions)
    _write_csv(run_dir / "validation_metrics.csv", observer.validation_metrics)
    _write_json(run_dir / "checkpoint_inventory.json", observer.checkpoint_inventory)
    _write_csv(run_dir / "resource_metrics.csv", observer.resource_metrics)
    (run_dir / "report.md").write_text(
        "# Formal 1.5B training run\n\n"
        f"- Status: {status}\n- Algorithm: {observer.contract.algorithm}\n"
        f"- Seed: {observer.contract.seed}\n- Reason: {reason}\n"
        "- Two-seed portfolio comparison; no statistical-significance claim.\n",
        encoding="utf-8",
    )
    (run_dir / "error_analysis.md").write_text(
        "# Error analysis\n\nGenerated only from persisted completion and verifier evidence.\n",
        encoding="utf-8",
    )
    if status != "success":
        _write_json(
            run_dir / "failure_report.json",
            {"status": status, "reason": reason, "counters": counters},
        )
    for name in REQUIRED_TRAINING_FILES:
        target = run_dir / name
        if not target.exists():
            raise FormalRuntimeError(f"formal artifact finalization missed {name}")
    return manifest


def execute_formal_training(
    config: dict[str, Any],
    backend: FormalTrainingBackend,
    *,
    run_dir: Path,
    run_id: str,
    resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    contract = formal_run_contract(config)
    observer = FormalRuntimeObserver(contract, run_dir, run_id)
    start_update = 1
    if resume_checkpoint is not None:
        if resume_checkpoint.parent.resolve() != run_dir.resolve():
            raise FormalRuntimeError(
                "formal resume checkpoint must belong to the same run directory"
            )
        observer.restore(resume_checkpoint)
        start_update = observer.guard.updates + 1
    try:
        backend.execute(contract, observer, start_update=start_update)
        return finalize_formal_training(config, observer, status="success")
    except Exception as exc:
        finalize_formal_training(config, observer, status="failure", reason=str(exc))
        raise


def create_formal_backup(run_dir: Path, archive: Path) -> dict[str, Any]:
    if not run_dir.is_dir() or archive.exists():
        raise FormalRuntimeError("formal backup source/destination state invalid")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(run_dir, arcname=run_dir.name)
    with tarfile.open(archive, "r:gz") as handle:
        names = handle.getnames()
    forbidden = {"model.safetensors", "pytorch_model.bin", "auth.json", "token.json"}
    if any(Path(name).name in forbidden or "huggingface" in Path(name).parts for name in names):
        raise FormalRuntimeError("formal backup contains cache, full weights, or credentials")
    digest = file_sha256(archive)
    checksum_path = archive.with_suffix(archive.suffix + ".sha256")
    checksum_path.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    if file_sha256(archive) != digest:
        raise FormalRuntimeError("formal backup SHA256 verification failed")
    return {"archive": str(archive), "sha256": digest, "entries": len(names)}
