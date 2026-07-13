"""Fail-closed, artifact-first GRPO single-update smoke runner."""

from __future__ import annotations

import hashlib
import math
import stat
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from math_rlvr.config import resolve_grpo_smoke_budget, validate_training_config
from math_rlvr.dataset import MathProblem, load_manifest
from math_rlvr.rewards.result import DEFAULT_REWARD_POLICY, RewardResult, RewardStatus
from math_rlvr.training.model_source import (
    DEFAULT_CACHE_ROOT,
    PINNED_REPO_ID,
    PINNED_REVISION,
    ValidatedModelSource,
)

MODEL = PINNED_REPO_ID
REVISION = PINNED_REVISION
SMOKE_CONFIG = Path("configs/smoke/grpo.yaml")
TIMEOUT_SECONDS = 900


class AuthorizationError(RuntimeError):
    pass


class BudgetExceededError(RuntimeError):
    pass


class CheckpointSafetyError(RuntimeError):
    pass


@dataclass
class BudgetGuard:
    max_completions: int
    max_tokens: int
    max_optimizer_steps: int
    max_global_step: int
    max_microsteps: int
    deadline: float
    clock: Callable[[], float] = time.monotonic
    completions: int = 0
    generated_tokens: int = 0
    optimizer_steps: int = 0
    global_step: int = 0
    microsteps: int = 0
    rewards: list[dict[str, Any]] = field(default_factory=list)
    started_at: float | None = None
    exceeded: bool = False
    exceeded_reason: str | None = None
    _last_time: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.started_at is None:
            self.started_at = self.deadline - TIMEOUT_SECONDS
        self._last_time = self.started_at

    def _raise(self, error_type, reason):
        self.exceeded = True
        self.exceeded_reason = reason
        raise error_type(reason)

    def _time(self):
        self._last_time = self.clock()
        if self._last_time > self.deadline:
            self._raise(TimeoutError, "GRPO smoke exceeded 15-minute deadline")

    def snapshot(self) -> dict[str, Any]:
        payload = {
            "limits": {
                "completions": self.max_completions,
                "generated_tokens": self.max_tokens,
                "microsteps": self.max_microsteps,
                "optimizer_steps": self.max_optimizer_steps,
                "global_step": self.max_global_step,
            },
            "completions": self.completions,
            "generated_tokens": self.generated_tokens,
            "microsteps": self.microsteps,
            "optimizer_steps": self.optimizer_steps,
            "global_step": self.global_step,
            "start_timestamp": self.started_at,
            "elapsed_seconds": max(
                0.0, (self._last_time or self.started_at) - self.started_at
            ),
            "exceeded": self.exceeded,
            "exceeded_reason": self.exceeded_reason,
            "rewards": [dict(row) for row in self.rewards],
        }
        assert_json_safe(payload)
        return payload

    to_json_dict = snapshot

    def record_generation(self, completions: int, tokens: int):
        self._time()
        if self.completions + completions > self.max_completions:
            self._raise(BudgetExceededError, "completion cap exceeded before update")
        if self.generated_tokens + tokens > self.max_tokens:
            self._raise(BudgetExceededError, "generated-token cap exceeded before update")
        self.completions += completions
        self.generated_tokens += tokens

    def record_reward(self, result: RewardResult, scalar: float):
        self._time()
        if result.status == RewardStatus.INFRA_ERROR:
            self._raise(RuntimeError, f"infra_error: {result.detail}")
        if not math.isfinite(scalar):
            self._raise(FloatingPointError, "non-finite reward")
        if len(self.rewards) >= self.max_completions:
            self._raise(BudgetExceededError, "ninth reward/completion refused before update")
        self.rewards.append(
            {"status": result.status.value, "detail": str(result.detail), "reward": scalar}
        )

    def record_microstep(self):
        self._time()
        self.microsteps += 1
        if self.microsteps > self.max_microsteps:
            self._raise(BudgetExceededError, "gradient microstep cap exceeded")

    def record_optimizer_step(self):
        self._time()
        if self.completions != self.max_completions or len(self.rewards) != self.max_completions:
            self._raise(
                BudgetExceededError,
                "optimizer update attempted before exactly 8 completions/rewards",
            )
        if self.generated_tokens > self.max_tokens or self.microsteps != self.max_microsteps:
            self._raise(BudgetExceededError, "optimizer preconditions failed")
        self.optimizer_steps += 1
        if self.optimizer_steps > self.max_optimizer_steps:
            self._raise(BudgetExceededError, "second optimizer step refused")

    def record_global_step(self, step: int):
        self._time()
        if step > self.max_global_step:
            self._raise(BudgetExceededError, "global_step cap exceeded")
        self.global_step = step

    def assert_success(self):
        expected = (self.max_completions, self.max_completions, 1, 1, self.max_microsteps)
        actual = (
            self.completions,
            len(self.rewards),
            self.optimizer_steps,
            self.global_step,
            self.microsteps,
        )
        if actual != expected or self.generated_tokens > self.max_tokens:
            self._raise(BudgetExceededError, f"incomplete smoke counters: {actual}")


def assert_json_safe(value: Any, path: str = "$") -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(f"non-finite float at {path}")
        return
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            assert_json_safe(item, f"{path}[{index}]")
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for key, item in value.items():
            assert_json_safe(item, f"{path}.{key}")
        return
    raise TypeError(f"non-JSON-safe value at {path}: {type(value).__name__}")


def primitive_failure_record(reason: str, phase: str) -> dict[str, str]:
    payload = {"status": "failure", "phase": str(phase), "reason": str(reason)}
    assert_json_safe(payload)
    return payload


class Backend(Protocol):
    def run(self, problems: list[MathProblem], guard: BudgetGuard, reward: Callable) -> dict: ...


def select_smoke_problems(config: dict) -> list[MathProblem]:
    problems = load_manifest(Path(config["data"]["manifest"]))[:2]
    if len(problems) != 2 or len({p.problem_id for p in problems}) != 2:
        raise AuthorizationError("smoke manifest must provide two unique problems")
    if any(p.split != "train" or p.source != "countdown" for p in problems):
        raise AuthorizationError("smoke problems must be countdown train records")
    return problems


def validate_smoke_authorization(config: dict, config_path: Path) -> None:
    if config_path.resolve() != SMOKE_CONFIG.resolve():
        raise AuthorizationError("only the frozen GRPO smoke config is authorized")
    validate_training_config(config, "grpo")
    budget = resolve_grpo_smoke_budget(config)
    model = config["model"]
    training = config["training"]
    generation = config["generation"]
    contract = (
        model["name_or_path"],
        model.get("revision"),
        model.get("local_files_only"),
        config["experiment"]["seed"],
        budget["unique_prompts"],
        training["per_device_train_batch_size"],
        training["gradient_accumulation_steps"],
        generation["num_generations"],
        generation["generation_batch_size"],
        budget["steps_per_generation"],
        training["num_iterations"],
        training["max_steps"],
        generation["max_completion_length"],
        budget["total_completions"],
        budget["total_generated_tokens"],
    )
    expected = (MODEL, REVISION, True, 42, 2, 2, 4, 4, 8, 4, 1, 1, 128, 8, 1024)
    if contract != expected:
        raise AuthorizationError("resolved GRPO smoke contract mismatch")


def require_clean_git() -> dict[str, str]:
    branch = subprocess.run(
        ["git", "branch", "--show-current"], capture_output=True, text=True, check=True
    ).stdout.strip()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain=v1"], capture_output=True, text=True, check=True
    ).stdout
    if branch != "pivot/math-rlvr" or dirty:
        raise AuthorizationError("GRPO smoke requires clean pivot/math-rlvr worktree")
    return {"branch": branch, "commit": commit}


def require_local_snapshot(
    *, cache_root: Path = DEFAULT_CACHE_ROOT, snapshot_resolver=None
) -> ValidatedModelSource:
    return ValidatedModelSource.resolve(
        MODEL, REVISION, cache_root=cache_root, snapshot_resolver=snapshot_resolver
    )


def reward_recorder(guard: BudgetGuard, verifier: Callable[[str], RewardResult]):
    def reward(completion: str) -> float:
        result = verifier(completion)
        scalar = DEFAULT_REWARD_POLICY.to_scalar(result)
        guard.record_reward(result, scalar)
        return scalar

    return reward


TRAINING_ARGS_MAX_BYTES = 1024 * 1024


def _regular_file_within(path: Path, root: Path) -> Path:
    if path.is_symlink():
        raise CheckpointSafetyError(f"checkpoint symlink forbidden: {path.name}")
    try:
        mode = path.lstat().st_mode
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, RuntimeError) as exc:
        raise CheckpointSafetyError(f"invalid checkpoint file: {path.name}") from exc
    if not stat.S_ISREG(mode) or not resolved.is_relative_to(root):
        raise CheckpointSafetyError(f"checkpoint path escape: {path.name}")
    return resolved


def authoritative_checkpoint(run_dir: Path, global_step: int) -> Path:
    """Return the sole Trainer-created checkpoint and reject duplicates/escapes."""
    if global_step != 1:
        raise CheckpointSafetyError("authoritative checkpoint requires global_step 1")
    if run_dir.is_symlink():
        raise CheckpointSafetyError("run directory symlink is forbidden")
    run_root = run_dir.resolve(strict=True)
    expected = run_dir / f"checkpoint-{global_step}"

    adapter_paths = sorted(run_dir.rglob("adapter_model.safetensors"))
    adapter_digests = []
    for path in adapter_paths:
        resolved = _regular_file_within(path, run_root)
        adapter_digests.append(hashlib.sha256(resolved.read_bytes()).hexdigest())
    if len(adapter_paths) != 1:
        if len(adapter_digests) != len(set(adapter_digests)):
            raise CheckpointSafetyError("duplicate adapter SHA256 detected")
        raise CheckpointSafetyError("exactly one adapter_model.safetensors is required")
    if len(list(run_dir.rglob("adapter_config.json"))) != 1:
        raise CheckpointSafetyError("exactly one adapter_config.json is required")

    candidates = [
        path
        for path in list(run_dir.glob("checkpoint-*"))
        + list((run_dir / "checkpoints").glob("checkpoint-*"))
        if path.is_dir() or path.is_symlink()
    ]
    if len(candidates) != 1 or candidates[0] != expected:
        raise CheckpointSafetyError("exactly one authoritative Trainer checkpoint is required")
    if expected.is_symlink():
        raise CheckpointSafetyError("authoritative checkpoint symlink is forbidden")
    resolved = expected.resolve(strict=True)
    if resolved.parent != run_root:
        raise CheckpointSafetyError("authoritative checkpoint escaped run directory")
    return resolved


def checkpoint_inventory(
    root: Path,
    *,
    run_dir: Path | None = None,
    full_weight_threshold: int = 500_000_000,
    training_args_max_bytes: int = TRAINING_ARGS_MAX_BYTES,
) -> dict[str, Any]:
    """Hash a canonical checkpoint without deserializing any checkpoint content."""
    if root.is_symlink():
        raise CheckpointSafetyError("checkpoint root symlink is forbidden")
    root_resolved = root.resolve(strict=True)
    if run_dir is not None:
        run_resolved = run_dir.resolve(strict=True)
        if root_resolved.parent != run_resolved or root_resolved.name != "checkpoint-1":
            raise CheckpointSafetyError("checkpoint root is outside the expected run directory")

    allowed_adapter_weights = {"adapter_model.safetensors", "adapter_model.bin"}
    inventory = []
    for path in sorted(root.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        resolved = _regular_file_within(path, root_resolved)
        size = resolved.stat().st_size
        name = path.name
        if size >= full_weight_threshold:
            raise CheckpointSafetyError(f"full-size weight-like file: {name}")
        if name == "training_args.bin":
            if resolved.parent != root_resolved:
                raise CheckpointSafetyError("training_args.bin must be at checkpoint root")
            if size > training_args_max_bytes:
                raise CheckpointSafetyError("training_args.bin exceeds 1 MiB safety limit")
            classification = "trainer_metadata"
        elif path.suffix == ".bin" and name not in allowed_adapter_weights:
            raise CheckpointSafetyError(f"non-adapter weight file: {name}")
        elif path.suffix == ".safetensors" and name not in allowed_adapter_weights:
            raise CheckpointSafetyError(f"non-adapter weight file: {name}")
        elif name in allowed_adapter_weights:
            classification = "lora_adapter"
        elif name == "adapter_config.json":
            classification = "adapter_config"
        elif name == "trainer_state.json":
            classification = "trainer_state"
        else:
            classification = "tokenizer_or_metadata"
        inventory.append(
            {
                "name": str(path.relative_to(root)),
                "size_bytes": size,
                "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
                "classification": classification,
            }
        )
    return {
        "checkpoint_root": root_resolved.name,
        "files": inventory,
        "total_size_bytes": sum(item["size_bytes"] for item in inventory),
        "duplicate_checkpoint_count": 0,
    }


def validate_completion_evidence(
    records: Any, guard: BudgetGuard
) -> list[dict[str, Any]]:
    if not isinstance(records, list) or len(records) != guard.max_completions:
        raise RuntimeError("exactly 8 persisted completion evidence records are required")
    per_problem: dict[str, list[int]] = {}
    total_tokens = 0
    for index, (record, reward) in enumerate(zip(records, guard.rewards, strict=True)):
        if record.get("completion_index") != index:
            raise RuntimeError("completion evidence order mismatch")
        ids = record.get("completion_ids")
        mask = record.get("completion_mask")
        if not isinstance(ids, list) or not isinstance(mask, list) or len(ids) != len(mask):
            raise RuntimeError("completion IDs/mask evidence missing")
        if any(not isinstance(value, int) for value in ids) or any(
            value not in (0, 1) for value in mask
        ):
            raise RuntimeError("invalid completion IDs/mask evidence")
        exact_count = sum(mask)
        if record.get("exact_token_count") != exact_count:
            raise RuntimeError("completion token count is not derived from mask")
        text = record.get("decoded_completion")
        if (
            not isinstance(text, str)
            or record.get("raw_completion") != text
            or record.get("verifier_input") != text
        ):
            raise RuntimeError("saved completion differs from verifier input")
        if (
            record.get("reward_status") != reward["status"]
            or record.get("scalar_reward") != reward["reward"]
            or record.get("verifier_detail") != reward["detail"]
        ):
            raise RuntimeError("completion/reward evidence order mismatch")
        problem_id = record.get("problem_id")
        prompt_hash = record.get("prompt_hash")
        generation_index = record.get("generation_index")
        if not isinstance(problem_id, str) or not isinstance(prompt_hash, str):
            raise RuntimeError("completion problem association missing")
        per_problem.setdefault(problem_id, []).append(generation_index)
        total_tokens += exact_count
        assert_json_safe(record)
    if total_tokens != guard.generated_tokens:
        raise RuntimeError("persisted completion token total mismatch")
    if sorted(sorted(values) for values in per_problem.values()) != [[0, 1, 2, 3]] * 2:
        raise RuntimeError("completion generation indexes mismatch")
    return records


def run_guarded(
    config: dict, backend: Backend, verifier: Callable, lifecycle, monitor, clock=time.monotonic
) -> dict:
    """Dependency-injected lifecycle used by fake tests and the real entry point."""
    validate_smoke_authorization(config, SMOKE_CONFIG)
    problems = select_smoke_problems(config)
    started_at = clock()
    guard = BudgetGuard(
        8, 1024, 1, 1, 4, started_at + TIMEOUT_SECONDS, clock, started_at=started_at
    )
    status = "failure"
    reason = None
    result = {}
    inventory = None
    lifecycle.start(config, problems)
    monitor.start()
    try:
        result = backend.run(problems, guard, reward_recorder(guard, verifier))
        guard.assert_success()
        lifecycle.persist("trainer_metrics.json", result.get("metrics", {}))
        lifecycle.persist(
            "trainer_log_history.json", result.get("trainer_log_history", [])
        )
        completions = validate_completion_evidence(result.get("completions"), guard)
        lifecycle.persist_jsonl("completions.jsonl", completions)
        inventory = checkpoint_inventory(
            Path(result["checkpoint_dir"]),
            run_dir=Path(result["run_dir"]) if result.get("run_dir") else None,
        )
        lifecycle.persist("checkpoint_inventory.json", inventory)
        status = "success"
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        failure = {
            **primitive_failure_record(reason, "guarded_execution"),
            "counters": guard.snapshot(),
        }
        assert_json_safe(failure)
        lifecycle.persist("failure_report.json", failure)
    finally:
        monitor.stop()
        counters = guard.snapshot()
        runtime_evidence = (
            lifecycle.runtime_summary() if hasattr(lifecycle, "runtime_summary") else {}
        )
        summary = {
            "status": status,
            "reason": reason,
            "counters": counters,
            "completion_evidence_count": len(result.get("completions", [])),
            "metrics": result.get("metrics", {}),
            "trainer_log_history": result.get("trainer_log_history", []),
            "checkpoint_inventory": inventory if status == "success" else None,
            "prompt_version": config.get("prompt_version"),
            "prompt_sha256": config.get("prompt_sha256"),
            "renderer_version": config.get("renderer_version"),
            "duplicate_checkpoint_count": (
                inventory["duplicate_checkpoint_count"]
                if status == "success" and inventory is not None
                else None
            ),
            **runtime_evidence,
        }
        try:
            assert_json_safe(summary)
            lifecycle.persist("summary.json", summary)
            lifecycle.finalize(summary)
            if status == "success":
                lifecycle.backup_and_verify()
                summary["backed_up"] = True
                lifecycle.persist("summary.json", summary)
                lifecycle.finalize(summary)
        except Exception as exc:
            status = "failure"
            fallback = primitive_failure_record(
                f"artifact_or_backup: {type(exc).__name__}: {exc}",
                "artifact_finalization",
            )
            summary = {**fallback, "backed_up": False, "counters": counters}
            try:
                assert_json_safe(summary)
                lifecycle.persist("summary.json", summary)
                lifecycle.persist("failure_report.json", fallback)
            except Exception:
                pass
    return {**summary, "status": status}


def real_backend(config: dict):
    """Delayed real construction path. Never call this from CPU gates."""
    source = require_local_snapshot()
    from math_rlvr.training.builders import build_grpo_trainer, load_policy_and_tokenizer
    from math_rlvr.training.trl_compat import guarded_trainer_class, optimizer_guard_callback

    return (
        source,
        build_grpo_trainer,
        load_policy_and_tokenizer,
        guarded_trainer_class,
        optimizer_guard_callback,
    )
