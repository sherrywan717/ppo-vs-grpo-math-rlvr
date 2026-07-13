"""Fail-closed, artifact-first GRPO single-update smoke runner."""

from __future__ import annotations

import math
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


def checkpoint_inventory(root: Path, full_weight_threshold=500_000_000) -> list[dict]:
    allowed_weights = {"adapter_model.safetensors", "adapter_model.bin"}
    inventory = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        size = path.stat().st_size
        name = path.name
        if size >= full_weight_threshold:
            raise CheckpointSafetyError(f"full-size weight-like file: {name}")
        if path.suffix in {".safetensors", ".bin"} and name not in allowed_weights:
            raise CheckpointSafetyError(f"non-adapter weight file: {name}")
        inventory.append({"path": str(path.relative_to(root)), "size": size})
    return inventory


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
    lifecycle.start(config, problems)
    monitor.start()
    try:
        result = backend.run(problems, guard, reward_recorder(guard, verifier))
        guard.assert_success()
        inventory = checkpoint_inventory(Path(result["checkpoint_dir"]))
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
        summary = {
            "status": status,
            "reason": reason,
            "counters": counters,
            "metrics": result.get("metrics", {}),
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
