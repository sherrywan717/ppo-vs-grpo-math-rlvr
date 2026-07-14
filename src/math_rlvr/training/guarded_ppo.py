"""Fail-closed PPO 0.5B single-update smoke contract and fake-test lifecycle."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from math_rlvr.config import resolve_ppo_smoke_contract, validate_training_config
from math_rlvr.dataset import MathProblem, load_manifest
from math_rlvr.rewards.result import RewardResult, RewardStatus
from math_rlvr.rewards.staged import STAGED_REWARD_VERSION, reward_policy_from_config
from math_rlvr.training.execution_contract import (
    ExpectedRunContract,
    expected_run_contract_for_config,
)
from math_rlvr.training.guarded_grpo import (
    AuthorizationError,
    BudgetExceededError,
    CheckpointSafetyError,
    assert_json_safe,
    primitive_failure_record,
)

PPO_SMOKE_CONFIG = Path("configs/smoke/ppo.yaml")
PPO_TIMEOUT_SECONDS = 1200
EXPECTED_PROMPT_SHA = "6842002e4591630a4105a4ca8fdf4cab91676b3902708ec0cb7f7b458864ecd7"
EXPECTED_REWARD_SHA = "90af0614676279eb8a47636acfdbeaded6d92237d3b16f027d79557057ca0e14"


@dataclass
class PPOBudgetGuard:
    max_completions: int
    max_tokens: int
    max_updates: int
    max_optimizer_steps: int
    max_global_steps: int
    max_epochs: int
    max_minibatches: int
    deadline: float
    clock: Callable[[], float] = time.monotonic
    completions: int = 0
    generated_tokens: int = 0
    updates: int = 0
    optimizer_steps: int = 0
    global_step: int = 0
    ppo_epochs: int = 0
    minibatches: int = 0
    rewards: list[dict[str, Any]] = field(default_factory=list)
    exceeded: bool = False
    exceeded_reason: str | None = None
    started_at: float | None = None
    _last_time: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.started_at is None:
            self.started_at = self.clock()
        self._last_time = self.started_at
        if not self.deadline:
            self.deadline = self.started_at + PPO_TIMEOUT_SECONDS

    def _fail(self, error, reason):
        self.exceeded, self.exceeded_reason = True, reason
        raise error(reason)

    def _time(self):
        self._last_time = self.clock()
        if self._last_time > self.deadline:
            self._fail(TimeoutError, "PPO smoke exceeded hard deadline")

    def record_generation(self, count: int, tokens: int):
        self._time()
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or count <= 0
            or not isinstance(tokens, int)
            or isinstance(tokens, bool)
            or tokens < 0
        ):
            self._fail(ValueError, "invalid PPO completion/token counters")
        if self.completions + count > self.max_completions:
            self._fail(BudgetExceededError, "PPO completion cap exceeded before update")
        if self.generated_tokens + tokens > self.max_tokens:
            self._fail(BudgetExceededError, "PPO generated-token cap exceeded before update")
        self.completions += count
        self.generated_tokens += tokens

    def record_reward(self, result: RewardResult, scalar: float, evidence: dict[str, Any]):
        self._time()
        if result.status == RewardStatus.INFRA_ERROR:
            self._fail(RuntimeError, f"infra_error: {result.detail}")
        if not math.isfinite(float(scalar)):
            self._fail(FloatingPointError, "non-finite PPO reward")
        if len(self.rewards) >= self.max_completions:
            self._fail(BudgetExceededError, "too many PPO reward records")
        if len(self.rewards) >= self.completions:
            self._fail(BudgetExceededError, "PPO reward arrived without a generated response")
        if evidence.get("canonical_status") != result.status.value or evidence.get(
            "scalar_reward"
        ) != float(scalar):
            self._fail(RuntimeError, "PPO reward evidence mismatch")
        assert_json_safe(evidence)
        self.rewards.append(
            {
                "status": result.status.value,
                "reward": float(scalar),
                "detail": str(result.detail),
                **evidence,
            }
        )

    def record_epoch_minibatch(self):
        self._time()
        self.ppo_epochs += 1
        self.minibatches += 1
        if self.ppo_epochs > self.max_epochs or self.minibatches > self.max_minibatches:
            self._fail(BudgetExceededError, "PPO epoch/minibatch cap exceeded")

    def record_optimizer_step(self):
        self._time()
        if self.completions != self.max_completions or len(self.rewards) != self.max_completions:
            self._fail(
                BudgetExceededError, "PPO optimizer step before all protected responses/rewards"
            )
        self.optimizer_steps += 1
        if self.optimizer_steps > self.max_optimizer_steps:
            self._fail(BudgetExceededError, "PPO optimizer-step cap exceeded")

    def record_update(self):
        self._time()
        if self.optimizer_steps != 1:
            self._fail(BudgetExceededError, "PPO update logged before one optimizer step")
        self.updates += 1
        if self.updates > self.max_updates:
            self._fail(BudgetExceededError, "PPO outer-update cap exceeded")

    def record_global_step(self, step: int):
        self._time()
        if not isinstance(step, int) or isinstance(step, bool) or step < self.global_step:
            self._fail(BudgetExceededError, "invalid or decreasing PPO global step")
        if step > self.max_global_steps:
            self._fail(BudgetExceededError, "PPO global-step cap exceeded")
        self.global_step = step

    def assert_success(self):
        actual = (
            self.completions,
            len(self.rewards),
            self.updates,
            self.optimizer_steps,
            self.global_step,
            self.ppo_epochs,
            self.minibatches,
        )
        expected = (
            self.max_completions,
            self.max_completions,
            self.max_updates,
            self.max_optimizer_steps,
            self.max_global_steps,
            self.max_epochs,
            self.max_minibatches,
        )
        if actual != expected or self.generated_tokens > self.max_tokens:
            self._fail(BudgetExceededError, f"incomplete PPO counters: {actual}")

    def snapshot(self):
        elapsed = max(0.0, (self._last_time or self.started_at) - self.started_at)
        payload = {
            "limits": {
                "completions": self.max_completions,
                "generated_tokens": self.max_tokens,
                "updates": self.max_updates,
                "optimizer_steps": self.max_optimizer_steps,
                "global_steps": self.max_global_steps,
                "ppo_epochs": self.max_epochs,
                "minibatches": self.max_minibatches,
            },
            "completions": self.completions,
            "generated_tokens": self.generated_tokens,
            "updates": self.updates,
            "optimizer_steps": self.optimizer_steps,
            "global_step": self.global_step,
            "ppo_epochs": self.ppo_epochs,
            "minibatches": self.minibatches,
            "reward_count": len(self.rewards),
            "elapsed_seconds": elapsed,
            "exceeded": self.exceeded,
            "exceeded_reason": self.exceeded_reason,
        }
        assert_json_safe(payload)
        return payload


class PPOBackend(Protocol):
    def run(self, problems: list[MathProblem], guard: PPOBudgetGuard) -> dict[str, Any]: ...


def validate_ppo_authorization(config: dict, config_path: Path) -> dict[str, Any]:
    if config_path.resolve() != PPO_SMOKE_CONFIG.resolve():
        raise AuthorizationError("only configs/smoke/ppo.yaml may enter PPO smoke")
    validate_training_config(config, "ppo")
    contract = resolve_ppo_smoke_contract(config)
    if (
        config["experiment"] != {"name": "smoke-ppo-qwen-0.5b", "algorithm": "ppo", "seed": 42}
        or config["model"]["name_or_path"] != "Qwen/Qwen2.5-0.5B-Instruct"
        or config["prompt_version"] != "prompt_v1_strict_concise"
        or config["prompt_sha256"] != EXPECTED_PROMPT_SHA
        or config["renderer_version"] != "math_rlvr.prompt.chat_template.v1"
        or config["reward_policy_version"] != STAGED_REWARD_VERSION
        or config["reward_policy_sha256"] != EXPECTED_REWARD_SHA
    ):
        raise AuthorizationError("frozen PPO smoke identity mismatch")
    reward_policy_from_config(config)
    return contract


def require_ppo_offline_environment() -> dict[str, str]:
    names = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    values = {name: os.environ.get(name, "") for name in names}
    if any(value != "1" for value in values.values()):
        raise AuthorizationError("PPO smoke requires both offline variables equal to 1")
    return values


def select_ppo_smoke_problems(config: dict) -> list[MathProblem]:
    problems = load_manifest(Path(config["data"]["manifest"]))[:4]
    if len(problems) != 4 or len({p.problem_id for p in problems}) != 4:
        raise AuthorizationError("PPO smoke requires four unique fixed records")
    if any(p.source != "countdown" or p.split != "train" for p in problems):
        raise AuthorizationError("PPO smoke records must be Countdown train")
    return problems


def ppo_execution_problems_and_episodes(
    config: dict[str, Any], contract: ExpectedRunContract
) -> tuple[list[MathProblem], list[dict[str, Any]]]:
    manifest_path = Path(
        config.get("data", {}).get("source_manifest") or config.get("data", {}).get("manifest")
    )
    unique = load_manifest(manifest_path)[: contract.expected_prompt_count]
    if tuple(problem.problem_id for problem in unique) != contract.problem_ids:
        raise AuthorizationError("PPO execution problems differ from protected profile")
    if any(problem.source != "countdown" or problem.split != "train" for problem in unique):
        raise AuthorizationError("PPO execution records must be Countdown train")
    if contract.profile == "ppo_matched_pilot":
        from math_rlvr.training.pilot import pilot_episode_records

        episodes = pilot_episode_records("ppo", config["experiment"]["seed"])
    else:
        from math_rlvr.training.pilot import (
            problem_contract_sha256,
            rendered_prompt_payload_sha256,
        )

        episodes = [
            {
                "episode_position": index,
                "problem_id": problem.problem_id,
                "generation_index": 0,
                "pair_key": f"{problem.problem_id}::generation:0",
                "problem_hash": problem_contract_sha256(problem),
                "rendered_prompt_hash": rendered_prompt_payload_sha256(
                    problem, contract.prompt_version
                ),
                "seed": config["experiment"]["seed"],
                "algorithm": "ppo",
            }
            for index, problem in enumerate(unique)
        ]
    problem_map = {problem.problem_id: problem for problem in unique}
    expanded = [problem_map[row["problem_id"]] for row in episodes]
    if tuple(row["pair_key"] for row in episodes) != contract.pair_keys:
        raise AuthorizationError("PPO episode order differs from protected profile")
    return expanded, episodes


_ALLOWED = {
    "policy_adapter/adapter_model.safetensors": "policy_adapter",
    "policy_adapter/adapter_config.json": "policy_adapter_config",
    "value_adapter/adapter_model.safetensors": "value_adapter",
    "value_adapter/adapter_config.json": "value_adapter_config",
    "value_head/value_head.safetensors": "value_head",
    "value_head/config.json": "value_head_config",
    "trainer_state.json": "trainer_state",
    "resume_manifest.json": "resume_manifest",
    "training_args.bin": "trainer_metadata",
}


def _safe_file(path: Path, root: Path) -> Path:
    if path.is_symlink():
        raise CheckpointSafetyError(f"checkpoint symlink forbidden: {path}")
    mode = path.lstat().st_mode
    resolved = path.resolve(strict=True)
    if not stat.S_ISREG(mode) or not resolved.is_relative_to(root):
        raise CheckpointSafetyError(f"checkpoint path escape: {path}")
    return resolved


def ppo_checkpoint_inventory(root: Path, full_weight_threshold: int = 100_000_000):
    if root.name != "checkpoint-1" or root.is_symlink():
        raise CheckpointSafetyError("PPO authoritative checkpoint must be checkpoint-1")
    resolved_root = root.resolve(strict=True)
    rows, adapter_hashes = [], []
    seen = set()
    allowed_directories = {Path(name).parent for name in _ALLOWED} - {Path(".")}
    for path in sorted(root.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            if path.relative_to(root) not in allowed_directories:
                raise CheckpointSafetyError(
                    f"unexpected PPO checkpoint directory: {path.relative_to(root)}"
                )
            continue
        resolved = _safe_file(path, resolved_root)
        relative = str(path.relative_to(root))
        if relative not in _ALLOWED:
            raise CheckpointSafetyError(f"unexpected PPO checkpoint file: {relative}")
        if relative in seen:
            raise CheckpointSafetyError(f"duplicate PPO checkpoint file: {relative}")
        seen.add(relative)
        size = resolved.stat().st_size
        if size >= full_weight_threshold:
            raise CheckpointSafetyError(f"full model-like PPO checkpoint file: {relative}")
        if relative == "training_args.bin" and size > 1024 * 1024:
            raise CheckpointSafetyError("training_args.bin exceeds 1 MiB")
        digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
        if relative.endswith("adapter_model.safetensors"):
            adapter_hashes.append(digest)
        rows.append(
            {
                "name": relative,
                "size_bytes": size,
                "sha256": digest,
                "classification": _ALLOWED[relative],
            }
        )
    required = set(_ALLOWED) - {"training_args.bin"}
    if not required <= seen:
        raise CheckpointSafetyError(f"missing PPO checkpoint roles: {sorted(required - seen)}")
    if len(adapter_hashes) != len(set(adapter_hashes)):
        raise CheckpointSafetyError("duplicate PPO adapter SHA256")
    return {
        "checkpoint_root": "checkpoint-1",
        "files": rows,
        "total_size_bytes": sum(row["size_bytes"] for row in rows),
        "duplicate_adapter_count": 0,
    }


def write_fake_ppo_checkpoint(root: Path, tensor_factory=None):
    """Create a tiny role-separated safetensors tree for CPU contract tests."""
    import torch
    from safetensors.torch import save_file

    root.mkdir(parents=True)
    tensor_factory = tensor_factory or (lambda value: torch.tensor([value], dtype=torch.float32))
    payloads = {
        "policy_adapter/adapter_model.safetensors": {"policy.lora": tensor_factory(1.0)},
        "value_adapter/adapter_model.safetensors": {"value.lora": tensor_factory(2.0)},
        "value_head/value_head.safetensors": {"score.weight": tensor_factory(3.0)},
    }
    for relative, tensors in payloads.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(path))
    (root / "policy_adapter/adapter_config.json").write_text('{"role":"policy"}\n')
    (root / "value_adapter/adapter_config.json").write_text('{"role":"value"}\n')
    (root / "value_head/config.json").write_text('{"shape":"scalar_per_token"}\n')
    (root / "trainer_state.json").write_text('{"global_step":1,"updates":1}\n')
    manifest = {
        "schema_version": 1,
        "checkpoint": "checkpoint-1",
        "required": sorted(set(_ALLOWED) - {"training_args.bin"}),
        "roles": {
            "policy_adapter": "policy LoRA",
            "value_adapter": "value LoRA",
            "value_head": "scalar score head",
        },
    }
    (root / "resume_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return root


def fake_reload_ppo_checkpoint(root: Path):
    from safetensors.torch import load_file

    inventory = ppo_checkpoint_inventory(root)
    loaded = {
        "policy_adapter": load_file(str(root / "policy_adapter/adapter_model.safetensors")),
        "value_adapter": load_file(str(root / "value_adapter/adapter_model.safetensors")),
        "value_head": load_file(str(root / "value_head/value_head.safetensors")),
    }
    manifest = json.loads((root / "resume_manifest.json").read_text())
    if manifest["checkpoint"] != "checkpoint-1" or any(not tensors for tensors in loaded.values()):
        raise CheckpointSafetyError("fake PPO checkpoint reload failed")
    return {"inventory": inventory, "roles": sorted(loaded), "manifest": manifest}


def validate_ppo_completion_evidence(records, guard, contract: ExpectedRunContract):
    if not isinstance(records, list) or len(records) != contract.expected_completions:
        raise RuntimeError(
            f"exactly {contract.expected_completions} PPO completion records required"
        )
    total = 0
    pair_keys = []
    for index, (row, reward) in enumerate(zip(records, guard.rewards, strict=True)):
        ids, mask = row.get("response_token_ids"), row.get("response_mask")
        if (
            row.get("completion_index") != index
            or not isinstance(ids, list)
            or not isinstance(mask, list)
        ):
            raise RuntimeError("PPO response evidence missing or out of order")
        if len(ids) != len(mask) or any(v not in (0, 1) for v in mask):
            raise RuntimeError("invalid PPO response mask")
        if row.get("exact_token_count") != sum(mask):
            raise RuntimeError("PPO token count not derived from response mask")
        text = row.get("decoded_completion")
        if (
            not isinstance(text, str)
            or text != row.get("reward_callback_text")
            or text != row.get("verifier_input")
        ):
            raise RuntimeError("PPO completion/reward/verifier text mismatch")
        if (
            row.get("scalar_reward") != reward["reward"]
            or row.get("canonical_status") != reward["status"]
        ):
            raise RuntimeError("PPO ordered reward mismatch")
        if not isinstance(row.get("prompt_token_ids"), list):
            raise RuntimeError("PPO prompt/response boundary evidence incomplete")
        if not isinstance(row.get("problem_id"), str) or not isinstance(
            row.get("prompt_hash"), str
        ):
            raise RuntimeError("PPO fixed-problem identity missing")
        pair_key = f"{row['problem_id']}::generation:{row.get('generation_index')}"
        if row.get("pair_key") != pair_key:
            raise RuntimeError("PPO comparison key is missing or inconsistent")
        pair_keys.append(pair_key)
        assert_json_safe(row)
        total += sum(mask)
    if total != guard.generated_tokens:
        raise RuntimeError("PPO persisted token total mismatch")
    if tuple(pair_keys) != contract.pair_keys:
        raise RuntimeError("PPO comparison keys differ from protected profile")
    return records


def run_guarded_ppo(config, backend: PPOBackend, lifecycle, monitor, clock=time.monotonic):
    contract = expected_run_contract_for_config(config, "ppo")
    started = clock()
    guard = PPOBudgetGuard(
        max_completions=contract.expected_completions,
        max_tokens=contract.generated_token_cap,
        max_updates=contract.expected_updates,
        max_optimizer_steps=contract.expected_optimizer_steps,
        max_global_steps=contract.expected_global_steps,
        max_epochs=contract.expected_ppo_epochs,
        max_minibatches=contract.expected_minibatches,
        deadline=started + config["budget"]["max_wall_time_seconds"],
        clock=clock,
        started_at=started,
    )
    status, reason, result, inventory = "failure", None, {}, None
    monitor_started = False
    try:
        if contract.profile == "ppo_matched_pilot":
            from math_rlvr.training.pilot import validate_pilot_execution_authorization

            validate_pilot_execution_authorization(config, Path(contract.config_path), "ppo")
        else:
            validate_ppo_authorization(config, PPO_SMOKE_CONFIG)
        problems, episode_records = ppo_execution_problems_and_episodes(config, contract)
        lifecycle.start(config, problems)
        lifecycle.persist("expected_run_contract.json", contract.to_dict())
        lifecycle.persist("ppo_episode_order.json", episode_records)
        monitor.start()
        monitor_started = True
        result = backend.run(problems, guard)
        if result.get("episode_records") != episode_records:
            raise RuntimeError("PPO backend episode records differ from protected order")
        if contract.profile == "ppo_matched_pilot":
            loader_contract = result.get("loader_contract")
            expected_loader = {
                "replacement_sampler_type": "SequentialSampler",
                "batch_size": contract.expected_completions,
                "drop_last": True,
                "num_workers": 0,
                "world_size": 1,
                "prepared_first_batch_pair_keys": list(contract.pair_keys),
            }
            if not isinstance(loader_contract, dict) or any(
                loader_contract.get(key) != value for key, value in expected_loader.items()
            ):
                raise RuntimeError("PPO pilot prepared-loader evidence mismatch")
        guard.assert_success()
        completions = validate_ppo_completion_evidence(result.get("completions"), guard, contract)
        metrics = dict(result.get("metrics", {}))
        metrics["evidence_counters"] = {
            "completions": guard.completions,
            "generated_tokens": guard.generated_tokens,
            "updates": guard.updates,
            "optimizer_steps": guard.optimizer_steps,
            "global_step": guard.global_step,
        }
        result["metrics"] = metrics
        lifecycle.persist_jsonl("completions.jsonl", completions)
        lifecycle.persist("trainer_metrics.json", metrics)
        lifecycle.persist("trainer_log_history.json", result.get("trainer_log_history", []))
        inventory = ppo_checkpoint_inventory(Path(result["checkpoint_dir"]))
        lifecycle.persist("checkpoint_inventory.json", inventory)
        status = "success"
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        try:
            lifecycle.persist(
                "failure_report.json",
                {
                    **primitive_failure_record(reason, "guarded_ppo_execution"),
                    "counters": guard.snapshot(),
                },
            )
        except Exception:
            pass
    finally:
        if monitor_started:
            try:
                monitor.stop()
            except Exception as exc:
                status = "failure"
                reason = f"monitor_stop: {type(exc).__name__}: {exc}"
        resolved_contract = (
            config.get("resolved_pilot_contract")
            if contract.profile == "ppo_matched_pilot"
            else resolve_ppo_smoke_contract(config)
        )
        summary = {
            "status": status,
            "reason": reason,
            "counters": guard.snapshot(),
            "completion_evidence_count": len(result.get("completions", [])),
            "metrics": result.get("metrics", {}),
            "checkpoint_inventory": inventory,
            "resolved_ppo_contract": resolved_contract,
            "expected_run_contract": contract.to_dict(),
            "model_roles": result.get("model_roles"),
            "loader_contract": result.get("loader_contract"),
            "prompt_version": config["prompt_version"],
            "prompt_sha256": config["prompt_sha256"],
            "renderer_version": config["renderer_version"],
            "reward_policy_version": config["reward_policy_version"],
            "reward_policy_sha256": config["reward_policy_sha256"],
            "reward_component_weights": config["reward_component_weights"],
            "smoke_disclaimer": config.get("reporting", {}).get(
                "disclaimer", "Smoke test - not a benchmark result"
            ),
        }
        try:
            lifecycle.persist("summary.json", summary)
            lifecycle.finalize(summary)
            lifecycle.backup_and_verify(failure=status != "success")
            summary["backed_up"] = True
            lifecycle.persist("summary.json", summary)
            lifecycle.finalize(summary)
        except Exception as exc:
            status = "failure"
            fallback = primitive_failure_record(
                f"artifact_or_backup: {type(exc).__name__}: {exc}",
                "artifact_finalization",
            )
            summary = {
                **fallback,
                "backed_up": False,
                "counters": guard.snapshot(),
                "resolved_ppo_contract": resolved_contract,
                "expected_run_contract": contract.to_dict(),
            }
            try:
                lifecycle.persist("failure_report.json", summary)
            except Exception:
                pass
    return {**summary, "status": status}
