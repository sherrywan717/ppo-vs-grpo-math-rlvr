"""CPU-safe frozen contracts for the model-bound GRPO-v2 seed-42 run."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from math_rlvr.artifacts.manager import atomic_text
from math_rlvr.contracts import formal_parser_verifier_metadata
from math_rlvr.prompt import PROMPT_RENDERER_VERSION, PROMPT_V2_FORMAL_MATH, PROMPT_V2_SHA256
from math_rlvr.rewards.formal import FORMAL_REWARD_SHA256, FORMAL_REWARD_VERSION
from math_rlvr.training.formal_runtime import (
    FormalOnlineGuard,
    FormalRuntimeError,
    ValidatedFormalResume,
    formal_checkpoint_inventory,
)
from math_rlvr.training.model_source import FORMAL_REPO_ID, FORMAL_REVISION
from math_rlvr.training.warmstart_runtime import (
    file_sha256,
    grpo_adapter_handoff,
    validate_checkpoint,
)

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = Path("configs/grpo_v2/grpo_v2_seed42.json")
REGISTRY_PATH = ROOT / "configs/grpo_v2/runtime_registry.json"
DATA_REGISTRY_PATH = ROOT / "configs/grpo_v2/data_registry.json"
RUN_ROOT = Path("/root/autodl-tmp/runs/math_rlvr")
EXPECTED_BRANCH = "improve/grpo-v2"
EXPECTED_WARMSTART_RUN_ID = "warmstart_grpo_v2_seed42_20260722T051218Z"
EXPECTED_CHECKPOINT_ARTIFACT_SHA256 = (
    "507749d393f38690915a76228b4c53a8b5c8927d40aada9f2768a90334d892f0"
)
WARMSTART_CHECKPOINT = RUN_ROOT / EXPECTED_WARMSTART_RUN_ID / "checkpoint-16"
EXPECTED_ADAPTER_SHA256 = "44066dd13d8cfa4f5c40f10cad705eea617c37ce2e2f85ff5407751fb5a972b9"
GRPO_V2_PROFILE = "grpo_v2_1p5b"
CHECKPOINT_STEPS = (32, 64, 96, 128)


class GRPOV2ContractError(FormalRuntimeError):
    """A frozen GRPO-v2 identity, budget, evidence, or resume invariant failed."""


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _registry() -> tuple[dict[str, Any], dict[str, Any]]:
    registry = json.loads(REGISTRY_PATH.read_text())
    body = dict(registry)
    claim = body.pop("registry_sha256", None)
    if claim != canonical_sha256(body):
        raise GRPOV2ContractError("GRPO-v2 runtime registry SHA mismatch")
    identity = registry.get("grpo_v2")
    if not isinstance(identity, dict):
        raise GRPOV2ContractError("GRPO-v2 runtime identity is not registered")
    return registry, identity


def _canonical_file(path: Path, *, under_root: bool = True) -> Path:
    target = ROOT / path if under_root else path
    if target.is_symlink() or target.resolve(strict=True) != target:
        raise GRPOV2ContractError(f"non-canonical frozen path: {path}")
    return target


@dataclass(frozen=True)
class GRPOV2Contract:
    algorithm: str
    seed: int
    config_path: str
    config_sha256: str
    registry_sha256: str
    data_registry_sha256: str
    manifest_sha256: str
    trusted_manifest_sha256: str
    curriculum_sha256: str
    curriculum_identity_sha256: str
    dev_manifest_sha256: str
    warmstart_checkpoint_sha256: str
    warmstart_adapter_sha256: str
    prompt_sha256: str
    reward_sha256: str
    parser_sha256: str
    verifier_sha256: str
    pair_keys: tuple[str, ...]
    problem_ids: tuple[str, ...]
    profile: str = GRPO_V2_PROFILE
    updates: int = 128
    optimizer_steps: int = 128
    global_steps: int = 128
    expected_microsteps: int = 512
    expected_completions: int = 2048
    completions_per_update: int = 16
    prompts_per_update: int = 4
    max_prompt_length: int = 928
    max_completion_length: int = 256
    max_sequence_length: int = 1184
    token_cap: int = 524_288
    checkpoint_steps: tuple[int, ...] = CHECKPOINT_STEPS
    validation_steps: tuple[int, ...] = CHECKPOINT_STEPS
    dev_problems_per_step: int = 128

    @property
    def active_suite_sha256(self) -> str:
        return self.registry_sha256

    @property
    def schedule_sha256(self) -> str:
        return self.curriculum_identity_sha256

    def pair_keys_for_update(self, update: int) -> tuple[str, ...]:
        start = (update - 1) * self.completions_per_update
        return self.pair_keys[start : start + self.completions_per_update]

    def checkpoint_identity(self, *, run_id: str, step: int) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "seed": self.seed,
            "run_id": run_id,
            "checkpoint_step": step,
            "config_sha256": self.config_sha256,
            "active_suite_sha256": self.registry_sha256,
            "manifest_sha256": self.manifest_sha256,
            "schedule_sha256": self.curriculum_identity_sha256,
            "model_repo": FORMAL_REPO_ID,
            "model_revision": FORMAL_REVISION,
            "prompt_sha256": self.prompt_sha256,
            "reward_sha256": self.reward_sha256,
            "parser_sha256": self.parser_sha256,
            "verifier_sha256": self.verifier_sha256,
            "warmstart_checkpoint_sha256": self.warmstart_checkpoint_sha256,
            "warmstart_adapter_sha256": self.warmstart_adapter_sha256,
        }

    def as_dict(self) -> dict[str, Any]:
        result = dict(vars(self))
        result["pair_keys"] = list(self.pair_keys)
        result["problem_ids"] = list(self.problem_ids)
        result["checkpoint_steps"] = list(self.checkpoint_steps)
        result["validation_steps"] = list(self.validation_steps)
        return result


def load_contract(
    config_path: Path = CONFIG_PATH,
) -> tuple[dict[str, Any], dict[str, Any], GRPOV2Contract]:
    if config_path != CONFIG_PATH or config_path.is_absolute() or ".." in config_path.parts:
        raise GRPOV2ContractError("GRPO-v2 requires the exact frozen config path")
    target = _canonical_file(config_path)
    registry, identity = _registry()
    if file_sha256(target) != identity.get("config_sha256"):
        raise GRPOV2ContractError("GRPO-v2 config SHA mismatch")
    config = json.loads(target.read_text())
    data_registry = json.loads(DATA_REGISTRY_PATH.read_text())
    data_body = dict(data_registry)
    data_claim = data_body.pop("registry_sha256", None)
    if (
        data_claim != canonical_sha256(data_body)
        or data_claim != identity.get("data_registry_sha256")
        or file_sha256(DATA_REGISTRY_PATH) != identity.get("data_registry_raw_sha256")
    ):
        raise GRPOV2ContractError("GRPO-v2 data registry identity mismatch")
    expected_top = {"schema_version": 1, "experiment": "grpo_v2_seed42", "seed": 42}
    if any(config.get(key) != value for key, value in expected_top.items()):
        raise GRPOV2ContractError("GRPO-v2 top-level identity mismatch")
    if config.get("model") != {
        "repo": FORMAL_REPO_ID,
        "revision": FORMAL_REVISION,
        "local_files_only": True,
        "dtype": "bfloat16",
    }:
        raise GRPOV2ContractError("GRPO-v2 model identity mismatch")
    if config.get("lora") != {
        "rank": 16,
        "alpha": 32,
        "dropout": 0,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    }:
        raise GRPOV2ContractError("GRPO-v2 LoRA identity mismatch")
    if config.get("generation") != {
        "generation_batch_size": 16,
        "num_generations": 4,
        "temperature": 0.8,
        "top_p": 0.95,
    } or config.get("prompt") != {
        "version": PROMPT_V2_FORMAL_MATH,
        "sha256": PROMPT_V2_SHA256,
        "max_prompt_length": 928,
        "max_completion_length": 256,
        "max_sequence_length": 1184,
    }:
        raise GRPOV2ContractError("GRPO-v2 prompt/sampling identity mismatch")
    expected_training = {
        "prompts_per_update": 4,
        "completions_per_prompt": 4,
        "completions_per_update": 16,
        "updates": 128,
        "training_completions": 2048,
        "per_device_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "expected_microsteps": 512,
        "optimizer_steps": 128,
        "global_steps": 128,
        "learning_rate": 1e-5,
        "trl_version": "0.24.0",
    }
    if config.get("training") != expected_training:
        raise GRPOV2ContractError("GRPO-v2 training budget drift")
    if config.get("checkpoint_steps") != list(CHECKPOINT_STEPS):
        raise GRPOV2ContractError("GRPO-v2 checkpoint cadence drift")
    if config.get("dev_validation") != {
        "candidates_per_problem": 1,
        "problems_per_step": 128,
        "steps": list(CHECKPOINT_STEPS),
        "training_budget_scope": False,
    }:
        raise GRPOV2ContractError("GRPO-v2 dev cadence drift")
    if config.get("budget", {}).get("max_generated_tokens") != 524_288:
        raise GRPOV2ContractError("GRPO-v2 token budget drift")
    contracts = formal_parser_verifier_metadata()
    expected_ids = {
        "parser_sha256": contracts["parser_contract"]["contract_sha256"],
        "verifier_sha256": contracts["verifier_contract"]["contract_sha256"],
    }
    if any(config.get(key) != value for key, value in expected_ids.items()):
        raise GRPOV2ContractError("GRPO-v2 parser/verifier identity mismatch")
    if config.get("reward") != {"policy": FORMAL_REWARD_VERSION, "sha256": FORMAL_REWARD_SHA256}:
        raise GRPOV2ContractError("GRPO-v2 reward identity mismatch")

    train_path = _canonical_file(Path(config["data"]["manifest"]))
    curriculum_path = _canonical_file(Path(config["data"]["curriculum"]))
    dev_path = _canonical_file(Path(config["data"]["dev_manifest"]))
    trusted_path = Path(identity["trusted_train_manifest_path"])
    if trusted_path.is_symlink() or trusted_path.resolve(strict=True) != trusted_path:
        raise GRPOV2ContractError("trusted train manifest path is not canonical")
    current = {
        "manifest_sha256": file_sha256(train_path),
        "trusted_manifest_sha256": file_sha256(trusted_path),
        "curriculum_sha256": file_sha256(curriculum_path),
        "dev_manifest_sha256": file_sha256(dev_path),
    }
    if any(identity.get(key) != value for key, value in current.items()):
        raise GRPOV2ContractError("GRPO-v2 data/curriculum SHA mismatch")
    curriculum = json.loads(curriculum_path.read_text())
    rows = curriculum.get("positions", [])
    if len(rows) != 512:
        raise GRPOV2ContractError("GRPO-v2 curriculum must contain 512 rows")
    positions = [row.get("position") for row in rows]
    updates = [row.get("update") for row in rows]
    slots = [row.get("slot") for row in rows]
    problem_ids = tuple(str(row.get("problem_id")) for row in rows)
    if (
        positions != list(range(1, 513))
        or updates != [index // 4 + 1 for index in range(512)]
        or slots != [index % 4 for index in range(512)]
        or len(set(problem_ids)) != 512
    ):
        raise GRPOV2ContractError("GRPO-v2 curriculum order/uniqueness drift")
    train_rows = _read_jsonl(train_path)
    if {row["problem_id"] for row in train_rows} != set(problem_ids):
        raise GRPOV2ContractError("GRPO-v2 curriculum/train universe mismatch")
    pair_keys = tuple(
        f"{problem_id}::generation:{generation}"
        for problem_id in problem_ids
        for generation in range(4)
    )
    contract = GRPOV2Contract(
        algorithm="grpo",
        seed=42,
        config_path=str(CONFIG_PATH),
        config_sha256=identity["config_sha256"],
        registry_sha256=registry["registry_sha256"],
        data_registry_sha256=identity["data_registry_sha256"],
        manifest_sha256=current["manifest_sha256"],
        trusted_manifest_sha256=current["trusted_manifest_sha256"],
        curriculum_sha256=current["curriculum_sha256"],
        curriculum_identity_sha256=curriculum["curriculum_sha256"],
        dev_manifest_sha256=current["dev_manifest_sha256"],
        warmstart_checkpoint_sha256=identity["warmstart_checkpoint_sha256"],
        warmstart_adapter_sha256=identity["warmstart_adapter_sha256"],
        prompt_sha256=PROMPT_V2_SHA256,
        reward_sha256=FORMAL_REWARD_SHA256,
        parser_sha256=expected_ids["parser_sha256"],
        verifier_sha256=expected_ids["verifier_sha256"],
        pair_keys=pair_keys,
        problem_ids=problem_ids,
    )
    return config, identity, contract


def normalized_training_config(config: dict[str, Any], contract: GRPOV2Contract) -> dict[str, Any]:
    """Translate the frozen design into the existing TRL builder schema without changing it."""
    return {
        "experiment": {"algorithm": "grpo", "name": "grpo_v2_seed42", "seed": 42},
        "grpo_v2": {"family": "grpo_v2_seed42_v1", "config_sha256": contract.config_sha256},
        "model": {
            "name_or_path": FORMAL_REPO_ID,
            "revision": FORMAL_REVISION,
            "local_files_only": True,
            "dtype": "bfloat16",
            "use_qlora": False,
            "gradient_checkpointing": True,
        },
        "lora": {
            "rank": 16,
            "alpha": 32,
            "dropout": 0,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
        "prompt": {"version": PROMPT_V2_FORMAL_MATH},
        "prompt_version": PROMPT_V2_FORMAL_MATH,
        "prompt_sha256": PROMPT_V2_SHA256,
        "renderer_version": PROMPT_RENDERER_VERSION,
        "reward": {"policy": FORMAL_REWARD_VERSION},
        "reward_policy_version": FORMAL_REWARD_VERSION,
        "reward_policy_sha256": FORMAL_REWARD_SHA256,
        "generation": {
            "max_prompt_length": 928,
            "max_completion_length": 256,
            "max_sequence_length": 1184,
            "num_generations": 4,
            "generation_batch_size": 16,
            "temperature": 0.8,
            "top_p": 0.95,
        },
        "data": {"manifest": config["data"]["manifest"], "max_train_samples": 512},
        "training": {
            "max_steps": 128,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "num_iterations": 1,
            "learning_rate": 1e-5,
            "logging_steps": 1,
            "save_strategy": "steps",
            "save_steps": 32,
            "save_total_limit": 4,
            "save_only_model": True,
            "report_to": [],
            "push_to_hub": False,
            "shuffle_dataset": False,
            "dataloader_drop_last": True,
            "dataloader_num_workers": 0,
        },
        "budget": {
            "max_completions": 2048,
            "max_generated_tokens": 524_288,
            "max_wall_time_seconds": config["budget"]["max_wall_time_seconds"],
            "max_gpu_hours": config["budget"]["max_gpu_hours"],
            "max_estimated_cost_cny": config["budget"]["max_cost_cny"],
        },
    }


def validate_normalized_training_config(config: dict[str, Any]) -> None:
    contract = config.get("grpo_v2", {})
    if contract.get("config_sha256") != file_sha256(ROOT / CONFIG_PATH):
        raise ValueError("GRPO-v2 normalized config identity mismatch")
    if config["data"].get("max_train_samples") != 512:
        raise ValueError("GRPO-v2 normalized train count mismatch")
    training = config["training"]
    budget = config["budget"]
    if (
        any(
            training.get(key) != value
            for key, value in {
                "max_steps": 128,
                "per_device_train_batch_size": 4,
                "gradient_accumulation_steps": 4,
                "num_iterations": 1,
                "save_steps": 32,
                "save_total_limit": 4,
                "save_only_model": True,
                "shuffle_dataset": False,
            }.items()
        )
        or budget.get("max_completions") != 2048
        or budget.get("max_generated_tokens") != 524_288
    ):
        raise ValueError("GRPO-v2 normalized training budget mismatch")


def validate_initial_checkpoint(checkpoint: Path, identity: dict[str, Any]) -> dict[str, Any]:
    if checkpoint != WARMSTART_CHECKPOINT:
        raise GRPOV2ContractError("GRPO-v2 requires the exact warm-start checkpoint")
    evidence = validate_checkpoint(
        checkpoint,
        expected_config_sha=identity["warmstart_config_sha256"],
        expected_run_id=EXPECTED_WARMSTART_RUN_ID,
    )
    handoff = grpo_adapter_handoff(evidence)
    adapter_config = json.loads((checkpoint / "adapter/adapter_config.json").read_text())
    lora_matches = (
        adapter_config.get("r") == 16
        and adapter_config.get("lora_alpha") == 32
        and adapter_config.get("lora_dropout") == 0
        and set(adapter_config.get("target_modules", []))
        == {"q_proj", "k_proj", "v_proj", "o_proj"}
    )
    if (
        evidence["manifest"].get("artifact_sha256") != EXPECTED_CHECKPOINT_ARTIFACT_SHA256
        or handoff["adapter_sha256"] != EXPECTED_ADAPTER_SHA256
        or handoff["inherit_sft_optimizer_state"] is not False
        or not lora_matches
    ):
        raise GRPOV2ContractError("warm-start adapter handoff identity mismatch")
    return {**evidence, "handoff": handoff}


def require_execution_environment() -> dict[str, str]:
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise GRPOV2ContractError("GRPO-v2 requires both offline variables")
    branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain=v1"], text=True)
    if branch != EXPECTED_BRANCH or dirty:
        raise GRPOV2ContractError("GRPO-v2 requires a clean improve/grpo-v2 worktree")
    return {
        "branch": branch,
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }


def validate_run_dir(path: Path) -> Path:
    if (
        not path.is_absolute()
        or path.parent != RUN_ROOT
        or not path.name.startswith("grpo_v2_seed42_")
        or path.exists()
        or path.is_symlink()
    ):
        raise GRPOV2ContractError("GRPO-v2 run directory identity/conflict mismatch")
    return path


class GRPOV2ProgressGuard:
    def __init__(self, contract: GRPOV2Contract, run_id: str):
        self.contract = contract
        self.run_id = run_id
        self.updates = self.optimizer_steps = self.global_steps = 0
        self.completions = self.generated_tokens = 0
        self.pair_keys: list[str] = []
        self.checkpoints: list[int] = []
        self.validations: list[int] = []
        self.dev_completions = self.dev_tokens = 0

    def record_update(
        self, update: int, rows: list[dict[str, Any]], metric: dict[str, Any]
    ) -> None:
        if update != self.updates + 1 or len(rows) != 16:
            raise GRPOV2ContractError("GRPO-v2 update continuity/count mismatch")
        expected = self.contract.pair_keys_for_update(update)
        actual = tuple(row.get("pair_key") for row in rows)
        if actual != expected or len(set(actual)) != 16:
            raise GRPOV2ContractError("GRPO-v2 comparison-key order mismatch")
        tokens = 0
        for row in rows:
            ids, mask = row.get("completion_ids"), row.get("completion_mask")
            if not isinstance(ids, list) or not isinstance(mask, list) or len(ids) != len(mask):
                raise GRPOV2ContractError("GRPO-v2 completion IDs/mask missing")
            if any(value not in (0, 1) for value in mask):
                raise GRPOV2ContractError("GRPO-v2 completion mask is not binary")
            count = sum(mask)
            if row.get("exact_token_count") != count or count > 256:
                raise GRPOV2ContractError("GRPO-v2 completion token count mismatch")
            if not isinstance(row.get("raw_completion"), str):
                raise GRPOV2ContractError("GRPO-v2 completion text missing")
            reward = row.get("scalar_reward")
            if (
                isinstance(reward, bool)
                or not isinstance(reward, (int, float))
                or not math.isfinite(reward)
            ):
                raise GRPOV2ContractError("GRPO-v2 reward is non-finite")
            tokens += count
        if self.generated_tokens + tokens > self.contract.token_cap:
            raise GRPOV2ContractError("GRPO-v2 generated-token cap exceeded")
        required = {
            "reward_mean",
            "reward_std",
            "reward_variance",
            "loss",
            "learning_rate",
            "generated_tokens",
            "cumulative_generated_tokens",
            "group_rewards",
            "zero_advantage_fraction",
            "canonical_pass_rate",
            "format_accuracy",
            "parseable_rate",
            "valid_answer_rate",
        }
        if required - metric.keys():
            raise GRPOV2ContractError("GRPO-v2 required update evidence missing")
        for key in ("reward_mean", "reward_std", "reward_variance", "loss", "learning_rate"):
            value = metric[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise GRPOV2ContractError(f"GRPO-v2 non-finite required metric: {key}")
        optional_metrics = (
            (
                "entropy",
                "policy_entropy_mean",
                "policy_entropy_mean_available",
                "policy_entropy_mean_reason",
            ),
            ("grad_norm", "grad_norm", "grad_norm_available", "grad_norm_reason"),
            ("kl", "kl", "kl_available", "kl_unavailable_reason"),
            ("ratio", "ratio_mean", "ratio_available", "ratio_reason"),
            (
                "clip_fraction",
                "clip_fraction",
                "clip_fraction_available",
                "clip_fraction_reason",
            ),
        )
        for label, value_key, available_key, reason_key in optional_metrics:
            availability = metric.get(available_key)
            value = metric.get(value_key)
            if availability is True and (
                not isinstance(value, (int, float)) or not math.isfinite(value)
            ):
                raise GRPOV2ContractError(f"GRPO-v2 invalid optional metric: {label}")
            if availability is False and (value is not None or not metric.get(reason_key)):
                raise GRPOV2ContractError(
                    f"GRPO-v2 unavailable metric lacks null/reason: {label}"
                )
        self.updates = self.optimizer_steps = self.global_steps = update
        self.completions += 16
        self.generated_tokens += tokens
        self.pair_keys.extend(actual)

    def record_checkpoint(self, step: int) -> None:
        index = len(self.checkpoints)
        if index >= 4 or step != CHECKPOINT_STEPS[index] or step > self.updates:
            raise GRPOV2ContractError("GRPO-v2 checkpoint cadence mismatch")
        self.checkpoints.append(step)

    def record_validation(self, step: int, rows: list[dict[str, Any]]) -> None:
        index = len(self.validations)
        if index >= 4 or step != CHECKPOINT_STEPS[index] or self.checkpoints[index] != step:
            raise GRPOV2ContractError("GRPO-v2 validation cadence mismatch")
        if len(rows) != 128 or any(row.get("checkpoint_step") != step for row in rows):
            raise GRPOV2ContractError("GRPO-v2 dev must contain 128 checkpoint-linked rows")
        tokens = sum(int(row.get("exact_token_count", -1)) for row in rows)
        if tokens < 0 or tokens > 128 * 256:
            raise GRPOV2ContractError("GRPO-v2 dev token ledger mismatch")
        self.validations.append(step)
        self.dev_completions += 128
        self.dev_tokens += tokens

    def snapshot(self) -> dict[str, Any]:
        return {
            "updates": self.updates,
            "optimizer_steps": self.optimizer_steps,
            "global_steps": self.global_steps,
            "completions": self.completions,
            "generated_tokens": self.generated_tokens,
            "pair_keys": list(self.pair_keys),
            "checkpoints": list(self.checkpoints),
            "validations": list(self.validations),
            "dev_completions": self.dev_completions,
            "dev_tokens": self.dev_tokens,
            "training_token_budget_excludes_dev": True,
        }

    def assert_complete(self) -> dict[str, Any]:
        if (
            self.updates != 128
            or self.completions != 2048
            or self.checkpoints != list(CHECKPOINT_STEPS)
            or self.validations != list(CHECKPOINT_STEPS)
            or self.dev_completions != 512
            or tuple(self.pair_keys) != self.contract.pair_keys
        ):
            raise GRPOV2ContractError("GRPO-v2 final counters are incomplete")
        return self.snapshot()


class GRPOV2Observer:
    def __init__(self, contract: GRPOV2Contract, run_dir: Path, run_id: str):
        self.contract = contract
        self.run_dir = run_dir
        self.guard = GRPOV2ProgressGuard(contract, run_id)
        self.completions: list[dict[str, Any]] = []
        self.metrics: list[dict[str, Any]] = []
        self.validation_rows: list[dict[str, Any]] = []
        self.checkpoint_inventory: list[dict[str, Any]] = []

    def restore(self, validated: ValidatedFormalResume) -> None:
        step = validated.step
        self.completions = [dict(row) for row in validated.completion_prefix]
        self.metrics = [dict(row) for row in validated.metrics_prefix]
        self.guard.updates = self.guard.optimizer_steps = self.guard.global_steps = step
        self.guard.completions = len(self.completions)
        self.guard.generated_tokens = sum(int(row["exact_token_count"]) for row in self.completions)
        self.guard.pair_keys = [str(row["pair_key"]) for row in self.completions]
        self.guard.checkpoints = [value for value in CHECKPOINT_STEPS if value <= step]
        self.checkpoint_inventory = [
            formal_checkpoint_inventory(self.run_dir / f"checkpoint-{value}", self.contract, value)
            for value in self.guard.checkpoints
        ]
        self.persist_prefix()

    def update(self, update: int, rows: list[dict[str, Any]], metric: dict[str, Any]) -> None:
        self.guard.record_update(update, rows, metric)
        self.completions.extend(rows)
        self.metrics.append(
            {
                "update": update,
                "optimizer_step": update,
                "global_step": update,
                "microsteps": update * 4,
                "cumulative_completions": update * 16,
                **metric,
            }
        )
        self.persist_prefix()

    def persist_prefix(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for path, rows in (
            ("completions.jsonl", self.completions),
            ("metrics.jsonl", self.metrics),
        ):
            text = "".join(
                json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows
            )
            atomic_text(self.run_dir / path, text)
        ledger = {
            "phase": "training",
            "algorithm": "grpo_v2",
            "seed": 42,
            "updates": self.guard.updates,
            "unique_problems": self.guard.updates * 4,
            "completions": self.guard.completions,
            "generated_tokens": self.guard.generated_tokens,
            "training_token_budget": 524_288,
            "dev_in_training_budget": False,
        }
        atomic_text(self.run_dir / "sample_ledger.json", json.dumps(ledger, indent=2) + "\n")

    def checkpoint(self, step: int, path: Path) -> None:
        inventory = formal_checkpoint_inventory(path, self.contract, step)
        self.guard.record_checkpoint(step)
        self.checkpoint_inventory.append(inventory)

    def validation(self, step: int, rows: list[dict[str, Any]]) -> None:
        self.guard.record_validation(step, rows)
        self.validation_rows.extend(rows)


def validate_resume_checkpoint(
    root: Path, contract: GRPOV2Contract, run_dir: Path
) -> ValidatedFormalResume:
    if not run_dir.is_dir() or root.parent.resolve(strict=True) != run_dir.resolve(strict=True):
        raise GRPOV2ContractError("GRPO-v2 resume must belong to the exact same run")
    try:
        step = int(root.name.removeprefix("checkpoint-"))
    except ValueError as exc:
        raise GRPOV2ContractError("GRPO-v2 resume step malformed") from exc
    if step not in CHECKPOINT_STEPS[:-1]:
        raise GRPOV2ContractError("GRPO-v2 resume is allowed only from 32/64/96")
    inventory = formal_checkpoint_inventory(root, contract, step)
    manifest = json.loads((root / "resume_manifest.json").read_text())
    expected_count = step * 16
    completions = _read_jsonl(root / "trainer_completion_prefix.jsonl")
    metrics = _read_jsonl(root / "metrics_prefix.jsonl")
    if (
        len(completions) != expected_count
        or len(metrics) != step
        or tuple(row.get("pair_key") for row in completions) != contract.pair_keys[:expected_count]
        or manifest.get("updates") != step
        or manifest.get("sampler_position", {}).get("grpo_prompt_rows") != step * 4
        or manifest.get("warmstart_adapter_sha256") != contract.warmstart_adapter_sha256
    ):
        raise GRPOV2ContractError("GRPO-v2 resume prefix/cursor/identity mismatch")
    FormalOnlineGuard.from_resume_manifest(contract, manifest)
    return ValidatedFormalResume(
        root.resolve(), step, manifest, tuple(completions), tuple(metrics), inventory
    )


def select_dev_checkpoint(rows: list[dict[str, Any]]) -> int:
    by_step = {int(row["checkpoint_step"]): row for row in rows}
    if set(by_step) != set(CHECKPOINT_STEPS):
        raise GRPOV2ContractError("checkpoint selection requires all four dev summaries")

    def finite(name: str, row: dict[str, Any]) -> float:
        value = row.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise GRPOV2ContractError(f"invalid dev selection metric: {name}")
        return float(value)

    return min(
        CHECKPOINT_STEPS,
        key=lambda step: (
            -finite("canonical_pass_rate", by_step[step]),
            -finite("parseable_rate", by_step[step]),
            -finite("format_rate", by_step[step]),
            finite("truncation_rate", by_step[step]),
            step,
        ),
    )
