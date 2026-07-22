"""CPU-safe contracts for the guarded GRPO-v2 completion-only warm-start."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from math_rlvr.training.formal_model import derive_static_parameter_contract
from math_rlvr.training.model_source import (
    DEFAULT_CACHE_ROOT,
    FORMAL_REPO_ID,
    FORMAL_REVISION,
    ValidatedModelSource,
)

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = Path("configs/grpo_v2/warmstart_seed42.json")
REGISTRY_PATH = ROOT / "configs/grpo_v2/runtime_registry.json"
RUN_ROOT = Path("/root/autodl-tmp/runs/math_rlvr")
BACKUP_ROOT = Path("/root/autodl-fs/math-rlvr-backups")
EXPECTED_BRANCH = "improve/grpo-v2"
REQUIRED_CHECKPOINT_FILES = {
    "adapter/adapter_config.json",
    "adapter/adapter_model.safetensors",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pt",
    "trainer_state.json",
    "runtime_state.json",
    "checkpoint_identity.json",
    "artifact_manifest.json",
}
FORBIDDEN_CHECKPOINT_FILES = {"pytorch_model.bin", "model.safetensors"}
ALLOWED_CHECKPOINT_FILES = REQUIRED_CHECKPOINT_FILES | {"adapter/README.md"}


class WarmstartContractError(RuntimeError):
    """A warm-start identity, budget, or evidence invariant failed."""


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract(config_path: Path = CONFIG_PATH) -> tuple[dict[str, Any], dict[str, Any]]:
    if config_path != CONFIG_PATH or config_path.is_absolute() or ".." in config_path.parts:
        raise WarmstartContractError("warm-start requires the exact repository config path")
    target = ROOT / config_path
    if target.is_symlink() or target.resolve(strict=True) != target:
        raise WarmstartContractError("warm-start config must be canonical and non-symlink")
    config = json.loads(target.read_text())
    registry = json.loads(REGISTRY_PATH.read_text())
    expected = registry["warmstart"]
    if file_sha256(target) != expected["config_sha256"]:
        raise WarmstartContractError("warm-start config SHA mismatch")
    identity = (
        config["seed"],
        config["data"]["samples"],
        config["data"]["epochs"],
        config["training"]["per_device_batch_size"],
        config["training"]["gradient_accumulation_steps"],
        config["training"]["effective_batch_size"],
        config["training"]["optimizer_steps"],
        config["training"]["scheduler_steps"],
        config["model"]["repo"],
        config["model"]["revision"],
        config["model"]["local_files_only"],
        config["model"]["dtype"],
        config["prompt"]["max_prompt_length"],
        config["prompt"]["max_target_length"],
        config["prompt"]["max_sequence_length"],
    )
    expected_identity = (
        42,
        256,
        1,
        4,
        4,
        16,
        16,
        16,
        FORMAL_REPO_ID,
        FORMAL_REVISION,
        True,
        "bfloat16",
        928,
        640,
        1088,
    )
    if identity != expected_identity:
        raise WarmstartContractError("warm-start resolved contract mismatch")
    if derive_static_parameter_contract()["policy_lora_trainable_parameters"] != 4_358_144:
        raise WarmstartContractError("static policy LoRA parameter contract drift")
    for key in ("warmstart_manifest_sha256", "train_manifest_sha256", "data_registry_sha256"):
        path = (
            ROOT
            / expected[
                key.removesuffix("_sha256")
                .replace("warmstart_manifest", "warmstart_manifest_path")
                .replace("train_manifest", "train_manifest_path")
                .replace("data_registry", "data_registry_path")
            ]
        )
        if file_sha256(path) != expected[key]:
            raise WarmstartContractError(f"{key} mismatch")
    return config, expected


def encode_completion_only(
    tokenizer,
    prompt_messages: list[dict[str, str]],
    target: str,
    *,
    max_prompt: int = 928,
    max_target: int = 640,
    max_sequence: int = 1088,
) -> dict[str, Any]:
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages, tokenize=True, add_generation_prompt=True
    )
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    eos = tokenizer.eos_token_id
    if eos is None:
        raise WarmstartContractError("tokenizer has no EOS token")
    active = target_ids + [eos]
    if len(prompt_ids) > max_prompt:
        raise WarmstartContractError("prompt exceeds max_prompt_length")
    if len(active) > max_target:
        raise WarmstartContractError("target including EOS exceeds max_target_length")
    if len(prompt_ids) + len(active) > max_sequence:
        raise WarmstartContractError("actual combined sequence exceeds max_sequence_length")
    return {
        "input_ids": prompt_ids + active,
        "attention_mask": [1] * (len(prompt_ids) + len(active)),
        "labels": [-100] * len(prompt_ids) + active,
        "prompt_tokens": len(prompt_ids),
        "active_label_tokens": len(active),
    }


def completion_only_collate(
    features: list[dict[str, Any]], *, pad_token_id: int
) -> dict[str, list[list[int]]]:
    if not features:
        raise WarmstartContractError("empty warm-start batch")
    width = max(len(row["input_ids"]) for row in features)
    result = {"input_ids": [], "attention_mask": [], "labels": []}
    for row in features:
        padding = width - len(row["input_ids"])
        result["input_ids"].append(row["input_ids"] + [pad_token_id] * padding)
        result["attention_mask"].append(row["attention_mask"] + [0] * padding)
        result["labels"].append(row["labels"] + [-100] * padding)
    return result


@dataclass
class WarmstartBudgetGuard:
    samples: int = 0
    batches: int = 0
    microsteps: int = 0
    optimizer_steps: int = 0
    global_steps: int = 0
    epochs: int = 0
    active_label_tokens: int = 0
    seen_sample_ids: set[str] = field(default_factory=set, repr=False)

    def record_batch(self, *, sample_ids: list[str], active_label_tokens: int) -> None:
        if len(sample_ids) != len(set(sample_ids)) or self.seen_sample_ids & set(sample_ids):
            raise WarmstartContractError("duplicate samples in warm-start batch")
        self.seen_sample_ids.update(sample_ids)
        self.samples += len(sample_ids)
        self.batches += 1
        self.active_label_tokens += active_label_tokens
        if self.samples > 256:
            raise WarmstartContractError("warm-start sample budget exceeded")

    def record_microstep(self, loss: float) -> None:
        if not math.isfinite(loss):
            raise WarmstartContractError("non-finite warm-start loss")
        self.microsteps += 1
        if self.microsteps > 64:
            raise WarmstartContractError("warm-start microstep budget exceeded")

    def record_optimizer_step(self, global_step: int) -> None:
        self.optimizer_steps += 1
        self.global_steps = global_step
        if self.optimizer_steps > 16 or global_step > 16:
            raise WarmstartContractError("warm-start optimizer/global-step budget exceeded")

    def record_epoch(self) -> None:
        self.epochs += 1
        if self.epochs > 1:
            raise WarmstartContractError("warm-start epoch budget exceeded")

    def finalize(self) -> dict[str, int]:
        actual = (
            self.samples,
            self.batches,
            self.microsteps,
            self.optimizer_steps,
            self.global_steps,
            self.epochs,
        )
        if actual != (256, 64, 64, 16, 16, 1):
            raise WarmstartContractError(f"incomplete warm-start counters: {actual}")
        return {key: value for key, value in self.__dict__.items() if key != "seen_sample_ids"}


def require_execution_environment() -> dict[str, str]:
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise WarmstartContractError("warm-start requires both offline variables")
    branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain=v1"], text=True)
    if branch != EXPECTED_BRANCH or dirty:
        raise WarmstartContractError("warm-start requires clean improve/grpo-v2 worktree")
    return {
        "branch": branch,
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }


def require_local_snapshot() -> ValidatedModelSource:
    return ValidatedModelSource.resolve(
        FORMAL_REPO_ID, FORMAL_REVISION, cache_root=DEFAULT_CACHE_ROOT
    )


def validate_checkpoint(
    checkpoint: Path, *, expected_config_sha: str, expected_run_id: str | None = None
) -> dict[str, Any]:
    if not checkpoint.is_absolute() or checkpoint.is_symlink():
        raise WarmstartContractError("checkpoint must be an absolute non-symlink path")
    resolved = checkpoint.resolve(strict=True)
    if resolved.name != "checkpoint-16":
        raise WarmstartContractError("warm-start checkpoint must be checkpoint-16")
    if resolved.parent.parent != RUN_ROOT.resolve(strict=True):
        raise WarmstartContractError("checkpoint path is outside direct project run")
    if not resolved.parent.name.startswith("warmstart_grpo_v2_seed42_"):
        raise WarmstartContractError("checkpoint run directory identity mismatch")
    files = set()
    for path in resolved.rglob("*"):
        if path.is_symlink():
            raise WarmstartContractError("checkpoint symlink forbidden")
        if path.is_dir():
            continue
        if not stat.S_ISREG(path.lstat().st_mode):
            raise WarmstartContractError("checkpoint non-regular file forbidden")
        files.add(path.relative_to(resolved).as_posix())
    if (
        not REQUIRED_CHECKPOINT_FILES <= files
        or not files <= ALLOWED_CHECKPOINT_FILES
        or any(Path(name).name in FORBIDDEN_CHECKPOINT_FILES for name in files)
    ):
        raise WarmstartContractError("checkpoint allowlist/required-state contract failed")
    identity = json.loads((resolved / "checkpoint_identity.json").read_text())
    if (
        identity.get("config_sha256") != expected_config_sha
        or identity.get("adapter_role") != "policy"
    ):
        raise WarmstartContractError("checkpoint identity mismatch")
    if expected_run_id is not None and identity.get("run_id") != expected_run_id:
        raise WarmstartContractError("checkpoint run identity mismatch")
    manifest = json.loads((resolved / "artifact_manifest.json").read_text())
    if manifest.get("base_weights_included") is not False:
        raise WarmstartContractError("checkpoint base-weight declaration invalid")
    inventory = manifest.get("files", {})
    for name, evidence in inventory.items():
        path = resolved / name
        if name not in files or file_sha256(path) != evidence.get("sha256"):
            raise WarmstartContractError("checkpoint inventory SHA mismatch")
    if not (REQUIRED_CHECKPOINT_FILES - {"artifact_manifest.json"}) <= set(inventory):
        raise WarmstartContractError("checkpoint inventory omits trusted resume state")
    return {"checkpoint": str(resolved), "identity": identity, "manifest": manifest}


def validate_postprocess_gpu_release(
    *, worker_exited: bool, compute_processes: int, used_memory_mib: int
) -> dict[str, Any]:
    if not worker_exited or compute_processes != 0 or used_memory_mib != 0:
        raise WarmstartContractError("post-process GPU release contract failed")
    return {"worker_exited": True, "compute_processes": 0, "used_memory_mib": 0}


def grpo_adapter_handoff(checkpoint_evidence: dict[str, Any]) -> dict[str, Any]:
    identity = checkpoint_evidence["identity"]
    return {
        "adapter_path": str(Path(checkpoint_evidence["checkpoint"]) / "adapter"),
        "adapter_sha256": checkpoint_evidence["manifest"]["files"][
            "adapter/adapter_model.safetensors"
        ]["sha256"],
        "source_checkpoint_sha256": checkpoint_evidence["manifest"]["artifact_sha256"],
        "adapter_role": "policy",
        "grpo_optimizer_initialization": "fresh_frozen_grpo_v2_contract",
        "inherit_sft_optimizer_state": False,
        "source_run_id": identity["run_id"],
    }


def backup_warmstart_run(run_dir: Path, *, failure: bool) -> dict[str, Any]:
    """Create the existing verified formal archive for one immutable run directory."""
    from math_rlvr.training.formal_runtime import create_formal_backup

    suffix = ".failure" if failure else ""
    archive = BACKUP_ROOT / f"{run_dir.name}{suffix}.tar.gz"
    return create_formal_backup(run_dir, archive)
