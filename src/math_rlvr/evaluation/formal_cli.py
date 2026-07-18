"""CPU-safe authorization and adapter selection for formal 1.5B evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from math_rlvr.evaluation.formal import DEFAULT_CONFIG_PATH
from math_rlvr.training.formal import validate_formal_config_file
from math_rlvr.training.formal_cli import FormalAuthorizationError
from math_rlvr.training.formal_runtime import formal_checkpoint_inventory, formal_run_contract

EVALUATION_RAW_SHA256 = "85100dd0f613f295a7219a45a42a03e3ad4a45e24893c7f296e1d8da9a1f4a35"


@dataclass(frozen=True)
class FormalEvaluationSelection:
    mode: str
    phase: str
    seed: int
    algorithm: str | None
    checkpoint_step: int | None
    checkpoint: Path | None
    policy_adapter: Path | None
    config_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "phase": self.phase,
            "seed": self.seed,
            "algorithm": self.algorithm,
            "checkpoint_step": self.checkpoint_step,
            "checkpoint": str(self.checkpoint) if self.checkpoint else None,
            "policy_adapter": str(self.policy_adapter) if self.policy_adapter else None,
            "config_sha256": self.config_sha256,
            "value_adapter_loaded_for_generation": False,
            "value_head_loaded_for_generation": False,
        }


def validate_evaluation_config_path(path: Path) -> str:
    expected = Path("configs/formal_1p5b/evaluation.json")
    if path.is_absolute() or path.as_posix() != expected.as_posix():
        raise FormalAuthorizationError(
            "formal evaluation execute requires the exact repository-relative config path"
        )
    if path.is_symlink() or not path.is_file() or path.resolve(strict=True) != DEFAULT_CONFIG_PATH:
        raise FormalAuthorizationError("formal evaluation config must be canonical and non-symlink")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != EVALUATION_RAW_SHA256:
        raise FormalAuthorizationError("formal evaluation raw config SHA256 mismatch")
    return actual


def validate_formal_evaluation_selection(
    *,
    config_path: Path,
    mode: str,
    phase: str,
    seed: int,
    checkpoint_step: int | None,
    checkpoint: Path | None,
) -> FormalEvaluationSelection:
    config_sha = validate_evaluation_config_path(config_path)
    if mode == "base":
        if phase != "baseline" or checkpoint is not None or checkpoint_step is not None:
            raise FormalAuthorizationError(
                "base evaluation requires phase=baseline and no checkpoint/adapter"
            )
        return FormalEvaluationSelection(mode, phase, seed, None, None, None, None, config_sha)
    if mode not in {"ppo", "grpo"} or phase not in {"validation", "final"}:
        raise FormalAuthorizationError("adapter evaluation requires an explicit PPO/GRPO mode")
    if checkpoint is None:
        raise FormalAuthorizationError("adapter evaluation requires a checkpoint")
    if phase == "final" and checkpoint_step != 32:
        raise FormalAuthorizationError("formal final evaluation is fixed to checkpoint-32")
    if phase == "validation" and checkpoint_step not in {8, 16, 24, 32}:
        raise FormalAuthorizationError("formal validation checkpoint step is not frozen")
    if (
        checkpoint.name != f"checkpoint-{checkpoint_step}"
        or checkpoint.is_symlink()
        or checkpoint.parent.is_symlink()
    ):
        raise FormalAuthorizationError("formal evaluation checkpoint path/step mismatch")
    training_path = Path(f"configs/formal_1p5b/resolved/{mode}_seed_{seed}.json")
    training_config = validate_formal_config_file(training_path, mode)[0]
    contract = formal_run_contract(training_config)
    formal_checkpoint_inventory(checkpoint, contract, int(checkpoint_step))
    resume = json.loads((checkpoint / "resume_manifest.json").read_text(encoding="utf-8"))
    for key, expected in {
        "algorithm": mode,
        "seed": seed,
        "config_sha256": contract.config_sha256,
        "active_suite_sha256": contract.active_suite_sha256,
        "updates": checkpoint_step,
        "base_weights_included": False,
    }.items():
        if resume.get(key) != expected:
            raise FormalAuthorizationError(f"formal evaluation checkpoint {key} mismatch")
    if resume.get("run_id") != checkpoint.parent.name:
        raise FormalAuthorizationError(
            "formal evaluation checkpoint run_id does not match its run directory"
        )
    policy_adapter = checkpoint / "policy_adapter"
    if policy_adapter.is_symlink() or not policy_adapter.is_dir():
        raise FormalAuthorizationError("formal policy adapter is missing or symlinked")
    return FormalEvaluationSelection(
        mode,
        phase,
        seed,
        mode,
        checkpoint_step,
        checkpoint.resolve(strict=True),
        policy_adapter.resolve(strict=True),
        config_sha,
    )
