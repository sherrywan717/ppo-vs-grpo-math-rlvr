"""CPU-safe authorization boundary for formal 1.5B model-bound execution."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from math_rlvr.training.formal import (
    FORMAL_MODEL,
    FORMAL_REVISION,
    validate_active_suite,
)
from math_rlvr.training.model_source import DEFAULT_CACHE_ROOT, ValidatedModelSource

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
FORMAL_RUN_ROOT = Path("/root/autodl-tmp/runs/math_rlvr")


class FormalAuthorizationError(RuntimeError):
    """A formal execute request does not match the frozen paid-stage boundary."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_repository_config_path(path: Path) -> str:
    """Reject aliases: execute paths accept only literal repository-relative files."""
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise FormalAuthorizationError(
            "formal execution requires an exact repository-relative config path"
        )
    relative = path.as_posix()
    target = REPOSITORY_ROOT / relative
    try:
        target.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise FormalAuthorizationError("formal config escaped the repository") from exc
    if not target.is_file() or target.is_symlink():
        raise FormalAuthorizationError("formal config must be a regular non-symlink file")
    current = target.parent
    while current != REPOSITORY_ROOT:
        if current.is_symlink():
            raise FormalAuthorizationError("formal config parent symlinks are forbidden")
        current = current.parent
    if target.resolve(strict=True) != target:
        raise FormalAuthorizationError("formal config path is not canonical")
    return relative


def validate_formal_training_authorization(
    config: dict[str, Any], config_path: Path, algorithm: str
) -> dict[str, Any]:
    if algorithm not in {"ppo", "grpo"}:
        raise FormalAuthorizationError("formal training algorithm must be PPO or GRPO")
    relative = exact_repository_config_path(config_path)
    suite = validate_active_suite()
    reserved = {row["config"] for row in suite["reserved_configs"]}
    if relative in reserved:
        raise FormalAuthorizationError("formal seed 2026 is reserved_not_scheduled")
    row = next(
        (
            item
            for item in suite["active_training_runs"]
            if item["config"] == relative and item["algorithm"] == algorithm
        ),
        None,
    )
    if row is None:
        raise FormalAuthorizationError("config is not in the active four-run suite")
    target = REPOSITORY_ROOT / relative
    actual_sha = _sha256(target)
    if actual_sha != row["config_sha256"]:
        raise FormalAuthorizationError("formal resolved config SHA256 mismatch")
    identity = (
        config.get("resolved_config_path"),
        config.get("resolved_config_sha256"),
        config.get("experiment", {}).get("algorithm"),
        config.get("experiment", {}).get("seed"),
        config.get("model", {}).get("name_or_path"),
        config.get("model", {}).get("revision"),
        config.get("model", {}).get("local_files_only"),
    )
    expected = (
        relative,
        actual_sha,
        algorithm,
        row["seed"],
        FORMAL_MODEL,
        FORMAL_REVISION,
        True,
    )
    if identity != expected:
        raise FormalAuthorizationError("formal resolved execution identity mismatch")

    from math_rlvr.training.execution_contract import expected_run_contract_for_config
    from math_rlvr.training.formal_runtime import formal_run_contract

    evidence_contract = expected_run_contract_for_config(config, algorithm)
    runtime_contract = formal_run_contract(config)
    if (
        evidence_contract.profile != runtime_contract.profile
        or evidence_contract.config_sha256 != runtime_contract.config_sha256
        or evidence_contract.expected_completions != runtime_contract.expected_completions
        or evidence_contract.generated_token_cap != runtime_contract.token_cap
        or evidence_contract.expected_updates != runtime_contract.updates
    ):
        raise FormalAuthorizationError(
            "ExpectedRunContract and formal runtime contract disagree"
        )
    return {
        "algorithm": algorithm,
        "seed": row["seed"],
        "config_path": relative,
        "config_sha256": actual_sha,
        "active_suite_sha256": suite["active_suite_sha256"],
        "expected_run_profile": evidence_contract.profile,
    }


def require_formal_offline_environment() -> dict[str, str]:
    names = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    values = {name: os.environ.get(name, "") for name in names}
    if any(value != "1" for value in values.values()):
        raise FormalAuthorizationError(
            "formal execution requires HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1"
        )
    return values


def require_formal_local_snapshot(
    *,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    snapshot_resolver: Callable[..., str] | None = None,
) -> ValidatedModelSource:
    return ValidatedModelSource.resolve(
        FORMAL_MODEL,
        FORMAL_REVISION,
        cache_root=cache_root,
        snapshot_resolver=snapshot_resolver,
    )

def validate_formal_resume_authorization(
    config: dict[str, Any], checkpoint: Path, algorithm: str
):
    """Validate a project-owned same-run checkpoint before snapshot/model handling."""
    from math_rlvr.training.formal_runtime import (
        formal_run_contract,
        validate_formal_resume_checkpoint,
    )

    if not checkpoint.is_absolute() or checkpoint.is_symlink():
        raise FormalAuthorizationError(
            "formal resume checkpoint must be an absolute canonical project run path"
        )
    try:
        resolved_root = FORMAL_RUN_ROOT.resolve(strict=True)
        resolved_checkpoint = checkpoint.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FormalAuthorizationError("formal resume checkpoint does not exist") from exc
    run_dir = resolved_checkpoint.parent
    expected_prefix = f"{algorithm}_formal_1p5b_seed{config['experiment']['seed']}_"
    if (
        run_dir.parent != resolved_root
        or not run_dir.name.startswith(expected_prefix)
        or resolved_checkpoint != run_dir / checkpoint.name
    ):
        raise FormalAuthorizationError(
            "formal resume checkpoint is not a direct child of its same project run"
        )
    return validate_formal_resume_checkpoint(
        resolved_checkpoint, formal_run_contract(config), run_dir=run_dir
    )
