"""Guarded matched Base/warm-start evaluation on the frozen GRPO-v2 dev split."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from math_rlvr.evaluation.grpo_v2_dev_runtime import (
    EXPECTED_BRANCH,
    RUN_ROOT,
    DevEvaluationContractError,
    build_dev_plan,
    load_dev_contract,
    validate_matched_plans,
    validate_warmstart_selection,
)


def require_dev_environment() -> dict[str, str]:
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise DevEvaluationContractError("dev evaluation requires both offline variables")
    branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain=v1"], text=True)
    if branch != EXPECTED_BRANCH or dirty:
        raise DevEvaluationContractError("dev evaluation requires a clean improve/grpo-v2 worktree")
    return {
        "branch": branch,
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }


def validate_run_dir(path: Path, mode: str) -> Path:
    prefix = "base_dev_grpo_v2_seed42_" if mode == "base" else "warmstart_dev_grpo_v2_seed42_"
    if (
        not path.is_absolute()
        or path.parent != RUN_ROOT
        or not path.name.startswith(prefix)
        or path.exists()
        or path.is_symlink()
    ):
        raise DevEvaluationContractError("dev run directory identity/conflict mismatch")
    return path


def main(
    argv: list[str] | None = None,
    *,
    execute_fn=None,
    environment_probe=None,
    snapshot_probe=None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("base", "warmstart"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-grpo-v2-dev", action="store_true")
    args = parser.parse_args(argv)

    config, identity, public_rows = load_dev_contract(args.config)
    plan = build_dev_plan(config, public_rows, mode=args.mode)
    validate_matched_plans(
        build_dev_plan(config, public_rows, mode="base"),
        build_dev_plan(config, public_rows, mode="warmstart"),
    )
    checkpoint_identity = validate_warmstart_selection(config, args.checkpoint, mode=args.mode)
    dry_run: dict[str, Any] = {
        "status": "dry_run",
        "mode": args.mode,
        "config_sha256": identity["config_sha256"],
        "data_registry_sha256": identity["data_registry_sha256"],
        "dev_manifest_sha256": identity["dev_manifest_sha256"],
        "problem_count": len(plan),
        "completion_count": len(plan),
        "candidate_indices": [0],
        "max_generated_tokens": config["budget"]["max_generated_tokens"],
        "model_or_tokenizer_loads": 0,
        "cuda_initialized": False,
        "generation_calls": 0,
        "train_calls": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "checkpoint_writes": 0,
        "checkpoint_identity": checkpoint_identity,
    }
    if not args.execute:
        print(json.dumps(dry_run, sort_keys=True))
        return 0
    if not args.confirm_grpo_v2_dev:
        raise RuntimeError("dev evaluation requires --execute --confirm-grpo-v2-dev")
    if args.run_dir is None:
        raise DevEvaluationContractError("dev evaluation execute requires --run-dir")
    run_dir = validate_run_dir(args.run_dir, args.mode)
    environment = (environment_probe or require_dev_environment)()
    if snapshot_probe is None:
        from math_rlvr.training.warmstart_runtime import require_local_snapshot

        snapshot_probe = require_local_snapshot
    model_source = snapshot_probe()
    if execute_fn is None:
        from math_rlvr.evaluation.grpo_v2_dev_supervisor import execute_supervised_dev

        execute_fn = execute_supervised_dev
    outcome = execute_fn(
        config=config,
        identity=identity,
        public_rows=public_rows,
        mode=args.mode,
        checkpoint_identity=checkpoint_identity,
        model_source=model_source,
        run_dir=run_dir,
        environment=environment,
    )
    if outcome.get("status") != "success":
        raise RuntimeError(outcome.get("reason", "dev evaluation failed"))
    print(json.dumps(outcome, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
