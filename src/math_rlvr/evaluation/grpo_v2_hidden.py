"""Guarded four-role evaluator for the frozen GRPO-v2 hidden test."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from math_rlvr.evaluation.grpo_v2_hidden_runtime import (
    EXPECTED_BRANCH,
    ROLES,
    RUN_ROOT,
    HiddenEvaluationContractError,
    artifact_schema,
    build_hidden_plan,
    load_hidden_contract,
    validate_four_model_plans,
    validate_role_selection,
)


def require_hidden_environment() -> dict[str, str]:
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get(
        "TRANSFORMERS_OFFLINE"
    ) != "1":
        raise HiddenEvaluationContractError(
            "hidden evaluation requires both offline variables"
        )
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], text=True
    ).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain=v1"], text=True)
    if branch != EXPECTED_BRANCH or dirty:
        raise HiddenEvaluationContractError(
            "hidden evaluation requires a clean improve/grpo-v2 worktree"
        )
    return {
        "branch": branch,
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    }


def validate_run_dir(path: Path, role: str) -> Path:
    prefix = f"{role}_hidden_grpo_v2_seed42_"
    if (
        not path.is_absolute()
        or path.parent != RUN_ROOT
        or not path.name.startswith(prefix)
        or path.exists()
        or path.is_symlink()
    ):
        raise HiddenEvaluationContractError(
            "hidden evaluation run directory identity/conflict mismatch"
        )
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
    parser.add_argument("--role", choices=ROLES, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-grpo-v2-hidden", action="store_true")
    args = parser.parse_args(argv)

    config, identity, public_rows, shared_problem_ids = load_hidden_contract(args.config)
    plan = build_hidden_plan(public_rows, shared_problem_ids)
    validate_four_model_plans({role: list(plan) for role in ROLES})
    checkpoint_identity = validate_role_selection(
        config, role=args.role, checkpoint=args.checkpoint
    )
    dry_run: dict[str, Any] = {
        "status": "dry_run",
        "role": args.role,
        "config_sha256": identity["config_sha256"],
        "public_manifest_sha256": identity["public_manifest_sha256"],
        "unique_problems": 400,
        "candidate0_rows": 400,
        "shared_n10_problems": 100,
        "completion_count": len(plan),
        "four_model_completion_count": 5_200,
        "candidate_indices": list(range(10)),
        "checkpoint_identity": checkpoint_identity,
        "artifact_schema": artifact_schema(),
        "trusted_manifest_opened": False,
        "model_or_tokenizer_loads": 0,
        "cuda_initialized": False,
        "generation_calls": 0,
        "train_calls": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "checkpoint_writes": 0,
    }
    if not args.execute:
        print(json.dumps(dry_run, sort_keys=True))
        return 0
    if not args.confirm_grpo_v2_hidden:
        raise RuntimeError(
            "hidden evaluation requires --execute --confirm-grpo-v2-hidden"
        )
    if args.run_dir is None:
        raise HiddenEvaluationContractError(
            "hidden evaluation execute requires --run-dir"
        )
    run_dir = validate_run_dir(args.run_dir, args.role)
    environment = (environment_probe or require_hidden_environment)()
    if snapshot_probe is None:
        from math_rlvr.training.warmstart_runtime import require_local_snapshot

        snapshot_probe = require_local_snapshot
    model_source = snapshot_probe()
    if execute_fn is None:
        from math_rlvr.evaluation.grpo_v2_hidden_supervisor import (
            execute_supervised_hidden,
        )

        execute_fn = execute_supervised_hidden
    outcome = execute_fn(
        config=config,
        identity=identity,
        public_rows=public_rows,
        shared_problem_ids=shared_problem_ids,
        role=args.role,
        checkpoint_identity=checkpoint_identity,
        model_source=model_source,
        run_dir=run_dir,
        environment=environment,
    )
    if outcome.get("status") != "success":
        raise RuntimeError(outcome.get("reason", "hidden evaluation failed"))
    print(json.dumps(outcome, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
