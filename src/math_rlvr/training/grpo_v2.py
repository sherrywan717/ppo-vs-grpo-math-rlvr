"""Guarded model-bound CLI for the frozen seed-42 GRPO-v2 training run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from math_rlvr.training.grpo_v2_runtime import (
    CONFIG_PATH,
    RUN_ROOT,
    GRPOV2ContractError,
    load_contract,
    require_execution_environment,
    validate_initial_checkpoint,
    validate_resume_checkpoint,
    validate_run_dir,
)


def main(
    argv: list[str] | None = None,
    *,
    execute_fn=None,
    environment_probe=None,
    snapshot_probe=None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--warmstart-checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-grpo-v2", action="store_true")
    args = parser.parse_args(argv)

    design, identity, contract = load_contract(args.config)
    warmstart = validate_initial_checkpoint(args.warmstart_checkpoint, identity)
    dry_run = {
        "status": "dry_run",
        "config_path": str(CONFIG_PATH),
        "config_sha256": contract.config_sha256,
        "runtime_registry_sha256": contract.registry_sha256,
        "warmstart_checkpoint_sha256": contract.warmstart_checkpoint_sha256,
        "warmstart_adapter_sha256": contract.warmstart_adapter_sha256,
        "initial_adapter_role": warmstart["handoff"]["adapter_role"],
        "sft_optimizer_inherited": False,
        "grpo_optimizer_initialization": "fresh",
        "updates": 128,
        "microsteps": 512,
        "unique_training_problems": 512,
        "training_completions": 2048,
        "training_generated_token_cap": 524288,
        "checkpoint_steps": [32, 64, 96, 128],
        "dev_problems_per_step": 128,
        "dev_budget_isolated": True,
        "hidden_test_accesses": 0,
        "model_or_tokenizer_loads": 0,
        "cuda_initialized": False,
        "generation_calls": 0,
        "train_calls": 0,
        "backward_calls": 0,
        "optimizer_steps_executed": 0,
    }
    if not args.execute:
        if args.confirm_grpo_v2:
            raise RuntimeError("GRPO-v2 confirmation is invalid without --execute")
        print(json.dumps(dry_run, sort_keys=True))
        return 0
    if not args.confirm_grpo_v2:
        raise RuntimeError("GRPO-v2 requires --execute --confirm-grpo-v2")
    if args.run_dir is None:
        raise GRPOV2ContractError("GRPO-v2 execute requires --run-dir")
    if args.resume_checkpoint is None:
        run_dir = validate_run_dir(args.run_dir)
        validated_resume = None
    else:
        if args.run_dir != args.resume_checkpoint.parent or args.run_dir.parent != RUN_ROOT:
            raise GRPOV2ContractError("GRPO-v2 resume run/checkpoint path mismatch")
        run_dir = args.run_dir.resolve(strict=True)
        validated_resume = validate_resume_checkpoint(args.resume_checkpoint, contract, run_dir)
    environment = (environment_probe or require_execution_environment)()
    if snapshot_probe is None:
        from math_rlvr.training.warmstart_runtime import require_local_snapshot

        snapshot_probe = require_local_snapshot
    model_source = snapshot_probe()
    if execute_fn is None:
        from math_rlvr.training.grpo_v2_supervisor import execute_supervised_grpo_v2

        execute_fn = execute_supervised_grpo_v2
    result = execute_fn(
        design,
        identity={**identity, "warmstart_handoff": warmstart["handoff"]},
        contract=contract,
        model_source=model_source,
        warmstart_checkpoint=args.warmstart_checkpoint,
        run_dir=run_dir,
        environment=environment,
        resume_checkpoint=(validated_resume.checkpoint if validated_resume else None),
    )
    if result.get("status") != "success":
        raise RuntimeError(result.get("reason", "GRPO-v2 execution failed"))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
