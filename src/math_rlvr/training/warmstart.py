"""Guarded model-bound CLI for the GRPO-v2 warm-start."""

from __future__ import annotations

import argparse
from pathlib import Path

from math_rlvr.training.warmstart_runtime import (
    load_contract,
    require_execution_environment,
    require_local_snapshot,
)


def main(argv=None, *, execute_fn=None, environment_probe=None, snapshot_probe=None) -> int:
    parser = argparse.ArgumentParser(description="GRPO-v2 warm-start preflight")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-grpo-v2-warmstart", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args(argv)
    config, identity = load_contract(args.config)
    if not args.execute:
        if args.confirm_grpo_v2_warmstart:
            raise RuntimeError("confirmation is invalid without --execute")
        print("GRPO-v2 warm-start preflight passed; model/training not started.")
        return 0
    if not args.confirm_grpo_v2_warmstart:
        raise RuntimeError("warm-start requires --execute --confirm-grpo-v2-warmstart")
    if args.run_dir is None or args.run_dir.exists():
        raise RuntimeError("warm-start execute requires a new non-existing --run-dir")
    environment = (environment_probe or require_execution_environment)()
    model_source = (snapshot_probe or require_local_snapshot)()
    if execute_fn is None:
        from math_rlvr.training.warmstart_model_runtime import execute_real_warmstart

        execute_fn = execute_real_warmstart
    result = execute_fn(
        config,
        identity=identity,
        model_source=model_source,
        run_dir=args.run_dir,
        environment=environment,
    )
    if result.get("status") != "success":
        raise RuntimeError(result.get("reason", "warm-start failed"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
