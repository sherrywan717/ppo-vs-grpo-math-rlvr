"""Fail-closed GRPO entry point; real imports occur only after dual confirmation."""

from pathlib import Path

from math_rlvr.training.common import parse_args, preflight, render_training_prompt

__all__ = ["main", "render_training_prompt"]


def main(argv=None, execute_fn=None, git_probe=None, snapshot_probe=None) -> int:
    args = parse_args("GRPO training preflight", argv)
    config = preflight(args.config, "grpo")
    if not args.execute:
        print(f"Preflight passed for {config['experiment']['name']}; no training started.")
        return 0
    if not args.confirm_single_update:
        raise RuntimeError("--execute alone is insufficient; add --confirm-single-update")
    from math_rlvr.training.guarded_grpo import (
        require_clean_git,
        require_local_snapshot,
        validate_smoke_authorization,
    )

    validate_smoke_authorization(config, Path(args.config))
    (git_probe or require_clean_git)()
    (snapshot_probe or require_local_snapshot)()
    if execute_fn is None:
        from math_rlvr.training.grpo_runtime import execute_real_smoke
        execute_fn = execute_real_smoke
    result = execute_fn(config)
    if result.get("status") != "success":
        raise RuntimeError(result.get("reason", "guarded GRPO smoke failed"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
