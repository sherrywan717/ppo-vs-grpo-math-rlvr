"""Fail-closed GRPO entry point; real imports occur only after dual confirmation."""

from pathlib import Path

from math_rlvr.training.common import (
    parse_args,
    preflight,
    render_candidate_training_prompt,
    render_training_prompt,
)

__all__ = ["main", "render_candidate_training_prompt", "render_training_prompt"]


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

    is_pilot = config.get("pilot", {}).get("family") == "matched_0p5b_v1"
    if is_pilot:
        from math_rlvr.training.guarded_ppo import require_ppo_offline_environment
        from math_rlvr.training.pilot import validate_pilot_execution_authorization

        validate_pilot_execution_authorization(config, Path(args.config), "grpo")
        require_ppo_offline_environment()
    else:
        validate_smoke_authorization(config, Path(args.config))
    (git_probe or require_clean_git)()
    (snapshot_probe or require_local_snapshot)()
    if execute_fn is None:
        if is_pilot:
            raise RuntimeError(
                "matched GRPO pilot execution remains disabled until the 16-completion "
                "guarded runtime and grouped artifact path are implemented and reviewed"
            )
        from math_rlvr.training.grpo_runtime import execute_real_smoke

        execute_fn = execute_real_smoke
    result = execute_fn(config)
    if result.get("status") != "success":
        raise RuntimeError(result.get("reason", "guarded GRPO smoke failed"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
