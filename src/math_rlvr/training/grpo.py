"""Fail-closed GRPO entry point; real imports occur only after dual confirmation."""

from pathlib import Path

from math_rlvr.training.common import (
    parse_args,
    preflight,
    render_candidate_training_prompt,
    render_training_prompt,
)

__all__ = ["main", "render_candidate_training_prompt", "render_training_prompt"]


def main(
    argv=None, execute_fn=None, git_probe=None, snapshot_probe=None, offline_probe=None
) -> int:
    args = parse_args("GRPO training preflight", argv)
    config = preflight(args.config, "grpo")
    if not args.execute:
        print(f"Preflight passed for {config['experiment']['name']}; no training started.")
        return 0
    if config.get("formal", {}).get("family") == "formal_1p5b_v1":
        if (
            not args.confirm_formal_grpo
            or args.confirm_formal_ppo
            or args.confirm_single_update
        ):
            raise RuntimeError(
                "formal GRPO requires --execute --confirm-formal-grpo only"
            )
        from math_rlvr.training.formal_cli import (
            require_formal_local_snapshot,
            require_formal_offline_environment,
            validate_formal_resume_authorization,
            validate_formal_training_authorization,
        )
        from math_rlvr.training.formal_runtime import (
            prepare_formal_runtime_prompt_preflight,
        )
        from math_rlvr.training.guarded_grpo import require_clean_git

        authorization = validate_formal_training_authorization(config, args.config, "grpo")
        validated_resume = (
            validate_formal_resume_authorization(
                config, args.resume_checkpoint, "grpo"
            )
            if args.resume_checkpoint is not None
            else None
        )
        (git_probe or require_clean_git)()
        (offline_probe or require_formal_offline_environment)()
        model_source = (snapshot_probe or require_formal_local_snapshot)()
        prompt_preflight = prepare_formal_runtime_prompt_preflight(config, "grpo")
        if execute_fn is None:
            from math_rlvr.training.formal_model_runtime import execute_real_formal_grpo

            execute_fn = execute_real_formal_grpo
        result = execute_fn(
            config,
            model_source=model_source,
            prompt_preflight=prompt_preflight,
            authorization=authorization,
            resume_checkpoint=args.resume_checkpoint,
            validated_resume=validated_resume,
        )
        if result.get("status") != "success":
            raise RuntimeError(result.get("reason", "formal GRPO execution failed"))
        return 0
    if args.confirm_formal_ppo or args.confirm_formal_grpo or args.resume_checkpoint:
        raise RuntimeError("formal-only flags cannot authorize smoke or pilot GRPO")
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
        from math_rlvr.training.grpo_runtime import execute_real_grpo

        execute_fn = execute_real_grpo
    result = execute_fn(config)
    if result.get("status") != "success":
        raise RuntimeError(result.get("reason", "guarded GRPO smoke failed"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
