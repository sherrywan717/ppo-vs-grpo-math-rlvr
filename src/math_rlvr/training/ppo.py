"""PPO-specific training entry point."""

from math_rlvr.training.common import (
    parse_args,
    preflight,
    refuse_unimplemented,
    render_training_prompt,
)

__all__ = ["main", "render_training_prompt"]


def main() -> int:
    args = parse_args("PPO training preflight")
    return refuse_unimplemented(preflight(args.config, "ppo"), args.execute)


if __name__ == "__main__":
    raise SystemExit(main())
