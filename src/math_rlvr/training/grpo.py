"""GRPO-specific training entry point."""

from math_rlvr.training.common import parse_args, preflight, refuse_unimplemented


def main() -> int:
    args = parse_args("GRPO training preflight")
    return refuse_unimplemented(preflight(args.config, "grpo"), args.execute)


if __name__ == "__main__":
    raise SystemExit(main())
