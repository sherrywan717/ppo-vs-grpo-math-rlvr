"""Common preflight logic; importing this module never loads a model."""

import argparse
from pathlib import Path
from typing import Any

from math_rlvr.config import load_config, resolve_training_config, validate_training_config
from math_rlvr.prompt import (
    render_candidate_prompt,
)
from math_rlvr.prompt import (
    render_training_prompt as _render_training_prompt,
)

render_training_prompt = _render_training_prompt
render_candidate_training_prompt = render_candidate_prompt


def parse_args(description: str, argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Request guarded execution")
    parser.add_argument(
        "--confirm-single-update",
        action="store_true",
        help="Second confirmation required only for the frozen GRPO smoke",
    )
    return parser.parse_args(argv)


def preflight(config_path: Path, algorithm: str) -> dict[str, Any]:
    config = load_config(config_path)
    validate_training_config(config, algorithm)
    return resolve_training_config(config)


def refuse_unimplemented(config: dict[str, Any], execute: bool) -> int:
    name = config["experiment"]["name"]
    if not execute:
        print(f"Preflight passed for {name}; no training started. Add --execute only when ready.")
        return 0
    raise RuntimeError("Training is disabled in phase 1; implement and review the adapter first")
