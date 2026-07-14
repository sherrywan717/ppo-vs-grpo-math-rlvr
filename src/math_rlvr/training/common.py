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
        help="Second confirmation required for a frozen single-update smoke",
    )
    return parser.parse_args(argv)


def preflight(config_path: Path, algorithm: str) -> dict[str, Any]:
    from math_rlvr.training.execution_contract import validated_experiment_scope

    scope = validated_experiment_scope(config_path, algorithm)
    config = load_config(config_path)
    if config.get("pilot", {}).get("family") == "matched_0p5b_v1":
        from math_rlvr.training.pilot import (
            enrich_pilot_config,
            validate_pilot_config_file,
        )

        frozen, contract = validate_pilot_config_file(config_path, algorithm)
        validate_training_config(frozen, algorithm, scope)
        return enrich_pilot_config(frozen, contract, config_path, scope)
    validate_training_config(config, algorithm, scope)
    return resolve_training_config(config, scope)


def refuse_unimplemented(config: dict[str, Any], execute: bool) -> int:
    name = config["experiment"]["name"]
    if not execute:
        print(f"Preflight passed for {name}; no training started. Add --execute only when ready.")
        return 0
    raise RuntimeError("Training is disabled in phase 1; implement and review the adapter first")
