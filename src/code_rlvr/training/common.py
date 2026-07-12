"""Common preflight logic; importing this module never loads a model."""

import argparse
from pathlib import Path
from typing import Any

from code_rlvr.config import load_config, validate_training_config


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Authorize training when implemented")
    return parser.parse_args()


def preflight(config_path: Path, algorithm: str) -> dict[str, Any]:
    config = load_config(config_path)
    validate_training_config(config, algorithm)
    return config


def refuse_unimplemented(config: dict[str, Any], execute: bool) -> int:
    name = config["experiment"]["name"]
    if not execute:
        print(f"Preflight passed for {name}; no training started. Add --execute only when ready.")
        return 0
    raise RuntimeError("Training is disabled in phase 1; implement and review the adapter first")

