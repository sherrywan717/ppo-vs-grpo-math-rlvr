"""Configuration loading and guardrails shared by all entry points."""

from pathlib import Path
from typing import Any

import yaml

TEMP_ROOT = Path("/root/autodl-tmp")


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {config_path}")
    return config


def validate_training_config(config: dict[str, Any], algorithm: str) -> None:
    if config.get("experiment", {}).get("algorithm") != algorithm:
        raise ValueError(f"Config is not for {algorithm}")
    model = config.get("model", {})
    if model.get("dtype") != "bfloat16" or model.get("use_qlora") is not False:
        raise ValueError("This project phase requires BF16 LoRA and forbids QLoRA")
    for section, key in (
        ("data", "max_train_samples"),
        ("generation", "max_new_tokens"),
        ("training", "max_steps"),
        ("training", "save_total_limit"),
        ("budget", "max_gpu_hours"),
        ("budget", "max_estimated_cost_cny"),
    ):
        value = config.get(section, {}).get(key)
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"Missing positive limit: {section}.{key}")


def validate_runtime_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_relative_to(TEMP_ROOT):
        raise ValueError(f"Runtime artifacts must be under {TEMP_ROOT}: {resolved}")
    return resolved

