"""Static configuration validation; never loads a model or initializes CUDA."""

from pathlib import Path
from typing import Any

import yaml

TEMP_ROOT = Path("/root/autodl-tmp")
POLICY_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SMOKE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
POLICY_LORA = {
    "rank": 16,
    "alpha": 32,
    "dropout": 0,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
}


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration must be a mapping")
    return config


def validate_training_config(config: dict[str, Any], algorithm: str) -> None:
    if config.get("experiment", {}).get("algorithm") != algorithm:
        raise ValueError(f"Config is not for {algorithm}")
    model = config.get("model", {})
    if model.get("name_or_path") not in {POLICY_MODEL, SMOKE_MODEL}:
        raise ValueError("unexpected model checkpoint")
    if model.get("dtype") != "bfloat16" or model.get("use_qlora") is not False:
        raise ValueError("BF16 LoRA required; QLoRA forbidden")
    if model.get("gradient_checkpointing") is not True:
        raise ValueError("gradient checkpointing required")
    if config.get("lora") != POLICY_LORA:
        raise ValueError("policy LoRA contract mismatch")
    generation = config.get("generation", {})
    expected = {
        "max_prompt_length": 512,
        "max_completion_length": 384,
        "temperature": 0.8,
        "top_p": 0.95,
        "num_generations": 4,
    }
    if any(generation.get(key) != value for key, value in expected.items()):
        raise ValueError("generation contract mismatch")
    for section, key in (
        ("data", "max_train_samples"),
        ("training", "max_steps"),
        ("training", "save_total_limit"),
        ("budget", "max_completions"),
        ("budget", "max_generated_tokens"),
        ("budget", "max_wall_time_seconds"),
        ("budget", "max_gpu_hours"),
        ("budget", "max_estimated_cost_cny"),
    ):
        value = config.get(section, {}).get(key)
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"Missing positive limit: {section}.{key}")
    if algorithm == "ppo":
        value = config.get("value_model", {})
        required = {
            "base_checkpoint": model["name_or_path"],
            "architecture": "AutoModelForSequenceClassification",
            "num_labels": 1,
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_target_modules": ["q_proj", "v_proj"],
            "trainable_score_head": True,
        }
        if value != required:
            raise ValueError("PPO value model contract mismatch")


def validate_runtime_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_relative_to(TEMP_ROOT):
        raise ValueError(f"Runtime artifacts must be under {TEMP_ROOT}: {resolved}")
    return resolved
