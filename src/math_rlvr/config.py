"""Static configuration validation; never loads a model or initializes CUDA."""

from pathlib import Path
from typing import Any

import yaml

from math_rlvr.prompt import (
    prompt_metadata,
    prompt_version_from_config,
)

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
    expected_prompt_metadata = prompt_metadata(prompt_version_from_config(config))
    for key in ("prompt_version", "prompt_sha256", "renderer_version"):
        if key in config and config[key] != expected_prompt_metadata[key]:
            raise ValueError(f"resolved prompt metadata mismatch: {key}")
    generation = config.get("generation", {})
    completion_length = 128 if "smoke" in config["experiment"]["name"] else 384
    expected = {
        "max_prompt_length": 512,
        "max_completion_length": completion_length,
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
    if algorithm == "grpo" and "smoke" in config["experiment"]["name"]:
        validate_grpo_smoke_budget(config)
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


def validate_grpo_smoke_budget(config: dict[str, Any]) -> None:
    """Fail closed when the smoke YAML's batching and hard budgets disagree."""
    generation = config.get("generation", {})
    training = config.get("training", {})
    data = config.get("data", {})
    budget = config.get("budget", {})
    batch = training.get("per_device_train_batch_size")
    accumulation = training.get("gradient_accumulation_steps")
    generations = generation.get("num_generations")
    generation_batch = generation.get("generation_batch_size")
    completion_length = generation.get("max_completion_length")
    required = (batch, accumulation, generations, generation_batch, completion_length)
    if not all(isinstance(value, int) and value > 0 for value in required):
        raise ValueError("incomplete GRPO smoke batching contract")
    expected = {
        "unique_prompts": generation_batch // generations,
        "total_completions": generation_batch,
        "total_generated_tokens": generation_batch * completion_length,
        "steps_per_generation": generation_batch // batch,
    }
    if generation_batch % batch or generation_batch % generations:
        raise ValueError("GRPO generation batch divisibility contract mismatch")
    if (
        data.get("max_train_samples") != expected["unique_prompts"]
        or accumulation != expected["steps_per_generation"]
        or training.get("max_steps") != 1
        or training.get("num_iterations") != 1
        or budget.get("max_completions") != expected["total_completions"]
        or budget.get("max_generated_tokens") != expected["total_generated_tokens"]
        or budget.get("max_optimizer_steps") != 1
        or budget.get("max_global_steps") != 1
        or training.get("save_strategy") != "steps"
        or training.get("save_steps") != 1
        or training.get("save_only_model") is not True
        or training.get("push_to_hub") is not False
        or training.get("report_to") != []
    ):
        raise ValueError("GRPO smoke hard budget mismatch")
    if "steps_per_generation" in generation or "steps_per_generation" in training:
        raise ValueError("steps_per_generation must be inferred by TRL")
    if config.get("model", {}).get("local_files_only") is not True:
        raise ValueError("GRPO smoke must be local-files-only")


def resolve_grpo_smoke_budget(config: dict[str, Any]) -> dict[str, int]:
    validate_grpo_smoke_budget(config)
    generation = config["generation"]
    training = config["training"]
    return {
        "unique_prompts": config["data"]["max_train_samples"],
        "total_completions": config["budget"]["max_completions"],
        "total_generated_tokens": config["budget"]["max_generated_tokens"],
        "expected_optimizer_updates": training["max_steps"],
        "generation_batch_size": generation["generation_batch_size"],
        "steps_per_generation": generation["generation_batch_size"]
        // training["per_device_train_batch_size"],
    }


def resolve_training_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a copy enriched with derived prompt identity metadata."""
    resolved = dict(config)
    resolved.update(prompt_metadata(prompt_version_from_config(config)))
    return resolved


def validate_runtime_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_relative_to(TEMP_ROOT):
        raise ValueError(f"Runtime artifacts must be under {TEMP_ROOT}: {resolved}")
    return resolved
