"""Static configuration validation; never loads a model or initializes CUDA."""

from pathlib import Path
from typing import Any

import yaml

from math_rlvr.contracts import parser_verifier_metadata
from math_rlvr.prompt import (
    prompt_metadata,
    prompt_version_from_config,
)
from math_rlvr.rewards.staged import (
    STAGED_REWARD_VERSION,
    reward_metadata_from_config,
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


def validate_training_config(config: dict[str, Any], algorithm: str, scope=None) -> None:
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
    is_pilot = config.get("pilot", {}).get("family") == "matched_0p5b_v1"
    is_formal = config.get("formal", {}).get("family") == "formal_1p5b_v1"
    is_grpo_v2 = config.get("grpo_v2", {}).get("family") == "grpo_v2_seed42_v1"
    if scope is None and config.get("validated_experiment_scope") is not None:
        from math_rlvr.training.execution_contract import validated_scope_from_config

        scope = validated_scope_from_config(config, algorithm)
    if scope is None:
        selected_prompt = config.get("prompt", {}).get("version") or "prompt_v0_grpo_smoke"
    else:
        selected_prompt = prompt_version_from_config(config, scope.scope)
    expected_prompt_metadata = prompt_metadata(selected_prompt)
    for key in ("prompt_version", "prompt_sha256", "renderer_version"):
        if key in config and config[key] != expected_prompt_metadata[key]:
            raise ValueError(f"resolved prompt metadata mismatch: {key}")
    generation = config.get("generation", {})
    is_smoke = (scope is not None and scope.scope.value == "stage_d_smoke") or (
        scope is None and model.get("name_or_path") == SMOKE_MODEL and not is_pilot
    )
    completion_length = 128 if is_smoke or is_pilot else 256 if is_formal or is_grpo_v2 else 384
    expected = {
        "max_prompt_length": 832 if is_formal or is_grpo_v2 else 512,
        "max_completion_length": completion_length,
        "temperature": 0.8,
        "top_p": 0.95,
    }
    if algorithm == "grpo" or not (is_pilot or is_formal):
        expected["num_generations"] = 4
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
    reward_selector = config.get("reward", {}).get("policy")
    if (is_smoke or is_pilot) and reward_selector != STAGED_REWARD_VERSION:
        raise ValueError("bounded 0.5B reward policy must be shaped_v2_staged")
    if not (is_smoke or is_pilot) and reward_selector == STAGED_REWARD_VERSION:
        raise ValueError("staged smoke reward must not activate main/formal config")
    if algorithm == "grpo" and is_smoke:
        validate_grpo_smoke_budget(config)
    if algorithm == "ppo" and is_smoke:
        validate_ppo_smoke_budget(config)
    if is_pilot:
        from math_rlvr.training.pilot import validate_pilot_config_content

        validate_pilot_config_content(config, algorithm)
    if is_formal:
        from math_rlvr.training.formal import validate_formal_config_content

        validate_formal_config_content(config, algorithm)
    if is_grpo_v2:
        from math_rlvr.training.grpo_v2_runtime import validate_normalized_training_config

        validate_normalized_training_config(config)
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


def resolve_ppo_smoke_contract(config: dict[str, Any]) -> dict[str, Any]:
    """Derive the single-device TRL 0.24.0 PPO loop contract."""
    training = config.get("training", {})
    generation = config.get("generation", {})
    per_device = training.get("per_device_train_batch_size")
    accumulation = training.get("gradient_accumulation_steps")
    epochs = training.get("num_ppo_epochs")
    minibatches = training.get("num_mini_batches")
    episodes = training.get("total_episodes")
    response_length = generation.get("max_new_tokens")
    required = (per_device, accumulation, epochs, minibatches, episodes, response_length)
    if not all(
        isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in required
    ):
        raise ValueError("incomplete PPO smoke loop contract")
    local_batch = per_device * accumulation
    if local_batch % minibatches:
        raise ValueError("PPO rollout batch must be divisible by minibatches")
    local_minibatch = local_batch // minibatches
    if local_minibatch % per_device:
        raise ValueError("PPO smoke minibatch must be divisible by micro-batch")
    microbatches = local_minibatch // per_device
    outer_updates = (episodes + local_batch - 1) // local_batch
    optimizer_steps_per_update = epochs * minibatches * microbatches // accumulation
    if epochs * minibatches * microbatches % accumulation:
        raise ValueError("PPO optimizer-step accumulation contract mismatch")
    return {
        "selected_dataset_records": config["data"]["max_train_samples"],
        "unique_prompts": episodes,
        "responses_per_prompt": 1,
        "rollout_batch_size": local_batch,
        "micro_batch_size": per_device,
        "gradient_accumulation_steps": accumulation,
        "num_ppo_epochs": epochs,
        "num_mini_batches": minibatches,
        "microbatches_per_minibatch": microbatches,
        "total_episodes": episodes,
        "outer_updates": outer_updates,
        "optimizer_steps_per_update": optimizer_steps_per_update,
        "total_optimizer_steps": outer_updates * optimizer_steps_per_update,
        "global_steps": outer_updates,
        "total_completions": episodes,
        "max_completion_length": response_length,
        "total_generated_tokens": episodes * response_length,
        "authoritative_checkpoints": outer_updates,
        "configured_num_generations": generation.get("num_generations"),
        "num_generations_effective_for_ppo": 1,
        "ignored_generation_fields": {
            "num_generations": (
                "TRL 0.24.0 PPO samples one response for each rollout dataset row; "
                "this shared-schema field is not passed to PPOConfig"
            )
        },
        "configured_top_p": generation.get("top_p"),
        "effective_top_p": generation.get("top_p"),
    }


def validate_ppo_smoke_budget(config: dict[str, Any]) -> None:
    """Fail closed unless YAML and the TRL-derived one-update contract agree."""
    model = config.get("model", {})
    training = config.get("training", {})
    budget = config.get("budget", {})
    generation = config.get("generation", {})
    contract = resolve_ppo_smoke_contract(config)
    expected = {
        "selected_dataset_records": 4,
        "unique_prompts": 4,
        "responses_per_prompt": 1,
        "rollout_batch_size": 4,
        "micro_batch_size": 4,
        "gradient_accumulation_steps": 1,
        "num_ppo_epochs": 1,
        "num_mini_batches": 1,
        "microbatches_per_minibatch": 1,
        "total_episodes": 4,
        "outer_updates": 1,
        "optimizer_steps_per_update": 1,
        "total_optimizer_steps": 1,
        "global_steps": 1,
        "total_completions": 4,
        "max_completion_length": 128,
        "total_generated_tokens": 512,
        "authoritative_checkpoints": 1,
        "configured_num_generations": 4,
        "num_generations_effective_for_ppo": 1,
        "ignored_generation_fields": {
            "num_generations": (
                "TRL 0.24.0 PPO samples one response for each rollout dataset row; "
                "this shared-schema field is not passed to PPOConfig"
            )
        },
        "configured_top_p": 0.95,
        "effective_top_p": 0.95,
    }
    if contract != expected:
        raise ValueError("derived PPO smoke single-update contract mismatch")
    if (
        model.get("revision") != "7ae557604adf67be50417f59c2c2f167def9a775"
        or model.get("local_files_only") is not True
        or training.get("max_steps") != 1
        or training.get("local_rollout_forward_batch_size") != 4
        or training.get("save_strategy") != "steps"
        or training.get("save_steps") != 1
        or training.get("save_total_limit") != 1
        or training.get("save_only_model") is not True
        or training.get("push_to_hub") is not False
        or training.get("report_to") != []
        or generation.get("max_completion_length") != 128
        or generation.get("max_new_tokens") != 128
        or budget.get("max_completions") != 4
        or budget.get("max_generated_tokens") != 512
        or budget.get("max_update_steps") != 1
        or budget.get("max_optimizer_steps") != 1
        or budget.get("max_global_steps") != 1
        or budget.get("max_ppo_epochs") != 1
        or budget.get("max_minibatches") != 1
        or budget.get("max_wall_time_seconds") != 1200
        or budget.get("max_vram_gib") != 14
        or budget.get("max_gpu_hours") != 0.3333333333
        or budget.get("max_estimated_cost_cny") != 2.96
        or budget.get("gpu_hour_price_cny") != 8.88
    ):
        raise ValueError("PPO smoke YAML and hard budgets disagree")


def resolve_training_config(config: dict[str, Any], scope) -> dict[str, Any]:
    """Return a copy enriched from one exact path/SHA-validated scope."""
    from math_rlvr.training.execution_contract import ValidatedExperimentScope

    if not isinstance(scope, ValidatedExperimentScope):
        raise ValueError("training config resolution requires validated experiment scope")
    resolved = dict(config)
    prompt_version = prompt_version_from_config(config, scope.scope)
    resolved.update(prompt_metadata(prompt_version))
    resolved["validated_experiment_scope"] = scope.to_dict()
    resolved["resolved_config_path"] = scope.config_path
    resolved["resolved_config_sha256"] = scope.config_sha256
    resolved.update(reward_metadata_from_config(config))
    if config.get("pilot", {}).get("family") == "matched_0p5b_v1":
        resolved.update(parser_verifier_metadata())
    return resolved


def validate_runtime_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_relative_to(TEMP_ROOT):
        raise ValueError(f"Runtime artifacts must be under {TEMP_ROOT}: {resolved}")
    return resolved
