import copy
from pathlib import Path

import pytest

from math_rlvr.config import (
    load_config,
    resolve_grpo_smoke_budget,
    resolve_ppo_smoke_contract,
    resolve_training_config,
    validate_runtime_path,
    validate_training_config,
)


@pytest.mark.parametrize(
    "path,algorithm",
    [
        ("configs/smoke/ppo.yaml", "ppo"),
        ("configs/smoke/grpo.yaml", "grpo"),
        ("configs/main/ppo.yaml", "ppo"),
        ("configs/main/grpo.yaml", "grpo"),
    ],
)
def test_training_configs_are_bounded(path: str, algorithm: str) -> None:
    validate_training_config(load_config(path), algorithm)


def test_runtime_paths_stay_on_temp_disk() -> None:
    assert validate_runtime_path("/root/autodl-tmp/math-rlvr-outputs").is_absolute()
    with pytest.raises(ValueError):
        validate_runtime_path(Path.home() / ".cache" / "models")


def test_grpo_smoke_budget_is_single_consistent_contract():
    config = load_config("configs/smoke/grpo.yaml")
    assert resolve_grpo_smoke_budget(config) == {
        "unique_prompts": 2,
        "total_completions": 8,
        "total_generated_tokens": 1024,
        "expected_optimizer_updates": 1,
        "generation_batch_size": 8,
        "steps_per_generation": 4,
    }


@pytest.mark.parametrize(
    ("section", "key", "bad_value"),
    [
        ("training", "max_steps", 2),
        ("budget", "max_completions", 64),
        ("budget", "max_generated_tokens", 8192),
        ("training", "gradient_accumulation_steps", 1),
        ("generation", "generation_batch_size", 4),
    ],
)
def test_grpo_smoke_budget_conflicts_fail_closed(section, key, bad_value):
    config = copy.deepcopy(load_config("configs/smoke/grpo.yaml"))
    config[section][key] = bad_value
    with pytest.raises(ValueError, match="GRPO smoke|generation batch"):
        validate_training_config(config, "grpo")


def test_grpo_rejects_explicit_steps_per_generation():
    config = copy.deepcopy(load_config("configs/smoke/grpo.yaml"))
    config["training"]["steps_per_generation"] = 4
    with pytest.raises(ValueError, match="must be inferred"):
        validate_training_config(config, "grpo")


def test_ppo_smoke_contract_is_one_update_and_num_generations_is_explicitly_ignored():
    config = load_config("configs/smoke/ppo.yaml")
    contract = resolve_ppo_smoke_contract(config)
    assert contract["total_episodes"] == contract["unique_prompts"] == 4
    assert contract["rollout_batch_size"] == 4
    assert contract["responses_per_prompt"] == 1
    assert contract["total_completions"] == 4
    assert contract["total_generated_tokens"] == 512
    assert contract["outer_updates"] == contract["total_optimizer_steps"] == 1
    assert contract["global_steps"] == contract["authoritative_checkpoints"] == 1
    assert contract["configured_num_generations"] == 4
    assert contract["num_generations_effective_for_ppo"] == 1
    assert "num_generations" in contract["ignored_generation_fields"]


@pytest.mark.parametrize(
    ("section", "key", "bad_value"),
    [
        ("training", "total_episodes", 8),
        ("training", "per_device_train_batch_size", 2),
        ("training", "num_ppo_epochs", 2),
        ("training", "num_mini_batches", 2),
        ("training", "local_rollout_forward_batch_size", 2),
        ("training", "save_total_limit", 2),
        ("budget", "max_completions", 16),
        ("budget", "max_generated_tokens", 2048),
        ("budget", "max_optimizer_steps", 2),
        ("budget", "max_global_steps", 2),
        ("budget", "max_wall_time_seconds", 2400),
        ("budget", "max_vram_gib", 28),
        ("budget", "max_gpu_hours", 1),
        ("budget", "max_estimated_cost_cny", 8.88),
    ],
)
def test_ppo_smoke_budget_conflicts_fail_closed(section, key, bad_value):
    config = copy.deepcopy(load_config("configs/smoke/ppo.yaml"))
    config[section][key] = bad_value
    with pytest.raises(ValueError, match="PPO smoke"):
        validate_training_config(config, "ppo")


def test_smoke_reward_selector_is_identical_and_resolved():
    grpo = resolve_training_config(load_config("configs/smoke/grpo.yaml"))
    ppo = resolve_training_config(load_config("configs/smoke/ppo.yaml"))
    assert grpo["reward"] == ppo["reward"] == {"policy": "shaped_v2_staged"}
    assert grpo["reward_policy_version"] == ppo["reward_policy_version"]
    assert grpo["reward_component_weights"] == ppo["reward_component_weights"]
    assert grpo["reward_policy_sha256"] == ppo["reward_policy_sha256"]


def test_smoke_reward_selector_mismatch_fails_closed():
    config = copy.deepcopy(load_config("configs/smoke/grpo.yaml"))
    config["reward"]["policy"] = "shaped_v1_legacy"
    with pytest.raises(ValueError, match="reward policy"):
        validate_training_config(config, "grpo")
