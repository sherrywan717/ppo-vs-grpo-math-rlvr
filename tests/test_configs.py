import copy
from pathlib import Path

import pytest

from math_rlvr.config import (
    load_config,
    resolve_grpo_smoke_budget,
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


def test_ppo_smoke_contract_is_unchanged():
    config = load_config("configs/smoke/ppo.yaml")
    assert config["training"] == {"max_steps": 2, "save_total_limit": 1}
    assert config["budget"]["max_completions"] == 64
    assert config["budget"]["max_generated_tokens"] == 8192


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
