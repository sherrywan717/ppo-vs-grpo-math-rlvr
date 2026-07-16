import copy
from pathlib import Path

import pytest
from trl.trainer.utils import RepeatSampler

from math_rlvr.rewards.result import RewardPolicyError
from math_rlvr.training.builders import grpo_config, ppo_config
from math_rlvr.training.common import preflight
from math_rlvr.training.formal import (
    FORMAL_ACTIVE_SEEDS,
    FORMAL_SEEDS,
    formal_pair_keys,
    formal_reserved_configs,
    formal_run_order,
    formal_training_schedule,
    validate_active_suite,
    validate_formal_config_content,
    validate_formal_config_file,
)
from math_rlvr.training.formal_model import derive_static_parameter_contract


@pytest.mark.parametrize("algorithm", ["ppo", "grpo"])
@pytest.mark.parametrize("seed", FORMAL_SEEDS)
def test_six_formal_configs_are_exact_cpu_dry_runs(algorithm, seed, tmp_path):
    path = Path(f"configs/formal_1p5b/resolved/{algorithm}_seed_{seed}.json")
    config, contract = validate_formal_config_file(path, algorithm)
    assert preflight(path, algorithm) == config
    assert contract["outer_updates"] == 32
    assert contract["optimizer_steps"] == contract["global_steps"] == 32
    assert contract["total_completions"] == 512
    assert contract["total_generated_tokens"] == 131_072
    assert contract["checkpoint_steps"] == [8, 16, 24, 32]
    args = (
        ppo_config(config, tmp_path / "ppo", cpu_only=True)
        if algorithm == "ppo"
        else grpo_config(config, tmp_path / "grpo", cpu_only=True)
    )
    assert args.max_steps == 32
    assert args.per_device_train_batch_size == 4
    assert args.gradient_accumulation_steps == 4
    assert args.save_steps == 8 and args.save_total_limit == 4


def test_ppo_trl_024_budget_is_32_matched_updates(tmp_path):
    config, contract = validate_formal_config_file(
        Path("configs/formal_1p5b/resolved/ppo_seed_42.json"), "ppo"
    )
    args = ppo_config(config, tmp_path, cpu_only=True)
    assert args.total_episodes == 512
    assert args.num_ppo_epochs == args.num_mini_batches == 1
    assert args.local_rollout_forward_batch_size == 4
    assert not hasattr(args, "num_generations")
    assert contract["rollout_batch_size"] == 16
    assert contract["microbatches_per_minibatch"] == 4
    assert contract["outer_updates"] == 512 // 16 == 32


def test_grpo_trl_024_budget_and_sequential_groups(tmp_path):
    config, contract = validate_formal_config_file(
        Path("configs/formal_1p5b/resolved/grpo_seed_42.json"), "grpo"
    )
    args = grpo_config(config, tmp_path, cpu_only=True)
    assert args.generation_batch_size == 16
    assert args.num_generations == 4
    assert args.steps_per_generation == 4
    assert args.num_iterations == 1
    assert args.shuffle_dataset is False
    assert args.dataloader_drop_last is True
    assert args.dataloader_num_workers == 0
    assert contract["training_microsteps"] == 128
    sampler = RepeatSampler(
        range(128),
        mini_repeat_count=4,
        batch_size=4,
        repeat_count=4,
        shuffle=args.shuffle_dataset,
        seed=args.seed,
    )
    indices = list(sampler)
    assert indices[:16] == [index for index in range(4) for _ in range(4)]
    assert indices[16:64] == indices[:16] * 3
    assert len(indices) == 2048


def test_formal_schedule_is_exact_two_plus_two_and_complete():
    schedule = formal_training_schedule()
    ids = schedule["ordered_problem_ids"]
    assert len(ids) == len(set(ids)) == 128
    for offset in range(0, 128, 4):
        group = ids[offset : offset + 4]
        assert [problem_id.split(":", 1)[0] for problem_id in group] == [
            "gsm8k",
            "gsm8k",
            "math",
            "math",
        ]
    pair_keys = formal_pair_keys()
    assert len(pair_keys) == len(set(pair_keys)) == 512
    assert all(
        pair_keys[index * 4 + generation].endswith(f"::generation:{generation}")
        for index in range(128)
        for generation in range(4)
    )


def test_formal_mutations_fail_closed():
    config, _ = validate_formal_config_file(
        Path("configs/formal_1p5b/resolved/ppo_seed_42.json"), "ppo"
    )
    for section, key, value in (
        ("training", "total_episodes", 511),
        ("training", "num_ppo_epochs", 2),
        ("budget", "max_completions", 513),
        ("budget", "max_generated_tokens", 131_073),
        ("data", "schedule_sha256", "0" * 64),
    ):
        mutated = copy.deepcopy(config)
        mutated[section][key] = value
        with pytest.raises(ValueError, match="formal"):
            validate_formal_config_content(mutated, "ppo")


def test_active_suite_hashes_four_configs_and_preserves_reserved_descriptors():
    suite = validate_active_suite()
    assert suite["active_suite_sha256"] == (
        "f6de8c555a70837d08c1e34e13a738a21ce247b3a09531df6244a2f1d3ef53bd"
    )
    assert len(suite["active_training_runs"]) == 4
    assert all(row["seed"] in FORMAL_ACTIVE_SEEDS for row in suite["active_training_runs"])
    assert [row["status"] for row in suite["reserved_configs"]] == [
        "reserved_not_scheduled",
        "reserved_not_scheduled",
    ]
    assert {row["seed"] for row in suite["reserved_configs"]} == {2026}


def test_fixed_four_run_formal_order_has_no_seed_override():
    assert [(row["seed"], row["algorithm"]) for row in formal_run_order()] == [
        (42, "ppo"),
        (42, "grpo"),
        (123, "grpo"),
        (123, "ppo"),
    ]
    assert all(row["automatic_retries"] == 0 for row in formal_run_order())
    assert {row["seed"] for row in formal_run_order()} == set(FORMAL_ACTIVE_SEEDS)
    assert {row["seed"] for row in formal_reserved_configs()} == {2026}
    assert all(row["status"] == "reserved_not_scheduled" for row in formal_reserved_configs())


def test_static_model_roles_and_parameter_counts_need_no_model_load():
    contract = derive_static_parameter_contract()
    assert contract["policy_lora_trainable_parameters"] == 4_358_144
    assert contract["value_lora_trainable_parameters"] == 1_089_536
    assert contract["value_scalar_head_trainable_parameters"] == 1_537
    assert contract["ppo_value_trainable_parameters"] == 1_091_073
    assert contract["ppo_optimizer_trainable_parameters"] == 5_449_217
    assert contract["grpo_optimizer_trainable_parameters"] == 4_358_144
    assert contract["policy_value_trainable_overlap"] == 0
    assert contract["reference_trainable_parameters"] == 0
    assert contract["reward_trainable_parameters"] == 0
    assert contract["parameter_counts_require_model_load"] is False


def test_resolved_formal_identity_mutations_fail_closed():
    config, _ = validate_formal_config_file(
        Path("configs/formal_1p5b/resolved/ppo_seed_42.json"), "ppo"
    )
    mutated = copy.deepcopy(config)
    mutated["prompt_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="prompt metadata|formal prompt"):
        validate_formal_config_content(mutated, "ppo")
    mutated = copy.deepcopy(config)
    mutated["reward_policy_sha256"] = "0" * 64
    with pytest.raises(RewardPolicyError, match="reward metadata"):
        validate_formal_config_content(mutated, "ppo")
    mutated = copy.deepcopy(config)
    mutated["verifier_contract"]["contract_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="parser/verifier identity"):
        validate_formal_config_content(mutated, "ppo")
