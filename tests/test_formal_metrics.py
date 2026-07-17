from pathlib import Path

import pytest

from math_rlvr.training.formal import validate_formal_config_file
from math_rlvr.training.formal_model_runtime import _normal_metrics
from math_rlvr.training.formal_runtime import formal_run_contract


def _contract(algorithm):
    path = Path(f"configs/formal_1p5b/resolved/{algorithm}_seed_42.json")
    return formal_run_contract(validate_formal_config_file(path, algorithm)[0])


def _records(contract):
    rewards = [0.1] * 4 + [0.0, 0.1, 0.2, 0.3] * 3
    rows = []
    for index, pair_key in enumerate(contract.pair_keys[:16]):
        problem_id, generation = pair_key.rsplit("::generation:", 1)
        within_group = index % 4
        text = ["same", "same", "third", "fourth"][within_group]
        rows.append(
            {
                "problem_id": problem_id,
                "generation_index": int(generation),
                "pair_key": pair_key,
                "raw_completion": text,
                "completion_ids": [10, 11],
                "completion_mask": [1, 1],
                "exact_token_count": 2,
                "eos_reached": index % 2 == 0,
                "truncated": index == 15,
                "scalar_reward": rewards[index],
                "canonical_status": "verified_pass" if index == 0 else "wrong_answer",
                "components": {"valid_answer": 0.1},
            }
        )
    return rows


def test_ppo_metric_schema_preserves_native_entropy_and_nullable_fields():
    contract = _contract("ppo")
    row = {
        "loss/policy_avg": 0.2,
        "loss/value_avg": 0.3,
        "grad_norm": 1.25,
        "policy/entropy_avg": 0.4,
        "policy/approxkl_avg": 0.01,
        "policy/clipfrac_avg": 0.125,
        "val/ratio": 1.01,
        "val/ratio_var": float("nan"),
        "lr": 1e-5,
    }
    metric = _normal_metrics([row], _records(contract), contract)[0]
    assert metric["total_loss"] == pytest.approx(0.23)
    assert metric["policy_loss"] == 0.2
    assert metric["value_loss"] == 0.3
    assert metric["kl"] == 0.01
    assert metric["clip_fraction"] == 0.125
    assert metric["ratio_mean"] == 1.01
    assert metric["ratio_variance"] is None
    assert metric["ratio_variance_available"] is False
    assert "non-finite" in metric["ratio_variance_reason"]
    assert metric["policy_entropy_mean"] == 0.4
    assert metric["entropy_raw_metric_key"] == "policy/entropy_avg"
    assert metric["entropy_excludes_prompt"] is True
    assert metric["entropy_excludes_pad"] is False
    assert metric["response_token_entropy_mean"] is None
    assert metric["response_token_entropy_mean_available"] is False
    assert metric["entropy_extra_model_forward"] is False
    assert metric["entropy_full_logits_persisted"] is False
    assert metric["completion_duplicate_rate"] == 0.25
    assert metric["unique_completion_rate"] == 0.75
    assert metric["completion_length_std"] == 0.0
    assert metric["eos_rate"] == 0.5
    assert metric["truncation_rate"] == 1 / 16


def test_grpo_metric_schema_records_groups_and_masked_native_entropy():
    contract = _contract("grpo")
    row = {
        "loss": 0.2,
        "grad_norm": 1.25,
        "entropy": 0.35,
        "clip_ratio/region_mean": 0.05,
        "learning_rate": 1e-5,
    }
    metric = _normal_metrics([row], _records(contract), contract)[0]
    assert metric["policy_loss"] is None
    assert metric["value_loss"] is None
    assert metric["policy_entropy_mean"] == 0.35
    assert metric["entropy_raw_metric_key"] == "entropy"
    assert metric["entropy_excludes_pad"] is True
    assert metric["entropy_excludes_eos"] is False
    assert metric["response_token_entropy_mean"] is None
    assert metric["zero_advantage_group_count"] == 1
    assert metric["zero_advantage_fraction"] == 0.25
    assert len(metric["group_rewards"]) == 4
    assert metric["clip_fraction"] == 0.05
    assert metric["kl"] is None
    assert "beta=0.0" in metric["kl_unavailable_reason"]


def test_missing_native_entropy_is_null_not_zero():
    contract = _contract("grpo")
    row = {"loss": 0.2, "grad_norm": 1.25, "learning_rate": 1e-5}
    metric = _normal_metrics([row], _records(contract), contract)[0]
    assert metric["entropy"] is None
    assert metric["policy_entropy_mean"] is None
    assert metric["policy_entropy_mean_available"] is False
    assert metric["policy_entropy_mean_reason"]
