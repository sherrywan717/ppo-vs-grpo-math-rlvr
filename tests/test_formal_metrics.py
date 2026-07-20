from pathlib import Path

import pytest

from math_rlvr.training.formal import validate_formal_config_file
from math_rlvr.training.formal_model_runtime import _normal_metrics
from math_rlvr.training.formal_runtime import (
    FormalProgressGuard,
    FormalRuntimeError,
    formal_run_contract,
    formal_valid_answer_metric,
)


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
                "valid_answer_component": 0.1,
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
    assert metric["grad_norm"] == 1.25
    assert metric["grad_norm_available"] is True
    assert metric["grad_norm_raw_metric_key"] == "grad_norm"
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


@pytest.mark.parametrize(
    ("status", "component", "expected"),
    [
        ("verified_pass", 0.1, 1.0),
        ("wrong_answer", 0.1, 1.0),
        ("format_error", 0.0, 0.0),
        ("format_error", 0.1, 1.0),
        ("parse_error", 0.0, 0.0),
        ("invalid_expression", 0.0, 0.0),
        ("invalid_number_usage", 0.0, 0.0),
    ],
)
def test_valid_answer_metric_uses_flat_reward_component(status, component, expected):
    metric = formal_valid_answer_metric(
        [{"canonical_status": status, "valid_answer_component": component}]
    )
    assert metric["valid_answer_rate"] == expected
    assert metric["valid_answer_rate_available"] is True
    assert metric["valid_answer_rate_numerator"] == int(expected)
    assert metric["valid_answer_rate_denominator"] == 1
    assert metric["valid_answer_rate_raw_source_field"] == "valid_answer_component"


def test_valid_answer_metric_mixed_and_all_invalid_batches():
    mixed = formal_valid_answer_metric(
        [
            {"valid_answer_component": 0.1},
            {"valid_answer_component": 0.0},
            {"valid_answer_component": 0.1},
            {"valid_answer_component": 0.0},
        ]
    )
    invalid = formal_valid_answer_metric(
        [{"valid_answer_component": 0.0}, {"valid_answer_component": 0.0}]
    )
    assert mixed["valid_answer_rate"] == 0.5
    assert mixed["valid_answer_rate_numerator"] == 2
    assert invalid["valid_answer_rate"] == 0.0
    assert invalid["valid_answer_rate_available"] is True


def test_valid_answer_metric_zero_denominator_and_missing_field_are_unavailable():
    empty = formal_valid_answer_metric([])
    missing = formal_valid_answer_metric([{"canonical_status": "wrong_answer"}])
    assert empty["valid_answer_rate"] is None
    assert empty["valid_answer_rate_available"] is False
    assert empty["valid_answer_rate_reason"] == "zero_denominator"
    assert missing["valid_answer_rate"] is None
    assert missing["valid_answer_rate_available"] is False
    assert missing["valid_answer_rate_reason"] == "valid_answer_component_missing"


def test_ppo_and_grpo_use_identical_valid_answer_mapping_without_training_changes():
    trainer_rows = {
        "ppo": {"loss/policy_avg": 0.2, "loss/value_avg": 0.3, "lr": 1e-5},
        "grpo": {"loss": 0.23, "learning_rate": 1e-5},
    }
    metrics = {}
    for algorithm in ("ppo", "grpo"):
        contract = _contract(algorithm)
        records = _records(contract)
        rewards_before = [row["scalar_reward"] for row in records]
        metrics[algorithm] = _normal_metrics(
            [trainer_rows[algorithm]], records, contract
        )[0]
        assert [row["scalar_reward"] for row in records] == rewards_before
    assert metrics["ppo"]["valid_answer_rate"] == metrics["grpo"]["valid_answer_rate"]
    assert metrics["ppo"]["valid_answer_rate_definition_version"] == metrics["grpo"][
        "valid_answer_rate_definition_version"
    ]
    assert metrics["ppo"]["total_loss"] == metrics["grpo"]["total_loss"] == 0.23
    assert metrics["ppo"]["advantage_mean"] is metrics["grpo"]["advantage_mean"] is None


def test_valid_answer_aggregate_mismatch_fails_runtime_finalization():
    contract = _contract("ppo")
    records = _records(contract)
    for index, row in enumerate(records):
        row["update"] = 1
        row["pair_key"] = contract.pair_keys[index]
    metric = _normal_metrics(
        [{"loss/policy_avg": 0.2, "loss/value_avg": 0.3, "lr": 1e-5}],
        records,
        contract,
    )[0]
    metric["valid_answer_rate"] = 0.0
    with pytest.raises(
        FormalRuntimeError, match="valid-answer aggregate contradicts completion evidence"
    ):
        FormalProgressGuard(contract, run_id="mismatch").record_update(
            update=1,
            completion_rows=records,
            metrics=metric,
            optimizer_step=1,
            global_step=1,
        )


def test_missing_native_entropy_is_null_not_zero():
    contract = _contract("grpo")
    row = {"loss": 0.2, "grad_norm": 1.25, "learning_rate": 1e-5}
    metric = _normal_metrics([row], _records(contract), contract)[0]
    assert metric["entropy"] is None
    assert metric["policy_entropy_mean"] is None
    assert metric["policy_entropy_mean_available"] is False
    assert metric["policy_entropy_mean_reason"]


def test_missing_grad_norm_is_null_unavailable_and_nonblocking():
    contract = _contract("ppo")
    row = {
        "loss/policy_avg": 0.2,
        "loss/value_avg": 0.3,
        "policy/entropy_avg": 0.4,
        "lr": 1e-5,
    }
    metric = _normal_metrics([row], _records(contract), contract)[0]
    for name in ("grad_norm", "policy_grad_norm", "value_grad_norm"):
        assert metric[name] is None
        assert metric[f"{name}_available"] is False
        assert metric[f"{name}_reason"]
        assert metric[f"{name}_raw_metric_key"] is None


def test_finite_role_grad_norms_preserve_raw_keys():
    contract = _contract("ppo")
    row = {
        "loss/policy_avg": 0.2,
        "loss/value_avg": 0.3,
        "policy_grad_norm": 1.5,
        "train/value_grad_norm": 2.5,
        "lr": 1e-5,
    }
    metric = _normal_metrics([row], _records(contract), contract)[0]
    assert metric["policy_grad_norm"] == 1.5
    assert metric["policy_grad_norm_available"] is True
    assert metric["policy_grad_norm_raw_metric_key"] == "policy_grad_norm"
    assert metric["value_grad_norm"] == 2.5
    assert metric["value_grad_norm_available"] is True
    assert metric["value_grad_norm_raw_metric_key"] == "train/value_grad_norm"


@pytest.mark.parametrize("key", ["grad_norm", "policy_grad_norm", "value_grad_norm"])
def test_nonfinite_grad_norm_fails_closed(key):
    contract = _contract("ppo")
    row = {
        "loss/policy_avg": 0.2,
        "loss/value_avg": 0.3,
        key: float("nan"),
        "lr": 1e-5,
    }
    with pytest.raises(Exception, match="invalid optional .*grad norm"):
        _normal_metrics([row], _records(contract), contract)
