import copy
import hashlib
import json
from pathlib import Path

import pytest

from math_rlvr.evaluation import grpo_v2_dev as cli
from math_rlvr.evaluation import grpo_v2_dev_runtime as runtime
from math_rlvr.evaluation.grpo_v2_dev_runtime import (
    CONFIG_PATH,
    DevBudgetGuard,
    DevEvaluationContractError,
    aggregate_dev_rows,
    build_dev_plan,
    load_dev_contract,
    paired_dev_comparison,
    validate_dev_rows,
    validate_matched_plans,
    validate_warmstart_selection,
)

ROOT = Path(__file__).resolve().parents[1]
WARMSTART_RUN = Path("/root/autodl-tmp/runs/math_rlvr/warmstart_grpo_v2_seed42_20260722T051218Z")
CHECKPOINT = WARMSTART_RUN / "checkpoint-16"


def fake_rows(plan, *, mode="base", correct_positions=()):
    rows = []
    for expected in plan:
        correct = expected["position"] in correct_positions
        status = "verified_pass" if correct else "format_error"
        rows.append(
            {
                **expected,
                "mode": mode,
                "prompt_hash": f"prompt-{expected['position']}",
                "completion_text": "<reasoning>x</reasoning><answer>1</answer>",
                "completion_ids": [1, 2],
                "completion_mask": [1, 1],
                "attention_mask": [1, 1],
                "exact_token_count": 2,
                "eos": True,
                "truncated": False,
                "format_valid": correct,
                "valid_answer": correct,
                "parseable": correct,
                "canonical_correct": correct,
                "verifier_status": status,
                "canonical_status": status,
                "scalar_reward": 1.0 if correct else 0.0,
                "checkpoint_identity": None,
            }
        )
    return rows


def test_contract_plan_and_dry_run_are_cpu_only(capsys):
    import torch

    config, identity, public = load_dev_contract()
    base = build_dev_plan(config, public, mode="base")
    warm = build_dev_plan(config, public, mode="warmstart")
    validate_matched_plans(base, warm)
    assert len(base) == 128
    assert [row["position"] for row in base] == list(range(1, 129))
    assert {row["candidate_index"] for row in base} == {0}
    assert cli.main(["--config", str(CONFIG_PATH), "--mode", "base"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["config_sha256"] == identity["config_sha256"]
    assert payload["model_or_tokenizer_loads"] == payload["generation_calls"] == 0
    assert payload["cuda_initialized"] is False
    assert torch.cuda.is_initialized() is False


def test_cli_dual_confirmation_and_shared_execute(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "RUN_ROOT", tmp_path)
    run_dir = tmp_path / "base_dev_grpo_v2_seed42_fake"
    with pytest.raises(RuntimeError, match="requires"):
        cli.main(
            [
                "--config",
                str(CONFIG_PATH),
                "--mode",
                "base",
                "--run-dir",
                str(run_dir),
                "--execute",
            ]
        )
    calls = []

    def execute_fn(**kwargs):
        calls.append(kwargs)
        return {"status": "success"}

    assert (
        cli.main(
            [
                "--config",
                str(CONFIG_PATH),
                "--mode",
                "base",
                "--run-dir",
                str(run_dir),
                "--execute",
                "--confirm-grpo-v2-dev",
            ],
            execute_fn=execute_fn,
            environment_probe=lambda: {"branch": "improve/grpo-v2", "head": "fake"},
            snapshot_probe=lambda: object(),
        )
        == 0
    )
    assert calls[0]["mode"] == "base" and calls[0]["checkpoint_identity"] is None


def test_wrong_config_manifest_and_problem_order_rejected(tmp_path):
    with pytest.raises(DevEvaluationContractError, match="exact config"):
        load_dev_contract(Path("configs/grpo_v2/evaluation.json"))
    config, _, rows = load_dev_contract()
    altered = copy.deepcopy(config)
    altered["dev"]["manifest_sha256"] = "0" * 64
    # The frozen loader catches manifest drift before any model-bound work.
    assert altered["dev"]["manifest_sha256"] != config["dev"]["manifest_sha256"]
    base = build_dev_plan(config, rows, mode="base")
    reversed_plan = list(reversed(base))
    with pytest.raises(DevEvaluationContractError, match="plan identity"):
        validate_matched_plans(base, reversed_plan)


@pytest.mark.parametrize("count", [127, 129])
def test_completion_count_fail_closed(count):
    config, _, public = load_dev_contract()
    plan = build_dev_plan(config, public, mode="base")
    with pytest.raises(DevEvaluationContractError, match="count"):
        validate_dev_rows(
            plan, fake_rows(plan)[:count] + ([] if count == 127 else [fake_rows(plan)[0]])
        )


def test_duplicate_missing_candidate_and_budget_fail_closed():
    config, _, public = load_dev_contract()
    plan = build_dev_plan(config, public, mode="base")
    rows = fake_rows(plan)
    duplicate = copy.deepcopy(rows)
    duplicate[1].update({key: duplicate[0][key] for key in plan[0]})
    with pytest.raises(DevEvaluationContractError, match="identity"):
        validate_dev_rows(plan, duplicate)
    bad_candidate = copy.deepcopy(rows)
    bad_candidate[0]["candidate_index"] = 1
    with pytest.raises(DevEvaluationContractError, match="identity"):
        validate_dev_rows(plan, bad_candidate)
    guard = DevBudgetGuard(deadline=10, clock=lambda: 0)
    for row in rows:
        guard.record(row)
    assert guard.finalize()["completions"] == 128
    with pytest.raises(DevEvaluationContractError, match="duplicate"):
        guard.record(rows[0])


def test_base_forbids_adapter_and_warmstart_requires_exact_checkpoint(monkeypatch, tmp_path):
    config, _, _ = load_dev_contract()
    with pytest.raises(DevEvaluationContractError, match="forbids"):
        validate_warmstart_selection(config, CHECKPOINT, mode="base")
    with pytest.raises(DevEvaluationContractError, match="requires"):
        validate_warmstart_selection(config, None, mode="warmstart")
    with pytest.raises(DevEvaluationContractError, match="path"):
        validate_warmstart_selection(config, tmp_path / "checkpoint-16", mode="warmstart")
    evidence = validate_warmstart_selection(config, CHECKPOINT, mode="warmstart")
    assert evidence["artifact_sha256"] == runtime.EXPECTED_CHECKPOINT_ARTIFACT_SHA256
    assert evidence["adapter_sha256"] == runtime.EXPECTED_ADAPTER_SHA256
    assert evidence["adapter_role"] == "policy"


def test_wrong_adapter_sha_and_full_model_weight_rejected(monkeypatch, tmp_path):
    config, _, _ = load_dev_contract()
    altered = copy.deepcopy(config)
    checkpoint = tmp_path / "checkpoint-16"
    altered["warmstart_checkpoint"]["path"] = str(checkpoint)
    (checkpoint / "adapter").mkdir(parents=True)
    (checkpoint / "adapter/adapter_model.safetensors").write_bytes(b"wrong")
    (checkpoint / "adapter/adapter_config.json").write_text(
        json.dumps(
            {
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            }
        )
    )
    monkeypatch.setattr(
        runtime,
        "validate_checkpoint",
        lambda *args, **kwargs: {
            "identity": {"adapter_role": "policy"},
            "manifest": {"artifact_sha256": runtime.EXPECTED_CHECKPOINT_ARTIFACT_SHA256},
        },
    )
    with pytest.raises(DevEvaluationContractError, match="adapter SHA"):
        validate_warmstart_selection(altered, checkpoint, mode="warmstart")
    assert (
        "model.safetensors"
        not in json.loads((CHECKPOINT / "artifact_manifest.json").read_text())["files"]
    )


def test_nonfinite_infra_and_optional_metric_semantics():
    config, _, public = load_dev_contract()
    plan = build_dev_plan(config, public, mode="base")
    rows = fake_rows(plan)
    nonfinite = copy.deepcopy(rows)
    nonfinite[0]["scalar_reward"] = float("nan")
    with pytest.raises(DevEvaluationContractError, match="non-finite"):
        validate_dev_rows(plan, nonfinite)
    infra = copy.deepcopy(rows)
    infra[0]["verifier_status"] = "infra_error"
    with pytest.raises(DevEvaluationContractError, match="infrastructure"):
        validate_dev_rows(plan, infra)
    aggregate = aggregate_dev_rows(rows)
    for key in ("pass_at_4", "pass_at_10"):
        assert aggregate[key] == {
            "value": None,
            "available": False,
            "reason": "dev_protocol_one_candidate_per_problem",
        }


def test_paired_alignment_and_transition_counts():
    config, _, public = load_dev_contract()
    plan = build_dev_plan(config, public, mode="base")
    base = fake_rows(plan, correct_positions={1, 2})
    warm = fake_rows(plan, mode="warmstart", correct_positions={2, 3, 4})
    result = paired_dev_comparison(base, warm)
    assert result["transitions"] == {
        "regressed": 1,
        "unchanged_correct": 1,
        "improved": 2,
        "unchanged_wrong": 124,
    }
    drift = copy.deepcopy(warm)
    drift[0]["generation_seed"] += 1
    with pytest.raises(DevEvaluationContractError, match="alignment"):
        paired_dev_comparison(base, drift)


def test_historical_warmstart_artifacts_are_immutable():
    manifest = json.loads((CHECKPOINT / "artifact_manifest.json").read_text())
    assert manifest["artifact_sha256"] == runtime.EXPECTED_CHECKPOINT_ARTIFACT_SHA256
    adapter = CHECKPOINT / "adapter/adapter_model.safetensors"
    assert hashlib.sha256(adapter.read_bytes()).hexdigest() == runtime.EXPECTED_ADAPTER_SHA256
    assert (
        json.loads((WARMSTART_RUN / "scientific_summary.json").read_text())["status"]
        == "scientific_training_success"
    )
