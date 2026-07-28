import copy
import json
from pathlib import Path

import pytest

from math_rlvr.evaluation import grpo_v2_hidden as cli
from math_rlvr.evaluation.grpo_v2_hidden_runtime import (
    CONFIG_PATH,
    FOUR_MODEL_COMPLETIONS,
    MODEL_COMPLETIONS,
    ROLES,
    HiddenEvaluationContractError,
    build_hidden_plan,
    load_hidden_contract,
    validate_four_model_plans,
    validate_role_selection,
)
from math_rlvr.evaluation.grpo_v2_hidden_supervisor import (
    MAX_IPC_BYTES,
    validate_ipc_payload,
)
from math_rlvr.grpo_v2_contract import (
    select_checkpoint,
    validate_model_evaluation_ledger,
    validate_shared_candidate_batch,
)

OLD = Path(
    "/root/autodl-tmp/runs/math_rlvr/"
    "grpo_formal_1p5b_seed42_20260720T031006Z/checkpoint-32"
)
WARM = Path(
    "/root/autodl-tmp/runs/math_rlvr/"
    "warmstart_grpo_v2_seed42_20260722T051218Z/checkpoint-16"
)
SELECTED = Path(
    "/root/autodl-tmp/runs/math_rlvr/"
    "grpo_v2_seed42_20260726T044303Z/checkpoint-96"
)


def _fake_rows(plan):
    rows = []
    for expected in plan:
        rows.append(
            {
                **expected,
                "model_identity": "base",
                "checkpoint_identity": "base:qwen",
                "prompt_hash": f"prompt:{expected['problem_id']}",
                "generate_call_id": f"call:{expected['problem_id']}",
                "sampling_config": {
                    "temperature": 0.8,
                    "top_p": 0.95,
                    "top_k": None,
                    "max_completion_length": 256,
                    "num_return_sequences": 10 if expected["shared_n10"] else 1,
                },
                "canonical_correct": False,
                "verifier_status": "FORMAT_ERROR",
                "evidence_ref": (
                    f"completions.jsonl:{expected['problem_id']}:"
                    f"{expected['candidate_index']}"
                ),
            }
        )
    return rows


def test_four_roles_checkpoint_identity_and_selected_step():
    config, _, rows, shared = load_hidden_contract()
    plan = build_hidden_plan(rows, shared)
    validate_four_model_plans({role: copy.deepcopy(plan) for role in ROLES})
    assert validate_role_selection(config, role="base", checkpoint=None) is None
    assert validate_role_selection(config, role="old_grpo_v1", checkpoint=OLD)[
        "checkpoint_step"
    ] == 32
    assert validate_role_selection(config, role="warmstart_only", checkpoint=WARM)[
        "checkpoint_step"
    ] == 16
    selected = validate_role_selection(
        config, role="selected_grpo_v2", checkpoint=SELECTED
    )
    assert selected["checkpoint_step"] == 96
    assert selected["artifact_sha256"] == (
        "73bb15a32911f490216be2a80eb0d112be0f79236a6d461fd81fbd0579639246"
    )
    with pytest.raises(HiddenEvaluationContractError, match="path/step"):
        validate_role_selection(
            config,
            role="selected_grpo_v2",
            checkpoint=SELECTED.parent / "checkpoint-128",
        )
    with pytest.raises(HiddenEvaluationContractError, match="forbids"):
        validate_role_selection(config, role="base", checkpoint=WARM)


def test_exact_ledger_and_four_model_key_fairness():
    _, _, public, shared = load_hidden_contract()
    plan = build_hidden_plan(public, shared)
    assert len(plan) == MODEL_COMPLETIONS == 1_300
    assert FOUR_MODEL_COMPLETIONS == 5_200
    assert sum(row["candidate_index"] == 0 for row in plan) == 400
    assert sum(row["shared_n10"] for row in plan) == 1_000
    rows = _fake_rows(plan)
    assert validate_model_evaluation_ledger(
        rows,
        all_problem_ids={row["problem_id"] for row in public},
        shared_problem_ids=shared,
    )["completion_rows"] == 1_300
    plans = {role: copy.deepcopy(plan) for role in ROLES}
    validate_four_model_plans(plans)
    plans["selected_grpo_v2"][0]["batch_seed"] += 1
    with pytest.raises(HiddenEvaluationContractError, match="key drift"):
        validate_four_model_plans(plans)


def test_missing_duplicate_n10_and_checkpoint_selection_separation():
    _, _, public, shared = load_hidden_contract()
    rows = _fake_rows(build_hidden_plan(public, shared))
    problem_id = next(iter(shared))
    one_problem = [row for row in rows if row["problem_id"] == problem_id]
    assert len(one_problem) == 10
    validate_shared_candidate_batch(one_problem)
    with pytest.raises(ValueError, match="exactly ten"):
        validate_shared_candidate_batch(one_problem[:9])
    duplicate = copy.deepcopy(one_problem)
    duplicate[-1]["candidate_index"] = 0
    with pytest.raises(ValueError, match="unique"):
        validate_shared_candidate_batch(duplicate)
    selection_rows = [
        {
            "checkpoint_step": step,
            "evaluation_split": "test_v2_hidden",
            "canonical_pass_at_1": 1,
            "parseable_rate": 1,
            "format_rate": 1,
            "truncation_rate": 0,
        }
        for step in (32, 64, 96, 128)
    ]
    with pytest.raises(ValueError, match="dev_v2 only"):
        select_checkpoint(selection_rows)


def test_file_backed_1300_rows_keep_ipc_primitive_and_small(tmp_path):
    _, _, public, shared = load_hidden_contract()
    evidence_path = tmp_path / "completions.jsonl"
    with evidence_path.open("w") as handle:
        for row in _fake_rows(build_hidden_plan(public, shared)):
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    outcome = {
        "status": "success",
        "run_id": "fake",
        "run_dir": str(tmp_path),
        "summary_path": str(tmp_path / "summary.json"),
        "counts": {
            "completion_rows": 1_300,
            "candidate0_rows": 400,
            "shared_problem_rows": 100,
        },
        "failure_reason": None,
    }
    payload = validate_ipc_payload({"outcome": outcome, "error": None})
    assert "completions" not in payload["outcome"]
    assert len(json.dumps(payload).encode()) < MAX_IPC_BYTES
    assert sum(1 for _ in evidence_path.open()) == 1_300


@pytest.mark.parametrize(
    ("role", "checkpoint"),
    [
        ("base", None),
        ("old_grpo_v1", OLD),
        ("warmstart_only", WARM),
        ("selected_grpo_v2", SELECTED),
    ],
)
def test_four_role_dry_runs_are_cpu_only(role, checkpoint, capsys):
    import torch

    argv = ["--config", str(CONFIG_PATH), "--role", role]
    if checkpoint is not None:
        argv += ["--checkpoint", str(checkpoint)]
    assert cli.main(argv) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["completion_count"] == 1_300
    assert result["four_model_completion_count"] == 5_200
    assert result["trusted_manifest_opened"] is False
    assert result["model_or_tokenizer_loads"] == result["generation_calls"] == 0
    assert result["cuda_initialized"] is False
    assert torch.cuda.is_initialized() is False


def test_wrong_artifact_sha_rejected():
    config, _, _, _ = load_hidden_contract()
    altered = copy.deepcopy(config)
    altered["roles"]["selected_grpo_v2"]["artifact_sha256"] = "0" * 64
    with pytest.raises(HiddenEvaluationContractError, match="artifact SHA"):
        validate_role_selection(
            altered, role="selected_grpo_v2", checkpoint=SELECTED
        )
