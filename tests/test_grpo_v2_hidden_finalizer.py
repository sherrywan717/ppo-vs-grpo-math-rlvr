import json
from pathlib import Path

import pytest

from math_rlvr.evaluation.grpo_v2_dev_runtime import (
    DevEvaluationContractError,
    aggregate_dev_rows,
)
from math_rlvr.evaluation.grpo_v2_hidden_recovery import (
    BASE_RUN,
    COMPLETIONS_SHA256,
    file_sha256,
    recover_base_evidence,
)
from math_rlvr.evaluation.grpo_v2_hidden_runtime import (
    HiddenEvaluationContractError,
    aggregate_hidden_candidate0,
)
from math_rlvr.evaluation.grpo_v2_hidden_supervisor import validate_ipc_payload


def _base_rows():
    return [
        json.loads(line)
        for line in (BASE_RUN / "completions.jsonl").read_text().splitlines()
        if line.strip()
    ]


def test_hidden_candidate0_400_contract_and_known_counts():
    candidate0 = [row for row in _base_rows() if row["candidate_index"] == 0]
    aggregate = aggregate_hidden_candidate0(candidate0)
    assert aggregate["completion_count"] == 400
    assert aggregate["candidate0_pass_at_1"]["numerator"] == 6
    assert aggregate["format_rate"]["numerator"] == 31
    assert aggregate["valid_answer_rate"]["numerator"] == 28
    assert aggregate["parseable_rate"]["numerator"] == 28
    assert aggregate["eos_rate"]["numerator"] == 357
    assert aggregate["truncation_rate"]["numerator"] == 43


@pytest.mark.parametrize("delta", (-1, 1))
def test_hidden_candidate0_399_401_rejected(delta):
    candidate0 = [row for row in _base_rows() if row["candidate_index"] == 0]
    altered = candidate0[:-1] if delta < 0 else candidate0 + [dict(candidate0[0])]
    with pytest.raises(HiddenEvaluationContractError, match="requires 400"):
        aggregate_hidden_candidate0(altered)


def test_dev_aggregator_remains_128_only_and_hidden_runtime_does_not_import_it():
    candidate0 = [row for row in _base_rows() if row["candidate_index"] == 0]
    assert aggregate_dev_rows(candidate0[:128])["completion_count"] == 128
    with pytest.raises(DevEvaluationContractError, match="requires 128"):
        aggregate_dev_rows(candidate0)
    source = Path("src/math_rlvr/evaluation/grpo_v2_hidden_model_runtime.py").read_text()
    assert "aggregate_dev_rows" not in source


def test_base_1300_row_offline_recovery_is_nonmutating(tmp_path):
    before = file_sha256(BASE_RUN / "completions.jsonl")
    output = tmp_path / "recovery"
    summary = recover_base_evidence(output_dir=output)
    assert summary["recovery_status"] == (
        "scientifically_complete_with_recovered_metric_finalization"
    )
    assert summary["ledger"] == {
        "all_problem_count": 400,
        "shared_problem_count": 100,
        "candidate0_rows": 400,
        "shared_candidate_rows": 1000,
        "completion_rows": 1300,
        "generated_tokens": 152567,
    }
    values = [
        summary["pass_k_summary"]["overall"]["metrics"][str(k)]["mean"]
        for k in (1, 4, 10)
    ]
    assert values == sorted(values)
    assert file_sha256(BASE_RUN / "completions.jsonl") == before == COMPLETIONS_SHA256
    assert (output / "checksums.sha256").is_file()
    with pytest.raises(Exception, match="non-overwriting"):
        recover_base_evidence(output_dir=output)


def test_recovery_comparisons_unavailable_and_ipc_primitive():
    payload = validate_ipc_payload(
        {
            "outcome": {
                "status": "success",
                "run_id": "future",
                "run_dir": "/tmp/future",
                "summary_path": "/tmp/future/summary.json",
                "counts": {"completion_rows": 1300},
                "failure_reason": None,
            },
            "error": None,
        }
    )
    assert "completions" not in payload["outcome"]
