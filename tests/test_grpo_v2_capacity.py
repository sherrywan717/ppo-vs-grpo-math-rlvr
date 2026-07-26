import copy
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from math_rlvr.training import grpo_v2_capacity as capacity
from math_rlvr.training.grpo_v2 import main
from math_rlvr.training.grpo_v2_capacity import (
    audit_prompt_capacity,
    deterministic_prompt_cap,
    validate_capacity_rows,
)
from math_rlvr.training.grpo_v2_runtime import CONFIG_PATH, WARMSTART_CHECKPOINT, load_contract
from math_rlvr.training.warmstart_runtime import require_local_snapshot

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "reports/grpo_v2/prompt_capacity_audit.json"


def _rows():
    return copy.deepcopy(json.loads(AUDIT.read_text())["rows"])


def _expected(rows):
    training = [row for row in rows if row["phase"] == "train"]
    dev = [row for row in rows if row["phase"] == "dev"]
    return (
        [(row["problem_id"], row["content_hash"]) for row in training],
        [(row["problem_id"], row["content_hash"]) for row in dev],
    )


def _validate(rows, *, prompt_cap=928, sequence_ceiling=1184):
    training, dev = _expected(_rows())
    validate_capacity_rows(
        rows,
        expected_training=training,
        expected_dev=dev,
        prompt_cap=prompt_cap,
        completion_cap=256,
        sequence_ceiling=sequence_ceiling,
    )


def test_deterministic_cap_and_known_failure_samples_pass():
    rows = _rows()
    known = next(
        row
        for row in rows
        if row["problem_id"] == "math:DigitalLearningGmbH/MATH-lighteval:train:4567"
    )
    maximum = max(rows, key=lambda row: row["prompt_tokens"])
    assert (known["position"], known["update"], known["slot"], known["prompt_tokens"]) == (
        83,
        21,
        2,
        914,
    )
    assert (maximum["problem_id"], maximum["prompt_tokens"]) == (
        "math:DigitalLearningGmbH/MATH-lighteval:train:4207",
        918,
    )
    assert deterministic_prompt_cap(maximum["prompt_tokens"]) == 928
    _validate(rows)


def test_count_order_and_hash_drift_fail_closed():
    rows = _rows()
    with pytest.raises(ValueError, match="exactly 512 train"):
        _validate(rows[:512] + [copy.deepcopy(rows[0])] + rows[512:])
    with pytest.raises(ValueError, match="exactly 512 train"):
        _validate(rows[1:])
    swapped = _rows()
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(ValueError, match="order/hash"):
        _validate(swapped)
    drifted = _rows()
    drifted[0]["content_hash"] = "0" * 64
    with pytest.raises(ValueError, match="order/hash"):
        _validate(drifted)


def test_exact_cap_passes_and_capacity_failures_are_rejected():
    rows = _rows()
    row = max(rows, key=lambda item: item["prompt_tokens"])
    row["prompt_tokens"] = 928
    row["combined_potential"] = 1184
    _validate(rows)

    over = _rows()
    row = max(over, key=lambda item: item["prompt_tokens"])
    row["prompt_tokens"] = 929
    row["combined_potential"] = 1185
    with pytest.raises(ValueError, match="prompt cap"):
        _validate(over)

    with pytest.raises(ValueError, match="sequence ceiling"):
        _validate(_rows(), sequence_ceiling=1173)

    truncated = _rows()
    truncated[0]["truncation"] = True
    with pytest.raises(ValueError, match="truncation"):
        _validate(truncated)


def test_real_pinned_tokenizer_full_curriculum_and_dev_preflight():
    design, identity, contract = load_contract()
    result = audit_prompt_capacity(
        __import__("transformers").AutoTokenizer.from_pretrained(
            require_local_snapshot().snapshot_path,
            local_files_only=True,
            trust_remote_code=False,
        ),
        design=design,
        identity=identity,
        contract=contract,
        model_source=require_local_snapshot(),
    )
    summary = result["summary"]
    assert summary["training"]["count"] == 512
    assert summary["dev"]["count"] == 128
    assert summary["new_cap_overflows"] == summary["truncation_count"] == 0
    assert summary["max_combined_potential"] == 1174
    assert summary["audit_identity_sha256"] == identity["capacity_audit_identity_sha256"]
    import torch

    assert torch.cuda.is_initialized() is False


def test_hidden_test_is_not_a_capacity_preflight_input():
    source = inspect.getsource(capacity)
    assert "test_v2_hidden" not in source
    hidden = {
        json.loads(line)["problem_id"]
        for line in (ROOT / "configs/grpo_v2/manifests/test_v2_hidden.jsonl")
        .read_text()
        .splitlines()
    }
    assert hidden.isdisjoint({row["problem_id"] for row in _rows()})


def test_noncapacity_scientific_identities_and_failure_evidence_unchanged():
    failure_dir = ROOT / "reports/grpo_v2/grpo_v2_training/grpo_v2_seed42_20260726T030733Z"
    expected = {
        ROOT / "configs/grpo_v2/manifests/train_v2.jsonl": (
            "ca3403ae7b0c1f2689e21aca3283348f89b2ffb65498329e74cc4fa7fde8b664"
        ),
        ROOT / "configs/grpo_v2/manifests/dev_v2.jsonl": (
            "bdf02e1202e564177fea59f80f0b0ac8a36649daf8636ed6dd5bf3e5f6356b80"
        ),
        ROOT / "configs/grpo_v2/manifests/test_v2_hidden.jsonl": (
            "1da04a0093382711d618f515261b417e7df085d8e4fe93ddb34d314868062285"
        ),
        ROOT / "configs/grpo_v2/curriculum.json": (
            "7f7dcfa1218828e72dd6d42783bc2c790897c7e2a8f2f84d59ce2189710e3b41"
        ),
        ROOT / "configs/grpo_v2/pass_k_contract.json": (
            "c6c7728de1273dbe87eb73f450ae49b09f03588999a7e9b96c1892e363901cc8"
        ),
        failure_dir / "summary.json": (
            "fec0070da88b7a866c2a85fd08f4578e9e274ae8bcad83e680d6062e1cf38747"
        ),
        failure_dir / "resource_metrics.csv": (
            "9f31e6ea7296b4adf655ee2b18921db02feb2f7c2cfca5861c86cab86b16c297"
        ),
        failure_dir / "report.md": (
            "7377cb8f26aceaf3f5954cf3db7268d6708b6bfe844de649fbf727b037bdd01b"
        ),
    }
    for path, digest in expected.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    design, _, _ = load_contract()
    assert design["model"]["revision"] == ("989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
    assert design["prompt"]["max_completion_length"] == 256
    assert design["training"]["updates"] == 128
    assert design["training"]["training_completions"] == 2048
    assert design["budget"]["max_generated_tokens"] == 524288


def test_dry_run_capacity_preflight_occurs_before_execute(capsys):
    events = []

    def snapshot():
        events.append("snapshot")
        return object()

    def probe(**_kwargs):
        events.append("capacity")
        return {"summary": {"status": "passed", "training": {"count": 512}, "dev": {"count": 128}}}

    assert (
        main(
            ["--config", str(CONFIG_PATH), "--warmstart-checkpoint", str(WARMSTART_CHECKPOINT)],
            snapshot_probe=snapshot,
            capacity_probe=probe,
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert events == ["snapshot", "capacity"]
    assert payload["model_weight_loads"] == 0
    assert payload["cuda_initialized"] is False
    assert payload["prompt_capacity_preflight"]["training"]["count"] == 512
