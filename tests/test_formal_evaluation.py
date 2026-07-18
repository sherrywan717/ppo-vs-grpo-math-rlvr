import json
from pathlib import Path

import pytest

from math_rlvr.artifacts import manager as artifact_manager
from math_rlvr.artifacts.manager import ArtifactManager
from math_rlvr.evaluation.formal import (
    load_evaluation_config,
    validate_evaluation_config,
)
from math_rlvr.evaluation.formal_model_runtime import (
    _formal_completion_record,
    _record_formal_completion,
)
from math_rlvr.rewards.formal import FORMAL_REWARD_POLICY
from math_rlvr.verifier import GSM8KVerifier, MathExpressionVerifier


def test_formal_evaluation_phases_are_cpu_only_and_matched() -> None:
    config = load_evaluation_config()
    baseline = validate_evaluation_config(config, "baseline", seed=42)
    final = validate_evaluation_config(config, "final", algorithm="ppo", seed=42)
    validation = validate_evaluation_config(
        config, "validation", algorithm="grpo", seed=123
    )
    assert baseline["completion_contract"]["completions_per_seed"] == 800
    assert final["completion_contract"]["completions_per_checkpoint_seed"] == 800
    assert validation["completion_contract"]["completions"] == 64
    for result in (baseline, final, validation):
        assert result["cuda_initialized"] is False
        assert result["model_or_tokenizer_loads"] == 0
        assert result["generation_calls"] == 0
        assert result["trainer_calls"] == 0


def test_reserved_seed_2026_is_not_an_active_evaluation_seed() -> None:
    config = load_evaluation_config()
    with pytest.raises(ValueError, match="unapproved formal evaluation seed"):
        validate_evaluation_config(config, "final", algorithm="ppo", seed=2026)


def test_formal_evaluation_rejects_test_tuning_and_identity_drift() -> None:
    config = load_evaluation_config()
    config["selection_policy"]["checkpoint_selection"] = "best_test_checkpoint"
    with pytest.raises(ValueError, match="test-selection"):
        validate_evaluation_config(config, "final", algorithm="ppo", seed=42)

    config = load_evaluation_config()
    config["reward"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="reward identity"):
        validate_evaluation_config(config, "baseline", seed=42)


def test_fake_artifact_finalization_uses_existing_manager(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "runs"
    report_root = tmp_path / "reports"
    monkeypatch.setattr(artifact_manager, "RUN_ROOT", run_root)
    monkeypatch.setattr(artifact_manager, "REPORT_ROOT", report_root)
    manager = ArtifactManager(
        stage="formal_1p5b_fake",
        algorithm="ppo",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        seed=42,
        command="cpu-only fake finalization",
        config={"dry_run": True},
        run_id="formal_fake_artifact_finalization",
    )
    summary = manager.finalize(
        status="success",
        counters={"completions": 512, "generated_tokens": 131072},
        summary={"optimizer_steps": 32, "global_steps": 32},
    )
    persisted = json.loads((manager.run_dir / "final_summary.json").read_text())
    assert (manager.run_dir / "checksums.sha256").is_file()
    assert persisted == summary
    assert persisted["counters"] == {"completions": 512, "generated_tokens": 131072}


def test_evaluation_artifact_manager_omits_empty_checkpoint_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(artifact_manager, "RUN_ROOT", tmp_path / "runs")
    monkeypatch.setattr(artifact_manager, "REPORT_ROOT", tmp_path / "reports")
    manager = ArtifactManager(
        stage="formal_1p5b_evaluation_fake",
        algorithm="base",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        seed=42,
        command="cpu-only fake evaluation",
        config={"dry_run": True},
        run_id="formal_fake_evaluation",
        create_checkpoints=False,
    )
    assert (manager.run_dir / "figures").is_dir()
    assert not (manager.run_dir / "checkpoints").exists()


@pytest.mark.parametrize(
    ("domain", "completion", "verifier"),
    [
        ("gsm8k", "<reasoning>brief</reasoning><answer>2</answer>", GSM8KVerifier("2")),
        (
            "math500",
            "<reasoning>brief</reasoning><answer>$x^2$</answer>",
            MathExpressionVerifier("$x^2$"),
        ),
    ],
)
def test_formal_reward_record_is_flat_json_and_immediately_persisted(
    domain,
    completion,
    verifier,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(artifact_manager, "RUN_ROOT", tmp_path / "runs")
    monkeypatch.setattr(artifact_manager, "REPORT_ROOT", tmp_path / "reports")
    manager = ArtifactManager(
        stage="formal_evaluation_record_fake",
        algorithm="base",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        seed=42,
        command="cpu-only fake record serialization",
        config={"dry_run": True},
        run_id=f"formal_record_{domain}",
        create_checkpoints=False,
    )
    evaluation = FORMAL_REWARD_POLICY.evaluate(completion, verifier)
    record = _formal_completion_record(
        item={
            "phase": "baseline",
            "seed": 42,
            "problem_id": f"{domain}:fake",
            "domain": domain,
            "sample_kind": "pass1",
            "generation_index": 0,
            "pair_key": f"{domain}:fake::pass1::generation:0",
        },
        generation_seed=123,
        completion_ids=[7, 8],
        text=completion,
        max_completion_length=256,
        eos_token_id=2,
        evaluation=evaluation,
    )
    rows = []
    _record_formal_completion(
        rows,
        record,
        lambda payload: manager.append_jsonl("completions.jsonl", payload),
    )

    persisted = json.loads((manager.run_dir / "completions.jsonl").read_text())
    assert persisted == record == rows[0]
    assert "components" not in record
    assert "reward_components" not in record
    assert record["canonical_status"] == "verified_pass"
    assert record["verifier_status"] == record["canonical_status"]
    assert record["scalar_reward"] == 1.0
    assert record["answer_block_component"] == 0.05
    assert record["strict_protocol_component"] == 0.05
    assert record["valid_answer_component"] == 0.10
    assert record["correctness_component"] == 0.80
    assert record["reward_policy_version"] == "shaped_v3_domain"
    assert len(record["reward_policy_sha256"]) == 64
    assert isinstance(record["reward_component_weights"], dict)
    assert isinstance(record["verifier_detail"], str)
