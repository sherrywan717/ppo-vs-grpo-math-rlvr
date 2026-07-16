import json
from pathlib import Path

import pytest

from math_rlvr.artifacts import manager as artifact_manager
from math_rlvr.artifacts.manager import ArtifactManager
from math_rlvr.evaluation.formal import (
    load_evaluation_config,
    validate_evaluation_config,
)


def test_formal_evaluation_phases_are_cpu_only_and_matched() -> None:
    config = load_evaluation_config()
    baseline = validate_evaluation_config(config, "baseline", seed=42)
    final = validate_evaluation_config(config, "final", algorithm="ppo", seed=42)
    validation = validate_evaluation_config(
        config, "validation", algorithm="grpo", seed=2026
    )
    assert baseline["completion_contract"]["completions_per_seed"] == 800
    assert final["completion_contract"]["completions_per_checkpoint_seed"] == 800
    assert validation["completion_contract"]["completions"] == 64
    for result in (baseline, final, validation):
        assert result["cuda_initialized"] is False
        assert result["model_or_tokenizer_loads"] == 0
        assert result["generation_calls"] == 0
        assert result["trainer_calls"] == 0


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
    assert persisted == summary
    assert persisted["counters"] == {"completions": 512, "generated_tokens": 131072}
    assert (manager.run_dir / "checksums.sha256").is_file()
