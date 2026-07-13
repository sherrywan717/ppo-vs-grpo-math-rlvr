import json
from pathlib import Path

import pytest

from math_rlvr.artifacts.manager import ArtifactManager, make_run_id
from math_rlvr.artifacts.plotting import generate
from math_rlvr.config import load_config
from math_rlvr.gold import assert_delimiter_only, normalize_gold_answer
from math_rlvr.training.builders import build_grpo_trainer, build_ppo_trainer, trainer_plan
from math_rlvr.training.grpo import render_training_prompt as grpo_renderer
from math_rlvr.training.ppo import render_training_prompt as ppo_renderer

SIX = (r"\sqrt{51}", r"\text{east}", "(a+5)(b+2)", "(15,-29)", r"(5,\infty)", r"\frac14")


def test_normalization_only_inserts_delimiters():
    for raw in SIX:
        normalized = normalize_gold_answer(raw)
        assert normalized == f"${raw}$"
        assert_delimiter_only(raw, normalized)
    for changed in ("$x+1$", "$ x $", "$$x$", "$y$"):
        with pytest.raises(ValueError):
            assert_delimiter_only("x", changed)


def test_audit_has_six_complete_records():
    payload = json.loads(
        Path("reports/preprocessing/math500_gold_normalization_audit.json").read_text()
    )
    assert len(payload["records"]) == 6 and payload["policy"]["solution_fallback"] is False
    for row in payload["records"]:
        assert_delimiter_only(row["raw_gold"], row["normalized_gold"])


def test_run_id_shape():
    assert "_smoke_grpo_qwen2.5-0.5b_seed42" in make_run_id(
        "smoke", "grpo", "Qwen/Qwen2.5-0.5B-Instruct", 42
    )


def test_artifact_failure_summary_and_secret_rejection(tmp_path, monkeypatch):
    monkeypatch.setattr("math_rlvr.artifacts.manager.RUN_ROOT", tmp_path / "runs")
    monkeypatch.setattr("math_rlvr.artifacts.manager.REPORT_ROOT", tmp_path / "reports")
    manager = ArtifactManager(
        "smoke",
        "grpo",
        "Qwen/Qwen2.5-0.5B-Instruct",
        42,
        "safe command",
        {"cost": 8.88},
        run_id="fixed",
    )
    with pytest.raises(ValueError):
        manager.write_text("bad.txt", "HF_TOKEN=secret")
    manager.finalize(
        "failed",
        "rollout",
        ValueError("x"),
        "contract_error",
        {"completions": 2, "generated_tokens": 8},
    )
    assert json.loads((manager.run_dir / "final_summary.json").read_text())["status"] == "failed"
    assert (manager.run_dir / "checksums.sha256").is_file()


def test_plotting_uses_csv_and_omits_unavailable(tmp_path):
    (tmp_path / "figures").mkdir()
    (tmp_path / "metrics.csv").write_text(
        "step,reward,policy_loss,correctness,format_accuracy,parse_success_rate,cumulative_generated_tokens,mean_completion_length\n1,0.2,1.0,0,1,1,10,10\n"
    )
    (tmp_path / "gpu_metrics.csv").write_text(
        "elapsed_seconds,gpu_memory_used_mb,gpu_utilization_pct\n0,100,20\n"
    )
    made, missing = generate(tmp_path, "run", "grpo", 42)
    assert "reward_curve" in made and "kl_curve" in missing
    assert (tmp_path / "figures/reward_curve.png").is_file() and (
        tmp_path / "figures/reward_curve.svg"
    ).is_file()


def test_dry_plan_and_fake_trainer_builders_do_not_train(tmp_path):
    grpo = load_config("configs/smoke/grpo.yaml")
    ppo = load_config("configs/smoke/ppo.yaml")
    assert trainer_plan(grpo, tmp_path).max_steps == 2
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        return object()

    build_grpo_trainer(
        grpo,
        [],
        lambda completions, **kwargs: [0.0] * len(completions),
        tmp_path,
        model=object(),
        tokenizer=object(),
        trainer_factory=factory,
    )
    build_ppo_trainer(
        ppo, [], object(), None, object(), object(), object(), tmp_path, trainer_factory=factory
    )
    assert len(calls) == 2 and all("train" not in call for call in calls)


def test_ppo_and_grpo_share_prompt_renderer():
    assert ppo_renderer is grpo_renderer
