import json
from pathlib import Path

import pytest
from formal_checkpoint_helpers import write_fake_trusted_checkpoint

from math_rlvr.evaluation.formal import load_evaluation_config
from math_rlvr.evaluation.formal_runtime import execute_formal_evaluation
from math_rlvr.training.formal import FORMAL_ACTIVE_SEEDS, validate_formal_config_file
from math_rlvr.training.formal_runtime import (
    FormalRuntimeError,
    create_formal_backup,
    execute_formal_training,
    formal_episode_records,
    formal_run_contract,
)


class FakeFormalBackend:
    def __init__(self, root: Path, *, stop_after: int = 32, token_width: int = 1):
        self.root = root
        self.stop_after = stop_after
        self.token_width = token_width

    def execute(self, contract, observer, *, start_update):
        if start_update > 1:
            for step in contract.validation_steps:
                if step >= start_update:
                    break
                rows = [
                    {
                        "checkpoint_step": step,
                        "problem_id": f"validation:{index}",
                        "canonical_correct": False,
                    }
                    for index in range(64)
                ]
                observer.guard.record_restored_validation(step, rows)
                observer.validation_metrics.extend(rows)
        for update in range(start_update, self.stop_after + 1):
            rows = []
            for pair_key in contract.pair_keys_for_update(update):
                problem_id, generation = pair_key.rsplit("::generation:", 1)
                rows.append(
                    {
                        "update": update,
                        "problem_id": problem_id,
                        "generation_index": int(generation),
                        "pair_key": pair_key,
                        "completion_ids": [7] * self.token_width,
                        "completion_mask": [1] * self.token_width,
                        "exact_token_count": self.token_width,
                        "eos_reached": False,
                        "truncated": self.token_width == contract.max_completion_length,
                        "raw_completion": "<reasoning>fake</reasoning><answer>0</answer>",
                        "scalar_reward": 0.1,
                        "canonical_status": "wrong_answer",
                    }
                )
            metrics = {
                "reward_mean": 0.1,
                "reward_std": 0.0,
                "reward_variance": 0.0,
                "loss": 0.25,
                "total_loss": 0.25,
                "grad_norm": 1.0,
                "entropy": 0.5,
                "policy_entropy_mean": 0.5,
                "policy_entropy_mean_available": True,
                "policy_entropy_std": None,
                "policy_entropy_std_available": False,
                "policy_entropy_std_reason": "fake backend exposes mean only",
                "response_token_entropy_mean": None,
                "response_token_entropy_mean_available": False,
                "response_token_entropy_mean_reason": "fake backend has no logits",
                "policy_grad_norm": None,
                "policy_grad_norm_available": False,
                "policy_grad_norm_reason": "fake backend exposes aggregate grad norm only",
                "value_grad_norm": None,
                "value_grad_norm_available": False,
                "value_grad_norm_reason": "fake backend exposes aggregate grad norm only",
                "learning_rate": 1e-5,
                "mean_completion_length": float(self.token_width),
                "completion_length_std": 0.0,
                "completion_duplicate_rate": 0.75,
                "unique_completion_rate": 0.25,
                "eos_rate": 0.0,
                "eos_rate_available": True,
                "truncation_rate": float(self.token_width == contract.max_completion_length),
                "truncation_rate_available": True,
                "zero_advantage_fraction": 1.0,
                "format_accuracy": 1.0,
                "valid_answer_rate": 1.0,
                "canonical_pass_rate": 0.0,
                "generated_tokens": 16 * self.token_width,
                "cumulative_generated_tokens": update * 16 * self.token_width,
                "kl": None,
                "kl_unavailable_reason": "fake backend does not compute KL",
                "clip_fraction": None,
                "clip_fraction_available": False,
                "clip_fraction_reason": "fake backend does not compute clip fraction",
                "ratio_mean": None,
                "ratio_available": False,
                "ratio_reason": "fake backend does not compute ratio",
                "ratio_variance": None,
                "ratio_variance_available": False,
                "ratio_variance_reason": "fake backend does not compute ratio variance",
                "advantage_mean": None,
                "advantage_available": False,
                "advantage_reason": "fake backend does not expose advantage",
                "return_mean": None,
                "return_available": False,
                "return_reason": "fake backend does not expose return",
            }
            if contract.algorithm == "ppo":
                metrics.update({"policy_loss": 0.1, "value_loss": 0.15})
            observer.update(
                update,
                rows,
                metrics,
                optimizer_step=update,
                global_step=update,
            )
            if update in contract.validation_steps:
                observer.validation(
                    update,
                    [
                        {
                            "checkpoint_step": update,
                            "problem_id": f"validation:{index}",
                            "canonical_correct": False,
                        }
                        for index in range(64)
                    ],
                )
                checkpoint = self.root / f"checkpoint-{update}"
                write_fake_trusted_checkpoint(
                    checkpoint,
                    contract,
                    update,
                    completion_prefix=observer.completions,
                    metric_prefix=observer.metrics,
                )
                observer.checkpoint(update, checkpoint)
        if self.stop_after < contract.updates:
            raise RuntimeError("intentional interruption")


class FakeEvaluationBackend:
    def generate(self, plan):
        return [
            {
                **row,
                "completion_ids": [1, 2],
                "completion_mask": [1, 1],
                "exact_token_count": 2,
                "raw_completion": "<reasoning>fake</reasoning><answer>0</answer>",
                "truncated": False,
                "format_valid": True,
                "valid_answer": True,
                "canonical_correct": row["problem_id"].endswith(":0"),
                "verifier_status": (
                    "verified_pass" if row["problem_id"].endswith(":0") else "wrong_answer"
                ),
                "scalar_reward": 0.1,
            }
            for row in plan
        ]


def load_training(algorithm: str, seed: int):
    return validate_formal_config_file(
        Path(f"configs/formal_1p5b/resolved/{algorithm}_seed_{seed}.json"), algorithm
    )[0]


@pytest.mark.parametrize("algorithm", ["ppo", "grpo"])
@pytest.mark.parametrize("seed", FORMAL_ACTIVE_SEEDS)
def test_fake_formal_32_step_finalization(algorithm, seed, tmp_path):
    config = load_training(algorithm, seed)
    run_dir = tmp_path / f"{algorithm}-{seed}"
    result = execute_formal_training(
        config,
        FakeFormalBackend(run_dir),
        run_dir=run_dir,
        run_id=f"formal-{algorithm}-{seed}",
    )
    assert result["status"] == "success"
    counters = result["counters"]
    assert counters["updates"] == counters["optimizer_steps"] == counters["global_steps"] == 32
    assert counters["completions"] == 512
    assert counters["generated_tokens"] == 512
    assert counters["checkpoints"] == counters["validations"] == [8, 16, 24, 32]
    assert len((run_dir / "completions.jsonl").read_text().splitlines()) == 512
    assert len(json.loads((run_dir / "checkpoint_inventory.json").read_text())) == 4
    for required in (
        "resolved_config.json",
        "run_manifest.json",
        "expected_run_contract.json",
        "prompt_scope_preflight.json",
        "model_roles.json",
        "metrics.csv",
        "metrics.jsonl",
        "completions.jsonl",
        "validation_metrics.csv",
        "checkpoint_inventory.json",
        "resource_metrics.csv",
        "report.md",
        "error_analysis.md",
        "figures",
    ):
        assert (run_dir / required).exists()


def test_token_overflow_fails_closed_and_writes_failure_artifacts(tmp_path):
    config = load_training("ppo", 42)
    run_dir = tmp_path / "overflow"
    with pytest.raises(FormalRuntimeError, match="max completion length|token hard cap"):
        execute_formal_training(
            config,
            FakeFormalBackend(run_dir, token_width=257),
            run_dir=run_dir,
            run_id="overflow",
        )
    failure = json.loads((run_dir / "failure_report.json").read_text())
    assert failure["status"] == "failure"
    assert failure["counters"]["optimizer_steps"] == 0
    backup = create_formal_backup(run_dir, tmp_path / "overflow.failure.tar.gz")
    assert len(backup["sha256"]) == 64
    assert Path(str(tmp_path / "overflow.failure.tar.gz") + ".sha256").is_file()


def test_resume_requires_same_run_and_continues_exact_counters(tmp_path):
    config = load_training("grpo", 42)
    run_dir = tmp_path / "resume"
    with pytest.raises(RuntimeError, match="intentional interruption"):
        execute_formal_training(
            config,
            FakeFormalBackend(run_dir, stop_after=8),
            run_dir=run_dir,
            run_id=run_dir.name,
        )
    checkpoint = run_dir / "checkpoint-8"
    result = execute_formal_training(
        config,
        FakeFormalBackend(run_dir),
        run_dir=run_dir,
        run_id=run_dir.name,
        resume_checkpoint=checkpoint,
    )
    assert result["counters"]["updates"] == 32
    assert result["counters"]["completions"] == 512
    alien = dict(json.loads((checkpoint / "resume_manifest.json").read_text()))
    alien["run_id"] = "other-run"
    (checkpoint / "resume_manifest.json").write_text(json.dumps(alien) + "\n")
    with pytest.raises(FormalRuntimeError, match="SHA256 inventory mismatch"):
        execute_formal_training(
            config,
            FakeFormalBackend(run_dir),
            run_dir=run_dir,
            run_id="same-run",
            resume_checkpoint=checkpoint,
        )


def test_formal_episode_order_and_32_ga4_backward_groups():
    from math_rlvr.training.trl_compat import PPOBackwardEventGuard

    episodes = formal_episode_records("ppo", 42)
    assert len(episodes) == 512
    assert [row["generation_index"] for row in episodes[:8]] == [0, 1, 2, 3] * 2
    assert len({row["pair_key"] for row in episodes}) == 512
    loop = {
        "microbatches_per_minibatch": 4,
        "per_device_train_batch_size": 4,
        "local_mini_batch_size": 16,
        "expected_optimizer_steps": 32,
    }
    guard = PPOBackwardEventGuard(loop, {"num_steps": 4, "sync_with_dataloader": False})
    for microbatch in range(128):
        guard.note_training_forward(4)
        event = guard.prepare_backward((microbatch + 1) % 4 == 0)
        guard.commit_backward(event)
        if (microbatch + 1) % 4 == 0:
            guard.assert_ready_for_optimizer()
            guard.commit_optimizer_step()
    evidence = guard.assert_complete(32)
    assert evidence["backward_events"] == 128
    assert evidence["processed_samples"] == 512
    assert evidence["underlying_optimizer_steps"] == 32


def test_reserved_2026_cannot_enter_formal_runtime():
    config = load_training("ppo", 2026)
    with pytest.raises(FormalRuntimeError, match="not in the four-run active suite"):
        formal_run_contract(config)


@pytest.mark.parametrize("phase", ["baseline", "final"])
def test_fake_baseline_and_final_evaluation_artifacts(phase, tmp_path):
    kwargs = {"algorithm": "ppo", "checkpoint_step": 32} if phase == "final" else {}
    result = execute_formal_evaluation(
        FakeEvaluationBackend(),
        phase=phase,
        seed=42,
        run_dir=tmp_path / phase,
        config=load_evaluation_config(),
        **kwargs,
    )
    assert result["completion_count"] == 800
    assert result["unique_problem_count"] == 400
    assert (tmp_path / phase / "figures").is_dir()
    assert len((tmp_path / phase / "completions.jsonl").read_text().splitlines()) == 800


def test_fake_checkpoint_validation_finalizes_64_rows_with_nullable_pass_metrics(tmp_path):
    result = execute_formal_evaluation(
        FakeEvaluationBackend(),
        phase="validation",
        seed=42,
        run_dir=tmp_path / "validation",
        algorithm="grpo",
        checkpoint_step=8,
        config=load_evaluation_config(),
    )
    assert result["completion_count"] == 64
    assert result["sampled_pass_at_1"] is None
    assert result["pass_at_4"] is None
    assert result["greedy_accuracy"] is None
    aggregate = json.loads((tmp_path / "validation" / "aggregate_metrics.json").read_text())[
        "aggregate"
    ]
    assert aggregate["greedy_accuracy_available"] is False
    resource = (tmp_path / "validation" / "resource_metrics.csv").read_text()
    assert "unavailable" in resource
    summary = json.loads((tmp_path / "validation" / "resource_summary.json").read_text())
    allocator = json.loads((tmp_path / "validation" / "pytorch_allocator.json").read_text())
    assert summary["available"] is False
    assert allocator["available"] is False


def test_formal_evaluation_rejects_reserved_seed(tmp_path):
    with pytest.raises(ValueError, match="unapproved formal evaluation seed"):
        execute_formal_evaluation(
            FakeEvaluationBackend(),
            phase="baseline",
            seed=2026,
            run_dir=tmp_path / "reserved",
            config=load_evaluation_config(),
        )
