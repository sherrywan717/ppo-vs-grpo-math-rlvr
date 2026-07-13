import inspect
import json
from pathlib import Path

import pytest
import torch

from math_rlvr.config import load_config, resolve_training_config
from math_rlvr.rewards.result import RewardResult, RewardStatus
from math_rlvr.rewards.staged import (
    STAGED_COMPONENT_WEIGHTS,
    STAGED_REWARD_SHA256,
    STAGED_REWARD_VERSION,
)
from math_rlvr.training.grpo import main
from math_rlvr.training.guarded_grpo import (
    BudgetGuard,
    checkpoint_inventory,
    run_guarded,
    select_smoke_problems,
)
from math_rlvr.training.trl_compat import TRLContractError, exact_completion_counts


class Monitor:
    def __init__(self):
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


class Lifecycle:
    def __init__(self, root, fail_write=False, fail_backup=False):
        self.root = Path(root)
        self.root.mkdir()
        self.fail_write = fail_write
        self.fail_backup = fail_backup
        self.backed_up = False

    def start(self, config, problems):
        self.persist("run_manifest.json", {"ids": [p.problem_id for p in problems]})

    def persist(self, name, payload):
        if self.fail_write and name == "summary.json":
            raise OSError("artifact write failed")
        (self.root / name).write_text(json.dumps(payload))

    def persist_jsonl(self, name, rows):
        (self.root / name).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        )

    def runtime_summary(self):
        return {}

    def finalize(self, summary):
        (self.root / "checksums.sha256").write_text("verified")

    def backup_and_verify(self):
        if self.fail_backup:
            raise OSError("backup failed")
        self.backed_up = True


class Backend:
    def __init__(
        self,
        checkpoint,
        *,
        completions=8,
        tokens=64,
        steps=1,
        global_step=1,
        reward=float(1),
        infra=False,
        microsteps=4,
    ):
        self.checkpoint = checkpoint
        self.values = (completions, tokens, steps, global_step, reward, infra, microsteps)
        self.calls = 0

    def run(self, problems, guard, reward_fn):
        self.calls += 1
        completions, tokens, steps, global_step, reward, infra, microsteps = self.values
        guard.record_generation(completions, tokens)
        for _ in range(completions):
            result = RewardResult(RewardStatus.INFRA_ERROR if infra else RewardStatus.VERIFIED_PASS)
            if infra or reward != 1:
                guard.record_reward(result, reward)
            else:
                reward_fn("<reasoning>x</reasoning><answer>1</answer>")
        for _ in range(microsteps):
            guard.record_microstep()
        for _ in range(steps):
            guard.record_optimizer_step()
        guard.record_global_step(global_step)
        evidence = []
        if completions == 8:
            per_completion = tokens // completions
            remainder = tokens % completions
            for index, (problem, recorded_reward) in enumerate(
                zip(
                    [problems[0]] * 4 + [problems[1]] * 4,
                    guard.rewards,
                    strict=True,
                )
            ):
                exact_count = per_completion + (index < remainder)
                text = "<reasoning>x</reasoning><answer>1</answer>"
                evidence.append(
                    {
                        "problem_id": problem.problem_id,
                        "prompt_hash": problem.content_hash,
                        "generation_index": index % 4,
                        "completion_index": index,
                        "completion_ids": [index + 1] * exact_count,
                        "completion_mask": [1] * exact_count,
                        "exact_token_count": exact_count,
                        "decoded_completion": text,
                        "raw_completion": text,
                        "verifier_input": text,
                        "reward_status": recorded_reward["status"],
                        "scalar_reward": recorded_reward["reward"],
                        "verifier_detail": recorded_reward["detail"],
                        **{
                            key: value
                            for key, value in recorded_reward.items()
                            if key not in {"status", "reward", "detail"}
                        },
                    }
                )
        return {
            "checkpoint_dir": str(self.checkpoint),
            "metrics": {"loss": 0.1},
            "completions": evidence,
        }


def verifier(_):
    return RewardResult(RewardStatus.VERIFIED_PASS)


def checkpoint(tmp_path):
    p = tmp_path / "checkpoint"
    p.mkdir(parents=True)
    (p / "adapter_model.safetensors").write_bytes(b"x")
    (p / "adapter_config.json").write_text("{}")
    return p


def test_cli_dry_run_and_execute_alone_never_call_executor():
    calls = []
    assert main(["--config", "configs/smoke/grpo.yaml"], execute_fn=lambda c: calls.append(c)) == 0
    with pytest.raises(RuntimeError, match="insufficient"):
        main(
            ["--config", "configs/smoke/grpo.yaml", "--execute"],
            execute_fn=lambda c: calls.append(c),
        )
    assert calls == []


def test_dual_confirmation_enters_fake_once():
    calls = []

    def fn(config):
        calls.append(config)
        return {"status": "success"}

    assert (
        main(
            ["--config", "configs/smoke/grpo.yaml", "--execute", "--confirm-single-update"],
            execute_fn=fn,
            git_probe=lambda: {},
            snapshot_probe=lambda: Path("/local"),
        )
        == 0
    )
    assert len(calls) == 1


def test_main_or_ppo_config_cannot_use_smoke_authorization():
    for config in ("configs/main/grpo.yaml", "configs/smoke/ppo.yaml"):
        with pytest.raises((ValueError, RuntimeError)):
            main(
                ["--config", config, "--execute", "--confirm-single-update"],
                execute_fn=lambda c: {"status": "success"},
                git_probe=lambda: {},
                snapshot_probe=lambda: Path("/local"),
            )


def test_fixed_two_training_problems_are_reusable():
    config = load_config("configs/smoke/grpo.yaml")
    a = select_smoke_problems(config)
    b = select_smoke_problems(config)
    assert [x.problem_id for x in a] == [x.problem_id for x in b]
    assert len({x.problem_id for x in a}) == 2 and all(x.split == "train" for x in a)


def test_fake_success_exact_counters_and_monitor_cleanup(tmp_path):
    life = Lifecycle(tmp_path / "run")
    monitor = Monitor()
    result = run_guarded(
        load_config("configs/smoke/grpo.yaml"),
        Backend(checkpoint(tmp_path)),
        verifier,
        life,
        monitor,
    )
    assert result["status"] == "success" and result["backed_up"] is True
    c = result["counters"]
    assert (
        c["completions"],
        c["generated_tokens"],
        c["microsteps"],
        c["optimizer_steps"],
        c["global_step"],
    ) == (8, 64, 4, 1, 1)
    assert monitor.stopped and life.backed_up


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"completions": 9}, "completion"),
        ({"tokens": 1025}, "token"),
        ({"steps": 2}, "second optimizer"),
        ({"global_step": 2}, "global_step"),
        ({"reward": float("nan")}, "non-finite"),
        ({"infra": True}, "infra_error"),
        ({"microsteps": 5}, "microstep"),
    ],
)
def test_fake_failures_stop_before_success(tmp_path, kwargs, reason):
    monitor = Monitor()
    result = run_guarded(
        load_config("configs/smoke/grpo.yaml"),
        Backend(checkpoint(tmp_path), **kwargs),
        verifier,
        Lifecycle(tmp_path / "run"),
        monitor,
    )
    assert result["status"] == "failure" and reason in result["reason"]
    assert monitor.stopped


def test_timeout_fails_and_stops_monitor(tmp_path):
    guard = BudgetGuard(8, 1024, 1, 1, 4, 900, clock=lambda: 901)
    with pytest.raises(TimeoutError):
        guard.record_generation(1, 1)


def test_runner_timeout_marks_failure_and_stops_monitor(tmp_path):
    values = iter((0, 901))
    monitor = Monitor()
    result = run_guarded(
        load_config("configs/smoke/grpo.yaml"),
        Backend(checkpoint(tmp_path)),
        verifier,
        Lifecycle(tmp_path / "run"),
        monitor,
        clock=lambda: next(values),
    )
    assert result["status"] == "failure" and "TimeoutError" in result["reason"]
    assert monitor.stopped


def test_checkpoint_inventory_rejects_full_or_non_adapter_weights(tmp_path):
    p = tmp_path / "c"
    p.mkdir()
    (p / "model.safetensors").write_bytes(b"x")
    with pytest.raises(Exception, match="non-adapter"):
        checkpoint_inventory(p)
    (p / "model.safetensors").unlink()
    (p / "adapter_model.safetensors").write_bytes(b"x" * 11)
    with pytest.raises(Exception, match="full-size"):
        checkpoint_inventory(p, full_weight_threshold=10)


def test_artifact_or_backup_failure_never_marks_success(tmp_path):
    for write, backup in ((True, False), (False, True)):
        monitor = Monitor()
        result = run_guarded(
            load_config("configs/smoke/grpo.yaml"),
            Backend(checkpoint(tmp_path / ("w" if write else "b"))),
            verifier,
            Lifecycle(tmp_path / ("rw" if write else "rb"), write, backup),
            monitor,
        )
        assert (
            result["status"] == "failure"
            and result.get("backed_up") is not True
            and monitor.stopped
        )


def test_trl_shim_exact_fields_shapes_and_version():
    ids = torch.ones((8, 128), dtype=torch.long)
    mask = torch.zeros_like(ids)
    mask[:, :3] = 1
    assert exact_completion_counts({"completion_ids": ids, "completion_mask": mask}) == (8, 24)
    with pytest.raises(TRLContractError):
        exact_completion_counts({"completion_ids": ids})
    with pytest.raises(TRLContractError):
        exact_completion_counts({"completion_ids": ids, "completion_mask": mask[:, 0]})
    assert torch.cuda.is_initialized() is False


def test_budget_snapshot_is_primitive_only_with_fake_clock():
    guard = BudgetGuard(8, 1024, 1, 1, 4, 910.0, clock=lambda: 12.5, started_at=10.0)
    guard.record_generation(2, 20)
    payload = guard.snapshot()
    assert json.loads(json.dumps(payload)) == payload
    assert payload["elapsed_seconds"] == 2.5
    assert "clock" not in payload
    assert not contains_callable(payload)


def contains_callable(value):
    if callable(value):
        return True
    if isinstance(value, dict):
        return any(contains_callable(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_callable(item) for item in value)
    return False


def test_success_and_failure_finalization_are_json_safe_and_checksum_complete(tmp_path):
    for name, kwargs, expected in (
        ("success", {}, "success"),
        ("failure", {"reward": float("nan")}, "failure"),
    ):
        root = tmp_path / name
        result = run_guarded(
            load_config("configs/smoke/grpo.yaml"),
            Backend(checkpoint(tmp_path / f"{name}-checkpoint"), **kwargs),
            verifier,
            Lifecycle(root),
            Monitor(),
        )
        assert result["status"] == expected
        summary = json.loads((root / "summary.json").read_text())
        assert "clock" not in json.dumps(summary)
        assert not contains_callable(summary)
        assert (root / "checksums.sha256").read_text() == "verified"


def test_json_safety_rejects_non_finite_and_callable_values():
    from math_rlvr.training.guarded_grpo import assert_json_safe

    with pytest.raises(TypeError, match="non-finite"):
        assert_json_safe({"loss": float("nan")})
    with pytest.raises(TypeError, match="function"):
        assert_json_safe({"clock": lambda: 0})


def test_non_json_metric_fails_closed_with_primitive_fallback(tmp_path):
    class UnsafeBackend(Backend):
        def run(self, problems, guard, reward_fn):
            result = super().run(problems, guard, reward_fn)
            result["metrics"]["callback"] = lambda: None
            return result

    root = tmp_path / "run"
    result = run_guarded(
        load_config("configs/smoke/grpo.yaml"),
        UnsafeBackend(checkpoint(tmp_path / "checkpoint")),
        verifier,
        Lifecycle(root),
        Monitor(),
    )
    assert result["status"] == "failure"
    fallback = json.loads((root / "failure_report.json").read_text())
    assert fallback["phase"] == "artifact_finalization"
    assert "callable" not in json.dumps(fallback)


def test_staged_reward_identity_and_components_persist_in_fake_artifacts(tmp_path):
    config = resolve_training_config(load_config("configs/smoke/grpo.yaml"))
    life = Lifecycle(tmp_path / "staged-run")
    result = run_guarded(
        config,
        Backend(checkpoint(tmp_path / "staged-checkpoint")),
        verifier,
        life,
        Monitor(),
    )
    assert result["status"] == "success"
    assert result["reward_policy_version"] == STAGED_REWARD_VERSION
    assert result["reward_component_weights"] == STAGED_COMPONENT_WEIGHTS
    assert result["reward_policy_sha256"] == STAGED_REWARD_SHA256
    rows = [
        json.loads(line)
        for line in (life.root / "completions.jsonl").read_text().splitlines()
    ]
    for row in rows:
        assert row["canonical_status"] == row["reward_status"]
        assert row["reward_policy_version"] == STAGED_REWARD_VERSION
        assert row["reward_policy_sha256"] == STAGED_REWARD_SHA256
        assert row["reward_component_weights"] == STAGED_COMPONENT_WEIGHTS
        assert row["scalar_reward"] == 1.0
        assert row["correctness_component"] == 0.8


def test_real_manifest_finalizer_records_reward_identity():
    from math_rlvr.training.grpo_runtime import RealLifecycle

    source = inspect.getsource(RealLifecycle.finalize)
    for field in (
        "reward_policy_version",
        "reward_component_weights",
        "reward_policy_sha256",
    ):
        assert field in source
