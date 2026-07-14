import copy
import csv
import hashlib
import inspect
import json
import math
from pathlib import Path

import pytest
import torch

from math_rlvr.config import load_config, resolve_training_config
from math_rlvr.rewards.result import RewardResult, RewardStatus
from math_rlvr.rewards.staged import STAGED_REWARD_POLICY
from math_rlvr.training.builders import (
    audit_ppo_parameter_roles,
    load_policy_and_tokenizer,
    load_value_model,
    ppo_config,
)
from math_rlvr.training.execution_contract import expected_run_contract
from math_rlvr.training.guarded_ppo import (
    PPOBudgetGuard,
    fake_reload_ppo_checkpoint,
    ppo_checkpoint_inventory,
    ppo_execution_problems_and_episodes,
    run_guarded_ppo,
    write_fake_ppo_checkpoint,
)
from math_rlvr.training.model_source import ConfigIdentity, ValidatedModelSource
from math_rlvr.training.ppo import main
from math_rlvr.training.ppo_runtime import (
    RealPPOMonitor,
    write_authoritative_ppo_checkpoint,
)
from math_rlvr.training.trl_compat import (
    PPOCompletionEvidenceRecorder,
    TRLContractError,
    enforce_ppo_generation_contract,
    extract_ppo_metrics,
    validate_ppo_value_shape,
)


def resolved_config():
    return resolve_training_config(load_config("configs/smoke/ppo.yaml"))


class Monitor:
    def __init__(self, fail_start=False, fail_stop=False):
        self.fail_start = fail_start
        self.fail_stop = fail_stop
        self.started = False
        self.stopped = False

    def start(self):
        if self.fail_start:
            raise OSError("monitor start failed")
        self.started = True

    def stop(self):
        self.stopped = True
        if self.fail_stop:
            raise OSError("monitor stop failed")


class Lifecycle:
    def __init__(self, root, fail_start=False, fail_write=False, fail_backup=False):
        self.root = Path(root)
        self.root.mkdir()
        self.fail_start = fail_start
        self.fail_write = fail_write
        self.fail_backup = fail_backup
        self.backed_up = False

    def start(self, config, problems):
        if self.fail_start:
            raise OSError("lifecycle start failed")
        self.persist("run_manifest.json", {"ids": [p.problem_id for p in problems]})

    def persist(self, name, payload):
        if self.fail_write and name == "summary.json":
            raise OSError("artifact write failed")
        (self.root / name).write_text(json.dumps(payload, allow_nan=False))

    def persist_jsonl(self, name, rows):
        (self.root / name).write_text(
            "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows)
        )

    def finalize(self, summary):
        (self.root / "checksums.sha256").write_text("verified")

    def backup_and_verify(self, failure=False):
        if self.fail_backup:
            raise OSError("backup failed")
        self.backed_up = True


def passing_evaluation(completion):
    return STAGED_REWARD_POLICY.evaluate(
        completion, lambda _: RewardResult(RewardStatus.VERIFIED_PASS, "fake-pass")
    )


class Backend:
    def __init__(self, checkpoint, **overrides):
        self.checkpoint = checkpoint
        self.overrides = overrides
        self.calls = 0

    def run(self, problems, guard):
        self.calls += 1
        contract = expected_run_contract(Path("configs/smoke/ppo.yaml"), "ppo")
        _, episode_records = ppo_execution_problems_and_episodes(resolved_config(), contract)
        completions = self.overrides.get("completions", 4)
        tokens = self.overrides.get("tokens", 8)
        guard.record_generation(completions, tokens)
        text = "<reasoning>x</reasoning><answer>1</answer>"
        for _ in range(completions):
            evaluation = passing_evaluation(text)
            guard.record_reward(
                evaluation.canonical_result,
                self.overrides.get("reward", evaluation.scalar_reward),
                evaluation.to_dict(),
            )
        for _ in range(self.overrides.get("minibatches", 1)):
            guard.record_epoch_minibatch()
        for _ in range(self.overrides.get("optimizer_steps", 1)):
            guard.record_optimizer_step()
        for _ in range(self.overrides.get("updates", 1)):
            guard.record_update()
        guard.record_global_step(self.overrides.get("global_step", 1))
        records = []
        if completions == 4:
            lengths = [tokens // 4 + (index < tokens % 4) for index in range(4)]
            for index, (problem, reward, length) in enumerate(
                zip(problems, guard.rewards, lengths, strict=True)
            ):
                records.append(
                    {
                        "problem_id": problem.problem_id,
                        "prompt_hash": problem.content_hash,
                        "generation_index": 0,
                        "pair_key": f"{problem.problem_id}::generation:0",
                        "completion_index": index,
                        "prompt_token_ids": [index + 1],
                        "response_token_ids": [10 + index] * length,
                        "response_mask": [1] * length,
                        "exact_token_count": length,
                        "decoded_completion": text,
                        "reward_callback_text": text,
                        "verifier_input": text,
                        "scalar_reward": reward["reward"],
                        "canonical_status": reward["status"],
                    }
                )
        return {
            "checkpoint_dir": str(self.checkpoint),
            "metrics": self.overrides.get("metrics", {"policy_loss": 0.1}),
            "trainer_log_history": self.overrides.get(
                "trainer_log_history", [{"loss/policy_avg": 0.1}]
            ),
            "completions": records,
            "model_roles": {"optimizer_exact_role_match": True},
            "episode_records": episode_records,
        }


def fake_checkpoint(tmp_path):
    return write_fake_ppo_checkpoint(tmp_path / "checkpoint-1")


def test_cli_dry_run_execute_alone_and_dual_confirmation():
    calls = []
    assert main(["--config", "configs/smoke/ppo.yaml"], execute_fn=calls.append) == 0
    with pytest.raises(RuntimeError, match="insufficient"):
        main(["--config", "configs/smoke/ppo.yaml", "--execute"], execute_fn=calls.append)
    assert calls == []

    def execute(config):
        calls.append(config)
        return {"status": "success"}

    assert (
        main(
            [
                "--config",
                "configs/smoke/ppo.yaml",
                "--execute",
                "--confirm-single-update",
            ],
            execute_fn=execute,
            git_probe=lambda: {},
            offline_probe=lambda: {},
            snapshot_probe=lambda: Path("/fixed-local-snapshot"),
        )
        == 0
    )
    assert len(calls) == 1


def test_fake_guarded_ppo_success_exact_contract(tmp_path):
    lifecycle = Lifecycle(tmp_path / "run")
    monitor = Monitor()
    result = run_guarded_ppo(
        resolved_config(), Backend(fake_checkpoint(tmp_path)), lifecycle, monitor
    )
    assert result["status"] == "success"
    assert result["backed_up"] is True
    assert result["counters"]["completions"] == 4
    assert result["counters"]["generated_tokens"] == 8
    assert result["counters"]["updates"] == 1
    assert result["counters"]["optimizer_steps"] == 1
    assert result["counters"]["global_step"] == 1
    assert result["counters"]["ppo_epochs"] == 1
    assert result["counters"]["minibatches"] == 1
    assert result["resolved_ppo_contract"]["total_completions"] == 4
    assert monitor.started and monitor.stopped and lifecycle.backed_up


def test_nullable_ratio_variance_warns_and_finalizes_successfully(tmp_path):
    history = [
        {
            "loss/policy_avg": -0.01,
            "loss/value_avg": 0.2,
            "objective/kl": 0.03,
            "objective/scores": 0.1,
            "val/ratio_var": float("nan"),
        }
    ]
    metrics = extract_ppo_metrics(history)
    ratio_variance = metrics["normalized"]["ratio_variance"]
    assert ratio_variance == {
        "available": False,
        "value": None,
        "raw_key": "val/ratio_var",
        "classification": "non_finite",
        "non_finite_kind": "nan",
        "reason": (
            "TRL 0.24.0 may emit an undefined sample variance when only one "
            "ratio observation is available; this diagnostic is not used for "
            "rewards, losses, optimization, checkpoint counters, or budgets"
        ),
    }
    assert metrics["raw_log_history"][0]["val/ratio_var"] is None
    assert metrics["nullable_telemetry"][0]["raw_key"] == "val/ratio_var"
    assert metrics["warnings"][0]["category"] == "nullable_nonessential_telemetry"
    assert math.isnan(history[0]["val/ratio_var"])
    json.dumps(metrics, allow_nan=False)

    lifecycle = Lifecycle(tmp_path / "run")
    result = run_guarded_ppo(
        resolved_config(),
        Backend(
            fake_checkpoint(tmp_path),
            metrics=metrics,
            trainer_log_history=metrics["raw_log_history"],
        ),
        lifecycle,
        Monitor(),
    )
    assert result["status"] == "success"
    assert result["backed_up"] is True
    persisted = json.loads((tmp_path / "run" / "trainer_metrics.json").read_text())
    assert persisted["normalized"]["ratio_variance"]["value"] is None


@pytest.mark.parametrize(
    ("raw_key", "value"),
    [
        ("loss/policy_avg", float("nan")),
        ("loss/value_avg", float("inf")),
        ("objective/kl", float("-inf")),
        ("objective/scores", float("nan")),
        ("val/unreviewed_metric", float("nan")),
    ],
)
def test_required_or_unreviewed_nonfinite_ppo_metrics_fail_closed(raw_key, value):
    row = {
        "loss/policy_avg": 0.1,
        "loss/value_avg": 0.2,
        "objective/kl": 0.0,
        "objective/scores": 0.1,
    }
    row[raw_key] = value
    with pytest.raises(TRLContractError, match=raw_key):
        extract_ppo_metrics([row])


def test_nullable_telemetry_normalization_does_not_modify_historical_ppo_evidence():
    root = Path("reports/runs/ppo_single_update_qwen25_05b_20260714T051538Z")
    protected = (
        "failure_report.json",
        "launcher_output.txt",
        "completions.jsonl",
        "checkpoint_inventory.json",
    )
    before = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in protected}
    extract_ppo_metrics([{"loss/policy_avg": 0.1, "val/ratio_var": float("nan")}])
    after = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in protected}
    assert after == before


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"completions": 3}, "all protected responses"),
        ({"completions": 5}, "completion cap"),
        ({"tokens": 513}, "token cap"),
        ({"minibatches": 2}, "minibatch cap"),
        ({"optimizer_steps": 2}, "optimizer-step cap"),
        ({"updates": 2}, "outer-update cap"),
        ({"global_step": 2}, "global-step cap"),
    ],
)
def test_fake_budget_failures_stop_monitor(tmp_path, overrides, reason):
    monitor = Monitor()
    result = run_guarded_ppo(
        resolved_config(),
        Backend(fake_checkpoint(tmp_path), **overrides),
        Lifecycle(tmp_path / "run"),
        monitor,
    )
    assert result["status"] == "failure"
    assert reason in result["reason"]
    assert monitor.stopped


def test_nonfinite_reward_and_timeout_fail_closed(tmp_path):
    result = run_guarded_ppo(
        resolved_config(),
        Backend(fake_checkpoint(tmp_path), reward=float("nan")),
        Lifecycle(tmp_path / "run"),
        Monitor(),
    )
    assert result["status"] == "failure"
    assert "non-finite" in result["reason"]

    times = iter((0.0, 1201.0))
    result = run_guarded_ppo(
        resolved_config(),
        Backend(fake_checkpoint(tmp_path / "timeout")),
        Lifecycle(tmp_path / "timeout-run"),
        Monitor(),
        clock=lambda: next(times),
    )
    assert result["status"] == "failure"
    assert "deadline" in result["reason"]


def test_lifecycle_monitor_and_artifact_failures_never_succeed(tmp_path):
    checkpoint = fake_checkpoint(tmp_path)
    cases = (
        (Lifecycle(tmp_path / "start", fail_start=True), Monitor()),
        (Lifecycle(tmp_path / "monitor"), Monitor(fail_start=True)),
        (Lifecycle(tmp_path / "stop"), Monitor(fail_stop=True)),
        (Lifecycle(tmp_path / "write", fail_write=True), Monitor()),
        (Lifecycle(tmp_path / "backup", fail_backup=True), Monitor()),
    )
    for lifecycle, monitor in cases:
        result = run_guarded_ppo(resolved_config(), Backend(checkpoint), lifecycle, monitor)
        assert result["status"] == "failure"
    assert (
        json.loads((tmp_path / "write" / "failure_report.json").read_text())["status"] == "failure"
    )


def test_fake_checkpoint_inventory_reload_and_full_weight_rejection(tmp_path):
    root = fake_checkpoint(tmp_path)
    reloaded = fake_reload_ppo_checkpoint(root)
    assert reloaded["roles"] == ["policy_adapter", "value_adapter", "value_head"]
    assert reloaded["inventory"]["total_size_bytes"] > 0
    (root / "model.safetensors").write_bytes(b"base")
    with pytest.raises(Exception, match="unexpected PPO checkpoint file"):
        ppo_checkpoint_inventory(root)


def test_authoritative_checkpoint_writer_partitions_roles(tmp_path, monkeypatch):
    class Config:
        def save_pretrained(self, root):
            Path(root, "adapter_config.json").write_text("{}\n")

    class FakeModel:
        def __init__(self, role):
            self.role = role
            self.peft_config = {"default": Config()}

    policy = FakeModel("policy")
    value = FakeModel("value")

    def state(model):
        if model.role == "policy":
            return {"base.q_proj.lora_A.weight": torch.ones(1)}
        return {
            "base.v_proj.lora_A.weight": torch.ones(1),
            "base.score.modules_to_save.default.weight": torch.ones(1),
        }

    monkeypatch.setattr("peft.get_peft_model_state_dict", state)
    root = write_authoritative_ppo_checkpoint(
        tmp_path / "checkpoint-1",
        policy,
        value,
        {"global_step": 1},
        {"optimizer_exact_role_match": True},
    )
    inventory = ppo_checkpoint_inventory(root)
    assert {row["classification"] for row in inventory["files"]} >= {
        "policy_adapter",
        "value_adapter",
        "value_head",
    }
    assert not any("model.safetensors" == row["name"] for row in inventory["files"])


def test_real_monitor_persists_resource_tables_and_plots(tmp_path):
    class Manager:
        run_dir = tmp_path
        run_id = "fake-ppo"

        def write_csv(self, name, rows, fieldnames):
            with (tmp_path / name).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

    class Lifecycle:
        config = resolved_config()
        manager = Manager()

        def persist(self, name, payload):
            (tmp_path / name).write_text(json.dumps(payload))

        def persist_jsonl(self, name, rows):
            (tmp_path / name).write_text("".join(json.dumps(row) + "\n" for row in rows))

    (tmp_path / "metrics.csv").write_text(
        "step,reward,policy_loss,value_loss,kl,entropy,correctness,"
        "format_accuracy,parse_success_rate,cumulative_generated_tokens,"
        "mean_completion_length\n1,0.1,0.2,0.3,0.01,0.4,0,0,0,8,2\n"
    )
    wrapper = RealPPOMonitor(Lifecycle())

    class Monitor:
        rows = [
            {
                "timestamp": "2026-07-13T00:00:00+00:00",
                "gpu_memory_used_mb": 10.0,
                "gpu_utilization_pct": 20.0,
                "power_draw_w": 30.0,
                "temperature_c": 40.0,
                "process_rss_mb": 50.0,
                "elapsed_seconds": 1.0,
            }
        ]

        def stop(self):
            return None

        def summary(self, price):
            return {
                "peak_vram_mb": 10.0,
                "mean_gpu_utilization": 20.0,
                "gpu_hours": 1 / 3600,
                "estimated_cost_cny": price / 3600,
            }

    wrapper.monitor = Monitor()
    wrapper.stop()
    assert (tmp_path / "gpu_metrics.csv").is_file()
    assert (tmp_path / "resource_metrics.jsonl").is_file()
    assert (tmp_path / "resource_summary.json").is_file()
    plots = json.loads((tmp_path / "plot_inventory.json").read_text())
    assert {"ppo_value_loss", "gpu_memory", "gpu_utilization"} <= set(plots["generated"])


def test_real_trl_024_ppo_config_has_four_responses_not_sixteen(tmp_path):
    import trl.trainer.ppo_trainer as ppo_module

    config = load_config("configs/smoke/ppo.yaml")
    args = ppo_config(config, tmp_path, cpu_only=True)
    assert args.total_episodes == 4
    assert args.per_device_train_batch_size == 4
    assert args.gradient_accumulation_steps == 1
    assert args.num_ppo_epochs == 1
    assert args.num_mini_batches == 1
    assert args.local_rollout_forward_batch_size == 4
    assert args.response_length == 128
    assert args.num_sample_generations == 0
    assert not hasattr(args, "num_generations")
    source = inspect.getsource(ppo_module.PPOTrainer)
    assert (
        "args.local_batch_size = args.per_device_train_batch_size "
        "* args.gradient_accumulation_steps" in source
    )
    assert "args.total_episodes / args.batch_size" in source
    assert "num_generations" not in source
    assert torch.cuda.is_initialized() is False


def test_completion_recorder_and_scalar_value_shape_contract():
    class Tokenizer:
        def decode(self, ids, skip_special_tokens=True):
            return ",".join(str(value) for value in ids)

    queries = torch.tensor([[0, 1], [0, 2], [0, 3], [0, 4]])
    responses = torch.tensor([[0, 1, 11, 0], [0, 2, 12, 0], [0, 3, 13, 0], [0, 4, 14, 0]])
    contract = expected_run_contract(Path("configs/smoke/ppo.yaml"), "ppo")
    _, episodes = ppo_execution_problems_and_episodes(resolved_config(), contract)
    lookup = {
        (index,): {
            "problem_id": episode["problem_id"],
            "prompt_hash": episode["rendered_prompt_hash"],
        }
        for index, episode in enumerate(episodes, start=1)
    }
    guard = PPOBudgetGuard(
        max_completions=contract.expected_completions,
        max_tokens=contract.generated_token_cap,
        max_updates=contract.expected_updates,
        max_optimizer_steps=contract.expected_optimizer_steps,
        max_global_steps=contract.expected_global_steps,
        max_epochs=contract.expected_ppo_epochs,
        max_minibatches=contract.expected_minibatches,
        deadline=1000,
        clock=lambda: 0,
    )
    recorder = PPOCompletionEvidenceRecorder(contract, episodes)
    count, tokens = recorder.capture_generation(
        queries, responses, Tokenizer(), lookup, pad_token_id=0
    )
    guard.record_generation(count, tokens)
    for index in range(1, 5):
        text = str(10 + index)
        recorder.record_reward(text, passing_evaluation(text), guard)
    assert len(recorder.records()) == 4
    validate_ppo_value_shape(torch.zeros(4, 8, 1), 4, 8)
    with pytest.raises(TRLContractError):
        validate_ppo_value_shape(torch.zeros(4, 8), 4, 8)


def test_ppo_generation_shim_applies_yaml_top_p_and_rejects_drift():
    class Generation:
        max_new_tokens = 128
        temperature = 0.8000001
        top_p = 1.0

    generation = Generation()
    contract = {"max_new_tokens": 128, "temperature": 0.8, "top_p": 0.95}
    enforce_ppo_generation_contract(generation, contract)
    assert generation.top_p == 0.95
    generation.max_new_tokens = 129
    with pytest.raises(TRLContractError, match="max_new_tokens"):
        enforce_ppo_generation_contract(generation, contract)


def test_optimizer_roles_are_exact_and_disjoint():
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_policy = torch.nn.Parameter(torch.ones(2))
            self.base = torch.nn.Parameter(torch.ones(2), requires_grad=False)

    class Value(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_value = torch.nn.Parameter(torch.ones(3))
            self.score = torch.nn.Parameter(torch.ones(1))
            self.base = torch.nn.Parameter(torch.ones(2), requires_grad=False)

    policy, value = Policy(), Value()
    reward = torch.nn.Identity()
    optimizer = torch.optim.AdamW(
        [
            parameter
            for model in (policy, value)
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
    )
    roles = audit_ppo_parameter_roles(policy, value, reward, ref_model=None, optimizer=optimizer)
    assert roles["optimizer_exact_role_match"] is True
    assert roles["policy_trainable_parameters"] == 2
    assert roles["value_trainable_parameters"] == 4

    wrong = torch.optim.AdamW([policy.lora_policy])
    with pytest.raises(RuntimeError, match="optimizer role mismatch"):
        audit_ppo_parameter_roles(policy, value, reward, ref_model=None, optimizer=wrong)


def test_policy_and_value_loaders_use_only_same_fixed_snapshot(monkeypatch, tmp_path):
    config = load_config("configs/smoke/ppo.yaml")
    snapshot = tmp_path / config["model"]["revision"]
    snapshot.mkdir()
    source = ValidatedModelSource(
        repo_id=config["model"]["name_or_path"],
        revision=config["model"]["revision"],
        cache_root=tmp_path,
        snapshot_path=snapshot,
        local_files_only=True,
        config_identity=ConfigIdentity("qwen2", ("Qwen2ForCausalLM",)),
    )
    calls = {}

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 1
        padding_side = "right"

        @classmethod
        def from_pretrained(cls, name, **kwargs):
            calls["tokenizer"] = (name, kwargs)
            return cls()

    class Model:
        class Config:
            use_cache = True
            pad_token_id = None
            eos_token_id = 1

        config = Config()

    def causal_loader(name, **kwargs):
        calls["policy"] = (name, kwargs)
        return Model()

    def value_loader(name, **kwargs):
        calls["value"] = (name, kwargs)
        return Model()

    peft_roles = []

    def peft_model(model, peft_config):
        peft_roles.append(peft_config)
        return model

    monkeypatch.setattr("transformers.AutoTokenizer", Tokenizer)
    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", causal_loader)
    monkeypatch.setattr(
        "transformers.AutoModelForSequenceClassification.from_pretrained", value_loader
    )
    monkeypatch.setattr("peft.get_peft_model", peft_model)
    policy, tokenizer = load_policy_and_tokenizer(config, source)
    value = load_value_model(config, source)
    assert policy is not value and tokenizer.padding_side == "left"
    assert calls["policy"][0] == calls["value"][0] == str(snapshot)
    assert calls["tokenizer"][0] == str(snapshot)
    assert calls["policy"][1]["local_files_only"] is True
    assert calls["value"][1]["local_files_only"] is True
    assert calls["tokenizer"][1]["local_files_only"] is True
    assert calls["value"][1]["num_labels"] == 1
    assert peft_roles[0].r == 16 and peft_roles[0].target_modules == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    }
    assert peft_roles[1].r == 8 and peft_roles[1].target_modules == {
        "q_proj",
        "v_proj",
    }
    assert peft_roles[1].modules_to_save == ["score"]


def test_resolved_prompt_and_reward_identity_matches_grpo():
    ppo = resolved_config()
    grpo = resolve_training_config(load_config("configs/smoke/grpo.yaml"))
    for key in (
        "prompt_version",
        "prompt_sha256",
        "renderer_version",
        "reward_policy_version",
        "reward_policy_sha256",
        "reward_component_weights",
    ):
        assert ppo[key] == grpo[key]


def test_generation_num_generations_mutation_is_rejected_not_multiplied():
    config = copy.deepcopy(load_config("configs/smoke/ppo.yaml"))
    config["generation"]["num_generations"] = 16
    with pytest.raises(ValueError, match="generation contract"):
        from math_rlvr.config import validate_training_config

        validate_training_config(config, "ppo")
