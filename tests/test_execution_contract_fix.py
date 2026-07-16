import copy
import csv
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

from math_rlvr.artifacts.plotting import generate
from math_rlvr.rewards.result import RewardResult, RewardStatus
from math_rlvr.rewards.staged import STAGED_REWARD_POLICY
from math_rlvr.training.builders import ppo_config
from math_rlvr.training.common import preflight
from math_rlvr.training.execution_contract import (
    expected_run_contract,
    expected_run_contract_for_config,
    protected_execution_profiles,
)
from math_rlvr.training.guarded_grpo import run_guarded
from math_rlvr.training.guarded_ppo import (
    ppo_execution_problems_and_episodes,
    run_guarded_ppo,
    write_fake_ppo_checkpoint,
)
from math_rlvr.training.pilot import PILOT_SEEDS, pilot_episode_records, pilot_pair_keys
from math_rlvr.training.trl_compat import (
    PPOBackwardEventGuard,
    TRLContractError,
    extract_ordered_episode_batch,
    install_sequential_ppo_dataloader,
    ppo_loop_position,
    ppo_train_loop_contract,
    require_sequential_sampler,
)


class FakePreparedLoader:
    def __init__(self, loader, mutate=None):
        self.loader = loader
        self.mutate = mutate

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            yield self.mutate(batch) if self.mutate else batch


class FakeAccelerator:
    num_processes = 1

    def __init__(self, mutate=None):
        self.mutate = mutate
        self.prepare_calls = 0

    def prepare_data_loader(self, loader):
        self.prepare_calls += 1
        return FakePreparedLoader(loader, self.mutate)


def collate_inputs(features):
    return {"input_ids": torch.tensor([row["input_ids"] for row in features])}


def fake_ppo_trainer(seed=42, *, rows=None, accelerator=None):
    records = pilot_episode_records("ppo", seed)
    dataset = rows or [{"input_ids": [row["episode_position"] + 1], **row} for row in records]
    return (
        SimpleNamespace(
            train_dataset=dataset,
            local_dataloader_batch_size=16,
            data_collator=collate_inputs,
            dataloader=DataLoader(dataset, batch_size=16, shuffle=True),
            accelerator=accelerator or FakeAccelerator(),
        ),
        records,
    )


def test_trl_024_train_consumes_replaced_self_dataloader_without_reprepare_models():
    import trl.trainer.ppo_trainer as ppo_module

    source = inspect.getsource(ppo_module.PPOTrainer)
    assert "shuffle=True" in source
    assert "self.model, self.optimizer, self.dataloader = accelerator.prepare" in source
    assert "dataloader = self.dataloader" in source
    assert "yield from dataloader" in source
    assert "data = next(iter_dataloader)" in source
    shim = inspect.getsource(install_sequential_ppo_dataloader)
    assert "prepare_data_loader(loader)" in shim
    assert "accelerator.prepare(" not in shim


@pytest.mark.parametrize("seed", PILOT_SEEDS)
def test_sequential_loader_preserves_prepared_prompt_major_order_for_every_seed(seed):
    trainer, records = fake_ppo_trainer(seed)
    contract = expected_run_contract(Path(f"configs/pilot/resolved/ppo_seed_{seed}.json"), "ppo")
    evidence = install_sequential_ppo_dataloader(trainer, records, contract)
    assert evidence["trl_original_sampler_type"] == "RandomSampler"
    assert evidence["replacement_sampler_type"] == "SequentialSampler"
    assert evidence["batch_size"] == 16
    assert evidence["drop_last"] is True
    assert evidence["num_workers"] == 0
    assert evidence["world_size"] == 1
    assert evidence["prepared_first_batch_pair_keys"] == pilot_pair_keys()
    assert trainer.accelerator.prepare_calls == 1
    assert require_sequential_sampler(trainer.dataloader) == "SequentialSampler"
    actual = extract_ordered_episode_batch(next(iter(trainer.dataloader)))
    assert actual == records
    assert [row["episode_position"] for row in actual] == list(range(16))
    assert [row["pair_key"] for row in actual] == pilot_pair_keys()


def test_random_sampler_is_rejected_and_accelerator_reordering_fails_closed():
    trainer, records = fake_ppo_trainer()
    with pytest.raises(TRLContractError, match="SequentialSampler"):
        require_sequential_sampler(trainer.dataloader)

    def swap_first_two(batch):
        changed = dict(batch)
        for key, values in batch.items():
            if hasattr(values, "clone"):
                values = values.clone()
                values[[0, 1]] = values[[1, 0]]
            else:
                values = list(values)
                values[0], values[1] = values[1], values[0]
            changed[key] = values
        return changed

    trainer, records = fake_ppo_trainer(accelerator=FakeAccelerator(swap_first_two))
    contract = expected_run_contract(Path("configs/pilot/resolved/ppo_seed_42.json"), "ppo")
    with pytest.raises(TRLContractError, match="order/identity mismatch"):
        install_sequential_ppo_dataloader(trainer, records, contract)


@pytest.mark.parametrize("mutation", ["swap", "missing", "duplicate", "hash"])
def test_dataset_order_missing_duplicate_and_hash_drift_are_rejected(mutation):
    trainer, records = fake_ppo_trainer()
    rows = list(trainer.train_dataset)
    if mutation == "swap":
        rows[0], rows[1] = rows[1], rows[0]
    elif mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[1] = dict(rows[0])
    else:
        rows[0] = {**rows[0], "rendered_prompt_hash": "0" * 64}
    trainer, _ = fake_ppo_trainer(rows=rows)
    contract = expected_run_contract(Path("configs/pilot/resolved/ppo_seed_42.json"), "ppo")
    with pytest.raises(TRLContractError):
        install_sequential_ppo_dataloader(trainer, records, contract)


def test_four_protected_profiles_are_exact_and_main_configs_are_rejected():
    profiles = {profile.profile: profile for profile in protected_execution_profiles()}
    assert {
        name: (profile.expected_completions, profile.generated_token_cap)
        for name, profile in profiles.items()
    } == {
        "ppo_stage_d_smoke": (4, 512),
        "grpo_stage_d_smoke": (8, 1024),
        "ppo_matched_pilot": (16, 2048),
        "grpo_matched_pilot": (16, 2048),
    }
    for profile in profiles.values():
        assert profile.expected_optimizer_steps == profile.expected_global_steps == 1
        assert len(profile.pair_keys) == profile.expected_completions
        assert len(set(profile.pair_keys)) == profile.expected_completions
        assert profile.parser_sha256 and profile.verifier_sha256
    with pytest.raises(ValueError, match="no protected"):
        expected_run_contract(Path("configs/main/ppo.yaml"), "ppo")


def test_profile_hash_and_config_budget_cannot_be_widened(monkeypatch):
    import math_rlvr.training.execution_contract as module

    config = preflight(Path("configs/pilot/resolved/ppo_seed_42.json"), "ppo")
    widened = copy.deepcopy(config)
    widened["budget"]["max_completions"] = 17
    with pytest.raises(ValueError, match="completion/token budget"):
        expected_run_contract_for_config(widened, "ppo")
    parser_drift = copy.deepcopy(config)
    parser_drift["parser_contract"]["contract_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="parser/verifier"):
        expected_run_contract_for_config(parser_drift, "ppo")
    manifest_drift = copy.deepcopy(config)
    manifest_drift["data"]["pilot_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="pilot manifest"):
        expected_run_contract_for_config(manifest_drift, "ppo")
    monkeypatch.setitem(module._CONFIG_SHA256, "configs/smoke/ppo.yaml", "0" * 64)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        expected_run_contract(Path("configs/smoke/ppo.yaml"), "ppo")


class FakeLifecycle:
    def __init__(self, root):
        self.root = root
        self.root.mkdir(parents=True)
        self.backed_up = False

    def start(self, config, problems):
        self.persist("started.json", {"problem_ids": [problem.problem_id for problem in problems]})

    def persist(self, name, payload):
        (self.root / name).write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")

    def persist_jsonl(self, name, rows):
        (self.root / name).write_text(
            "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows),
            encoding="utf-8",
        )

    def finalize(self, summary):
        self.persist("final_summary.json", summary)

    def backup_and_verify(self, failure=False):
        if failure:
            raise RuntimeError("fake success backup cannot be a failure archive")
        self.backed_up = True


class FakeMonitor:
    def __init__(self):
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


def passing_evaluation(text):
    return STAGED_REWARD_POLICY.evaluate(
        text, lambda _: RewardResult(RewardStatus.VERIFIED_PASS, "fake-pass")
    )


class PilotPPOBackend:
    def __init__(
        self,
        config,
        checkpoint,
        completions=16,
        tokens=16,
        optimizer_steps=1,
        global_step=1,
    ):
        self.config = config
        self.checkpoint = checkpoint
        self.completions = completions
        self.tokens = tokens
        self.optimizer_steps = optimizer_steps
        self.global_step = global_step

    def run(self, problems, guard):
        contract = expected_run_contract_for_config(self.config, "ppo")
        _, episodes = ppo_execution_problems_and_episodes(self.config, contract)
        trainer, records = fake_ppo_trainer(self.config["experiment"]["seed"])
        loader = install_sequential_ppo_dataloader(trainer, records, contract)
        next(iter(trainer.dataloader))
        guard.record_generation(self.completions, self.tokens)
        text = "<reasoning>x</reasoning><answer>1</answer>"
        for _ in range(self.completions):
            evaluation = passing_evaluation(text)
            guard.record_reward(
                evaluation.canonical_result, evaluation.scalar_reward, evaluation.to_dict()
            )
        args = ppo_config(self.config, self.checkpoint.parent / "fake-ppo-config", cpu_only=True)
        loop_contract = ppo_train_loop_contract(args, contract, world_size=1)
        backward_guard = PPOBackwardEventGuard(
            loop_contract,
            {"num_steps": 4, "sync_with_dataloader": False},
        )
        for microbatch_index in range(loop_contract["microbatches_per_minibatch"]):
            backward_guard.note_training_forward(args.per_device_train_batch_size)
            event = backward_guard.prepare_backward(
                microbatch_index == loop_contract["microbatches_per_minibatch"] - 1
            )
            backward_guard.commit_backward(event)
        for optimizer_step_index in range(self.optimizer_steps):
            backward_guard.assert_ready_for_optimizer()
            guard.record_loop_position(*ppo_loop_position(optimizer_step_index, args))
            guard.record_optimizer_step()
        backward_guard.assert_complete(self.optimizer_steps)
        guard.record_update()
        guard.record_global_step(self.global_step)
        lengths = [self.tokens // self.completions] * self.completions
        rows = []
        if self.completions == 16:
            for index, (episode, reward, length) in enumerate(
                zip(episodes, guard.rewards, lengths, strict=True)
            ):
                rows.append(
                    {
                        **episode,
                        "prompt_hash": episode["rendered_prompt_hash"],
                        "completion_index": index,
                        "prompt_token_ids": [index + 1],
                        "response_token_ids": [100 + index] * length,
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
            "completions": rows,
            "metrics": {"policy_loss": 0.1},
            "trainer_log_history": [{"loss/policy_avg": 0.1}],
            "model_roles": {"optimizer_exact_role_match": True},
            "episode_records": episodes,
            "loader_contract": loader,
        }


class PilotGRPOBackend:
    def __init__(self, checkpoint, completions=16, tokens=16, optimizer_steps=1, global_step=1):
        self.checkpoint = checkpoint
        self.completions = completions
        self.tokens = tokens
        self.optimizer_steps = optimizer_steps
        self.global_step = global_step

    def run(self, problems, guard, reward_fn):
        guard.record_generation(self.completions, self.tokens)
        text = "<reasoning>x</reasoning><answer>1</answer>"
        for _ in range(self.completions):
            reward_fn(text)
        for _ in range(4):
            guard.record_microstep()
        for _ in range(self.optimizer_steps):
            guard.record_optimizer_step()
        guard.record_global_step(self.global_step)
        rows = []
        if self.completions == 16:
            for index, (problem, reward) in enumerate(
                zip(
                    [problem for problem in problems for _ in range(4)],
                    guard.rewards,
                    strict=True,
                )
            ):
                generation_index = index % 4
                rows.append(
                    {
                        "problem_id": problem.problem_id,
                        "prompt_hash": problem.content_hash,
                        "generation_index": generation_index,
                        "pair_key": f"{problem.problem_id}::generation:{generation_index}",
                        "completion_index": index,
                        "completion_ids": [index + 1],
                        "completion_mask": [1],
                        "exact_token_count": 1,
                        "decoded_completion": text,
                        "raw_completion": text,
                        "verifier_input": text,
                        "reward_status": reward["status"],
                        "scalar_reward": reward["reward"],
                        "verifier_detail": reward["detail"],
                        **{
                            key: value
                            for key, value in reward.items()
                            if key not in {"status", "reward", "detail"}
                        },
                    }
                )
        return {
            "checkpoint_dir": str(self.checkpoint),
            "completions": rows,
            "metrics": {"loss": 0.1},
            "trainer_log_history": [{"loss": 0.1}],
        }


def grpo_checkpoint(root):
    root.mkdir(parents=True)
    (root / "adapter_model.safetensors").write_bytes(b"adapter")
    (root / "adapter_config.json").write_text("{}", encoding="utf-8")
    return root


def test_fake_pilot_ppo_and_grpo_execute_finalize_exactly_sixteen(tmp_path):
    ppo_config = preflight(Path("configs/pilot/resolved/ppo_seed_42.json"), "ppo")
    ppo_lifecycle = FakeLifecycle(tmp_path / "ppo")
    ppo = run_guarded_ppo(
        ppo_config,
        PilotPPOBackend(
            ppo_config, write_fake_ppo_checkpoint(tmp_path / "ppo-checkpoint" / "checkpoint-1")
        ),
        ppo_lifecycle,
        FakeMonitor(),
    )
    assert ppo["status"] == "success", ppo
    assert ppo["counters"]["completions"] == 16
    assert ppo["counters"]["generated_tokens"] == 16
    assert ppo["completion_evidence_count"] == 16
    assert ppo["metrics"]["evidence_counters"] == {
        "completions": 16,
        "generated_tokens": 16,
        "updates": 1,
        "optimizer_steps": 1,
        "global_step": 1,
    }
    assert len((tmp_path / "ppo/completions.jsonl").read_text().splitlines()) == 16

    grpo_config = preflight(Path("configs/pilot/resolved/grpo_seed_42.json"), "grpo")
    grpo_lifecycle = FakeLifecycle(tmp_path / "grpo")
    grpo = run_guarded(
        grpo_config,
        PilotGRPOBackend(grpo_checkpoint(tmp_path / "grpo-checkpoint")),
        lambda _: RewardResult(RewardStatus.VERIFIED_PASS, "fake-pass"),
        grpo_lifecycle,
        FakeMonitor(),
    )
    assert grpo["status"] == "success"
    assert grpo["counters"]["completions"] == 16
    assert grpo["counters"]["generated_tokens"] == 16
    assert grpo["completion_evidence_count"] == 16
    assert grpo["metrics"]["evidence_counters"] == {
        "completions": 16,
        "generated_tokens": 16,
        "updates": 1,
        "optimizer_steps": 1,
        "global_step": 1,
    }
    assert len((tmp_path / "grpo/completions.jsonl").read_text().splitlines()) == 16
    assert ppo["expected_run_contract"]["pair_keys"] == grpo["expected_run_contract"]["pair_keys"]


@pytest.mark.parametrize("algorithm,count", [("ppo", 15), ("ppo", 17), ("grpo", 15), ("grpo", 17)])
def test_pilot_final_count_under_and_over_expected_fail_closed(tmp_path, algorithm, count):
    config = preflight(Path(f"configs/pilot/resolved/{algorithm}_seed_42.json"), algorithm)
    lifecycle = FakeLifecycle(tmp_path / algorithm)
    if algorithm == "ppo":
        result = run_guarded_ppo(
            config,
            PilotPPOBackend(
                config,
                write_fake_ppo_checkpoint(tmp_path / f"{algorithm}-checkpoint" / "checkpoint-1"),
                completions=count,
                tokens=count,
            ),
            lifecycle,
            FakeMonitor(),
        )
    else:
        result = run_guarded(
            config,
            PilotGRPOBackend(
                grpo_checkpoint(tmp_path / f"{algorithm}-checkpoint"),
                completions=count,
                tokens=count,
            ),
            lambda _: RewardResult(RewardStatus.VERIFIED_PASS, "fake-pass"),
            lifecycle,
            FakeMonitor(),
        )
    assert result["status"] == "failure"


@pytest.mark.parametrize(
    "algorithm,overrides",
    [
        ("ppo", {"tokens": 2049}),
        ("ppo", {"optimizer_steps": 2}),
        ("ppo", {"global_step": 2}),
        ("grpo", {"tokens": 2049}),
        ("grpo", {"optimizer_steps": 2}),
        ("grpo", {"global_step": 2}),
    ],
)
def test_pilot_token_optimizer_and_global_step_caps_fail_closed(tmp_path, algorithm, overrides):
    config = preflight(Path(f"configs/pilot/resolved/{algorithm}_seed_42.json"), algorithm)
    lifecycle = FakeLifecycle(tmp_path / algorithm)
    if algorithm == "ppo":
        backend = PilotPPOBackend(
            config,
            write_fake_ppo_checkpoint(tmp_path / f"{algorithm}-checkpoint" / "checkpoint-1"),
            **overrides,
        )
        result = run_guarded_ppo(config, backend, lifecycle, FakeMonitor())
    else:
        backend = PilotGRPOBackend(
            grpo_checkpoint(tmp_path / f"{algorithm}-checkpoint"), **overrides
        )
        result = run_guarded(
            config,
            backend,
            lambda _: RewardResult(RewardStatus.VERIFIED_PASS, "fake-pass"),
            lifecycle,
            FakeMonitor(),
        )
    assert result["status"] == "failure"


def test_sixteen_row_csv_json_and_figures_finalize_without_fixed_four_or_eight(tmp_path):
    rows = [
        {
            "step": index + 1,
            "reward": index / 16,
            "policy_loss": 0.1,
            "value_loss": 0.2,
            "kl": 0.01,
            "entropy": 0.5,
            "correctness": 0.0,
            "format_accuracy": 1.0,
            "parse_success_rate": 1.0,
            "cumulative_generated_tokens": index + 1,
            "mean_completion_length": 1.0,
        }
        for index in range(16)
    ]
    with (tmp_path / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    (tmp_path / "completions.jsonl").write_text(
        "".join(json.dumps({"completion_index": index}) + "\n" for index in range(16))
    )
    made, _ = generate(tmp_path, "fake-pilot", "ppo", 42)
    assert "reward_curve" in made and "ppo_value_loss" in made
    assert (tmp_path / "figures/reward_curve.png").is_file()
    assert len((tmp_path / "completions.jsonl").read_text().splitlines()) == 16


def test_committed_order_and_comparison_csv_match_frozen_records():
    records = pilot_episode_records("ppo", 42)
    with Path("reports/pilot_0p5b/ppo_episode_order.csv").open(newline="") as handle:
        order = list(csv.DictReader(handle))
    assert len(order) == 16
    for expected, actual in zip(records, order, strict=True):
        assert int(actual["episode_position"]) == expected["episode_position"]
        assert actual["problem_id"] == expected["problem_id"]
        assert int(actual["generation_index"]) == expected["generation_index"]
        assert actual["pair_key"] == expected["pair_key"]
        assert actual["problem_hash"] == expected["problem_hash"]
        assert actual["rendered_prompt_hash"] == expected["rendered_prompt_hash"]
        assert actual["seed_scope"] == "42|123|2026"
    with Path("reports/pilot_0p5b/comparison_keys.csv").open(newline="") as handle:
        comparison = list(csv.DictReader(handle))
    assert [row["pair_key"] for row in comparison] == pilot_pair_keys()
    assert all(
        row["ppo_expected_occurrences"] == row["grpo_expected_occurrences"] == "1"
        for row in comparison
    )


def test_cpu_contract_tests_never_initialize_cuda():
    assert torch.cuda.is_initialized() is False
