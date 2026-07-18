import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import trl
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from math_rlvr.config import load_config
from math_rlvr.training.builders import ppo_config
from math_rlvr.training.execution_contract import expected_run_contract
from math_rlvr.training.guarded_grpo import BudgetExceededError
from math_rlvr.training.guarded_ppo import PPOBudgetGuard
from math_rlvr.training.trl_compat import (
    PPOBackwardEventGuard,
    TRLContractError,
    configure_ppo_gradient_accumulation,
    ppo_guarded_trainer_class,
    ppo_loop_position,
    ppo_train_loop_contract,
)


def budget_guard(completions=16):
    guard = PPOBudgetGuard(
        max_completions=completions,
        max_tokens=2048,
        max_updates=1,
        max_optimizer_steps=1,
        max_global_steps=1,
        max_epochs=1,
        max_minibatches=1,
        deadline=10**18,
    )
    guard.completions = completions
    guard.rewards = [{} for _ in range(completions)]
    return guard


def test_seed42_resolved_builder_and_real_ppo_config_fields_match(monkeypatch, tmp_path):
    config = load_config("configs/pilot/resolved/ppo_seed_42.json")
    original = trl.PPOConfig
    captured = {}

    def capture_ppo_config(*args, **kwargs):
        captured.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(trl, "PPOConfig", capture_ppo_config)
    args = ppo_config(config, tmp_path, cpu_only=True)
    expected = {
        "total_episodes": 16,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "local_rollout_forward_batch_size": 4,
        "num_ppo_epochs": 1,
        "num_mini_batches": 1,
    }
    assert {key: config["training"][key] for key in expected} == expected
    assert {key: captured[key] for key in expected} == expected
    assert {key: getattr(args, key) for key in expected} == expected

    contract = expected_run_contract(Path("configs/pilot/resolved/ppo_seed_42.json"), "ppo")
    derived = ppo_train_loop_contract(args, contract, world_size=1)
    assert derived == {
        **expected,
        "local_batch_size": 16,
        "micro_batch_size": 4,
        "batch_size": 16,
        "local_mini_batch_size": 16,
        "mini_batch_size": 16,
        "microbatches_per_minibatch": 4,
        "num_total_batches": 1,
        "outer_updates": 1,
        "expected_optimizer_steps": 1,
        "expected_global_steps": 1,
    }
    source = inspect.getsource(trl.PPOTrainer.train)
    assert "for ppo_epoch_idx in range(args.num_ppo_epochs)" in source
    assert (
        "for mini_batch_start in range(0, args.local_batch_size, "
        "args.local_mini_batch_size)" in source
    )
    assert (
        "for micro_batch_start in range(0, args.local_mini_batch_size, "
        "args.per_device_train_batch_size)" in source
    )
    assert torch.cuda.is_initialized() is False


def test_real_ppo_trainer_constructor_derives_frozen_cpu_loop_args(tmp_path):
    class TinyPolicy(nn.Module):
        is_gradient_checkpointing = False

        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(2, 2)
            self.config = SimpleNamespace()
            self.generation_config = SimpleNamespace(eos_token_id=2)

        def forward(self, *args, **kwargs):
            return self.proj(torch.ones(1, 2))

    class TinyValue(nn.Module):
        base_model_prefix = "backbone"

        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(2, 2)
            self.score = nn.Linear(2, 1)

    class TinyOther(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(2, 2)

    class Rows(Dataset):
        def __len__(self):
            return 16

        def __getitem__(self, index):
            return {"input_ids": [index + 1]}

    class Processing:
        eos_token_id = 2
        pad_token_id = 0

    def collate(rows):
        return {"input_ids": torch.tensor([row["input_ids"] for row in rows])}

    config = load_config("configs/pilot/resolved/ppo_seed_42.json")
    args = ppo_config(config, tmp_path, cpu_only=True)
    trainer = trl.PPOTrainer(
        args=args,
        processing_class=Processing(),
        model=TinyPolicy(),
        ref_model=TinyPolicy(),
        reward_model=TinyOther(),
        train_dataset=Rows(),
        value_model=TinyValue(),
        data_collator=collate,
        eval_dataset=Rows(),
    )
    contract = expected_run_contract(Path("configs/pilot/resolved/ppo_seed_42.json"), "ppo")
    derived = ppo_train_loop_contract(
        trainer.args, contract, world_size=trainer.accelerator.num_processes
    )
    assert trainer.accelerator.device.type == "cpu"
    assert {
        name: getattr(trainer.args, name)
        for name in (
            "total_episodes",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "local_batch_size",
            "micro_batch_size",
            "batch_size",
            "local_mini_batch_size",
            "mini_batch_size",
            "local_rollout_forward_batch_size",
            "num_ppo_epochs",
            "num_mini_batches",
            "num_total_batches",
        )
    } == {
        "total_episodes": 16,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "local_batch_size": 16,
        "micro_batch_size": 4,
        "batch_size": 16,
        "local_mini_batch_size": 16,
        "mini_batch_size": 16,
        "local_rollout_forward_batch_size": 4,
        "num_ppo_epochs": 1,
        "num_mini_batches": 1,
        "num_total_batches": 1,
    }
    assert derived["microbatches_per_minibatch"] == 4
    assert derived["expected_optimizer_steps"] == 1
    assert torch.cuda.is_initialized() is False


def test_real_accelerate_cpu_ga4_updates_only_on_fourth_microbatch():
    events = []

    class CountingSGD(torch.optim.SGD):
        def step(self, closure=None):
            events.append(bool(accelerator.sync_gradients))
            return super().step(closure)

    accelerator = Accelerator(gradient_accumulation_steps=4, cpu=True)
    model = nn.Linear(2, 1, bias=False)
    optimizer = CountingSGD(model.parameters(), lr=0.1)
    model, optimizer = accelerator.prepare(model, optimizer)
    initial = model.weight.detach().clone()
    trace = []
    for microbatch_index in range(4):
        inputs = torch.ones(4, 2)
        targets = torch.full((4, 1), float(microbatch_index + 1))
        before = model.weight.detach().clone()
        with accelerator.accumulate(model):
            sync_gradients = bool(accelerator.sync_gradients)
            accelerator.backward(((model(inputs) - targets) ** 2).mean())
            optimizer.step()
            optimizer.zero_grad()
        trace.append(
            {
                "microbatch_index": microbatch_index,
                "microbatch_size": len(inputs),
                "sync_gradients": sync_gradients,
                "underlying_steps": len(events),
                "parameter_changed": not torch.equal(before, model.weight.detach()),
            }
        )
    assert [row["sync_gradients"] for row in trace] == [False, False, False, True]
    assert [row["parameter_changed"] for row in trace] == [False, False, False, True]
    assert [row["microbatch_size"] for row in trace] == [4, 4, 4, 4]
    assert len(events) == 1
    assert events == [True]
    assert not torch.equal(initial, model.weight.detach())
    assert torch.cuda.is_initialized() is False


def test_consumed_single_batch_disables_end_of_dataloader_early_sync():
    events = []

    class CountingSGD(torch.optim.SGD):
        def step(self, closure=None):
            events.append(bool(accelerator.sync_gradients))
            return super().step(closure)

    accelerator = Accelerator(gradient_accumulation_steps=4, cpu=True)
    model = nn.Linear(2, 1, bias=False)
    optimizer = CountingSGD(model.parameters(), lr=0.1)
    dataloader = DataLoader(TensorDataset(torch.ones(16, 2), torch.ones(16, 1)), batch_size=16)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    iterator = iter(dataloader)
    inputs, targets = next(iterator)
    assert accelerator.gradient_state.end_of_dataloader is True
    loop_contract = {
        "microbatches_per_minibatch": 4,
        "per_device_train_batch_size": 4,
        "local_mini_batch_size": 16,
        "expected_optimizer_steps": 1,
    }
    accumulation = configure_ppo_gradient_accumulation(accelerator, loop_contract)
    assert accumulation == {
        "num_steps": 4,
        "sync_with_dataloader_before": True,
        "sync_with_dataloader": False,
    }
    backward_guard = PPOBackwardEventGuard(loop_contract, accumulation)
    parameter_changes = []
    for microbatch_index in range(4):
        micro_inputs = inputs[microbatch_index * 4 : (microbatch_index + 1) * 4]
        micro_targets = targets[microbatch_index * 4 : (microbatch_index + 1) * 4]
        before = model.weight.detach().clone()
        with accelerator.accumulate(model):
            backward_guard.note_training_forward(len(micro_inputs))
            event = backward_guard.prepare_backward(accelerator.sync_gradients)
            accelerator.backward(((model(micro_inputs) - micro_targets) ** 2).mean())
            backward_guard.commit_backward(event)
            optimizer.step()
            optimizer.zero_grad()
        parameter_changes.append(not torch.equal(before, model.weight.detach()))
    evidence = backward_guard.assert_complete(len(events))
    assert evidence["backward_events"] == 4
    assert evidence["microbatch_sizes"] == [4, 4, 4, 4]
    assert evidence["processed_samples"] == 16
    assert evidence["sync_gradients"] == [False, False, False, True]
    assert evidence["underlying_optimizer_steps"] == 1
    assert parameter_changes == [False, False, False, True]
    assert events == [True]
    assert torch.cuda.is_initialized() is False


def test_guarded_trainer_shim_counts_real_backward_and_underlying_step(monkeypatch, tmp_path):
    import trl.trainer.ppo_trainer as ppo_module

    class TinyPolicy(nn.Module):
        is_gradient_checkpointing = False

        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(2, 2)
            self.config = SimpleNamespace()
            self.generation_config = SimpleNamespace(eos_token_id=2)

    class TinyValue(nn.Module):
        base_model_prefix = "backbone"

        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(2, 2)
            self.score = nn.Linear(2, 1)

    class TinyOther(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(2, 2)

    class Rows(Dataset):
        def __len__(self):
            return 16

        def __getitem__(self, index):
            return {"input_ids": [index + 1]}

    class Processing:
        eos_token_id = 2
        pad_token_id = 0

    def collate(rows):
        return {"input_ids": torch.tensor([row["input_ids"] for row in rows])}

    def fake_forward(model, query_responses, pad_token_id):
        return None

    def fake_parent_train(self):
        iterator = iter(self.dataloader)
        next(iterator)
        assert self.accelerator.gradient_state.end_of_dataloader is True
        for _microbatch_index in range(4):
            with self.accelerator.accumulate(self.model):
                ppo_module.forward(self.model, torch.ones(4, 2), 0)
                loss = sum(parameter.float().sum() for parameter in self.model.parameters())
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.optimizer.zero_grad()
        self.state.global_step = 1
        self.log({"loss/policy_avg": 0.1, "loss/value_avg": 0.2, "lr": 1e-5})
        return SimpleNamespace(metrics={})

    monkeypatch.setattr(ppo_module, "forward", fake_forward)
    monkeypatch.setattr(trl.PPOTrainer, "train", fake_parent_train)
    config = load_config("configs/pilot/resolved/ppo_seed_42.json")
    args = ppo_config(config, tmp_path, cpu_only=True)
    contract = expected_run_contract(Path("configs/pilot/resolved/ppo_seed_42.json"), "ppo")
    guard = budget_guard()
    update_calls = []
    trainer_class = ppo_guarded_trainer_class(
        guard,
        evidence_recorder=SimpleNamespace(),
        prompt_lookup={},
        generation_contract={
            "max_new_tokens": 128,
            "temperature": 0.8,
            "top_p": 0.95,
        },
        expected_contract=contract,
        update_callback=lambda trainer, step: update_calls.append(
            (step, len(trainer.state.log_history))
        ),
    )
    trainer = trainer_class(
        args=args,
        processing_class=Processing(),
        model=TinyPolicy(),
        ref_model=TinyPolicy(),
        reward_model=TinyOther(),
        train_dataset=Rows(),
        value_model=TinyValue(),
        data_collator=collate,
        eval_dataset=Rows(),
    )
    trainer.train()
    assert trainer.ppo_backward_evidence["backward_events"] == 4
    assert trainer.ppo_backward_evidence["microbatch_sizes"] == [4, 4, 4, 4]
    assert trainer.ppo_backward_evidence["processed_samples"] == 16
    assert trainer.ppo_backward_evidence["sync_gradients"] == [False, False, False, True]
    assert trainer.ppo_backward_evidence["underlying_optimizer_steps"] == 1
    assert guard.optimizer_steps == guard.ppo_epochs == guard.minibatches == 1
    assert guard.updates == guard.global_step == 1
    assert update_calls == [(1, 1)]
    assert torch.cuda.is_initialized() is False


def test_backward_events_guard_one_epoch_one_minibatch_and_optimizer():
    loop_contract = {
        "microbatches_per_minibatch": 4,
        "per_device_train_batch_size": 4,
        "local_mini_batch_size": 16,
        "expected_optimizer_steps": 1,
    }
    backward_guard = PPOBackwardEventGuard(
        loop_contract, {"num_steps": 4, "sync_with_dataloader": False}
    )
    guard = budget_guard()
    for microbatch_index in range(4):
        backward_guard.note_training_forward(4)
        event = backward_guard.prepare_backward(microbatch_index == 3)
        backward_guard.commit_backward(event)
    backward_guard.assert_ready_for_optimizer()
    guard.record_loop_position(0, 0, 0)
    guard.record_optimizer_step()
    evidence = backward_guard.assert_complete(guard.optimizer_steps)
    guard.record_update()
    guard.record_global_step(1)
    guard.assert_success()
    assert evidence["backward_events"] == 4
    assert evidence["processed_samples"] == 16
    assert evidence["underlying_optimizer_steps"] == 1


@pytest.mark.parametrize("event_count", [1, 3])
def test_incomplete_backward_event_counts_fail_closed(event_count):
    loop_contract = {
        "microbatches_per_minibatch": 4,
        "per_device_train_batch_size": 4,
        "local_mini_batch_size": 16,
        "expected_optimizer_steps": 1,
    }
    backward_guard = PPOBackwardEventGuard(loop_contract, {})
    for _microbatch_index in range(event_count):
        backward_guard.note_training_forward(4)
        event = backward_guard.prepare_backward(False)
        backward_guard.commit_backward(event)
    with pytest.raises(TRLContractError, match="before all expected backward events"):
        backward_guard.assert_complete(1)


def test_fifth_backward_event_and_wrong_sample_total_fail_closed():
    loop_contract = {
        "microbatches_per_minibatch": 4,
        "per_device_train_batch_size": 4,
        "local_mini_batch_size": 16,
        "expected_optimizer_steps": 1,
    }
    backward_guard = PPOBackwardEventGuard(loop_contract, {})
    for microbatch_index in range(4):
        backward_guard.note_training_forward(4)
        event = backward_guard.prepare_backward(microbatch_index == 3)
        backward_guard.commit_backward(event)
    backward_guard.note_training_forward(4)
    with pytest.raises(TRLContractError, match="too many PPO backward"):
        backward_guard.prepare_backward(True)

    wrong_total = dict(loop_contract)
    wrong_total["per_device_train_batch_size"] = 3
    sample_guard = PPOBackwardEventGuard(wrong_total, {})
    for microbatch_index in range(4):
        sample_guard.note_training_forward(3)
        event = sample_guard.prepare_backward(microbatch_index == 3)
        sample_guard.commit_backward(event)
    with pytest.raises(TRLContractError, match="sample total mismatch"):
        sample_guard.assert_complete(1)


def test_early_sync_and_first_underlying_sync_true_semantics():
    loop_contract = {
        "microbatches_per_minibatch": 4,
        "per_device_train_batch_size": 4,
        "local_mini_batch_size": 16,
        "expected_optimizer_steps": 1,
    }
    early = PPOBackwardEventGuard(loop_contract, {})
    early.note_training_forward(4)
    with pytest.raises(TRLContractError, match="sync_gradients at microbatch 0"):
        early.prepare_backward(True)

    accepted = PPOBackwardEventGuard(loop_contract, {})
    for microbatch_index in range(4):
        accepted.note_training_forward(4)
        event = accepted.prepare_backward(microbatch_index == 3)
        accepted.commit_backward(event)
    accepted.assert_ready_for_optimizer()
    assert accepted.assert_complete(1)["underlying_optimizer_steps"] == 1


def test_loop_key_is_idempotent_but_real_second_loop_positions_fail_closed():
    guard = budget_guard()
    assert guard.record_loop_position(0, 0, 0) is True
    assert guard.record_loop_position(0, 0, 0) is False
    assert guard.ppo_epochs == guard.minibatches == 1

    epoch_guard = budget_guard()
    epoch_guard.record_loop_position(0, 0, 0)
    with pytest.raises(BudgetExceededError, match="epoch/minibatch cap"):
        epoch_guard.record_loop_position(0, 1, 0)

    minibatch_guard = budget_guard()
    minibatch_guard.record_loop_position(0, 0, 0)
    with pytest.raises(BudgetExceededError, match="epoch/minibatch cap"):
        minibatch_guard.record_loop_position(0, 0, 1)

    outer_guard = budget_guard()
    outer_guard.record_loop_position(0, 0, 0)
    with pytest.raises(BudgetExceededError, match="epoch/minibatch cap"):
        outer_guard.record_loop_position(1, 0, 0)

    optimizer_guard = budget_guard()
    optimizer_guard.record_loop_position(0, 0, 0)
    optimizer_guard.record_optimizer_step()
    with pytest.raises(BudgetExceededError, match="optimizer-step cap"):
        optimizer_guard.record_optimizer_step()


def test_trl_loop_index_mapping_exposes_real_epoch_or_minibatch_two():
    epoch_args = SimpleNamespace(num_ppo_epochs=2, num_mini_batches=1)
    assert ppo_loop_position(0, epoch_args) == (0, 0, 0)
    assert ppo_loop_position(1, epoch_args) == (0, 1, 0)

    minibatch_args = SimpleNamespace(num_ppo_epochs=1, num_mini_batches=2)
    assert ppo_loop_position(0, minibatch_args) == (0, 0, 0)
    assert ppo_loop_position(1, minibatch_args) == (0, 0, 1)


def test_stage_d_ppo_loop_contract_remains_one_microbatch(tmp_path):
    config = load_config("configs/smoke/ppo.yaml")
    args = ppo_config(config, tmp_path, cpu_only=True)
    contract = expected_run_contract(Path("configs/smoke/ppo.yaml"), "ppo")
    derived = ppo_train_loop_contract(args, contract, world_size=1)
    assert derived["total_episodes"] == 4
    assert derived["microbatches_per_minibatch"] == 1
    assert derived["num_ppo_epochs"] == derived["num_mini_batches"] == 1
    assert derived["expected_optimizer_steps"] == derived["expected_global_steps"] == 1
    backward_guard = PPOBackwardEventGuard(derived, {"num_steps": 1, "sync_with_dataloader": False})
    backward_guard.note_training_forward(4)
    event = backward_guard.prepare_backward(True)
    backward_guard.commit_backward(event)
    evidence = backward_guard.assert_complete(1)
    assert evidence["backward_events"] == 1
    assert evidence["processed_samples"] == 4
    assert evidence["sync_gradients"] == [True]


def test_frozen_pilot_and_third_failure_hashes_are_unchanged():
    expected_hashes = {
        "configs/pilot/matched_0p5b_manifest.json": (
            "a79ea8ee9d8bdc8f3d6fba8307995cba0c4516b90331cb13ba18ef1b55fa1b0d"
        ),
        "configs/pilot/resolved/ppo_seed_42.json": (
            "1daeba7e6cd5e0af43c7f7cb9db87b46d44608adf9fdf432dc7b2c34ea059fdd"
        ),
        "configs/pilot/resolved/ppo_seed_123.json": (
            "9da0ad35e943cdeda2da410c20eec73e6d105f0ef66f7f67b1be22950a0e43c5"
        ),
        "configs/pilot/resolved/ppo_seed_2026.json": (
            "d3255ddb849224a4d87a069d981fcacf85cb98a7afa986ff0a9fb284b7698044"
        ),
        "configs/pilot/resolved/grpo_seed_42.json": (
            "83992a9c312b3ea6ab87f33dce1d4e9572a9647bbdb72bd67a6e98e90c182ac8"
        ),
        "configs/pilot/resolved/grpo_seed_123.json": (
            "edec9ce1265dfaec8c712b2c65046fe860cbd3e10aab52cf31b6d5e0350c2a28"
        ),
        "configs/pilot/resolved/grpo_seed_2026.json": (
            "1d558da6ea57cfa074fee30868f1772c76c617920e4e93a4897be2e2b48d6b00"
        ),
        "reports/runs/ppo_matched_0p5b_seed42_20260714T085240Z/failure_report.json": (
            "585f6cb1b7a2d79c1b180e01c26b17f2570d7e61a1a6514904f28969aab0461c"
        ),
        "reports/runs/ppo_matched_0p5b_seed42_20260714T085240Z/summary.json": (
            "ed6a5c2d425b2710eb7cdb8e0cbf537391925ff8ef47298b226f09ce4bcffc6d"
        ),
        "reports/runs/ppo_matched_0p5b_seed42_20260714T085240Z/completions.jsonl": (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        ),
        "reports/runs/ppo_matched_0p5b_seed42_20260714T085240Z/final_summary.json": (
            "ef3c3540e128ae894a24be414596ba8ecc45bb6acf13f851a093b122e8c92080"
        ),
    }
    for name, expected in expected_hashes.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected


def test_four_historical_seed42_failure_trees_are_immutable():
    expected = {
        "ppo_matched_0p5b_seed42_20260714T073357Z": (
            "e11a3660473f29586b9211a9a01f3f19ae053ff29e5b2231dea1d96c1fb0d687"
        ),
        "ppo_matched_0p5b_seed42_20260714T082003Z": (
            "df8e9d9217d36042ba82fdc387e982c97754645190a1fb3dd68a4a22bd77c48a"
        ),
        "ppo_matched_0p5b_seed42_20260714T085240Z": (
            "18266be5c66c20dd10c73e239c68be8f71cbc8f6c39a7f795593df4fef2129c5"
        ),
        "ppo_matched_0p5b_seed42_20260716T111934Z": (
            "02142db5c449cca4af4f01d8ace585ee0813ef40e992aa66ec2e010649c289a7"
        ),
    }
    for run_id, expected_hash in expected.items():
        root = Path("reports/runs") / run_id
        digest = hashlib.sha256()
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        assert digest.hexdigest() == expected_hash
