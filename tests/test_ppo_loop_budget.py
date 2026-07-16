import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import trl
from torch import nn
from torch.utils.data import Dataset

from math_rlvr.config import load_config
from math_rlvr.training.builders import ppo_config
from math_rlvr.training.execution_contract import expected_run_contract
from math_rlvr.training.guarded_grpo import BudgetExceededError
from math_rlvr.training.guarded_ppo import PPOBudgetGuard
from math_rlvr.training.trl_compat import (
    ppo_loop_position,
    ppo_train_loop_contract,
    record_ppo_optimizer_call,
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


def test_one_epoch_one_minibatch_four_microbatches_records_one_optimizer_step(tmp_path):
    config = load_config("configs/pilot/resolved/ppo_seed_42.json")
    args = ppo_config(config, tmp_path, cpu_only=True)
    contract = expected_run_contract(Path("configs/pilot/resolved/ppo_seed_42.json"), "ppo")
    loop = ppo_train_loop_contract(args, contract, world_size=1)
    guard = budget_guard()
    microbatch_calls = synchronized_steps = 0
    trace = ["update:0:begin", "epoch:0:begin", "minibatch:0:begin"]
    for microbatch_index in range(4):
        sync_gradients = microbatch_index == 3
        microbatch_calls, synchronized_steps = record_ppo_optimizer_call(
            guard,
            args,
            loop,
            microbatch_calls=microbatch_calls,
            synchronized_steps=synchronized_steps,
            sync_gradients=sync_gradients,
        )
        trace.append(f"microbatch:{microbatch_index}:sync={str(sync_gradients).lower()}")
    trace.extend(["optimizer_step:0", "minibatch:0:end", "epoch:0:end", "update:0:end"])
    guard.record_update()
    guard.record_global_step(1)
    guard.assert_success()
    assert trace == [
        "update:0:begin",
        "epoch:0:begin",
        "minibatch:0:begin",
        "microbatch:0:sync=false",
        "microbatch:1:sync=false",
        "microbatch:2:sync=false",
        "microbatch:3:sync=true",
        "optimizer_step:0",
        "minibatch:0:end",
        "epoch:0:end",
        "update:0:end",
    ]
    assert (microbatch_calls, synchronized_steps) == (0, 1)
    assert guard.snapshot()["optimizer_steps"] == 1


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
