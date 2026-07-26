import hashlib
import json
from pathlib import Path

import pytest

from math_rlvr.training.formal_runtime import (
    FormalOnlineGuard,
    FormalRuntimeError,
    write_formal_checkpoint_artifact_manifest,
)
from math_rlvr.training.grpo_v2 import main
from math_rlvr.training.grpo_v2_runtime import (
    CHECKPOINT_STEPS,
    CONFIG_PATH,
    EXPECTED_ADAPTER_SHA256,
    EXPECTED_CHECKPOINT_ARTIFACT_SHA256,
    WARMSTART_CHECKPOINT,
    GRPOV2ContractError,
    GRPOV2ProgressGuard,
    load_contract,
    normalized_training_config,
    select_dev_checkpoint,
    validate_initial_checkpoint,
    validate_normalized_training_config,
    validate_resume_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]


def metric(reward=0.1):
    return {
        "reward_mean": reward,
        "reward_std": 0.0,
        "reward_variance": 0.0,
        "group_rewards": [[reward] * 4 for _ in range(4)],
        "zero_advantage_fraction": 1.0,
        "canonical_pass_rate": 0.0,
        "format_accuracy": 0.0,
        "parseable_rate": 0.0,
        "valid_answer_rate": 0.0,
        "loss": 0.1,
        "learning_rate": 1e-5,
        "generated_tokens": 16,
        "cumulative_generated_tokens": 16,
        "entropy": None,
        "policy_entropy_mean": None,
        "policy_entropy_mean_available": False,
        "policy_entropy_mean_reason": "not exposed",
        "grad_norm": None,
        "grad_norm_available": False,
        "grad_norm_reason": "not exposed",
        "kl": None,
        "kl_available": False,
        "kl_unavailable_reason": "not exposed",
        "ratio": None,
        "ratio_mean": None,
        "ratio_available": False,
        "ratio_reason": "not exposed",
        "clip_fraction": None,
        "clip_fraction_available": False,
        "clip_fraction_reason": "not exposed",
    }


def test_native_formal_optional_metric_schema_does_not_block():
    from math_rlvr.training.formal_model_runtime import _normal_metrics
    from math_rlvr.training.grpo_v2_model_runtime import _with_parseable_metric

    _, _, contract = load_contract()
    rows = completion_rows(contract, 1)
    for index, row in enumerate(rows):
        row.update(
            {
                "canonical_status": "format_error",
                "eos_reached": False,
                "truncated": False,
                "reward_components": {"format": 0.0},
                "verifier_detail": "fake",
                "problem_id": contract.problem_ids[index // 4],
            }
        )
    native = _with_parseable_metric(
        _normal_metrics([{"loss": 0.25, "learning_rate": 1e-5}], rows, contract)[0],
        rows,
    )
    assert native["policy_entropy_mean_available"] is False
    assert native["kl_available"] is False
    GRPOV2ProgressGuard(contract, "fake").record_update(1, rows, native)


def completion_rows(contract, update, token_count=1):
    return [
        {
            "pair_key": key,
            "update": update,
            "completion_ids": [1] * token_count,
            "completion_mask": [1] * token_count,
            "exact_token_count": token_count,
            "raw_completion": "x",
            "scalar_reward": 0.1,
        }
        for key in contract.pair_keys_for_update(update)
    ]


def test_frozen_contract_curriculum_and_handoff():
    design, identity, contract = load_contract()
    assert contract.problem_ids == tuple(dict.fromkeys(contract.problem_ids))
    assert len(contract.problem_ids) == 512
    assert len(contract.pair_keys) == 2048
    assert [contract.problem_ids[(step - 1) * 4] for step in (1, 128)]
    assert contract.checkpoint_steps == (32, 64, 96, 128)
    handoff = validate_initial_checkpoint(WARMSTART_CHECKPOINT, identity)["handoff"]
    assert handoff["adapter_role"] == "policy"
    assert handoff["adapter_sha256"] == EXPECTED_ADAPTER_SHA256
    assert handoff["source_checkpoint_sha256"] == EXPECTED_CHECKPOINT_ARTIFACT_SHA256
    assert handoff["inherit_sft_optimizer_state"] is False
    normalized = normalized_training_config(design, contract)
    validate_normalized_training_config(normalized)
    assert normalized["training"]["max_steps"] == 128
    assert normalized["training"]["gradient_accumulation_steps"] == 4
    assert normalized["training"]["shuffle_dataset"] is False


def test_trl_024_batch_update_contract_is_statically_derived(tmp_path):
    from math_rlvr.training.builders import grpo_config

    design, _, contract = load_contract()
    normalized = normalized_training_config(design, contract)
    args = grpo_config(normalized, tmp_path, cpu_only=True)
    assert args.max_steps == 128
    assert args.per_device_train_batch_size == 4
    assert args.gradient_accumulation_steps == 4
    assert args.generation_batch_size == 16
    assert args.num_generations == 4
    assert args.save_steps == 32
    assert args.shuffle_dataset is False


def test_cli_dry_run_and_dual_confirmation(tmp_path, monkeypatch):
    checkpoint = str(WARMSTART_CHECKPOINT)
    assert main(["--config", str(CONFIG_PATH), "--warmstart-checkpoint", checkpoint]) == 0
    with pytest.raises(RuntimeError, match="requires"):
        main(
            [
                "--config",
                str(CONFIG_PATH),
                "--warmstart-checkpoint",
                checkpoint,
                "--run-dir",
                str(tmp_path / "grpo_v2_seed42_x"),
                "--execute",
            ]
        )
    called = {}
    import math_rlvr.training.grpo_v2_runtime as runtime

    monkeypatch.setattr(runtime, "RUN_ROOT", tmp_path)
    run_dir = tmp_path / "grpo_v2_seed42_fake"

    def execute_fn(_design, **kwargs):
        called.update(kwargs)
        return {"status": "success"}

    assert (
        main(
            [
                "--config",
                str(CONFIG_PATH),
                "--warmstart-checkpoint",
                checkpoint,
                "--run-dir",
                str(run_dir),
                "--execute",
                "--confirm-grpo-v2",
            ],
            execute_fn=execute_fn,
            environment_probe=lambda: {"branch": "improve/grpo-v2"},
            snapshot_probe=lambda: object(),
        )
        == 0
    )
    assert called["identity"]["warmstart_handoff"]["inherit_sft_optimizer_state"] is False
    with pytest.raises(RuntimeError, match="out of memory"):
        main(
            [
                "--config",
                str(CONFIG_PATH),
                "--warmstart-checkpoint",
                checkpoint,
                "--run-dir",
                str(tmp_path / "grpo_v2_seed42_oom"),
                "--execute",
                "--confirm-grpo-v2",
            ],
            execute_fn=lambda *_args, **_kwargs: {
                "status": "failure",
                "reason": "CUDA out of memory",
            },
            environment_probe=lambda: {"branch": "improve/grpo-v2"},
            snapshot_probe=lambda: object(),
        )


def test_exact_128_update_completion_token_and_dev_ledgers():
    _, _, contract = load_contract()
    guard = GRPOV2ProgressGuard(contract, "fake")
    for update in range(1, 129):
        guard.record_update(update, completion_rows(contract, update), metric())
        if update in CHECKPOINT_STEPS:
            guard.record_checkpoint(update)
            guard.record_validation(
                update, [{"checkpoint_step": update, "exact_token_count": 1} for _ in range(128)]
            )
    result = guard.assert_complete()
    assert result["updates"] == result["optimizer_steps"] == result["global_steps"] == 128
    assert result["completions"] == 2048
    assert result["generated_tokens"] == 2048
    assert result["dev_completions"] == 512
    assert result["dev_tokens"] == 512
    assert result["training_token_budget_excludes_dev"] is True
    for invalid_count in (2047, 2049):
        guard.completions = invalid_count
        with pytest.raises(GRPOV2ContractError, match="incomplete"):
            guard.assert_complete()


@pytest.mark.parametrize("bad_update", [127, 129])
def test_update_count_fail_closed(bad_update):
    _, _, contract = load_contract()
    guard = GRPOV2ProgressGuard(contract, "fake")
    for update in range(1, min(bad_update, 128) + 1):
        guard.record_update(update, completion_rows(contract, update), metric())
    if bad_update == 127:
        with pytest.raises(GRPOV2ContractError, match="incomplete"):
            guard.assert_complete()
    else:
        with pytest.raises(GRPOV2ContractError):
            guard.record_update(129, completion_rows(contract, 128), metric())


def test_completion_and_token_overflow_and_nonfinite_fail_closed():
    _, _, contract = load_contract()
    guard = GRPOV2ProgressGuard(contract, "fake")
    with pytest.raises(GRPOV2ContractError, match="count"):
        guard.record_update(1, completion_rows(contract, 1)[:-1], metric())
    rows = completion_rows(contract, 1, token_count=256)
    guard.generated_tokens = 524_288 - 4095
    with pytest.raises(GRPOV2ContractError, match="token cap"):
        guard.record_update(1, rows, metric())
    bad = metric()
    bad["loss"] = float("nan")
    with pytest.raises(GRPOV2ContractError, match="non-finite"):
        GRPOV2ProgressGuard(contract, "fake").record_update(1, completion_rows(contract, 1), bad)


def _fake_checkpoint(tmp_path, contract, step):
    run = tmp_path / "grpo_v2_seed42_fake"
    root = run / f"checkpoint-{step}"
    (root / "policy_adapter").mkdir(parents=True)
    payloads = {
        "policy_adapter/adapter_model.safetensors": b"adapter",
        "policy_adapter/adapter_config.json": b"{}",
        "optimizer.pt": b"optimizer",
        "scheduler.pt": b"scheduler",
        "rng_state.json": b"{}",
        "torch_rng.safetensors": b"rng",
        "trainer_state.json": json.dumps({"global_step": step}).encode(),
    }
    for name, data in payloads.items():
        (root / name).write_bytes(data)
    completions = []
    for index, key in enumerate(contract.pair_keys[: step * 16]):
        completions.append(
            {
                "pair_key": key,
                "update": index // 16 + 1,
                "exact_token_count": 1,
                "completion_ids": [1],
                "completion_mask": [1],
                "raw_completion": "x",
                "scalar_reward": 0.1,
            }
        )
    (root / "trainer_completion_prefix.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in completions)
    )
    (root / "metrics_prefix.jsonl").write_text(
        "".join(json.dumps({"update": update}) + "\n" for update in range(1, step + 1))
    )
    online = {
        "completions": step * 16,
        "generated_tokens": step * 16,
        "rewards": step * 16,
        "microsteps": step * 4,
        "optimizer_steps": step,
        "global_steps": step,
        "updates": step,
        "loop_positions": [],
    }
    resume = {
        **contract.checkpoint_identity(run_id=run.name, step=step),
        "schema": "math_rlvr.formal_resume.v1",
        "project_created": True,
        "updates": step,
        "optimizer_steps": step,
        "global_steps": step,
        "completions": step * 16,
        "generated_tokens": step * 16,
        "base_weights_included": False,
        "optimizer_state_included": True,
        "scheduler_state_included": True,
        "rng_state_included": True,
        "sampler_position": {"comparison_key_count": step * 16, "grpo_prompt_rows": step * 4},
        "online_counters": online,
    }
    (root / "resume_manifest.json").write_text(json.dumps(resume))
    write_formal_checkpoint_artifact_manifest(root, contract, step)
    return root


@pytest.mark.parametrize("step", [32, 64, 96])
def test_same_run_resume_checkpoint_prefix_and_inventory(tmp_path, step):
    _, _, contract = load_contract()
    root = _fake_checkpoint(tmp_path, contract, step)
    checked = validate_resume_checkpoint(root, contract, root.parent)
    assert checked.step == step
    assert checked.inventory["completion_prefix_count"] == step * 16
    with pytest.raises(GRPOV2ContractError, match="same run"):
        validate_resume_checkpoint(root, contract, tmp_path / "other")


def test_continuous_128_equals_64_resume_fake_state(tmp_path):
    _, _, contract = load_contract()
    continuous = GRPOV2ProgressGuard(contract, "continuous")
    for update in range(1, 129):
        continuous.record_update(update, completion_rows(contract, update), metric())
    root = _fake_checkpoint(tmp_path, contract, 64)
    checked = validate_resume_checkpoint(root, contract, root.parent)
    resumed = GRPOV2ProgressGuard(contract, root.parent.name)
    resumed.updates = resumed.optimizer_steps = resumed.global_steps = checked.step
    resumed.completions = len(checked.completion_prefix)
    resumed.generated_tokens = sum(row["exact_token_count"] for row in checked.completion_prefix)
    resumed.pair_keys = [row["pair_key"] for row in checked.completion_prefix]
    for update in range(65, 129):
        resumed.record_update(update, completion_rows(contract, update), metric())
    for key in (
        "updates",
        "optimizer_steps",
        "global_steps",
        "completions",
        "generated_tokens",
        "pair_keys",
    ):
        assert resumed.snapshot()[key] == continuous.snapshot()[key]


def test_online_guard_exact_512_microsteps_and_2048_completions():
    _, _, contract = load_contract()
    guard = FormalOnlineGuard(contract)
    for update in range(1, 129):
        guard.record_generation(16, 16)
        for _ in range(16):
            guard.record_reward(object(), 0.1, {})
        for _ in range(4):
            guard.record_microstep()
        guard.record_optimizer_step()
        guard.record_global_step(update)
        guard.record_update()
    result = guard.assert_complete()
    assert result["microsteps"] == 512
    assert result["completions"] == result["rewards"] == 2048
    overflow = FormalOnlineGuard(contract)
    overflow.microsteps = 512
    with pytest.raises(FormalRuntimeError, match="microstep cap"):
        overflow.record_microstep()


def test_fake_parameter_state_continuous_equals_resume():
    def advance(weight, start, stop):
        for update in range(start, stop + 1):
            weight = weight * 0.999 + update * 1e-6
        return weight

    continuous = advance(0.125, 1, 128)
    checkpoint64 = advance(0.125, 1, 64)
    resumed = advance(checkpoint64, 65, 128)
    assert resumed == continuous


def test_dev_selection_lexicographic_and_hidden_manifest_not_read(monkeypatch):
    rows = [
        {
            "checkpoint_step": 32,
            "canonical_pass_rate": 0.1,
            "parseable_rate": 0.2,
            "format_rate": 0.3,
            "truncation_rate": 0.1,
        },
        {
            "checkpoint_step": 64,
            "canonical_pass_rate": 0.2,
            "parseable_rate": 0.2,
            "format_rate": 0.3,
            "truncation_rate": 0.1,
        },
        {
            "checkpoint_step": 96,
            "canonical_pass_rate": 0.2,
            "parseable_rate": 0.3,
            "format_rate": 0.3,
            "truncation_rate": 0.1,
        },
        {
            "checkpoint_step": 128,
            "canonical_pass_rate": 0.2,
            "parseable_rate": 0.3,
            "format_rate": 0.3,
            "truncation_rate": 0.1,
        },
    ]
    assert select_dev_checkpoint(rows) == 96
    original = Path.read_text

    def guarded_read(path, *args, **kwargs):
        if "test_v2_hidden" in str(path):
            raise AssertionError("training runtime accessed hidden test")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read)
    load_contract()


def test_historical_warmstart_checkpoint_hashes_unchanged():
    manifest = json.loads((WARMSTART_CHECKPOINT / "artifact_manifest.json").read_text())
    adapter = WARMSTART_CHECKPOINT / "adapter/adapter_model.safetensors"
    assert manifest["artifact_sha256"] == EXPECTED_CHECKPOINT_ARTIFACT_SHA256
    assert hashlib.sha256(adapter.read_bytes()).hexdigest() == EXPECTED_ADAPTER_SHA256
