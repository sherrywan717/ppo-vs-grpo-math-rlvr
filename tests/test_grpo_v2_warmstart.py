import hashlib
import json
from pathlib import Path

import pytest

from math_rlvr.grpo_v2_contract import validate_nested_success
from math_rlvr.training import warmstart_runtime as runtime
from math_rlvr.training.warmstart import main
from math_rlvr.training.warmstart_runtime import (
    WarmstartBudgetGuard,
    WarmstartContractError,
    backup_warmstart_run,
    completion_only_collate,
    encode_completion_only,
    grpo_adapter_handoff,
    load_contract,
    validate_checkpoint,
    validate_postprocess_gpu_release,
)

ROOT = Path(__file__).resolve().parents[1]


class FakeTokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def __init__(self, prompt_tokens=10, target_tokens=10):
        self.prompt_tokens = prompt_tokens
        self.target_tokens = target_tokens

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        return list(range(1, self.prompt_tokens + 1))

    def encode(self, target, add_special_tokens=False):
        return [7] * self.target_tokens


def encoded(prompt, target):
    return encode_completion_only(FakeTokenizer(prompt, target), [], "target")


def test_capacity_boundaries_and_actual_combined_gate():
    assert encoded(914, 100)["prompt_tokens"] == 914
    assert encoded(100, 608)["active_label_tokens"] == 609
    assert encoded(500, 518)["active_label_tokens"] + 500 == 1019
    with pytest.raises(WarmstartContractError, match="prompt"):
        encoded(929, 10)
    with pytest.raises(WarmstartContractError, match="target"):
        encoded(10, 640)
    with pytest.raises(WarmstartContractError, match="combined"):
        encoded(500, 588)


def test_completion_only_labels_padding_and_eos():
    first = encoded(4, 3)
    second = encoded(2, 2)
    assert first["labels"][:4] == [-100] * 4
    assert first["labels"][-1] == 99
    batch = completion_only_collate([first, second], pad_token_id=0)
    assert batch["labels"][1][-2:] == [-100, -100]
    assert batch["attention_mask"][1][-2:] == [0, 0]
    assert batch["input_ids"][1][-2:] == [0, 0]


def test_real_tokenizer_three_completion_boundaries():
    from transformers import AutoTokenizer

    from math_rlvr.dataset import MathProblem
    from math_rlvr.prompt import format_problem_v2

    snapshot = (
        "/root/autodl-tmp/cache/huggingface/hub/"
        "models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/"
        "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False
    )
    rows = [
        json.loads(x)
        for x in (ROOT / "configs/grpo_v2/manifests/warmstart_v2.jsonl").read_text().splitlines()
    ]
    targets = {
        x["problem_id"]: x["target_text"]
        for x in map(
            json.loads,
            Path("/root/autodl-tmp/datasets/math_rlvr/grpo_v2/trusted/warmstart_targets.jsonl")
            .read_text()
            .splitlines(),
        )
    }
    samples = [
        rows[0],
        next(r for r in rows if r["source"] == "math" and r["difficulty"] == "1"),
        next(r for r in rows if r["source"] == "math" and r["difficulty"] == "3"),
    ]
    features = []
    for row in samples:
        problem = MathProblem(
            **{
                k: row[k]
                for k in (
                    "problem_id",
                    "source",
                    "prompt",
                    "category",
                    "difficulty",
                    "split",
                    "source_index",
                    "content_hash",
                    "metadata",
                )
            },
            gold_answer="0",
        )
        feature = encode_completion_only(
            tokenizer, format_problem_v2(problem), targets[row["problem_id"]]
        )
        active = feature["labels"][feature["prompt_tokens"] :]
        assert active[-1] == tokenizer.eos_token_id
        assert tokenizer.decode(active[:-1]) == targets[row["problem_id"]]
        assert set(feature["labels"][: feature["prompt_tokens"]]) == {-100}
        features.append(feature)
    batch = completion_only_collate(features, pad_token_id=tokenizer.pad_token_id)
    for labels, mask in zip(batch["labels"], batch["attention_mask"], strict=True):
        assert all(label == -100 for label, active in zip(labels, mask, strict=True) if not active)


def test_budget_guard_exact_and_fail_closed():
    guard = WarmstartBudgetGuard()
    for batch in range(64):
        guard.record_batch(
            sample_ids=[f"p{batch * 4 + i}" for i in range(4)], active_label_tokens=10
        )
        guard.record_microstep(1.0)
        if (batch + 1) % 4 == 0:
            guard.record_optimizer_step((batch + 1) // 4)
    guard.record_epoch()
    assert guard.finalize()["samples"] == 256
    incomplete = WarmstartBudgetGuard(
        samples=255, batches=64, microsteps=64, optimizer_steps=16, global_steps=16, epochs=1
    )
    with pytest.raises(WarmstartContractError, match="incomplete"):
        incomplete.finalize()
    overflow = WarmstartBudgetGuard(samples=253)
    with pytest.raises(WarmstartContractError, match="sample budget"):
        overflow.record_batch(sample_ids=["a", "b", "c", "d"], active_label_tokens=4)
    for value in (float("nan"), float("inf")):
        with pytest.raises(WarmstartContractError, match="non-finite"):
            WarmstartBudgetGuard().record_microstep(value)
    with pytest.raises(WarmstartContractError, match="optimizer"):
        WarmstartBudgetGuard(optimizer_steps=16).record_optimizer_step(17)
    for steps in (15, 17):
        candidate = WarmstartBudgetGuard(
            samples=256,
            batches=64,
            microsteps=64,
            optimizer_steps=steps,
            global_steps=steps,
            epochs=1,
        )
        with pytest.raises(WarmstartContractError):
            candidate.finalize()
    for epochs in (0, 2):
        candidate = WarmstartBudgetGuard(
            samples=256,
            batches=64,
            microsteps=64,
            optimizer_steps=16,
            global_steps=16,
            epochs=epochs,
        )
        with pytest.raises(WarmstartContractError):
            candidate.finalize()


def test_runtime_registry_and_cli_dual_confirmation(tmp_path):
    config, identity = load_contract()
    assert (
        identity["config_sha256"]
        == hashlib.sha256((ROOT / runtime.CONFIG_PATH).read_bytes()).hexdigest()
    )
    assert main(["--config", str(runtime.CONFIG_PATH)]) == 0
    with pytest.raises(RuntimeError, match="requires"):
        main(
            ["--config", str(runtime.CONFIG_PATH), "--execute", "--run-dir", str(tmp_path / "run")]
        )
    called = {}

    def execute_fn(config, **kwargs):
        called.update(kwargs)
        return {"status": "success"}

    assert (
        main(
            [
                "--config",
                str(runtime.CONFIG_PATH),
                "--execute",
                "--confirm-grpo-v2-warmstart",
                "--run-dir",
                str(tmp_path / "run"),
            ],
            execute_fn=execute_fn,
            environment_probe=lambda: {"branch": "improve/grpo-v2"},
            snapshot_probe=lambda: object(),
        )
        == 0
    )
    assert called["run_dir"] == tmp_path / "run"
    with pytest.raises(WarmstartContractError):
        load_contract(Path("configs/grpo_v2/grpo_v2_seed42.json"))


def make_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "RUN_ROOT", tmp_path)
    checkpoint = tmp_path / "warmstart_grpo_v2_seed42_fake" / "checkpoint-16"
    (checkpoint / "adapter").mkdir(parents=True)
    files = {
        "adapter/adapter_config.json": b"{}",
        "adapter/adapter_model.safetensors": b"adapter",
        "optimizer.pt": b"optimizer",
        "scheduler.pt": b"scheduler",
        "rng_state.pt": b"rng",
        "trainer_state.json": b"{}",
        "runtime_state.json": b"{}",
    }
    for name, data in files.items():
        (checkpoint / name).write_bytes(data)
    (checkpoint / "checkpoint_identity.json").write_text(
        json.dumps(
            {
                "run_id": "warmstart_grpo_v2_seed42_fake",
                "adapter_role": "policy",
                "config_sha256": "cfg",
            }
        )
    )
    inventory = {}
    for path in checkpoint.rglob("*"):
        if path.is_file():
            inventory[path.relative_to(checkpoint).as_posix()] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size": path.stat().st_size,
            }
    (checkpoint / "artifact_manifest.json").write_text(
        json.dumps(
            {"files": inventory, "artifact_sha256": "artifact", "base_weights_included": False}
        )
    )
    return checkpoint


def test_checkpoint_allowlist_inventory_and_grpo_handoff(tmp_path, monkeypatch):
    checkpoint = make_checkpoint(tmp_path, monkeypatch)
    evidence = validate_checkpoint(
        checkpoint, expected_config_sha="cfg", expected_run_id="warmstart_grpo_v2_seed42_fake"
    )
    handoff = grpo_adapter_handoff(evidence)
    assert handoff["adapter_role"] == "policy"
    assert handoff["inherit_sft_optimizer_state"] is False
    assert handoff["grpo_optimizer_initialization"] == "fresh_frozen_grpo_v2_contract"
    (checkpoint / "model.safetensors").write_bytes(b"forbidden")
    with pytest.raises(WarmstartContractError, match="allowlist"):
        validate_checkpoint(checkpoint, expected_config_sha="cfg")


def test_nested_pass10_subset_and_monotonicity():
    pass4 = json.loads((ROOT / "configs/grpo_v2/manifests/pass4_nested_subset.json").read_text())[
        "problems"
    ]
    pass10 = json.loads((ROOT / "configs/grpo_v2/manifests/pass10_nested_subset.json").read_text())[
        "problems"
    ]
    assert len(pass10) == 50 and {r["problem_id"] for r in pass10} < {
        r["problem_id"] for r in pass4
    }
    assert sum(r["source"] == "gsm8k" for r in pass10) == 25
    assert {
        i: sum(r["source"] == "math" and r["difficulty"] == str(i) for r in pass10)
        for i in range(1, 6)
    } == {1: 2, 2: 4, 3: 5, 4: 7, 5: 7}
    assert validate_nested_success([False, False, False, False, True] + [False] * 5) == {
        "success_at_1": False,
        "success_at_4": False,
        "success_at_10": True,
    }
    with pytest.raises(ValueError):
        validate_nested_success([False] * 9)


def test_postprocess_gpu_release_contract():
    assert (
        validate_postprocess_gpu_release(
            worker_exited=True, compute_processes=0, used_memory_mib=0
        )["used_memory_mib"]
        == 0
    )
    with pytest.raises(WarmstartContractError):
        validate_postprocess_gpu_release(worker_exited=True, compute_processes=1, used_memory_mib=1)


def test_verified_failure_backup(tmp_path, monkeypatch):
    run_dir = tmp_path / "warmstart_grpo_v2_seed42_fake"
    run_dir.mkdir()
    (run_dir / "failure.json").write_text('{"status":"failure"}\n')
    backup_root = tmp_path / "backups"
    monkeypatch.setattr(runtime, "BACKUP_ROOT", backup_root)
    evidence = backup_warmstart_run(run_dir, failure=True)
    archive = Path(evidence["archive"])
    assert archive.name.endswith(".failure.tar.gz")
    assert archive.is_file()
