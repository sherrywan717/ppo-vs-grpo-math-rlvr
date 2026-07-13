import hashlib
import inspect
from pathlib import Path

import pytest
import torch
from test_guarded_grpo import Backend, Lifecycle, Monitor, checkpoint, verifier

from math_rlvr.config import load_config
from math_rlvr.rewards.result import RewardResult, RewardStatus
from math_rlvr.training.grpo_runtime import validate_backup_inventory
from math_rlvr.training.guarded_grpo import (
    CheckpointSafetyError,
    authoritative_checkpoint,
    checkpoint_inventory,
    run_guarded,
)
from math_rlvr.training.resource_evidence import CudaAllocatorEvidence
from math_rlvr.training.trl_compat import (
    CompletionEvidenceRecorder,
    TRLContractError,
    extract_kl_metric,
)


def make_authoritative_checkpoint(tmp_path: Path, training_args_size: int = 7441):
    run = tmp_path / "run"
    root = run / "checkpoint-1"
    root.mkdir(parents=True)
    (root / "adapter_model.safetensors").write_bytes(b"adapter")
    (root / "adapter_config.json").write_text("{}")
    (root / "training_args.bin").write_bytes(b"x" * training_args_size)
    return run, root


def test_exact_training_args_bin_is_allowed_without_deserialization(tmp_path):
    run, root = make_authoritative_checkpoint(tmp_path)
    assert authoritative_checkpoint(run, 1) == root.resolve()
    inventory = checkpoint_inventory(root, run_dir=run)
    row = next(item for item in inventory["files"] if item["name"] == "training_args.bin")
    assert row == {
        "name": "training_args.bin",
        "size_bytes": 7441,
        "sha256": hashlib.sha256(b"x" * 7441).hexdigest(),
        "classification": "trainer_metadata",
    }


@pytest.mark.parametrize("name", ["unknown.bin", "training_state.bin", "pytorch_model.bin"])
def test_unknown_bin_files_are_rejected(tmp_path, name):
    run, root = make_authoritative_checkpoint(tmp_path)
    (root / name).write_bytes(b"x")
    with pytest.raises(CheckpointSafetyError, match="non-adapter"):
        checkpoint_inventory(root, run_dir=run)


def test_oversized_training_args_is_rejected(tmp_path):
    run, root = make_authoritative_checkpoint(tmp_path, 1024 * 1024 + 1)
    with pytest.raises(CheckpointSafetyError, match="exceeds 1 MiB"):
        checkpoint_inventory(root, run_dir=run)


def test_symlink_training_args_is_rejected(tmp_path):
    run, root = make_authoritative_checkpoint(tmp_path)
    (root / "training_args.bin").unlink()
    target = root / "metadata"
    target.write_bytes(b"x")
    (root / "training_args.bin").symlink_to(target)
    with pytest.raises(CheckpointSafetyError, match="symlink"):
        checkpoint_inventory(root, run_dir=run)


def test_checkpoint_root_escape_is_rejected(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (run / "checkpoint-1").symlink_to(outside, target_is_directory=True)
    with pytest.raises(CheckpointSafetyError, match="symlink"):
        checkpoint_inventory(run / "checkpoint-1", run_dir=run)


def test_only_one_authoritative_checkpoint_and_adapter(tmp_path):
    run, root = make_authoritative_checkpoint(tmp_path)
    selected = authoritative_checkpoint(run, 1)
    inventory = checkpoint_inventory(selected, run_dir=run)
    assert selected == root.resolve()
    assert inventory["duplicate_checkpoint_count"] == 0
    assert sum(row["name"] == "adapter_model.safetensors" for row in inventory["files"]) == 1
    assert sum(row["name"] == "adapter_config.json" for row in inventory["files"]) == 1


def test_duplicate_adapter_sha_is_rejected(tmp_path):
    run, root = make_authoritative_checkpoint(tmp_path)
    duplicate = run / "checkpoints" / "checkpoint-1"
    duplicate.mkdir(parents=True)
    (duplicate / "adapter_model.safetensors").write_bytes(
        (root / "adapter_model.safetensors").read_bytes()
    )
    (duplicate / "adapter_config.json").write_text("{}")
    with pytest.raises(CheckpointSafetyError, match="duplicate adapter SHA256"):
        authoritative_checkpoint(run, 1)


class FakeTokenizer:
    def __init__(self, texts):
        self.texts = texts

    def batch_decode(self, ids, skip_special_tokens=True):
        assert skip_special_tokens is True
        assert ids.shape[0] == len(self.texts)
        return list(self.texts)


def make_ordered_recorder():
    texts = [f"<reasoning>思考{i}</reasoning><answer>{i}</answer>" for i in range(8)]
    recorder = CompletionEvidenceRecorder()
    problem_ids = ["p0"] * 4 + ["p1"] * 4
    for problem_id, text in zip(problem_ids, texts, strict=True):
        recorder.record_reward(
            problem_id,
            text,
            RewardResult(RewardStatus.VERIFIED_PASS, detail="ok"),
            1.0,
        )
    inputs = [
        {
            "problem_id": problem_id,
            "prompt_hash": "hash0" if problem_id == "p0" else "hash1",
        }
        for problem_id in problem_ids
    ]
    ids = torch.arange(24, dtype=torch.long).reshape(8, 3)
    mask = torch.tensor(
        [[1, 1, 0], [1, 1, 1], [1, 0, 0], [1, 1, 1]] * 2,
        dtype=torch.long,
    )
    return recorder, texts, inputs, {"completion_ids": ids, "completion_mask": mask}


def test_completion_ids_text_reward_are_exactly_associated():
    recorder, texts, inputs, payload = make_ordered_recorder()
    recorder.capture_generation(inputs, payload, FakeTokenizer(texts))
    rows = recorder.records()
    assert len(rows) == 8
    assert [row["completion_index"] for row in rows] == list(range(8))
    assert [row["generation_index"] for row in rows] == [0, 1, 2, 3] * 2
    assert sum(row["exact_token_count"] for row in rows) == int(
        payload["completion_mask"].sum().item()
    )
    for row, text in zip(rows, texts, strict=True):
        assert row["decoded_completion"] == text
        assert row["raw_completion"] == text
        assert row["verifier_input"] == text
        assert row["exact_token_count"] == sum(row["completion_mask"])


def test_completion_reward_order_mismatch_fails_closed():
    recorder, texts, inputs, payload = make_ordered_recorder()
    inputs[0]["problem_id"] = "p1"
    with pytest.raises(TRLContractError, match="order mismatch"):
        recorder.capture_generation(inputs, payload, FakeTokenizer(texts))


def test_decoded_text_must_equal_verifier_input():
    recorder, texts, inputs, payload = make_ordered_recorder()
    changed = list(texts)
    changed[3] += "changed"
    with pytest.raises(TRLContractError, match="differs from verifier"):
        recorder.capture_generation(inputs, payload, FakeTokenizer(changed))


def test_missing_completion_evidence_cannot_succeed(tmp_path):
    class MissingEvidenceBackend(Backend):
        def run(self, problems, guard, reward_fn):
            result = super().run(problems, guard, reward_fn)
            result["completions"] = []
            return result

    result = run_guarded(
        load_config("configs/smoke/grpo.yaml"),
        MissingEvidenceBackend(checkpoint(tmp_path)),
        verifier,
        Lifecycle(tmp_path / "lifecycle"),
        Monitor(),
    )
    assert result["status"] == "failure"
    assert "8 persisted completion evidence" in result["reason"]


@pytest.mark.parametrize("key", ["kl", "train/kl", "objective/kl"])
def test_reviewed_kl_aliases_preserve_raw_key(key):
    result = extract_kl_metric([{"step": 1, key: 0.125}], beta=0.1)
    assert result["kl_available"] is True
    assert result["kl"] == 0.125
    assert result["kl_raw_key"] == key


def test_checkpoint_inventory_never_deserializes_files():
    source = inspect.getsource(checkpoint_inventory)
    assert "torch.load" not in source
    assert "pickle" not in source
    assert ".read_bytes()" in source


def test_kl_metric_alias_and_beta_zero_semantics():
    available = extract_kl_metric([{"step": 1, "kl": 0.25}], beta=0.1)
    assert available == {
        "beta": 0.1,
        "kl_available": True,
        "kl": 0.25,
        "kl_raw_key": "kl",
        "kl_unavailable_reason": None,
    }
    unavailable = extract_kl_metric([{"step": 1, "loss": 0.0}], beta=0.0)
    assert unavailable["kl_available"] is False
    assert unavailable["kl"] is None
    assert unavailable["kl_raw_key"] is None
    assert "beta=0.0" in unavailable["kl_unavailable_reason"]


class FakeCuda:
    def __init__(self, available=True):
        self.available = available
        self.reset_calls = []

    def is_available(self):
        return self.available

    def device_count(self):
        return 1

    def current_device(self):
        return 0

    def get_device_name(self, device):
        assert device == 0
        return "Fake CUDA"

    def reset_peak_memory_stats(self, device):
        self.reset_calls.append(device)

    def max_memory_allocated(self, device):
        return 10 * 1024 * 1024

    def max_memory_reserved(self, device):
        return 12 * 1024 * 1024

    def memory_allocated(self, device):
        return 2 * 1024 * 1024

    def memory_reserved(self, device):
        return 3 * 1024 * 1024


def test_fake_cuda_allocator_peak_is_recorded():
    cuda = FakeCuda()
    evidence = CudaAllocatorEvidence(cuda)
    evidence.start()
    payload = evidence.snapshot()
    assert cuda.reset_calls == [0]
    assert payload["available"] is True
    assert payload["max_memory_allocated"] == {
        "bytes": 10 * 1024 * 1024,
        "mib": 10.0,
    }
    assert payload["max_memory_reserved"]["mib"] == 12.0
    assert payload["memory_allocated"]["mib"] == 2.0
    assert payload["memory_reserved"]["mib"] == 3.0


def test_cpu_allocator_unavailable_does_not_touch_cuda():
    before = torch.cuda.is_initialized()
    cuda = FakeCuda(available=False)
    evidence = CudaAllocatorEvidence(cuda)
    evidence.start()
    payload = evidence.snapshot()
    assert cuda.reset_calls == []
    assert payload["available"] is False
    assert payload["max_memory_allocated"]["bytes"] is None
    assert payload["unavailable_reason"]
    assert torch.cuda.is_initialized() is before is False


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (
            "reports/runs/grpo_single_update_qwen25_05b_20260713T050407Z/summary.json",
            "4f67de1f42e2c8862a3918d3e4b7b4daa93b04b1b622ec007c2f412f8a2f2d77",
        ),
        (
            "reports/runs/grpo_single_update_qwen25_05b_20260713T050407Z/metrics.csv",
            "abedf37ea026caad0c15e4b0631330fe922c7ba6fd8550ef4b10eba681169e66",
        ),
        (
            "reports/runs/grpo_single_update_qwen25_05b_20260713T053852Z/summary.json",
            "e2a1ad0a7403e34ac15ecc2e08fdef0ed2829174b279d4180d57a0a811b62a3c",
        ),
        (
            "reports/runs/grpo_single_update_qwen25_05b_20260713T053852Z/metrics.csv",
            "de2bc84ff5e1680aef8d4aa36247a73d4c0ee45ed43b0a228802797d0ad73376",
        ),
        (
            "reports/runs/grpo_single_update_qwen25_05b_20260713T053852Z/checkpoint_inventory.json",
            "4572a9c888f31640f8320ec44429e93cc920815df967c7a1e5975f7ffc91e07d",
        ),
    ],
)
def test_historical_failure_evidence_is_immutable(path, expected):
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected


def test_backup_inventory_allows_adapter_and_tokenizer_but_rejects_secrets_and_base():
    validate_backup_inventory(
        [
            "run/checkpoint-1/adapter_model.safetensors",
            "run/checkpoint-1/tokenizer.json",
            "run/checkpoint-1/training_args.bin",
        ]
    )
    for name in (
        "run/checkpoint-1/model.safetensors",
        "run/auth.json",
        "run/cache/huggingface/blob",
        "run/proxy.txt",
    ):
        with pytest.raises(RuntimeError, match="prohibited"):
            validate_backup_inventory([name])
