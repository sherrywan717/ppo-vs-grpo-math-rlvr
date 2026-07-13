import csv
import hashlib
import json
from pathlib import Path

import pytest
import torch

from math_rlvr.evaluation.prompt_ab import DiagnosticAuthorizationError, main
from math_rlvr.evaluation.prompt_ab_evidence import (
    CAPABILITY_FIELDS,
    CAPABILITY_SCHEMA_VERSION,
    EvidenceContractError,
    build_group_reward_evidence,
    build_paired_comparison,
    validate_capability_manifest,
    validate_cross_file_consistency,
    write_paired_csv,
)
from math_rlvr.evaluation.prompt_ab_supervisor import verify_post_worker_exit
from math_rlvr.training.resource_evidence import CudaAllocatorEvidence

CONFIG = Path("configs/diagnostics/prompt_ab.yaml")


def completion_rows():
    rows = []
    for condition_index, condition in enumerate(("v0", "v1")):
        for problem_index, problem_id in enumerate(("countdown:train:0", "countdown:train:1")):
            for generation_index in range(4):
                seed = 42 + problem_index * 4 + generation_index
                reward = float(generation_index % 2) if problem_index == 1 else 0.0
                rows.append(
                    {
                        "condition": condition,
                        "problem_id": problem_id,
                        "generation_index": generation_index,
                        "seed": seed,
                        "matched_seed": seed,
                        "prompt_hash": f"problem-hash-{problem_index}",
                        "rendered_prompt_sha256": f"{condition}-rendered-{problem_index}",
                        "completion_index": condition_index * 8
                        + problem_index * 4
                        + generation_index,
                        "exact_completion_token_count": 8 + generation_index,
                        "decoded_raw_text": f"{condition}-{problem_id}-{generation_index}",
                        "reward_status": "verified_pass" if reward else "wrong_answer",
                        "scalar_reward": reward,
                        "format_valid": bool(reward),
                        "truncated_at_128": False,
                        "expression_valid": True,
                        "number_usage_valid": True,
                        "final_answer_correct": bool(reward),
                    }
                )
    return rows


def test_paired_rows_are_exact_stable_and_csv_matches(tmp_path):
    pairs = build_paired_comparison(completion_rows())
    assert len(pairs) == 8
    assert [(row["problem_id"], row["generation_index"], row["matched_seed"]) for row in pairs] == [
        (problem, generation, 42 + problem_index * 4 + generation)
        for problem_index, problem in enumerate(("countdown:train:0", "countdown:train:1"))
        for generation in range(4)
    ]
    path = tmp_path / "paired.csv"
    write_paired_csv(path, pairs)
    with path.open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == len(pairs)
    assert csv_rows[0]["status_transition"] == pairs[0]["status_transition"]
    assert [row["status_transition"] for row in pairs] == [
        "wrong_answer -> wrong_answer",
    ] * 4 + [
        "wrong_answer -> wrong_answer",
        "verified_pass -> verified_pass",
        "wrong_answer -> wrong_answer",
        "verified_pass -> verified_pass",
    ]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "seed", "problem"])
def test_missing_duplicate_or_mismatched_pair_is_rejected(mutation):
    rows = completion_rows()
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(dict(rows[-1]))
    elif mutation == "seed":
        rows[-1]["matched_seed"] += 1
    else:
        rows[-1]["problem_id"] = "countdown:train:other"
    with pytest.raises(EvidenceContractError):
        build_paired_comparison(rows)


def test_per_problem_reward_order_variance_and_zero_advantage_counts():
    evidence = build_group_reward_evidence(completion_rows())
    for condition in ("v0", "v1"):
        groups = evidence["conditions"][condition]
        assert groups["problems"]["countdown:train:0"]["scalar_rewards"] == [0.0] * 4
        assert groups["problems"]["countdown:train:0"]["zero_advantage_group"] is True
        assert groups["problems"]["countdown:train:1"]["scalar_rewards"] == [0.0, 1.0, 0.0, 1.0]
        assert groups["problems"]["countdown:train:1"]["variance"] == 0.25
        assert groups["zero_advantage_group_count"] == 1
        assert groups["nonzero_variance_group_count"] == 1


class FakeCuda:
    def __init__(self):
        self.calls = []

    def is_available(self):
        return True

    def device_count(self):
        return 1

    def current_device(self):
        return 0

    def get_device_name(self, index):
        self.calls.append(("name", index))
        return "fake H800"

    def reset_peak_memory_stats(self, index):
        self.calls.append(("reset", index))

    def max_memory_allocated(self, index):
        self.calls.append(("max_allocated", index))
        return 2 * 1024 * 1024

    def max_memory_reserved(self, index):
        self.calls.append(("max_reserved", index))
        return 3 * 1024 * 1024

    def memory_allocated(self, index):
        self.calls.append(("allocated", index))
        return 0

    def memory_reserved(self, index):
        self.calls.append(("reserved", index))
        return 0


def test_fake_allocator_current_peak_and_lifecycle_use_index_zero():
    fake = FakeCuda()
    evidence = CudaAllocatorEvidence(fake, device=0)
    evidence.start()
    payload = evidence.finalize()
    assert payload["state"] == "finalized"
    assert payload["device_index"] == 0
    assert payload["memory_allocated"]["bytes"] == 0
    assert payload["memory_reserved"]["bytes"] == 0
    assert payload["max_memory_allocated"]["mib"] == 2.0
    assert payload["max_memory_reserved"]["mib"] == 3.0
    assert all(index == 0 for _, index in fake.calls)


def test_post_worker_proves_pid_exit_compute_absence_and_memory_restore():
    evidence = verify_post_worker_exit(
        worker_pid=123,
        baseline={"memory_used_mib": {"0": 7}, "compute_pids": []},
        current={"memory_used_mib": {"0": 7}, "compute_pids": []},
        pid_exists=lambda _pid: False,
    )
    assert evidence["worker_pid_exited"]
    assert evidence["gpu_memory_restored_to_baseline"]
    assert evidence["parent_cuda_initialized"] is False


@pytest.mark.parametrize("field", CAPABILITY_FIELDS)
def test_each_missing_capability_rejects_before_execute(field):
    capabilities = {name: True for name in CAPABILITY_FIELDS}
    capabilities[field] = False
    payload = {"schema_version": CAPABILITY_SCHEMA_VERSION, "capabilities": capabilities}
    calls = []
    with pytest.raises(DiagnosticAuthorizationError, match="capability unavailable"):
        main(
            ["--config", str(CONFIG), "--generate-only", "--confirm-prompt-diagnostic"],
            execute_fn=lambda **kwargs: calls.append(kwargs),
            capability_probe=lambda: validate_capability_manifest(payload),
        )
    assert calls == []


def make_consistent_artifacts(root: Path):
    rows = completion_rows()
    pairs = build_paired_comparison(rows)
    groups = build_group_reward_evidence(rows)
    metrics = {}
    for condition in ("v0", "v1"):
        selected = [row for row in rows if row["condition"] == condition]
        metrics[condition] = {
            "reward_status_counts": dict(
                __import__("collections").Counter(row["reward_status"] for row in selected)
            ),
            "zero_advantage_group_count": 1,
        }
    token_total = sum(row["exact_completion_token_count"] for row in rows)
    summary = {
        "status": "pending_backup",
        "completion_count": 16,
        "budget": {
            "limits": {"total_completions": 16, "total_generated_tokens": 2048},
            "total_generated_tokens": token_total,
        },
        "safety_counters": {
            "backward_count": 0,
            "optimizer_steps": 0,
            "global_training_steps": 0,
            "checkpoint_writes": 0,
            "model_or_adapter_writes": 0,
        },
    }
    payloads = {
        "summary.json": summary,
        "run_manifest.json": {"completion_count": 16},
        "per_condition_metrics.json": metrics,
        "paired_comparison.json": pairs,
        "per_problem_rewards.json": groups,
        "seed_map.json": [
            {key: row[key] for key in ("condition", "problem_id", "generation_index", "seed")}
            for row in rows
        ],
        "prompt_hashes.json": [
            {"problem_id": f"countdown:train:{index}", "prompt_hash": f"problem-hash-{index}"}
            for index in range(2)
        ],
        "resource_cost.json": {},
        "pytorch_allocator.json": {},
        "post_worker_gpu_verification.json": {},
        "backup_manifest.json": {"verified": True},
    }
    for name, payload in payloads.items():
        (root / name).write_text(json.dumps(payload) + "\n")
    (root / "completions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    write_paired_csv(root / "paired_comparison.csv", pairs)
    with (root / "per_condition_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("condition", "completions"))
        writer.writeheader()
        writer.writerows({"condition": condition, "completions": 8} for condition in ("v0", "v1"))
    checksum_lines = []
    for path in sorted(root.iterdir()):
        if path.is_file() and path.name != "checksums.sha256":
            checksum_lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
    (root / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n")


def test_cross_file_consistency_accepts_complete_contract(tmp_path):
    make_consistent_artifacts(tmp_path)
    result = validate_cross_file_consistency(tmp_path, require_backup=True)
    assert result["valid"] and result["paired_row_count"] == 8


@pytest.mark.parametrize("mutation", ["totals", "status", "tokens", "seed"])
def test_cross_file_inconsistencies_fail_closed(tmp_path, mutation):
    make_consistent_artifacts(tmp_path)
    if mutation == "totals":
        path = tmp_path / "summary.json"
        payload = json.loads(path.read_text())
        payload["completion_count"] = 15
        path.write_text(json.dumps(payload))
    elif mutation == "status":
        path = tmp_path / "per_condition_metrics.json"
        payload = json.loads(path.read_text())
        payload["v0"]["reward_status_counts"] = {"wrong_answer": 8}
        path.write_text(json.dumps(payload))
    elif mutation == "tokens":
        path = tmp_path / "summary.json"
        payload = json.loads(path.read_text())
        payload["budget"]["total_generated_tokens"] += 1
        path.write_text(json.dumps(payload))
    else:
        path = tmp_path / "seed_map.json"
        payload = json.loads(path.read_text())
        payload[0]["seed"] = 99
        path.write_text(json.dumps(payload))
    # Refresh only the mutated file checksum so the semantic validator is exercised.
    lines = []
    for item in sorted(tmp_path.iterdir()):
        if item.is_file() and item.name != "checksums.sha256":
            lines.append(f"{hashlib.sha256(item.read_bytes()).hexdigest()}  {item.name}")
    (tmp_path / "checksums.sha256").write_text("\n".join(lines) + "\n")
    with pytest.raises(EvidenceContractError):
        validate_cross_file_consistency(tmp_path, require_backup=True)


def test_cpu_evidence_tests_do_not_initialize_cuda():
    assert torch.cuda.is_initialized() is False
