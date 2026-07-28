"""Non-CUDA file-backed supervisor for one frozen hidden-test model role."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
from pathlib import Path
from typing import Any

from math_rlvr.artifacts.manager import atomic_text
from math_rlvr.evaluation.prompt_ab_supervisor import (
    _pid_exists,
    query_gpu_state,
    verify_post_worker_exit,
)
from math_rlvr.training.formal_runtime import create_formal_backup

MAX_IPC_BYTES = 4096
REQUIRED_SUCCESS_FILES = {
    "candidate0_metrics.csv",
    "candidate0_metrics.json",
    "checksums.sha256",
    "completions.jsonl",
    "evaluation_identity.json",
    "final_summary.json",
    "model_roles.json",
    "pass_k_per_problem.csv",
    "pass_k_summary.csv",
    "pass_k_summary.json",
    "per_problem.csv",
    "report.md",
    "resource_metrics.csv",
    "resource_summary.json",
    "status_distribution.csv",
    "summary.json",
    "truncation_analysis.csv",
}


def _primitive_tree(value: Any) -> bool:
    if value is None or type(value) in {str, int, float, bool}:
        return True
    if isinstance(value, dict):
        return all(isinstance(key, str) and _primitive_tree(item) for key, item in value.items())
    if isinstance(value, list):
        return all(_primitive_tree(item) for item in value)
    return False


def validate_ipc_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not _primitive_tree(payload):
        raise RuntimeError("hidden evaluator IPC payload must be primitive-only")
    encoded = json.dumps(payload, sort_keys=True).encode()
    if len(encoded) > MAX_IPC_BYTES:
        raise RuntimeError("hidden evaluator IPC payload exceeds 4 KiB")
    outcome = payload.get("outcome")
    if outcome is not None and set(outcome) != {
        "status",
        "run_id",
        "run_dir",
        "summary_path",
        "counts",
        "failure_reason",
    }:
        raise RuntimeError("hidden evaluator IPC success schema drift")
    return payload


def _worker_entry(queue, kwargs):
    try:
        from math_rlvr.evaluation.grpo_v2_hidden_model_runtime import execute_hidden_worker

        outcome = execute_hidden_worker(**kwargs)
        queue.put(validate_ipc_payload({"outcome": outcome, "error": None}))
    except BaseException as exc:
        reason = str(exc)
        if len(reason) > 2_048:
            reason = reason[:2_048] + "...[truncated]"
        queue.put(
            validate_ipc_payload(
                {
                    "outcome": None,
                    "error": {"type": type(exc).__name__, "reason": reason},
                }
            )
        )


def _refresh_checksums(run_dir: Path) -> None:
    lines = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            lines.append(
                f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(run_dir)}"
            )
    atomic_text(run_dir / "checksums.sha256", "\n".join(lines) + "\n")


def _validate_success_artifacts(run_dir: Path) -> dict[str, Any]:
    missing = sorted(name for name in REQUIRED_SUCCESS_FILES if not (run_dir / name).is_file())
    if missing:
        raise RuntimeError(f"hidden evaluation success artifacts missing: {missing}")
    summary = json.loads((run_dir / "summary.json").read_text())
    if summary.get("status") != "scientific_evaluation_success":
        raise RuntimeError("hidden evaluation on-disk summary status mismatch")
    return summary


def execute_supervised_hidden(
    *,
    context_factory=None,
    gpu_probe=query_gpu_state,
    pid_probe=_pid_exists,
    **kwargs,
) -> dict[str, Any]:
    config = kwargs["config"]
    run_dir = kwargs["run_dir"]
    context = (context_factory or multiprocessing.get_context)("spawn")
    queue = context.Queue()
    baseline = gpu_probe()
    process = context.Process(
        target=_worker_entry,
        args=(queue, kwargs),
        name="math-rlvr-grpo-v2-hidden-worker",
    )
    process.start()
    worker_pid = process.pid
    try:
        payload = validate_ipc_payload(
            queue.get(timeout=config["budget"]["max_wall_time_seconds"] + 5)
        )
    except Exception:
        process.terminate()
        process.join(5)
        payload = {
            "outcome": None,
            "error": {"type": "TimeoutError", "reason": "hidden worker returned no status"},
        }
    else:
        process.join(30)
        if process.is_alive():
            process.terminate()
            process.join(5)
            payload = {
                "outcome": None,
                "error": {
                    "type": "WorkerExitError",
                    "reason": "hidden worker did not exit after file-backed finalization",
                },
            }
    current = gpu_probe()
    post_worker = verify_post_worker_exit(
        worker_pid=worker_pid,
        baseline=baseline,
        current=current,
        pid_exists=pid_probe,
    )
    backup = None
    disk_summary = None
    if run_dir.is_dir():
        if payload["error"] is None:
            disk_summary = _validate_success_artifacts(run_dir)
        atomic_text(
            run_dir / "post_worker_gpu_release.json",
            json.dumps(post_worker, indent=2, sort_keys=True) + "\n",
        )
        _refresh_checksums(run_dir)
        suffix = ".failure" if payload["error"] else ""
        archive = (
            Path("/root/autodl-fs/math-rlvr-backups") / f"{run_dir.name}{suffix}.tar.gz"
        )
        backup = create_formal_backup(run_dir, archive)
        atomic_text(
            run_dir / "backup_manifest.json",
            json.dumps({"verified": True, **backup}, indent=2, sort_keys=True) + "\n",
        )
        _refresh_checksums(run_dir)
    if payload["error"]:
        return {
            "status": "failure",
            "reason": payload["error"]["reason"],
            "error_type": payload["error"]["type"],
            "post_worker_gpu_release": post_worker,
            "backup": backup,
        }
    return {
        **payload["outcome"],
        "summary": disk_summary,
        "post_worker_gpu_release": post_worker,
        "backup": backup,
    }
