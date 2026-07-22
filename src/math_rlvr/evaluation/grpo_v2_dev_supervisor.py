"""Non-CUDA parent supervisor for one fixed GRPO-v2 matched-dev worker."""

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

REQUIRED_SUCCESS_FILES = {
    "aggregate_metrics.json",
    "completions.jsonl",
    "domain_level_metrics.csv",
    "evaluation_identity.json",
    "final_summary.json",
    "model_roles.json",
    "per_problem.csv",
    "report.md",
    "resource_metrics.csv",
    "resource_summary.json",
    "sample_ledger.json",
    "status_distribution.csv",
}


def validate_success_artifacts(run_dir: Path) -> None:
    missing = sorted(name for name in REQUIRED_SUCCESS_FILES if not (run_dir / name).is_file())
    if missing:
        raise RuntimeError(f"dev evaluation success artifacts missing: {missing}")


def validate_backup_result(backup: dict[str, Any]) -> None:
    archive = Path(backup.get("archive", "missing"))
    expected = backup.get("sha256")
    if (
        not archive.is_file()
        or not isinstance(expected, str)
        or hashlib.sha256(archive.read_bytes()).hexdigest() != expected
    ):
        raise RuntimeError("dev evaluation backup verification failed")


def _worker_entry(queue, kwargs):
    try:
        from math_rlvr.evaluation.grpo_v2_dev_model_runtime import execute_dev_worker

        queue.put({"outcome": execute_dev_worker(**kwargs), "error": None})
    except BaseException as exc:
        queue.put(
            {
                "outcome": None,
                "error": {"type": type(exc).__name__, "reason": str(exc)},
            }
        )


def _refresh_checksums(run_dir: Path) -> None:
    lines = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            lines.append(
                f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(run_dir)}"
            )
    atomic_text(run_dir / "checksums.sha256", "\n".join(lines) + "\n")


def execute_supervised_dev(
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
        name="math-rlvr-grpo-v2-dev-worker",
    )
    process.start()
    worker_pid = process.pid
    process.join(config["budget"]["max_wall_time_seconds"] + 5)
    if process.is_alive():
        process.terminate()
        process.join(5)
        payload = {
            "outcome": None,
            "error": {"type": "TimeoutError", "reason": "dev worker exceeded deadline"},
        }
    else:
        try:
            payload = queue.get(timeout=2)
        except Exception:
            payload = {
                "outcome": None,
                "error": {"type": "WorkerExitError", "reason": "dev worker exited without result"},
            }
    current = gpu_probe()
    post_worker = verify_post_worker_exit(
        worker_pid=worker_pid,
        baseline=baseline,
        current=current,
        pid_exists=pid_probe,
    )
    backup = None
    if run_dir.is_dir():
        if payload["error"] is None:
            validate_success_artifacts(run_dir)
        atomic_text(
            run_dir / "post_worker_gpu_release.json",
            json.dumps(post_worker, indent=2, sort_keys=True) + "\n",
        )
        suffix = ".failure" if payload["error"] else ""
        archive = Path("/root/autodl-fs/math-rlvr-backups") / f"{run_dir.name}{suffix}.tar.gz"
        _refresh_checksums(run_dir)
        backup = create_formal_backup(run_dir, archive)
        validate_backup_result(backup)
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
        "post_worker_gpu_release": post_worker,
        "backup": backup,
    }
