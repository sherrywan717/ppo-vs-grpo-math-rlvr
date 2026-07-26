"""Non-CUDA parent supervisor for one frozen GRPO-v2 training worker."""

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
    "checkpoint_inventory.json",
    "completions.jsonl",
    "dev_validation.jsonl",
    "dev_validation_summary.json",
    "metrics.jsonl",
    "run_manifest.json",
    "sample_ledger.json",
    "selected_checkpoint.json",
    "summary.json",
}


def _worker_entry(queue, kwargs):
    try:
        from math_rlvr.training.grpo_v2_model_runtime import execute_real_grpo_v2

        queue.put({"outcome": execute_real_grpo_v2(**kwargs), "error": None})
    except BaseException as exc:
        queue.put(
            {
                "outcome": None,
                "error": {"type": type(exc).__name__, "reason": str(exc)},
            }
        )


def _verify_backup(backup: dict[str, Any] | None) -> None:
    if not isinstance(backup, dict):
        raise RuntimeError("GRPO-v2 backup evidence is missing")
    path = Path(backup.get("archive", "missing"))
    digest = backup.get("sha256")
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise RuntimeError("GRPO-v2 backup SHA verification failed")


def _refresh_checksums(run_dir: Path) -> None:
    lines = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(f"{digest}  {path.relative_to(run_dir)}")
    atomic_text(run_dir / "checksums.sha256", "\n".join(lines) + "\n")


def execute_supervised_grpo_v2(
    design: dict[str, Any],
    *,
    context_factory=None,
    gpu_probe=query_gpu_state,
    pid_probe=_pid_exists,
    **kwargs,
) -> dict[str, Any]:
    run_dir: Path = kwargs["run_dir"]
    timeout = int(design["budget"]["max_wall_time_seconds"]) + 5
    context = (context_factory or multiprocessing.get_context)("spawn")
    queue = context.Queue()
    baseline = gpu_probe()
    worker_kwargs = {"design": design, **kwargs}
    process = context.Process(
        target=_worker_entry,
        args=(queue, worker_kwargs),
        name="math-rlvr-grpo-v2-worker",
    )
    process.start()
    worker_pid = process.pid
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(5)
        payload = {
            "outcome": None,
            "error": {"type": "TimeoutError", "reason": "GRPO-v2 worker exceeded deadline"},
        }
    else:
        try:
            payload = queue.get(timeout=2)
        except Exception:
            payload = {
                "outcome": None,
                "error": {
                    "type": "WorkerExitError",
                    "reason": "GRPO-v2 worker exited without result",
                },
            }
    post_worker = verify_post_worker_exit(
        worker_pid=worker_pid,
        baseline=baseline,
        current=gpu_probe(),
        pid_exists=pid_probe,
    )
    backup = payload["outcome"].get("backup") if payload["outcome"] else None
    backup_manifest = run_dir / "backup_manifest.json"
    if backup is None and backup_manifest.is_file():
        saved = json.loads(backup_manifest.read_text())
        backup = {key: saved[key] for key in ("archive", "sha256") if key in saved}
    if run_dir.is_dir():
        atomic_text(
            run_dir / "post_worker_gpu_release.json",
            json.dumps(post_worker, indent=2, sort_keys=True) + "\n",
        )
        if payload["error"] is None:
            missing = sorted(
                name for name in REQUIRED_SUCCESS_FILES if not (run_dir / name).is_file()
            )
            if missing:
                raise RuntimeError(f"GRPO-v2 success artifacts missing: {missing}")
        elif backup is None:
            archive = Path("/root/autodl-fs/math-rlvr-backups") / f"{run_dir.name}.failure.tar.gz"
            backup = create_formal_backup(run_dir, archive)
            atomic_text(
                run_dir / "backup_manifest.json",
                json.dumps({"verified": True, **backup}, indent=2, sort_keys=True) + "\n",
            )
        _refresh_checksums(run_dir)
    _verify_backup(backup)
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
