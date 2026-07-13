"""Non-CUDA parent supervisor for the fixed prompt A/B worker."""

from __future__ import annotations

import multiprocessing
import os
import subprocess
from collections.abc import Callable
from typing import Any

from math_rlvr.evaluation.prompt_ab_evidence import minimal_failure_record


class PostWorkerVerificationError(RuntimeError):
    pass


def query_gpu_state() -> dict[str, Any]:
    memory = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    memory_by_index = {}
    for line in memory.stdout.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) == 2 and all(value.isdigit() for value in fields):
            memory_by_index[fields[0]] = int(fields[1])
    compute_pids = []
    for line in processes.stdout.splitlines():
        pid = line.split(",", maxsplit=1)[0].strip()
        if pid.isdigit():
            compute_pids.append(int(pid))
    return {"memory_used_mib": memory_by_index, "compute_pids": sorted(compute_pids)}


def verify_post_worker_exit(
    *,
    worker_pid: int,
    baseline: dict[str, Any],
    current: dict[str, Any],
    pid_exists: Callable[[int], bool],
) -> dict[str, Any]:
    process_exited = not pid_exists(worker_pid)
    absent_from_compute = worker_pid not in current["compute_pids"]
    baseline_memory = int(baseline["memory_used_mib"].get("0", 0))
    current_memory = int(current["memory_used_mib"].get("0", 0))
    memory_restored = current_memory <= baseline_memory
    evidence = {
        "worker_pid": worker_pid,
        "worker_pid_exited": process_exited,
        "worker_absent_from_nvidia_smi_compute_processes": absent_from_compute,
        "baseline_gpu_memory_mib": baseline_memory,
        "post_worker_gpu_memory_mib": current_memory,
        "gpu_memory_restored_to_baseline": memory_restored,
        "parent_cuda_initialized": False,
    }
    if not (process_exited and absent_from_compute and memory_restored):
        raise PostWorkerVerificationError(f"post-worker GPU verification failed: {evidence}")
    return evidence


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def fixed_worker_entry(queue, config, source, git_info):
    """Fixed callable: no command string and no user-selected target."""
    try:
        from math_rlvr.evaluation.prompt_ab_runtime import execute_worker_diagnostic

        queue.put(execute_worker_diagnostic(config=config, source=source, git_info=git_info))
    except BaseException as exc:
        queue.put({"result": minimal_failure_record(exc, phase="worker", run_id=None)})


def execute_supervised_diagnostic(
    *,
    config,
    source,
    git_info,
    context_factory=None,
    gpu_probe=query_gpu_state,
    pid_probe=_pid_exists,
):
    """Run the fixed worker once; parent never imports torch or initializes CUDA."""
    context = (context_factory or multiprocessing.get_context)("spawn")
    queue = context.Queue()
    baseline = gpu_probe()
    process = context.Process(
        target=fixed_worker_entry,
        args=(queue, config, source, git_info),
        name="math-rlvr-prompt-ab-worker",
    )
    process.start()
    worker_pid = process.pid
    process.join(config["budget"]["max_wall_time_seconds"] + 5)
    if process.is_alive():
        process.terminate()
        process.join(5)
        payload = {
            "result": minimal_failure_record(
                TimeoutError("generation diagnostic worker exceeded deadline"),
                phase="worker_timeout",
                run_id=None,
            )
        }
    else:
        payload = queue.get(timeout=2)
    current = gpu_probe()
    post_worker = verify_post_worker_exit(
        worker_pid=worker_pid,
        baseline=baseline,
        current=current,
        pid_exists=pid_probe,
    )
    from math_rlvr.evaluation.prompt_ab_runtime import finalize_parent_diagnostic

    return finalize_parent_diagnostic(
        payload=payload,
        post_worker=post_worker,
        config=config,
        git_info=git_info,
    )
