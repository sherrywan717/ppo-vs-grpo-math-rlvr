#!/usr/bin/env python3
"""Fast offline environment check; never initializes CUDA or downloads assets."""

import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from code_rlvr.execution.capabilities import detect_capabilities

PACKAGES = ("torch", "transformers", "trl", "peft", "accelerate", "datasets", "PyYAML")
PATHS_FILE = Path(__file__).parents[1] / "configs" / "paths.yaml"


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def gpu_inventory() -> list[str]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return []
    result = subprocess.run(
        [nvidia_smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True, check=False, text=True, timeout=5,
    )
    return result.stdout.strip().splitlines() if result.returncode == 0 else []


def main() -> int:
    capabilities = detect_capabilities()
    report = {
        "python": platform.python_version(),
        "packages": package_versions(),
        "gpus": gpu_inventory(),
        "disk": shutil.disk_usage("/root/autodl-tmp")._asdict(),
        "paths_config_present": PATHS_FILE.is_file(),
        "untrusted_execution": {
            "allowed": capabilities.can_execute_untrusted,
            "safe_backend": capabilities.safe_backend,
            "policy": "fail_closed",
        },
    }
    print(json.dumps(report, indent=2))
    missing = [name for name, version in report["packages"].items() if version is None]
    if missing:
        print(f"INFO: dependencies not installed: {', '.join(missing)}", file=sys.stderr)
    if not capabilities.can_execute_untrusted:
        print("SAFE: untrusted code execution is disabled", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

