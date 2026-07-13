#!/usr/bin/env python3
"""Fast offline metadata check; never imports torch or initializes CUDA."""

import importlib.metadata
import json
import platform
import shutil
import sys
from pathlib import Path

PACKAGES = (
    "torch",
    "transformers",
    "trl",
    "peft",
    "accelerate",
    "datasets",
    "PyYAML",
    "math-verify",
)
PATHS_FILE = Path(__file__).parents[1] / "configs" / "paths.yaml"


def package_versions():
    result = {}
    for package in PACKAGES:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def main():
    report = {
        "python": platform.python_version(),
        "packages": package_versions(),
        "disk": shutil.disk_usage("/root/autodl-tmp")._asdict(),
        "paths_config_present": PATHS_FILE.is_file(),
        "cuda_initialized": False,
        "model_or_tokenizer_loaded": False,
        "generated_code_execution": False,
    }
    print(json.dumps(report, indent=2))
    missing = [name for name, version in report["packages"].items() if version is None]
    if missing:
        print(f"INFO: dependencies not installed: {', '.join(missing)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
