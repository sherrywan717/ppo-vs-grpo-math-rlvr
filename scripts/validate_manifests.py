#!/usr/bin/env python3
"""Validate frozen manifests and intentional pass@4 subset relationships."""

import json
from pathlib import Path

from math_rlvr.dataset import load_manifest, validate_manifests

ROOT = Path("/root/autodl-tmp/datasets/math_rlvr/manifests")


def read(name):
    return load_manifest(ROOT / name)


def main() -> int:
    train = read("countdown_train.json") + read("train_core_128.json")
    validation = read("countdown_validation.json") + read("validation_64.json")
    gsm_test = read("gsm8k_test_200.json")
    math_test = read("math500_test_200.json")
    test = read("countdown_test.json") + gsm_test + math_test
    validate_manifests({"train": train, "validation": validation, "test": test})
    train_hashes = {x.content_hash for x in train}
    math500_hashes = {x.content_hash for x in math_test}
    if train_hashes & math500_hashes:
        raise ValueError("MATH500 leakage detected")
    for subset_name, parent in (
        ("gsm8k_pass4_50.json", gsm_test),
        ("math500_pass4_50.json", math_test),
    ):
        subset = read(subset_name)
        parent_ids = {x.problem_id for x in parent}
        if len(subset) != 50 or not {x.problem_id for x in subset} <= parent_ids:
            raise ValueError(f"invalid subset: {subset_name}")
    counts = {path.stem: len(load_manifest(path)) for path in sorted(ROOT.glob("*.json"))}
    print(json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
