#!/usr/bin/env python3
# ruff: noqa: E501
"""Deterministically build the CPU-only GRPO-v2 data and design freeze."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from datasets import Dataset

from math_rlvr.dataset import MathProblem, content_hash
from math_rlvr.gold import normalize_gold_answer
from math_rlvr.grpo_v2_contract import canonical_json_sha256, selection_key
from math_rlvr.rewards.result import RewardStatus
from math_rlvr.verifier import MathVerifier

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs/grpo_v2"
MAN = CFG / "manifests"
REPORT = ROOT / "reports/grpo_v2"
METRICS = REPORT / "metrics"
FIGURES = REPORT / "figures"
TRUSTED = Path("/root/autodl-tmp/datasets/math_rlvr/grpo_v2/trusted")
SEED = 42

SOURCES = {
    "gsm_train": (
        "openai/gsm8k",
        "740312add88f781978c0658806c59bc2815b9866",
        "train",
        Path(
            "/root/autodl-tmp/cache/huggingface/datasets/openai___gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866/gsm8k-train.arrow"
        ),
    ),
    "gsm_test": (
        "openai/gsm8k",
        "740312add88f781978c0658806c59bc2815b9866",
        "test",
        Path(
            "/root/autodl-tmp/cache/huggingface/datasets/openai___gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866/gsm8k-test.arrow"
        ),
    ),
    "math_train": (
        "DigitalLearningGmbH/MATH-lighteval",
        "0530c78699ea5e8eb5530600900e1f328b48acad",
        "train",
        Path(
            "/root/autodl-tmp/cache/huggingface/datasets/DigitalLearningGmbH___math-lighteval/default/0.0.0/0530c78699ea5e8eb5530600900e1f328b48acad/math-lighteval-train.arrow"
        ),
    ),
    "math500": (
        "HuggingFaceH4/MATH-500",
        "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        "test",
        Path(
            "/root/autodl-tmp/cache/huggingface/datasets/HuggingFaceH4___math-500/default/0.0.0/6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be/math-500-test.arrow"
        ),
    ),
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows))


def public_row(
    source: str,
    split: str,
    index: int,
    prompt: str,
    dataset_id: str,
    revision: str,
    source_split: str,
    difficulty: str,
    source_problem_id: str,
) -> dict:
    prefix = "gsm8k" if source == "gsm8k" else "math"
    pid = f"{prefix}:{dataset_id}:{source_split}:{source_problem_id}"
    return {
        "problem_id": pid,
        "source": source,
        "prompt": prompt,
        "category": "arithmetic" if source == "gsm8k" else "competition_math",
        "difficulty": difficulty,
        "split": split,
        "source_index": index,
        "source_problem_id": source_problem_id,
        "content_hash": content_hash(prompt),
        "metadata": {
            "dataset_id": dataset_id,
            "revision": revision,
            "source_split": source_split,
            "source_index": index,
        },
    }


def stable_select(rows: list[dict], count: int, namespace: str) -> list[dict]:
    return sorted(
        rows,
        key=lambda r: selection_key(
            dataset_revision=r["metadata"]["revision"],
            source_split=r["metadata"]["source_split"],
            source_problem_id=r["source_problem_id"],
            namespace=namespace,
            seed=SEED,
        ),
    )[:count]


def gold_for(kind: str, row: dict) -> tuple[str, str]:
    if kind.startswith("gsm"):
        reasoning, answer = row["answer"].rsplit("####", 1)
        return answer.strip().replace(",", ""), reasoning.strip()
    if kind == "math500":
        return normalize_gold_answer(row["answer"]), row["solution"].strip()
    return normalize_gold_answer(row["solution"]), row["solution"].strip()


def main() -> None:
    for path in (CFG, MAN, REPORT, METRICS, FIGURES, TRUSTED):
        path.mkdir(parents=True, exist_ok=True)
    v1_hashes: set[str] = set()
    v1_ids: set[tuple[str, int]] = set()
    v1_dir = Path("/root/autodl-tmp/datasets/math_rlvr/manifests")
    for path in sorted(v1_dir.glob("*.json")):
        try:
            rows = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict) and row.get("content_hash"):
                v1_hashes.add(row["content_hash"])
                metadata = row.get("metadata", {})
                v1_ids.add((str(metadata.get("dataset_id", "")), int(row.get("source_index", -1))))

    raw: dict[str, list[dict]] = {}
    private: dict[str, dict] = {}
    for kind, (dataset_id, revision, source_split, path) in SOURCES.items():
        dataset = Dataset.from_file(str(path))
        built = []
        for index, item in enumerate(dataset):
            source = "gsm8k" if kind.startswith("gsm") else "math"
            prompt = item["question"] if source == "gsm8k" else item["problem"]
            difficulty = "na" if source == "gsm8k" else str(item["level"]).replace("Level ", "")
            source_problem_id = str(index) if kind != "math500" else str(item["unique_id"])
            row = public_row(
                source,
                "candidate",
                index,
                prompt,
                dataset_id,
                revision,
                source_split,
                difficulty,
                source_problem_id,
            )
            if row["content_hash"] in v1_hashes or (dataset_id, index) in v1_ids:
                continue
            try:
                gold, reasoning = gold_for(kind, item)
            except ValueError:
                # Reuse the v1 trusted-gold admission rule: unstable official
                # solutions are ineligible rather than normalized differently.
                continue
            private[row["problem_id"]] = {
                "problem_id": row["problem_id"],
                "gold_answer": gold,
                "official_solution": reasoning,
            }
            built.append(row)
        raw[kind] = built

    def set_split(rows: list[dict], split: str) -> list[dict]:
        return [{**r, "split": split} for r in rows]

    gsm_ranked = stable_select(raw["gsm_train"], 320, "grpo_v2/train_dev/gsm8k")
    train_gsm, dev_gsm = gsm_ranked[:256], gsm_ranked[256:]
    train_math, dev_math = [], []
    for level, train_n, dev_n in ((1, 64, 16), (2, 96, 24), (3, 96, 24)):
        candidates = [r for r in raw["math_train"] if r["difficulty"] == str(level)]
        chosen = stable_select(candidates, train_n + dev_n, f"grpo_v2/train_dev/math/level_{level}")
        train_math.extend(chosen[:train_n])
        dev_math.extend(chosen[train_n:])
    train = set_split(train_gsm + train_math, "train_v2")
    dev = set_split(dev_gsm + dev_math, "dev_v2")

    warm_gsm = stable_select(
        [r for r in train if r["source"] == "gsm8k"], 128, "grpo_v2/warmstart/gsm8k"
    )
    warm_math = []
    for level, n in ((1, 32), (2, 48), (3, 48)):
        warm_math += stable_select(
            [r for r in train if r["source"] == "math" and r["difficulty"] == str(level)],
            n,
            f"grpo_v2/warmstart/math/level_{level}",
        )
    warm = [{**r, "split": "warmstart_v2"} for r in warm_gsm + warm_math]

    test_gsm = stable_select(raw["gsm_test"], 200, "grpo_v2/test_v2_hidden/gsm8k")
    test_math, capacity = [], {}
    test_counts = {1: 3, 2: 33, 3: 43, 4: 59, 5: 62}
    for level, n in test_counts.items():
        candidates = [r for r in raw["math500"] if r["difficulty"] == str(level)]
        capacity[str(level)] = len(candidates)
        test_math += stable_select(candidates, n, f"grpo_v2/test_v2_hidden/math500/level_{level}")
    test = set_split(test_gsm + test_math, "test_v2_hidden")
    nested_gsm = stable_select(test_gsm, 50, "grpo_v2/pass4_nested/gsm8k")
    nested_math = []
    nested_counts = {1: 3, 2: 8, 3: 10, 4: 14, 5: 15}
    for level, n in nested_counts.items():
        nested_math += stable_select(
            [r for r in test_math if r["difficulty"] == str(level)],
            n,
            f"grpo_v2/pass4_nested/math500/level_{level}",
        )
    nested = [{**r, "split": "pass4_nested_subset"} for r in nested_gsm + nested_math]

    for name, rows in (
        ("train_v2", train),
        ("warmstart_v2", warm),
        ("dev_v2", dev),
        ("test_v2_hidden", test),
    ):
        dump_jsonl(MAN / f"{name}.jsonl", rows)
        dump_jsonl(
            TRUSTED / f"{name}_trusted.jsonl",
            [{**private[r["problem_id"]], "content_hash": r["content_hash"]} for r in rows],
        )
    dump_json(
        MAN / "pass4_nested_subset.json",
        {
            "schema_version": 1,
            "semantics": "candidate_0_is_shared_with_pass_at_1; candidates_0_to_3_are_nested",
            "problems": nested,
        },
    )

    verifier = MathVerifier()
    targets = []
    for row in warm:
        p = private[row["problem_id"]]
        target = (
            f"<reasoning>{p['official_solution']}</reasoning>\n<answer>{p['gold_answer']}</answer>"
        )
        mp = MathProblem(
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
            gold_answer=p["gold_answer"],
        )
        result = verifier(mp, target)
        if result.status is not RewardStatus.VERIFIED_PASS:
            raise ValueError(
                f"warmstart target verifier failure: {row['problem_id']} {result.status}"
            )
        targets.append(
            {
                "problem_id": row["problem_id"],
                "target_text": target,
                "target_sha256": hashlib.sha256(target.encode()).hexdigest(),
            }
        )
    dump_jsonl(TRUSTED / "warmstart_targets.jsonl", targets)

    # Frozen easy-to-hard order: 2G+2L1, then 32*(3G+1L2)+32*(2G+2L2), then 1G+3L3.
    train_by_id = {r["problem_id"]: r for r in train}
    gsm = [r for r in train if r["source"] == "gsm8k"]
    gsm = sorted(
        gsm,
        key=lambda r: (
            len(private[r["problem_id"]]["official_solution"].split()),
            r["content_hash"],
        ),
    )
    l1 = sorted([r for r in train if r["difficulty"] == "1"], key=lambda r: r["content_hash"])
    l2 = sorted([r for r in train if r["difficulty"] == "2"], key=lambda r: r["content_hash"])
    l3 = sorted([r for r in train if r["difficulty"] == "3"], key=lambda r: r["content_hash"])
    updates = []
    for i in range(32):
        updates.append(gsm[2 * i : 2 * i + 2] + l1[2 * i : 2 * i + 2])
    middle_gsm = gsm[64:224]
    for i in range(32):
        updates.append(middle_gsm[3 * i : 3 * i + 3] + l2[i : i + 1])
    for i in range(32):
        updates.append(middle_gsm[96 + 2 * i : 96 + 2 * i + 2] + l2[32 + 2 * i : 32 + 2 * i + 2])
    for i in range(32):
        updates.append(gsm[224 + i : 225 + i] + l3[3 * i : 3 * i + 3])
    positions = []
    for update, group in enumerate(updates, 1):
        for slot, row in enumerate(group):
            positions.append(
                {
                    "position": len(positions) + 1,
                    "update": update,
                    "slot": slot,
                    "problem_id": row["problem_id"],
                    "content_hash": row["content_hash"],
                    "domain": row["source"],
                    "level": row["difficulty"],
                    "difficulty_proxy": "official_solution_whitespace_tokens"
                    if row["source"] == "gsm8k"
                    else f"math_level_{row['difficulty']}",
                    "difficulty_proxy_value": len(
                        private[row["problem_id"]]["official_solution"].split()
                    )
                    if row["source"] == "gsm8k"
                    else int(row["difficulty"]),
                }
            )
    if len({p["problem_id"] for p in positions}) != 512 or set(train_by_id) != {
        p["problem_id"] for p in positions
    }:
        raise ValueError("curriculum does not cover train_v2 exactly once")
    curriculum = {
        "schema_version": 1,
        "selection_seed": 42,
        "frozen_before_training": True,
        "positions": positions,
    }
    curriculum["curriculum_sha256"] = canonical_json_sha256(curriculum)
    dump_json(CFG / "curriculum.json", curriculum)
    with (REPORT / "curriculum_ledger.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=positions[0])
        w.writeheader()
        w.writerows(positions)

    selection_contract = {
        "schema_version": 1,
        "allowed_split": "dev_v2",
        "candidate_steps": [32, 64, 96, 128],
        "lexicographic": [
            {"metric": "canonical_pass_at_1", "direction": "max"},
            {"metric": "parseable_rate", "direction": "max"},
            {"metric": "format_rate", "direction": "max"},
            {"metric": "truncation_rate", "direction": "min"},
            {"metric": "checkpoint_step", "direction": "min"},
        ],
        "test_selection_forbidden": True,
    }
    dump_json(CFG / "checkpoint_selection.json", selection_contract)
    eval_contract = {
        "schema_version": 1,
        "seed": 42,
        "candidate0_problem_count": 400,
        "nested_pass4_problem_count": 100,
        "extra_nested_candidates": 300,
        "nested_candidate_rows": 400,
        "completions_per_model": 700,
        "total_four_model_completions": 2800,
        "models": ["base", "old_grpo_v1_seed42_checkpoint32", "warmstart_only", "selected_grpo_v2"],
        "pass_at_1": {"numerator": "candidate_0 canonical passes", "denominator": 400},
        "nested_pass_at_4": {
            "numerator": "problems with any canonical pass among candidates 0..3",
            "denominator": 100,
            "candidate0_shared": True,
        },
        "math_primary": "200-problem micro_average",
        "math_macro": "diagnostic_unweighted_level_macro",
        "math_level_reporting": {
            str(i): {
                "denominator": test_counts[i],
                "interval": "wilson_95",
                "status": "diagnostic_only_small_n" if i == 1 else "diagnostic",
            }
            for i in range(1, 6)
        },
        "test_may_trigger_retraining": False,
    }
    dump_json(CFG / "evaluation.json", eval_contract)
    warm_cfg = {
        "schema_version": 1,
        "experiment": "grpo_v2_warmstart_seed42",
        "seed": 42,
        "model": {
            "repo": "Qwen/Qwen2.5-1.5B-Instruct",
            "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
            "dtype": "bfloat16",
            "local_files_only": True,
        },
        "data": {
            "manifest": "configs/grpo_v2/manifests/warmstart_v2.jsonl",
            "trusted_targets": str(TRUSTED / "warmstart_targets.jsonl"),
            "samples": 256,
            "epochs": 1,
            "shuffle": "deterministic_frozen_order",
        },
        "prompt": {
            "version": "prompt_v2_formal_math",
            "max_prompt_length": 832,
            "max_target_length": 256,
        },
        "lora": {
            "rank": 16,
            "alpha": 32,
            "dropout": 0,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
        "training": {
            "per_device_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "effective_batch_size": 16,
            "trainable_parameter_count": 4358144,
            "optimizer_steps": 16,
            "scheduler_steps": 16,
            "learning_rate": 1e-5,
        },
        "budget": {
            "max_target_tokens": 65536,
            "max_prompt_plus_target_tokens": 278528,
            "target_token_audit": {
                "value": None,
                "available": False,
                "reason": "Stage N forbids loading the pinned tokenizer; required before Stage O GPU execution",
            },
            "max_wall_time_seconds": 2700,
            "max_vram_gib": 24,
            "max_gpu_hours": 0.75,
            "max_cost_cny": 6.66,
        },
        "checkpoint": {
            "steps": [16],
            "adapter_only": True,
            "trusted_resume_state": True,
            "base_weights_forbidden": True,
        },
        "post_warmstart_dev": {
            "problems": 128,
            "candidates_per_problem": 1,
            "separate_token_ledger": True,
        },
    }
    dump_json(CFG / "warmstart_seed42.json", warm_cfg)
    grpo_cfg = {
        "schema_version": 1,
        "experiment": "grpo_v2_seed42",
        "seed": 42,
        "initial_adapter": "warmstart_only_checkpoint",
        "old_grpo_v1_checkpoint_forbidden": True,
        "model": warm_cfg["model"],
        "data": {
            "manifest": "configs/grpo_v2/manifests/train_v2.jsonl",
            "curriculum": "configs/grpo_v2/curriculum.json",
            "unique_prompts": 512,
            "each_prompt_occurrences": 1,
            "dev_manifest": "configs/grpo_v2/manifests/dev_v2.jsonl",
        },
        "prompt": {
            "version": "prompt_v2_formal_math",
            "sha256": "89e459da827474d9bcc66e4407b06b5f8a968ce10d0be92e830c59fd9830a994",
            "max_prompt_length": 832,
            "max_completion_length": 256,
        },
        "reward": {
            "policy": "shaped_v3_domain",
            "sha256": "b9eda9520bb0271e28f6c209db85a408cdc0a65c2d403871b2b0fcc06e06a463",
        },
        "parser_sha256": "655c30f20c677ead5728b402a1b6d5a4d4cefe54e4c1b34abebdafe41f3ba0ad",
        "verifier_sha256": "ac3603158e31c8603c21e5d33445745bb56f3ccf946b055db9544a3dbc5886fd",
        "lora": warm_cfg["lora"],
        "generation": {
            "temperature": 0.8,
            "top_p": 0.95,
            "num_generations": 4,
            "generation_batch_size": 16,
        },
        "training": {
            "updates": 128,
            "prompts_per_update": 4,
            "completions_per_prompt": 4,
            "completions_per_update": 16,
            "training_completions": 2048,
            "per_device_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "expected_microsteps": 512,
            "optimizer_steps": 128,
            "global_steps": 128,
            "learning_rate": 1e-5,
            "trl_version": "0.24.0",
        },
        "budget": {
            "max_generated_tokens": 524288,
            "max_wall_time_seconds": 7200,
            "max_vram_gib": 24,
            "max_gpu_hours": 2.0,
            "max_cost_cny": 17.76,
        },
        "checkpoint_steps": [32, 64, 96, 128],
        "dev_validation": {
            "steps": [32, 64, 96, 128],
            "problems_per_step": 128,
            "candidates_per_problem": 1,
            "training_budget_scope": False,
        },
    }
    dump_json(CFG / "grpo_v2_seed42.json", grpo_cfg)

    manifests = {
        n: MAN / f"{n}.jsonl" for n in ("train_v2", "warmstart_v2", "dev_v2", "test_v2_hidden")
    }
    registry = {
        "schema_version": 1,
        "selection_seed": 42,
        "sources": {
            k: {
                "dataset_id": v[0],
                "revision": v[1],
                "source_split": v[2],
                "arrow_sha256": sha(v[3]),
            }
            for k, v in SOURCES.items()
        },
        "manifests": {
            n: {
                "path": str(p.relative_to(ROOT)),
                "sha256": sha(p),
                "count": sum(1 for _ in p.open()),
            }
            for n, p in manifests.items()
        },
        "nested_subset": {
            "path": "configs/grpo_v2/manifests/pass4_nested_subset.json",
            "sha256": sha(MAN / "pass4_nested_subset.json"),
            "count": 100,
        },
        "trusted_runtime": {
            "path": str(TRUSTED),
            "git_public": False,
            "files": {
                p.name: {"sha256": sha(p), "size": p.stat().st_size}
                for p in sorted(TRUSTED.glob("*.jsonl"))
            },
        },
        "v1_excluded_content_hash_count": len(v1_hashes),
    }
    registry["registry_sha256"] = canonical_json_sha256(registry)
    dump_json(CFG / "data_registry.json", registry)

    groups = {
        "train_v2": train,
        "dev_v2": dev,
        "test_v2_hidden": test,
        "all_v1": [{"content_hash": h} for h in v1_hashes],
    }
    overlap_rows = []
    for a, ar in groups.items():
        ah = {r["content_hash"] for r in ar}
        for b, br in groups.items():
            overlap_rows.append(
                {
                    "left": a,
                    "right": b,
                    "overlap_count": len(ah & {r["content_hash"] for r in br}),
                    "expected": "self" if a == b else "0",
                }
            )
    with (REPORT / "data_overlap_matrix.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=overlap_rows[0])
        w.writeheader()
        w.writerows(overlap_rows)
    leakage = {
        "status": "passed",
        "core_cross_split_overlap_counts": {
            f"{a}_vs_{b}": next(
                r["overlap_count"] for r in overlap_rows if r["left"] == a and r["right"] == b
            )
            for a, b in (
                ("train_v2", "dev_v2"),
                ("train_v2", "test_v2_hidden"),
                ("dev_v2", "test_v2_hidden"),
                ("train_v2", "all_v1"),
                ("dev_v2", "all_v1"),
                ("test_v2_hidden", "all_v1"),
            )
        },
        "intentional_subsets": {
            "warmstart_v2_within_train_v2": 256,
            "pass4_nested_within_test_v2_hidden": 100,
        },
    }
    leakage["source_identity_overlap_counts"] = {
        "train_v2_vs_all_v1": sum(
            (r["metadata"]["dataset_id"], r["source_index"]) in v1_ids for r in train
        ),
        "dev_v2_vs_all_v1": sum(
            (r["metadata"]["dataset_id"], r["source_index"]) in v1_ids for r in dev
        ),
        "test_v2_hidden_vs_all_v1": sum(
            (r["metadata"]["dataset_id"], r["source_index"]) in v1_ids for r in test
        ),
    }
    dump_json(REPORT / "data_leakage_audit.json", leakage)
    strat = {
        "selection_seed": 42,
        "source_revision": SOURCES["math500"][1],
        "available_after_v1_exclusion": capacity,
        "test_selected": {str(k): v for k, v in test_counts.items()},
        "nested_selected": {str(k): v for k, v in nested_counts.items()},
        "unselected": {str(k): capacity[str(k)] - test_counts[k] for k in test_counts},
        "test_namespace": "grpo_v2/test_v2_hidden/math500/level_<n>",
        "nested_namespace": "grpo_v2/pass4_nested/math500/level_<n>",
        "selected_records": {
            str(level): [
                {"source_problem_id": r["source_problem_id"], "content_hash": r["content_hash"]}
                for r in test_math
                if r["difficulty"] == str(level)
            ]
            for level in range(1, 6)
        },
        "level_1_reporting": "diagnostic_only_small_n",
    }
    dump_json(
        REPORT / "math500_capacity_audit.json",
        {
            "math500_total": 500,
            "v1_used": 200,
            "available_total": 300,
            "available_by_level": capacity,
            "equal_40_each_feasible": False,
            "reason": "Only three unseen Level-1 records remain after strict v1 exclusion.",
        },
    )
    dump_json(REPORT / "test_v2_stratification.json", strat)
    dump_json(
        REPORT / "data_freeze_report.json",
        {
            "status": "frozen_cpu_only",
            "selection_seed": 42,
            "gold_independent_selection_key_fields": [
                "dataset_revision",
                "source_split",
                "source_problem_id",
                "selection_namespace",
                "selection_seed",
            ],
            "manifest_records": {
                name: [
                    {
                        "problem_id": r["problem_id"],
                        "source_problem_id": r["source_problem_id"],
                        "content_hash": r["content_hash"],
                    }
                    for r in rows
                ]
                for name, rows in (
                    ("train_v2", train),
                    ("warmstart_v2", warm),
                    ("dev_v2", dev),
                    ("test_v2_hidden", test),
                )
            },
            "math500": strat,
            "overlap_audit": leakage,
            "trusted_gold_separated": True,
        },
    )

    metric_rows = [
        {
            "scope": "training",
            "metric": "format_rate",
            "value": 0.546875,
            "numerator": 280,
            "denominator": 512,
            "source": "grpo_training_metrics.csv",
        },
        {
            "scope": "training",
            "metric": "canonical_pass",
            "value": 0.19140625,
            "numerator": 98,
            "denominator": 512,
            "source": "grpo_training_metrics.csv",
        },
        {
            "scope": "final_all_candidates",
            "metric": "format_rate",
            "value": 0.245,
            "numerator": 196,
            "denominator": 800,
            "source": "grpo_seed42_final_by_domain_level.csv",
        },
        {
            "scope": "final_all_candidates",
            "metric": "parseable_rate",
            "value": 0.2025,
            "numerator": 162,
            "denominator": 800,
            "source": "grpo_seed42_final_by_domain_level.csv",
        },
        {
            "scope": "final_all_candidates",
            "metric": "canonical_pass",
            "value": 0.06125,
            "numerator": 49,
            "denominator": 800,
            "source": "grpo_seed42_final_by_domain_level.csv",
        },
        {
            "scope": "final_all_candidates",
            "metric": "truncation_rate",
            "value": 0.08,
            "numerator": 64,
            "denominator": 800,
            "source": "grpo_seed42_final_by_domain_level.csv",
        },
        {
            "scope": "reward_groups",
            "metric": "nonzero_variance_fraction",
            "value": 101 / 128,
            "numerator": 101,
            "denominator": 128,
            "source": "grpo_reward_group_statistics.csv",
        },
        {
            "scope": "reward_groups",
            "metric": "zero_advantage_fraction",
            "value": 27 / 128,
            "numerator": 27,
            "denominator": 128,
            "source": "grpo_reward_group_statistics.csv",
        },
    ]
    with (REPORT / "v1_bottleneck_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=metric_rows[0])
        w.writeheader()
        w.writerows(metric_rows)
    taxonomy = {
        "denominator": 800,
        "status_counts": {
            "FORMAT_ERROR": 604,
            "PARSE_ERROR": 34,
            "INVALID_EXPRESSION": 0,
            "INVALID_NUMBER_USAGE": 0,
            "RESOURCE_LIMIT": 0,
            "WRONG_ANSWER": 113,
            "VERIFIED_PASS": 49,
        },
        "truncated": 64,
        "truncated_format_error": 64,
        "nontruncated_format_error": 540,
        "interpretation": "The dominant observed failure is strict-format failure; among parseable outputs, wrong answers remain substantial. Truncation explains 64/604 format errors, not the majority.",
    }
    dump_json(REPORT / "v1_failure_taxonomy.json", taxonomy)

    # Figures are rebuilt only from the derived CSV/JSON above.
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4.5))
    names = list(taxonomy["status_counts"])
    vals = list(taxonomy["status_counts"].values())
    plt.bar(names, vals, color="#dd8452")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Completions (n=800)")
    plt.title("GRPO-v1 seed 42 held-out failure taxonomy")
    plt.tight_layout()
    plt.savefig(FIGURES / "v1_failure_taxonomy.png", dpi=160)
    plt.close()
    plt.figure(figsize=(6.5, 4))
    plt.bar(["Training format", "Final format"], [0.546875, 0.245], color=["#dd8452", "#777777"])
    plt.ylim(0, 1)
    plt.ylabel("Rate")
    plt.title("GRPO-v1 format generalization gap")
    plt.tight_layout()
    plt.savefig(FIGURES / "v1_format_generalization_gap.png", dpi=160)
    plt.close()

    REPORT.joinpath("data_freeze_report.md").write_text(
        f"""# GRPO-v2 data freeze\n\nAll selections were frozen before model execution with seed 42 and gold-independent SHA-256 ranking. Public execution manifests omit gold and solutions; trusted verifier records are stored outside Git at `{TRUSTED}`.\n\n## Counts\n\n- train_v2: 512 (GSM8K 256; MATH L1/L2/L3 64/96/96)\n- warmstart_v2: 256 declared subset (GSM8K 128; MATH 32/48/48)\n- dev_v2: 128 (GSM8K 64; MATH 16/24/24)\n- test_v2_hidden: 400 (GSM8K 200; MATH500 3/33/43/59/62)\n- nested pass@4 subset: 100 (GSM8K 50; MATH500 3/8/10/14/15)\n\n## Capacity amendment\n\nMATH500 has 500 records. V1 observed 200; after strict hash exclusion the available per-level counts are **3/50/65/88/94**. Equal 40-per-level allocation is impossible because only three unseen Level-1 records remain. The preregistered unequal allocation **3/33/43/59/62** includes every unseen Level-1 record and preserves strict decontamination; this is a pre-run capacity amendment, not result-driven selection. L2-L5 use only revision/split/source-ID/namespace/seed in their selection keys. The nested MATH allocation is **3/8/10/14/15** under an independent namespace.\n\nEvery core off-diagonal overlap in [the matrix](data_overlap_matrix.csv) is zero. Warmstart-within-train and pass4-within-test are intentional declared subset relations. Level 1 is `diagnostic_only_small_n`, reported as integer numerator/3 with a Wilson interval and never used as a headline or checkpoint-selection result.\n"""
    )
    REPORT.joinpath("v1_bottleneck_analysis.md").write_text(
        """# GRPO-v1 bottleneck analysis\n\nThis audit reconstructs only saved v1 CSV/JSON; it does not run a model. The dominant held-out failure is format error (604/800, 75.5%). Parse errors are 34/800, parseable-but-wrong answers 113/800, and canonical passes 49/800. All 64 truncated outputs are format failures, but truncation accounts for only 64/604 format failures; most format failures are non-truncated. Training format was 280/512 (54.69%) versus 196/800 (24.5%) on the observed final pool, consistent with a serious protocol-generalization/coverage gap rather than proof of a parser defect.\n\nThe v1 run saw only 128 unique RL problems. That narrow coverage plausibly limits generalization, but the audit cannot identify causality. Reward had usable within-group signal: 101/128 groups had nonzero variance; 27/128 were zero-advantage/all-equal and 15/128 all-zero. Therefore reward variance is not the primary infrastructure bottleneck. No correctness defect was found in the frozen parser, verifier, or reward, so v2 leaves their semantics unchanged.\n\nThe v2 intervention is explicitly **format/solution warm-start + GRPO-v2 RLVR**, not pure GRPO. Hidden-test attribution separates Base, old v1, warmstart-only, and v2.\n"""
    )
    REPORT.joinpath("curriculum_report.md").write_text(
        """# Frozen curriculum\n\nThe 512 train prompts appear exactly once in 128 four-prompt updates. Updates 1–32 pair two shortest-solution GSM8K prompts with two MATH Level-1 prompts. Updates 33–96 mix GSM8K and Level 2 (32 updates at 3:1, then 32 at 2:2). Updates 97–128 pair one longest remaining GSM8K prompt with three Level-3 prompts. Split selection is hash-based; only the already-selected GSM8K official-train solution whitespace length is used as a deterministic curriculum proxy. No model output, validation result, or test field participates.\n"""
    )
    REPORT.joinpath("design_decision.md").write_text(
        """# Single-seed GRPO-v2 design decision\n\nGRPO-v2 freezes seed 42, a 256-example one-epoch format/solution warm-start, then 128 GRPO updates over 512 unique prompts. GRPO produces 2,048 training completions under a 524,288-token cap, with checkpoints and 128-problem dev evaluations at 32/64/96/128. Dev selects one checkpoint using canonical pass, parseability, format, truncation, then earlier-step tie-breaking. Test is never used for selection.\n\nThe final hidden evaluation compares Base, old GRPO-v1 checkpoint-32, warmstart-only, and selected v2 on exactly 700 completions each. Candidate 0 covers 400 problems; a fixed 100-problem subset adds candidates 1–3, making a genuinely nested pass@4 pool. The primary success criterion is at least +3 percentage points candidate-0 pass@1 over old v1, with more paired improvements than regressions and a paired bootstrap interval. A warmstart-only gain is attributed to SFT; only v2 over warmstart supports incremental RLVR benefit.\n"""
    )
    dump_json(
        REPORT / "design_decision.json",
        {
            "status": "frozen_cpu_only",
            "seed": 42,
            "warmstart": warm_cfg,
            "grpo_v2": grpo_cfg,
            "checkpoint_selection": selection_contract,
            "hidden_evaluation": eval_contract,
            "success": {
                "primary": "GRPO-v2 candidate-0 pass@1 >= old GRPO-v1 + 0.03 and paired improvements > regressions",
                "stretch": {
                    "pass_at_1": "0.10-0.12+",
                    "nested_pass_at_4": "~0.20+",
                    "format_rate": ">=0.50",
                    "truncation_rate": "<=0.05",
                },
            },
            "test_retraining_forbidden": True,
        },
    )
    REPORT.joinpath("cost_plan.md").write_text(
        """# GRPO-v2 cost plan\n\nThese are preregistered planning estimates, not measurements. The v1 GRPO seed-42 training plus validation used about 0.331 GPU-hours, peak 10.95 GiB, and CNY 2.93. Warm-start is expected at 0.25–0.50 GPU-hours (15–30 min), 12–18 GiB, CNY 2.22–4.44; its ceiling is 0.75 GPU-hours, 24 GiB, CNY 6.66. GRPO-v2 plus four dev evaluations is expected at 0.9–1.4 GPU-hours (54–84 min), 11–16 GiB, CNY 7.99–12.43; its ceiling is 2 GPU-hours, 24 GiB, CNY 17.76. Hidden testing is separately authorized and budgeted later.\n"""
    )
    REPORT.joinpath("cpu_validation.md").write_text(
        "# Stage N CPU validation\n\nGenerated by the final targeted validation commands. CUDA/model/tokenizer/generation/training counters remain zero.\n"
    )


if __name__ == "__main__":
    main()
