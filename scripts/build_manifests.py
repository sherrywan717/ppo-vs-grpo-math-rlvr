#!/usr/bin/env python3
"""Build deterministic frozen manifests; never loads a model or tokenizer."""

import random
from pathlib import Path

from datasets import load_dataset

from math_rlvr.dataset import MathProblem, content_hash, save_manifest, validate_manifests
from math_rlvr.gold import normalize_gold_answer

SEED = 20260712
ROOT = Path("/root/autodl-tmp/datasets/math_rlvr/manifests")
REVISIONS = {
    "openai/gsm8k": "740312add88f781978c0658806c59bc2815b9866",
    "DigitalLearningGmbH/MATH-lighteval": "0530c78699ea5e8eb5530600900e1f328b48acad",
    "HuggingFaceH4/MATH-500": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
}


def revision(name):
    return REVISIONS[name]


def gsm_problem(row, i, split, rev):
    gold = row["answer"].rsplit("####", 1)[-1].strip().replace(",", "")
    return MathProblem(
        f"gsm8k:{split}:{i}",
        "gsm8k",
        row["question"],
        gold,
        "word_problem",
        "gsm8k",
        split,
        i,
        content_hash(row["question"]),
        {"dataset_id": "openai/gsm8k", "revision": rev, "source_split": split, "source_index": i},
    )


def math_problem(row, i, split, rev, dataset_id, answer=None):
    gold = normalize_gold_answer(answer if answer is not None else row["solution"])
    return MathProblem(
        f"math:{dataset_id}:{split}:{i}",
        "math",
        row["problem"],
        gold,
        row.get("type", row.get("subject", "unknown")),
        str(row["level"]),
        split,
        i,
        content_hash(row["problem"]),
        {"dataset_id": dataset_id, "revision": rev, "source_split": split, "source_index": i},
    )


def stable(rows, n, rng, exclude=None, level_counts=None):
    exclude = exclude or set()
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    chosen = []
    cats = {}
    for i in indices:
        row = rows[i]
        h = content_hash(row["problem"])
        level_text = str(row["level"]).split()[-1]
        if not level_text.isdigit():
            continue
        level = int(level_text)
        if h in exclude:
            continue
        try:
            normalize_gold_answer(row.get("answer", row.get("solution", "")))
        except ValueError:
            continue
        if level_counts and sum(
            1 for x in chosen if int(str(x[0]["level"]).split()[-1]) == level
        ) >= level_counts.get(level, 0):
            continue
        cat = row.get("type", row.get("subject", "unknown"))
        count = cats.get(cat, 0)
        chosen.append((row, i))
        cats[cat] = count + 1
        if len(chosen) == n:
            return chosen
    if len(chosen) < n:
        raise RuntimeError(f"insufficient stable samples: {len(chosen)}/{n}")
    return chosen


def main():
    rng = random.Random(SEED)
    gsm = load_dataset("openai/gsm8k", "main")
    math = load_dataset("DigitalLearningGmbH/MATH-lighteval")
    m500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rg = revision("openai/gsm8k")
    rm = revision("DigitalLearningGmbH/MATH-lighteval")
    r5 = revision("HuggingFaceH4/MATH-500")
    m500_hashes = {content_hash(x["problem"]) for x in m500}
    gidx = list(range(len(gsm["train"])))
    rng.shuffle(gidx)
    gtrain = [gsm_problem(gsm["train"][i], i, "train", rg) for i in gidx[:64]]
    gval = [gsm_problem(gsm["train"][i], i, "validation", rg) for i in gidx[64:96]]
    mt = stable(math["train"], 64, rng, m500_hashes, {1: 16, 2: 24, 3: 24})
    used = {content_hash(x[0]["problem"]) for x in mt}
    mv = stable(math["train"], 32, rng, m500_hashes | used)
    train = gtrain + [
        math_problem(x, i, "train", rm, "DigitalLearningGmbH/MATH-lighteval") for x, i in mt
    ]
    val = gval + [
        math_problem(x, i, "validation", rm, "DigitalLearningGmbH/MATH-lighteval") for x, i in mv
    ]
    gtidx = list(range(len(gsm["test"])))
    rng.shuffle(gtidx)
    ge = [gsm_problem(gsm["test"][i], i, "test", rg) for i in gtidx[:200]]
    me_sel = stable(m500, 200, rng, set(), {1: 40, 2: 40, 3: 40, 4: 40, 5: 40})
    me = [math_problem(x, i, "test", r5, "HuggingFaceH4/MATH-500", x["answer"]) for x, i in me_sel]
    validate_manifests({"train": train, "validation": val, "test": ge + me})
    save_manifest(ROOT / "train_core_128.json", train)
    save_manifest(ROOT / "validation_64.json", val)
    save_manifest(ROOT / "gsm8k_test_200.json", ge)
    save_manifest(ROOT / "math500_test_200.json", me)
    gp = ge[:50]
    mp = []
    for level in range(1, 6):
        mp += [x for x in me if int(x.difficulty) == level][:10]
    save_manifest(ROOT / "gsm8k_pass4_50.json", gp)
    save_manifest(ROOT / "math500_pass4_50.json", mp)
    print(
        {
            "revisions": {"gsm8k": rg, "math": rm, "math500": r5},
            "counts": {
                "train": len(train),
                "validation": len(val),
                "gsm8k_test": len(ge),
                "math500": len(me),
                "gsm8k_pass4": len(gp),
                "math500_pass4": len(mp),
            },
        }
    )


if __name__ == "__main__":
    main()
