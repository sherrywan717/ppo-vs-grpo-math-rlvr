#!/usr/bin/env python3
"""Write the six reviewed MATH500 Level-1 delimiter normalization records."""

import importlib.metadata
import json
from pathlib import Path

from datasets import load_dataset

from math_rlvr.dataset import content_hash
from math_rlvr.gold import assert_delimiter_only, normalize_gold_answer

INDICES = (14, 97, 102, 144, 312, 362)
OUTPUT = Path("reports/preprocessing/math500_gold_normalization_audit.json")


def main():
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    records = []
    versions = {
        name: importlib.metadata.version(name)
        for name in ("math-verify", "latex2sympy2-extended", "antlr4-python3-runtime")
    }
    for index in INDICES:
        row = dataset[index]
        raw = row["answer"]
        normalized = normalize_gold_answer(raw)
        assert_delimiter_only(raw, normalized)
        records.append(
            {
                "problem_id": f"math:HuggingFaceH4/MATH-500:test:{index}",
                "source_index": index,
                "level": row["level"],
                "category": row["subject"],
                "raw_gold": raw,
                "normalized_gold": normalized,
                "character_diff": [
                    {"operation": "insert", "offset": 0, "text": "$"},
                    {"operation": "insert", "offset": len(raw), "text": "$"},
                ],
                "normalization_reason": (
                    "missing LaTeX math-mode delimiters; expression characters unchanged"
                ),
                "parser_versions": versions,
                "content_hash": content_hash(row["problem"]),
            }
        )
    payload = {
        "policy": {
            "trusted_gold_only": True,
            "solution_fallback": False,
            "prediction_relaxation": False,
            "allowed_change": "insert one leading and one trailing $ only",
        },
        "records": records,
    }
    temp = OUTPUT.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temp.replace(OUTPUT)
    print(OUTPUT, len(records))


if __name__ == "__main__":
    main()
