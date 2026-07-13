"""Frozen math dataset schema and manifest validation."""

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def normalize_problem(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize_problem(text).encode()).hexdigest()


@dataclass(frozen=True)
class MathProblem:
    problem_id: str
    source: str
    prompt: str
    gold_answer: str
    category: str
    difficulty: str
    split: str
    source_index: int
    content_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.source not in {"countdown", "gsm8k", "math"}:
            raise ValueError("unsupported source")
        if self.content_hash != content_hash(self.prompt):
            raise ValueError("content_hash mismatch")
        required = {"dataset_id", "revision", "source_split", "source_index"}
        if not required <= self.metadata.keys():
            raise ValueError("incomplete provenance metadata")


def save_manifest(path: Path, problems: Iterable[MathProblem]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(p) for p in problems], indent=2, ensure_ascii=False) + "\n")


def load_manifest(path: Path) -> list[MathProblem]:
    return [MathProblem(**x) for x in json.loads(path.read_text())]


def validate_manifests(groups: dict[str, list[MathProblem]]):
    ids = set()
    hashes = {}
    for split, problems in groups.items():
        for p in problems:
            if p.problem_id in ids:
                raise ValueError(f"duplicate id: {p.problem_id}")
            if p.content_hash in hashes and hashes[p.content_hash] != split:
                raise ValueError(f"cross-split leak: {p.problem_id}")
            ids.add(p.problem_id)
            hashes[p.content_hash] = split
