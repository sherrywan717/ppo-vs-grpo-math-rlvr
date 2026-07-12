"""Shared dataset schema and bounded selection helpers."""

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class CodeProblem:
    problem_id: str
    prompt: str
    tests: tuple[str, ...]


def take_bounded(problems: Iterable[CodeProblem], limit: int) -> list[CodeProblem]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    result: list[CodeProblem] = []
    for problem in problems:
        if len(result) == limit:
            break
        result.append(problem)
    return result

