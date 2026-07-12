"""Deterministic reward derived from the common verifier result."""

from dataclasses import dataclass

from code_rlvr.verifier.verifier import VerificationResult


@dataclass(frozen=True)
class RewardWeights:
    syntax: float = 0.1
    tests: float = 0.9


def score(result: VerificationResult, weights: RewardWeights = RewardWeights()) -> float:
    return float(result.syntax_valid) * weights.syntax + float(result.tests_passed) * weights.tests

