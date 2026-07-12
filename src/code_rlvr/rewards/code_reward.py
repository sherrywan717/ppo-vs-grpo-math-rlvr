"""Deterministic reward derived from the common verifier result."""

from dataclasses import dataclass

from code_rlvr.verifier.verifier import VerificationResult


@dataclass(frozen=True)
class RewardWeights:
    syntax: float = 0.1
    tests: float = 0.9


DEFAULT_WEIGHTS = RewardWeights()


def score(result: VerificationResult, weights: RewardWeights = DEFAULT_WEIGHTS) -> float:
    return float(result.syntax_valid) * weights.syntax + float(result.tests_passed) * weights.tests
