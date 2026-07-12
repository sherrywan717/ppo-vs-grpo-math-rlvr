"""Algorithm-neutral aggregation for verifier outputs."""

from dataclasses import dataclass

from code_rlvr.verifier.verifier import VerificationResult


@dataclass(frozen=True)
class EvaluationSummary:
    attempted: int
    syntax_rate: float
    pass_at_1: float


def summarize(results: list[VerificationResult]) -> EvaluationSummary:
    if not results:
        return EvaluationSummary(0, 0.0, 0.0)
    count = len(results)
    return EvaluationSummary(
        attempted=count,
        syntax_rate=sum(item.syntax_valid for item in results) / count,
        pass_at_1=sum(item.tests_passed for item in results) / count,
    )
