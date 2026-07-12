from code_rlvr.evaluation.evaluator import summarize
from code_rlvr.verifier.verifier import VerificationResult


def test_summary_uses_common_verifier_results() -> None:
    summary = summarize([VerificationResult(True, True, ""), VerificationResult(True, False, "")])
    assert (summary.attempted, summary.syntax_rate, summary.pass_at_1) == (2, 1.0, 0.5)

