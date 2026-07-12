import pytest

from code_rlvr.evaluation.evaluator import summarize
from code_rlvr.rewards.result import RewardInfrastructureError, RewardResult, RewardStatus


def test_summary_uses_common_verifier_results() -> None:
    summary = summarize([
        RewardResult(RewardStatus.VERIFIED_PASS),
        RewardResult(RewardStatus.VERIFIED_FAIL),
    ])
    assert (summary.attempted, summary.syntax_rate, summary.pass_at_1) == (2, 1.0, 0.5)


def test_evaluation_aborts_on_infrastructure_error() -> None:
    with pytest.raises(RewardInfrastructureError, match="sandbox_unavailable"):
        summarize([RewardResult(RewardStatus.SANDBOX_UNAVAILABLE, "missing")])
