from code_rlvr.rewards.code_reward import score
from code_rlvr.verifier.verifier import VerificationResult


def test_full_reward_requires_syntax_and_tests() -> None:
    assert score(VerificationResult(True, True, "")) == 1.0
    assert score(VerificationResult(True, False, "")) == 0.1

