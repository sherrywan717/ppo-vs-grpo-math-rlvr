import inspect
from collections import Counter
from fractions import Fraction
from pathlib import Path

import pytest

import math_rlvr.verifier.verifier as verifier_module
from math_rlvr.budget import RolloutBudget, RolloutState
from math_rlvr.countdown import generate_countdown
from math_rlvr.dataset import load_manifest, validate_manifests
from math_rlvr.gold import normalize_gold_answer
from math_rlvr.parser import parse_completion
from math_rlvr.rewards.result import RewardResult, RewardStatus
from math_rlvr.verifier import CountdownVerifier, GSM8KVerifier, MathExpressionVerifier


def envelope(answer: str) -> str:
    return f"<reasoning>brief</reasoning><answer>{answer}</answer>"


def test_countdown_reproducible_and_disjoint():
    first, second = generate_countdown(), generate_countdown()
    assert first == second
    validate_manifests(first)
    assert {key: len(value) for key, value in first.items()} == {
        "train": 32,
        "validation": 16,
        "test": 64,
    }


def test_frozen_manifest_counts_levels_and_no_leakage():
    root = Path("/root/autodl-tmp/datasets/math_rlvr/manifests")
    train, validation = (
        load_manifest(root / "train_core_128.json"),
        load_manifest(root / "validation_64.json"),
    )
    gsm, math500 = (
        load_manifest(root / "gsm8k_test_200.json"),
        load_manifest(root / "math500_test_200.json"),
    )
    validate_manifests({"train": train, "validation": validation, "test": gsm + math500})
    assert (len(train), len(validation), len(gsm), len(math500)) == (128, 64, 200, 200)
    assert Counter(int(x.difficulty) for x in math500) == {1: 40, 2: 40, 3: 40, 4: 40, 5: 40}
    assert {x.content_hash for x in train}.isdisjoint(x.content_hash for x in math500)


def test_countdown_fraction_parentheses_negative_and_number_usage():
    verifier = CountdownVerifier([2, 3, 4, 6], Fraction("13/6"))
    assert verifier(envelope("6/4+2/3")).status == RewardStatus.VERIFIED_PASS
    negative = CountdownVerifier([1, 2, 3, 4], -4)
    assert negative(envelope("((1-3)*4)/2")).status == RewardStatus.VERIFIED_PASS
    assert verifier(envelope("(2+2+3+4)")).status == RewardStatus.INVALID_NUMBER_USAGE
    assert verifier(envelope("2/(3-3)+4")).status == RewardStatus.INVALID_NUMBER_USAGE


def test_countdown_rejects_unsafe_ast_nodes_and_zero_division():
    verifier = CountdownVerifier([1, 2, 3, 4], 0)
    for answer in ("x+1+2+3", "f(1)+2+3+4", "x.y+1+2+3+4", "x[1]+2+3+4", "1**2+3+4"):
        assert verifier(envelope(answer)).status == RewardStatus.INVALID_EXPRESSION
    zero = CountdownVerifier([1, 2, 3, 3], 0)
    assert zero(envelope("1/(3-3)*2")).status == RewardStatus.INVALID_EXPRESSION


def test_gsm8k_numeric_forms_and_multiple_candidate_rejection():
    for gold, pred in (("-12", "-12"), ("1.25", "1.250"), ("3/4", "0.75"), ("25%", "1/4")):
        assert GSM8KVerifier(gold)(envelope(pred)).status == RewardStatus.VERIFIED_PASS
    assert GSM8KVerifier("2")(envelope("2 or 3")).status == RewardStatus.PARSE_ERROR


def test_math_equivalence_and_inequality():
    verifier = MathExpressionVerifier("$x^2+2x+1$")
    assert verifier(envelope("$(x+1)^2$")).status == RewardStatus.VERIFIED_PASS
    assert verifier(envelope("$x^2+1$")).status == RewardStatus.WRONG_ANSWER


def test_format_contract():
    bad = (
        "<answer>1</answer>",
        "<reasoning>x</reasoning><answer></answer>",
        "<reasoning>x</reasoning><answer>1</answer>tail",
        "<reasoning>x</reasoning><answer>1</answer><answer>1</answer>",
    )
    assert all(isinstance(parse_completion(x), RewardResult) for x in bad)


def test_gold_delimiter_regressions():
    six = (r"\sqrt{51}", r"\text{east}", "(a+5)(b+2)", "(15,-29)", r"(5,\infty)", r"\frac14")
    assert all(normalize_gold_answer(x) == f"${x}$" for x in six)


def test_budget_stops_and_resume(tmp_path):
    budget = RolloutBudget(4, 10, 5)
    state = RolloutState()
    state.record(1, 2, 6, 1)
    path = tmp_path / "state.json"
    state.save(path)
    resumed = RolloutState.load(path)
    assert resumed.completion_count == 2
    resumed.record(1, 2, 2, 1)
    assert resumed.stop_reason(budget) == "max_completions"
    assert RolloutState(generated_tokens=10).stop_reason(budget) == "max_generated_tokens"
    assert RolloutState(wall_time_seconds=5).stop_reason(budget) == "max_wall_time_seconds"


def test_verifier_source_never_calls_eval_exec():
    source = inspect.getsource(verifier_module)
    assert "eval(" not in source and "exec(" not in source


def test_smoke_runtime_counters_fail_before_exceeding_hard_budget():
    budget = RolloutBudget(8, 1024, 900, max_prompts=2, max_optimizer_steps=1, max_global_steps=1)
    state = RolloutState()
    state.record(2, 8, 1024, 1, optimizer_steps=1, global_steps=1, budget=budget)
    snapshot = state.__dict__.copy()
    for kwargs in (
        {"prompts": 1, "completions": 0, "tokens": 0, "elapsed": 0},
        {"prompts": 0, "completions": 1, "tokens": 0, "elapsed": 0},
        {"prompts": 0, "completions": 0, "tokens": 1, "elapsed": 0},
        {"prompts": 0, "completions": 0, "tokens": 0, "elapsed": 0, "optimizer_steps": 1},
        {"prompts": 0, "completions": 0, "tokens": 0, "elapsed": 0, "global_steps": 1},
    ):
        with pytest.raises(RuntimeError, match="hard budget exceeded"):
            state.record(**kwargs, budget=budget)
        assert state.__dict__ == snapshot
