"""Safe text-only math verifiers. No expression is ever executed."""

import ast
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from fractions import Fraction

from math_verify import parse, verify

from math_rlvr.dataset import MathProblem
from math_rlvr.parser import parse_completion
from math_rlvr.rewards.result import RewardResult, RewardStatus


class CountdownVerifier:
    def __init__(self, numbers, target, max_nodes=64, max_answer_length=512):
        self.numbers = numbers
        self.target = Fraction(target)
        self.max_nodes = max_nodes
        self.max_answer_length = max_answer_length

    def __call__(self, completion):
        parsed = parse_completion(completion, self.max_answer_length)
        if isinstance(parsed, RewardResult):
            return parsed
        try:
            tree = ast.parse(parsed.answer, mode="eval")
        except (SyntaxError, ValueError):
            return RewardResult(RewardStatus.INVALID_EXPRESSION)
        nodes = list(ast.walk(tree))
        if len(nodes) > self.max_nodes:
            return RewardResult(RewardStatus.RESOURCE_LIMIT)
        allowed = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Constant,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.UAdd,
            ast.USub,
            ast.Load,
        )
        if any(
            not isinstance(n, allowed)
            or isinstance(n, ast.Constant)
            and (type(n.value) is not int or n.value < 0)
            for n in nodes
        ):
            return RewardResult(RewardStatus.INVALID_EXPRESSION)
        used = [n.value for n in nodes if isinstance(n, ast.Constant)]
        if Counter(used) != Counter(self.numbers):
            return RewardResult(RewardStatus.INVALID_NUMBER_USAGE)

        def visit(n):
            if isinstance(n, ast.Expression):
                return visit(n.body)
            if isinstance(n, ast.Constant):
                return Fraction(n.value)
            if isinstance(n, ast.UnaryOp):
                return visit(n.operand) if isinstance(n.op, ast.UAdd) else -visit(n.operand)
            a, b = visit(n.left), visit(n.right)
            if isinstance(n.op, ast.Add):
                return a + b
            if isinstance(n.op, ast.Sub):
                return a - b
            if isinstance(n.op, ast.Mult):
                return a * b
            return a / b

        try:
            value = visit(tree)
        except ZeroDivisionError:
            return RewardResult(RewardStatus.INVALID_EXPRESSION, "division by zero")
        return RewardResult(
            RewardStatus.VERIFIED_PASS if value == self.target else RewardStatus.WRONG_ANSWER
        )


_NUM = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\d*\.\d+)(?:/[+-]?\d+)?%?$")


def parse_number(text):
    text = text.strip().replace(",", "")
    if not _NUM.fullmatch(text):
        raise ValueError("not one numeric answer")
    percent = text.endswith("%")
    if percent:
        text = text[:-1]
    try:
        value = Fraction(text) if "/" in text else Fraction(Decimal(text))
    except (InvalidOperation, ValueError, ZeroDivisionError) as e:
        raise ValueError from e
    return value / 100 if percent else value


class GSM8KVerifier:
    def __init__(self, gold):
        self.gold = parse_number(gold)

    def __call__(self, completion):
        parsed = parse_completion(completion)
        if isinstance(parsed, RewardResult):
            return parsed
        try:
            value = parse_number(parsed.answer)
        except ValueError:
            return RewardResult(RewardStatus.PARSE_ERROR)
        return RewardResult(
            RewardStatus.VERIFIED_PASS if value == self.gold else RewardStatus.WRONG_ANSWER
        )


class MathExpressionVerifier:
    def __init__(self, gold):
        self.gold = parse(gold)
        if not self.gold:
            raise ValueError("unparseable gold")

    def __call__(self, completion):
        parsed = parse_completion(completion)
        if isinstance(parsed, RewardResult):
            return parsed
        try:
            prediction = parse(parsed.answer)
        except Exception as e:
            raise RuntimeError("math-verify infrastructure failure") from e
        if not prediction:
            return RewardResult(RewardStatus.PARSE_ERROR)
        try:
            correct = verify(self.gold, prediction)
        except Exception as e:
            raise RuntimeError("math-verify infrastructure failure") from e
        return RewardResult(RewardStatus.VERIFIED_PASS if correct else RewardStatus.WRONG_ANSWER)


class MathVerifier:
    def __call__(self, problem: MathProblem, completion: str):
        if problem.source == "gsm8k":
            verifier = GSM8KVerifier(problem.gold_answer)
        elif problem.source == "math":
            verifier = MathExpressionVerifier(problem.gold_answer)
        else:
            verifier = CountdownVerifier(problem.metadata["numbers"], problem.metadata["target"])
        try:
            return verifier(completion)
        except RuntimeError as e:
            return RewardResult(RewardStatus.INFRA_ERROR, str(e))
