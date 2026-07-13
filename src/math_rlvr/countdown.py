"""Deterministic solvable Countdown smoke data."""

import random
from fractions import Fraction

from math_rlvr.dataset import MathProblem, content_hash

SEED = 20260712


def generate_countdown(seed: int = SEED) -> dict[str, list[MathProblem]]:
    rng = random.Random(seed)
    output = {split: [] for split in ("train", "validation", "test")}
    seen = set()
    for split, count in (("train", 32), ("validation", 16), ("test", 64)):
        while len(output[split]) < count:
            numbers = [rng.randint(1, 20) for _ in range(4)]
            expression = str(numbers[0])
            value = Fraction(numbers[0])
            operations = []
            for number in numbers[1:]:
                operation = rng.choice("+-*/")
                operations.append(operation)
                expression = f"({expression} {operation} {number})"
                if operation == "+":
                    value += number
                elif operation == "-":
                    value -= number
                elif operation == "*":
                    value *= number
                else:
                    value /= number
            key = (tuple(sorted(numbers)), value)
            if key in seen:
                continue
            seen.add(key)
            index = len(output[split])
            prompt = f"Use each of {numbers} exactly once with +, -, *, / to make {value}."
            metadata = {
                "dataset_id": "generated/countdown",
                "revision": f"seed-{seed}",
                "source_split": split,
                "source_index": index,
                "numbers": numbers,
                "target": str(value),
                "construction": expression,
            }
            output[split].append(
                MathProblem(
                    f"countdown:{split}:{index}",
                    "countdown",
                    prompt,
                    expression,
                    "arithmetic",
                    f"{len(set(operations))}-op-types",
                    split,
                    index,
                    content_hash(prompt),
                    metadata,
                )
            )
    return output
