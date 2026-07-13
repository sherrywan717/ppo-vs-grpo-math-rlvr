"""Persistent rollout accounting aligned by completions and generated tokens."""

import json
import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RolloutBudget:
    max_completions: int
    max_generated_tokens: int
    max_wall_time_seconds: float
    max_prompts: int | None = None
    max_optimizer_steps: int | None = None
    max_global_steps: int | None = None


@dataclass
class RolloutState:
    prompt_count: int = 0
    completion_count: int = 0
    generated_tokens: int = 0
    wall_time_seconds: float = 0.0
    checkpoint_stage: str = "start"
    optimizer_steps: int = 0
    global_step: int = 0

    @property
    def average_completion_length(self):
        return self.generated_tokens / self.completion_count if self.completion_count else 0.0

    def record(
        self, prompts, completions, tokens, elapsed, stage="rollout", *,
        optimizer_steps=0, global_steps=0, budget=None
    ):
        if any(not math.isfinite(x) for x in (tokens, elapsed)):
            raise RuntimeError("NaN/Inf accounting")
        projected = {
            "prompts": self.prompt_count + prompts,
            "completions": self.completion_count + completions,
            "tokens": self.generated_tokens + tokens,
            "optimizer_steps": self.optimizer_steps + optimizer_steps,
            "global_steps": self.global_step + global_steps,
        }
        if budget is not None:
            limits = {
                "prompts": budget.max_prompts,
                "completions": budget.max_completions,
                "tokens": budget.max_generated_tokens,
                "optimizer_steps": budget.max_optimizer_steps,
                "global_steps": budget.max_global_steps,
            }
            exceeded = [
                key
                for key, limit in limits.items()
                if limit is not None and projected[key] > limit
            ]
            if exceeded:
                raise RuntimeError(f"hard budget exceeded: {', '.join(exceeded)}")
        self.prompt_count = projected["prompts"]
        self.completion_count = projected["completions"]
        self.generated_tokens = projected["tokens"]
        self.optimizer_steps = projected["optimizer_steps"]
        self.global_step = projected["global_steps"]
        self.wall_time_seconds += elapsed
        self.checkpoint_stage = stage

    def stop_reason(self, budget):
        if self.completion_count >= budget.max_completions:
            return "max_completions"
        if self.generated_tokens >= budget.max_generated_tokens:
            return "max_generated_tokens"
        if self.wall_time_seconds >= budget.max_wall_time_seconds:
            return "max_wall_time_seconds"
        return None

    def save(self, path):
        path.write_text(json.dumps(asdict(self)))

    @classmethod
    def load(cls, path):
        return cls(**json.loads(path.read_text()))
