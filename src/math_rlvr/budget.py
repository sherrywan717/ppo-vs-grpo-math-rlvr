"""Persistent rollout accounting aligned by completions and generated tokens."""

import json
import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RolloutBudget:
    max_completions: int
    max_generated_tokens: int
    max_wall_time_seconds: float


@dataclass
class RolloutState:
    prompt_count: int = 0
    completion_count: int = 0
    generated_tokens: int = 0
    wall_time_seconds: float = 0.0
    checkpoint_stage: str = "start"

    @property
    def average_completion_length(self):
        return self.generated_tokens / self.completion_count if self.completion_count else 0.0

    def record(self, prompts, completions, tokens, elapsed, stage="rollout"):
        if any(not math.isfinite(x) for x in (tokens, elapsed)):
            raise RuntimeError("NaN/Inf accounting")
        self.prompt_count += prompts
        self.completion_count += completions
        self.generated_tokens += tokens
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
