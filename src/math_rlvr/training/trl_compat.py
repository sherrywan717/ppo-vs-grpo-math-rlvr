"""Narrow TRL 0.24.0 compatibility hooks for exact rollout accounting."""

from typing import Any

import trl

TRL_VERSION = "0.24.0"


class TRLContractError(RuntimeError):
    pass


def assert_trl_version() -> None:
    if trl.__version__ != TRL_VERSION:
        raise TRLContractError(f"requires trl=={TRL_VERSION}, found {trl.__version__}")


def exact_completion_counts(payload: dict[str, Any]) -> tuple[int, int]:
    """Validate TRL's private rollout fields and count non-padding tokens exactly."""
    assert_trl_version()
    if not {"completion_ids", "completion_mask"} <= payload.keys():
        raise TRLContractError("missing completion_ids/completion_mask")
    ids, mask = payload["completion_ids"], payload["completion_mask"]
    if getattr(ids, "shape", None) != getattr(mask, "shape", None) or len(ids.shape) != 2:
        raise TRLContractError("invalid completion tensor shapes")
    if not bool(((mask == 0) | (mask == 1)).all().item()):
        raise TRLContractError("completion mask must be binary")
    return int(ids.shape[0]), int(mask.sum().item())


def guarded_trainer_class(guard):
    """Create the sole private-API subclass; importing this does not construct a model."""
    assert_trl_version()
    from trl import GRPOTrainer

    class GuardedGRPOTrainer(GRPOTrainer):
        def _generate_and_score_completions(self, inputs):
            payload = super()._generate_and_score_completions(inputs)
            completions, tokens = exact_completion_counts(payload)
            guard.record_generation(completions, tokens)
            return payload

        def training_step(self, model, inputs, num_items_in_batch=None):
            guard.record_microstep()
            return super().training_step(model, inputs, num_items_in_batch)

    return GuardedGRPOTrainer


def optimizer_guard_callback(guard):
    assert_trl_version()
    from transformers import TrainerCallback

    class GuardCallback(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            guard.record_optimizer_step()

        def on_step_end(self, args, state, control, **kwargs):
            guard.record_global_step(int(state.global_step))

    return GuardCallback()
