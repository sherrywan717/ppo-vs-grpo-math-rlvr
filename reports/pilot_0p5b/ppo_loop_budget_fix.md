# PPO pilot loop budget CPU repair

`Matched 0.5B pilot - not the final benchmark`

## Root cause

The frozen PPO seed-42 config did not fall back to TRL's default four PPO epochs. The
resolved JSON, builder kwargs and constructed `PPOConfig` all contain
`num_ppo_epochs=1` and `num_mini_batches=1`. On one GPU, TRL derives a local rollout
batch of 16, one local minibatch of 16 and four microbatches of four.

The failed run's `2/2` counters were guard duplication, not a second real epoch. TRL
calls the Accelerator optimizer wrapper once per microbatch. The old hook counted each
call as a new epoch, minibatch and optimizer step, so it accepted microbatch 0 and
rejected microbatch 1 while `sync_gradients=false`.

## Four-layer contract

| Value | Resolved JSON | Builder kwargs | PPOConfig | Trainer-derived |
|---|---:|---:|---:|---:|
| total episodes | 16 | 16 | 16 | 16 |
| per-device batch | 4 | 4 | 4 | 4 |
| gradient accumulation | 4 | 4 | 4 | 4 |
| local batch | - | - | - | 16 |
| local minibatch | - | - | - | 16 |
| rollout forward batch | 4 | 4 | 4 | 4 |
| PPO epochs | 1 | 1 | 1 | 1 |
| minibatches | 1 | 1 | 1 | 1 |
| microbatches/minibatch | - | - | - | 4 |
| outer updates | - | - | - | 1 |
| synced optimizer steps | - | - | - | 1 |
| global steps | - | - | - | 1 |

## Minimal repair

`PPOBudgetGuard` now records an explicit zero-based
`(outer_update, epoch_index, minibatch_index)` key. Re-observing the same key is
idempotent. The sole TRL compatibility shim counts microbatch calls but records the
logical loop key and optimizer step only when `accelerator.sync_gradients` is true.
It verifies exactly four microbatches at that boundary and validates TRL's actual
post-construction derived fields against the protected run contract.

The accepted trace is outer 0 → epoch 0 → minibatch 0 → microbatches 0,1,2 with
`sync=false` → microbatch 3 with `sync=true` → optimizer step 1 → global/update 1.
A real epoch index 1, minibatch index 1, outer index 1 or optimizer step 2 remains a
hard failure before a second synchronized optimizer update.

No frozen config, manifest, prompt, reward, parser, verifier, budget or TRL
site-package changed. GRPO code is unchanged. The three historical failed PPO runs,
including the 16-completion evidence from `ppo_matched_0p5b_seed42_20260714T085240Z`,
remain immutable and excluded from scientific aggregation.

## CPU verification

- Targeted loop/guard/fake-finalization tests passed.
- Full pytest passed: 350 tests.
- Ruff, compileall, `check_env`, manifest validation, three PPO pilot dry-runs and the
  Stage D PPO dry-run passed.
- Fake PPO finalization constructs the real CPU `PPOConfig` and exercises the four
  microbatch events before one protected optimizer count.
- CUDA remained uninitialized; real model loads, generation, Trainer training,
  backward and real optimizer calls were zero.
