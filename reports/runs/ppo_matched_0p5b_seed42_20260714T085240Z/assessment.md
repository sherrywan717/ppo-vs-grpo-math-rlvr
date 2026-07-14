# Matched PPO pilot seed 42 budget-guard failure

`Matched 0.5B pilot - not the final benchmark`

- Run ID: `ppo_matched_0p5b_seed42_20260714T085240Z`
- Suite sequence: 1 of 6
- Status: `failure_during_update/budget_guard`
- Reason: `BudgetExceededError: PPO epoch/minibatch cap exceeded`
- Completions / generated tokens / rewards: `16 / 574 / 16`
- Update / optimizer / global steps: `0 / 1 / 0`
- Observed PPO epoch / minibatch counters: `2 / 2` with frozen caps `1 / 1`
- Completion evidence rows persisted: `0`
- Checkpoint: none

The Stage A collator repair worked. The real Accelerator loader was a sequential
`DataLoaderShard` with batch size 16, `drop_last=true`, `num_workers=0`, and the exact
16 prompt-major comparison keys. Frozen identities and four prompt-scope layers
matched. Policy/value/reward/reference roles and the optimizer parameter union passed.

The run generated and rewarded one 16-response rollout, then recorded one optimizer
step. Before a completed update/global-step log, a second epoch/minibatch accounting
attempt exceeded the frozen 1/1 cap. The runner therefore failed closed. Because the
backend did not reach completion-evidence finalization, completion texts, reward rows,
losses and a checkpoint are unavailable and are not reconstructed or fabricated.
This partial-update failure is excluded from scientific aggregation.

- Guard elapsed: `16.786310022696853 s`
- Resource window: `17.00034363567829 s`
- Peak nvidia-smi VRAM: `10903 MiB`
- Peak PyTorch allocated/reserved: `9984.15625 / 10528 MiB`
- GPU-hours / cost: `0.004722317676577303 / CNY 0.041934180968006454`
- Post-process GPU: `0 MiB`, no compute process
- Backup: `/root/autodl-fs/math-rlvr-backups/ppo_matched_0p5b_seed42_20260714T085240Z.failure.tar.gz`
- Backup SHA256: `dbc4377f5ba30294aab31f7d7cdb5efe0f289b58b3a2160004dd51086b4d0de4`

The command was not retried. Runs 2--6 were not executed, and no further code repair
was attempted in this authorized task. The two older historical PPO seed-42 failures
remain immutable and separate.
