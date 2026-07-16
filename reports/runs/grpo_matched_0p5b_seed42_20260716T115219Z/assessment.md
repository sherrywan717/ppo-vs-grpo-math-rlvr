# Matched GRPO pilot seed 42 success

`Matched 0.5B pilot - not the final benchmark`

- Run ID: `grpo_matched_0p5b_seed42_20260716T115219Z`
- Status: `execution_success/learning_signal_present`
- Completions / tokens / evidence: `16 / 545 / 16`
- Microsteps / optimizer / update / global: `4/1/1/1`
- Reward mean/population std: `0.078125 / 0.03940475066537028`
- Group variances: `0.00171875, 0.00171875, 0.00171875, 0.000625`; zero-advantage groups: `0`
- Canonical statuses: `13 FORMAT_ERROR`, `2 INVALID_EXPRESSION`, `1 INVALID_NUMBER_USAGE`; pass@1/pass@4: `0/0`
- Loss / grad norm / entropy: `0.30371785163879395 / 1.9378588199615479 / 0.4157172292470932`
- KL: unavailable because frozen GRPO beta is `0.0`; it is recorded as `null`, not zero

All 16 protected comparison keys occur exactly once, although TRL consumed prompt
groups in its internal order `2, 3, 0, 1`; each completion is explicitly mapped to
its `problem_id::generation_index`, so the comparison-key set matches PPO exactly.
The frozen identities and exact 16-completion/2,048-token/one-update contract passed.
The sole 24,570,925-byte checkpoint contains one policy LoRA adapter plus tokenizer
and Trainer metadata; no full base-model weight file or duplicate adapter is present.
This is a valid single-update pilot result, not evidence of task learning or GRPO
superiority.

- nvidia-smi / PyTorch allocated/reserved peaks: `3301 / 1973.27 / 2628 MiB`
- Resource window / GPU-hours / cost: `11.3425 s / 0.00315071 / CNY 0.02798`
- Worker pre-exit allocator residue: `64/330 MiB`; parent post-exit: `0 MiB`, no process
- Backup: `/root/autodl-fs/math-rlvr-backups/grpo_matched_0p5b_seed42_20260716T115219Z.tar.gz`
- Backup SHA256: `6c31e369554cc4272235981722c96ff65de69614eb435b580673b061003322fb`

The command ran once with no retry. All four historical PPO seed-42 failures remain
immutable and excluded. The suite may proceed to frozen Run 3, GRPO seed 123.
