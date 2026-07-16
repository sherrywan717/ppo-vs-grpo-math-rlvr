# Matched GRPO pilot seed 123 success

`Matched 0.5B pilot - not the final benchmark`

- Run ID: `grpo_matched_0p5b_seed123_20260716T115714Z`
- Status: `execution_success/learning_signal_present`
- Completions / tokens / evidence: `16 / 724 / 16`
- Microsteps / optimizer / update / global: `4/1/1/1`
- Reward mean/population std: `0.084375 / 0.03840064289826409`
- Group variances: `0.001875, 0.00125, 0.00125, 0.00046875`; zero-advantage groups: `0`
- Canonical statuses: `12 FORMAT_ERROR`, `4 INVALID_EXPRESSION`; pass@1/pass@4: `0/0`
- Loss / grad norm / entropy: `0.21626996994018555 / 1.620953917503357 / 0.42529723793268204`
- KL: unavailable/null because frozen beta is `0.0`

All 16 protected comparison keys occur exactly once and match the PPO key set. Frozen
identities, 16-completion/2,048-token/one-update counters, policy-only optimizer role
and the adapter-only checkpoint passed. This is a valid single-update pilot result,
not evidence of task learning or GRPO superiority.

- nvidia-smi / PyTorch allocated/reserved peaks: `3301 / 1973.27 / 2628 MiB`
- Wall / GPU-hours / cost: `11.0505 s / 0.00306958 / CNY 0.02726`
- Worker residue / parent post-exit: `64/330 MiB / 0 MiB and no process`
- Backup: `/root/autodl-fs/math-rlvr-backups/grpo_matched_0p5b_seed123_20260716T115714Z.tar.gz`
- Backup SHA256: `63d4e598169ec655a5d2c52e023606e1fb8b6b6915345d8a948d3273b45bf6f3`

The command ran once with no retry. Proceed next only to frozen Run 4, PPO seed 123.
