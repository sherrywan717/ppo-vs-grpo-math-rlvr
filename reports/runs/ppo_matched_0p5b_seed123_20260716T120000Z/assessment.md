# Matched PPO pilot seed 123 success

`Matched 0.5B pilot - not the final benchmark`

- Run ID: `ppo_matched_0p5b_seed123_20260716T120000Z`
- Status: `execution_success/learning_signal_present`
- Completions / tokens / evidence: `16 / 565 / 16`
- Backward events / sizes / samples: `4 / [4,4,4,4] / 16`
- Sync trace: `[false,false,false,true]`
- Epoch / minibatch / bottom optimizer / update / global: `1/1/1/1/1`
- Reward mean/population std: `0.09375 / 0.03903123748998998`
- Group variances: `0.00171875, 0, 0.00171875, 0.001875`; zero-advantage groups: `1`
- Canonical statuses: `13 FORMAT_ERROR`, `2 INVALID_EXPRESSION`, `1 INVALID_NUMBER_USAGE`; pass@1/pass@4: `0/0`
- Policy/value loss: `0.012560875155031681 / 3.9441425800323486`
- Objective/approximate KL: `0.6488973498344421 / 0.0007031410932540894`
- Entropy / ratio variance: `0.17659686505794525 / 7.748671123408712e-06`

The prompt-major 16 comparison keys, frozen identities, policy/value/ref/reward roles
and exact optimizer union passed. The safe role-separated checkpoint contains policy
and value adapters plus scalar head only. One group had zero variance, which is a real
scientific outcome and not an execution blocker. This does not prove task learning or
PPO superiority.

- nvidia-smi / PyTorch allocated/reserved peaks: `10015 / 8554.24 / 9342 MiB`
- Wall / GPU-hours / cost: `15.2371 s / 0.00423253 / CNY 0.03758`
- Worker residue / parent post-exit: `64/86 MiB / 0 MiB and no process`
- Backup: `/root/autodl-fs/math-rlvr-backups/ppo_matched_0p5b_seed123_20260716T120000Z.tar.gz`
- Backup SHA256: `2ac5a29a453e8e71bda7aea2d498e798b0f6d29393c20286bd0e585fe7f326f9`

The command ran once with no retry. Proceed next only to frozen Run 5, PPO seed 2026.
