# Matched PPO pilot seed 42 success

`Matched 0.5B pilot - not the final benchmark`

- Run ID: `ppo_matched_0p5b_seed42_20260716T114710Z`
- Status: `execution_success/learning_signal_present`
- Completions / tokens / evidence: `16 / 574 / 16`
- Backward events: `4`, sizes `[4,4,4,4]`, samples `16`
- Sync trace: `[false,false,false,true]`
- Epoch / minibatch / bottom optimizer / update / global: `1/1/1/1/1`
- Reward mean/std: `0.078125 / 0.03940475066537028`
- Group variances: `0.00171875, 0.00171875, 0.00125, 0.00046875`; zero-advantage groups: `0`
- Canonical statuses: `14 FORMAT_ERROR`, `2 INVALID_EXPRESSION`; pass@1/pass@4: `0/0`
- Policy/value loss: `0.0406283400952816 / 6.805111885070801`
- Objective/approximate KL: `0.5889581441879272 / 0.00042952707735821605`
- Entropy: `0.2690500020980835`; ratio variance finite: `2.229446408819058e-06`

The 16 prompt-major comparison keys, frozen identities, policy/value/ref/reward roles
and exact optimizer union passed. Policy/value trainables were `2,162,688 / 541,568`;
ref/reward trainables were zero. The sole 10,880,009-byte checkpoint contains policy
and value adapters plus scalar head and metadata only; no full base weights. This is a
valid single-update pilot result, not evidence of task learning or PPO superiority.

- nvidia-smi / PyTorch allocated/reserved peaks: `11189 / 9976.72 / 10516 MiB`
- Wall / GPU-hours / cost: `17.9055 s / 0.00497375 / CNY 0.04417`
- Worker pre-exit allocator residue: `64/108 MiB`; parent post-exit: `0 MiB`, no process
- Backup: `/root/autodl-fs/math-rlvr-backups/ppo_matched_0p5b_seed42_20260716T114710Z.tar.gz`
- Backup SHA256: `dd1833ea6fa75a6a8af1d7fba366b05e498b0524ee726a526eb7fa294f89b7f6`

The command ran once with no retry. All four historical PPO seed-42 failures remain
immutable and excluded. The suite may proceed to frozen Run 2, GRPO seed 42.
