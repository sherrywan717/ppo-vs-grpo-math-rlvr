# Formal 1.5B four-run budget amendment

This amendment preserves the six-run Stage E decision frozen at commit
`499fea9f6de6f991229f15b949e23e63c496e6cc` as historical evidence. The approved
active suite is now four matched training runs using seeds 42 and 123. It is a
job-portfolio-scale comparison, not a high-powered benchmark.

The active training order is PPO seed 42, GRPO seed 42, a seed-42 validation and
learning-signal review, GRPO seed 123, and PPO seed 123. Step-32 final evaluation then
runs on those four checkpoints before CPU-only aggregation. The exact active-suite
contract is `configs/formal_1p5b/active_suite.json`, canonical SHA256
`f6de8c555a70837d08c1e34e13a738a21ce247b3a09531df6244a2f1d3ef53bd`.

The existing PPO/GRPO seed-2026 descriptors are deliberately retained byte-for-byte
and marked `reserved_not_scheduled`. They are excluded from the active registry,
execution queue, cost totals, final evaluations, and formal statistics.

Per-run scientific identities and budgets are unchanged: 128 ordered training
problems, prompt v2, shaped-v3 domain reward, fixed parser/verifiers, matched policy
LoRA and sampling, 32 updates, 512 completions, a 131,072-token hard cap, and
checkpoint/validation at steps 8, 16, 24, and 32.

With two independent seeds, the report may show raw seed values, seed mean, sample SD,
paired algorithm deltas, and problem-level bootstrap 95% confidence intervals. It
must not claim statistical significance or general proof that PPO or GRPO is better.

At CNY 8.88/GPU-hour, four training runs are estimated at 2.96 GPU-hours / CNY 26.28,
with a 6.1667 GPU-hour / CNY 54.76 ceiling. Including one sanity, two baseline seeds,
16 checkpoint validations, and four final evaluations gives 6.8767 expected
GPU-hours / CNY 61.06 and a 14.0 GPU-hour / CNY 124.32 ceiling. These are planning
figures, not measured 1.5B results.
