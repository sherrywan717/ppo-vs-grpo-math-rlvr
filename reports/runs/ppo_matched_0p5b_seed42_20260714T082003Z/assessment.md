# Matched PPO pilot seed 42 failure assessment

`Matched 0.5B pilot - not the final benchmark`

- Run ID: `ppo_matched_0p5b_seed42_20260714T082003Z`
- Suite sequence: 1 of 6
- Scientific status: `failure_before_generation/no_update`
- Stop category: `execution_contract`
- Reason: `TRLContractError: PPO data collator must return a mapping`
- Completions / generated tokens: `0 / 0`
- Update / optimizer / global steps: `0 / 0 / 0`
- Checkpoint: none
- Backup: `/root/autodl-fs/math-rlvr-backups/ppo_matched_0p5b_seed42_20260714T082003Z.failure.tar.gz`
- Backup SHA256: `8fc1800417dc79ee22b6b8880986de8b4ea92efa48e48d869f2ee69ee6e34118`
- Peak nvidia-smi VRAM: `2451 MiB`
- Peak PyTorch allocated / reserved: `1910.64990234375 / 1992 MiB`
- GPU-hours / estimated cost: `0.001623465743743711 / CNY 0.014416375804444156`
- Post-process GPU state: `0 MiB`, no compute process

The frozen prompt, reward, parser, verifier, model revision, pilot manifest,
resolved config, expected 16 comparison keys, and four validated prompt-scope
layers all matched before model loading. The failure occurred while auditing the
actual PPO DataLoader boundary after trainer construction. No completion was
generated and no training update occurred, so this run is not a PPO scientific
result and is excluded from any six-run aggregate.

This is a correctness blocker under the frozen suite rules. The command was not
retried and suite runs 2--6 were not executed. The older immutable failure
`ppo_matched_0p5b_seed42_20260714T073357Z` remains separate and unchanged.
