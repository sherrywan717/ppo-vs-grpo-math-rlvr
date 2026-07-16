# Matched PPO pilot seed 42 sync-boundary failure

`Matched 0.5B pilot - not the final benchmark`

- Run ID: `ppo_matched_0p5b_seed42_20260716T111934Z`
- Suite sequence: 1 of 6
- Status: `failure_during_update/microbatch_sync_contract`
- Launcher exit code: `1`
- Reason: `TRLContractError: unexpected PPO microbatch count at optimizer boundary: 1 != 4`
- Completions / generated tokens / rewards: `16 / 574 / 16`
- Update / optimizer / global steps: `0 / 0 / 0`
- Completion evidence rows persisted: `0`
- Checkpoint: none

All frozen identities, prompt scope layers, the real sequential `DataLoaderShard`, 16
comparison keys, model roles and optimizer parameter union passed. The run generated
and rewarded one 16-response rollout. At the first optimizer-wrapper call, the newly
added CPU-audited guard observed `accelerator.sync_gradients=true` after only one call,
not the expected four, and failed before recording or performing a protected optimizer
step. This disproves the CPU assumption that the wrapper's sync flag changes across
four TRL calls; it does not show a frozen config drift or a second real epoch.

Because the backend did not reach completion-evidence finalization, completion texts,
losses and reward metrics are unavailable and are not reconstructed. The run is not a
scientific PPO result and is excluded from aggregation.

- Guard/resource window: `16.242839865386486 / 16.82423112168908 s`
- Peak nvidia-smi VRAM: `10881 MiB`
- Peak PyTorch allocated/reserved: `9640.32177734375 / 10208 MiB`
- GPU-hours / cost: `0.004673397533802522 / CNY 0.0414997701001664`
- Post-process GPU: `0 MiB`, no compute process
- Backup: `/root/autodl-fs/math-rlvr-backups/ppo_matched_0p5b_seed42_20260716T111934Z.failure.tar.gz`
- Backup SHA256: `29a6e478a6692782700c23900b2c0836af5dd132961f835867834461490014e1`

The command was executed once and was not retried. Runs 2--6 were not executed. Per
the authorization, no further code repair was attempted in this task; all three older
PPO seed-42 failures remain immutable and separate.
