# PPO single-update smoke

- Run ID: ppo_single_update_qwen25_05b_20260714T051538Z
- Source commit: ef74e46d375b943d490d89bb33d5d3421f0378c6
- Assessment: execution_success/nonessential_telemetry_warning
- Runner process: exit 1 during artifact finalization
- Invocation/retry count: 1 / 0

## Disposition

The frozen training chain completed exactly one PPO update, optimizer step, and global step. TRL emitted val/ratio_var=NaN for a single observation; the finite-JSON gate rejected that nonessential raw telemetry. Mandatory evidence was recovered without retraining. Per experiment blocking policy this is a warning, not a training failure. The original failure_report.json and exit code remain preserved.

Reward variance was nonzero, but one smoke update is not evidence that the model learned Countdown or that PPO outperformed GRPO.

## Budget and actual

- Prompts/responses/completions: 4 / 4 / 4
- Generated tokens: 141 / 512
- PPO epochs/minibatches/update/optimizer/global: 1 / 1 / 1 / 1 / 1
- Guarded/command wall time: 9.020524367690086 / 13.142639939 seconds
- PyTorch peak allocated/reserved: 6695.84228515625 / 6984.0 MiB
- nvidia-smi sampled peak: 5291 MiB
- GPU-hours/cost: 0.002613292737967438 / CNY 0.023206039513150853

## Metrics

- Rewards: [0.05, 0.10, 0.10, 0.10]
- Reward mean/population variance: 0.0875 / 0.00046875000000000004
- Policy/value loss: -0.003038088558241725 / 24.101951599121094
- Objective KL: 0.0792231559753418
- Policy entropy: 0.16412295401096344
- Aggregate loss, grad norm, zero-advantage fraction: null with explicit unavailable reasons
- val/ratio_var: null; raw TRL value was NaN and was not coerced to zero

## Evidence

- Four completion texts, token IDs/counts, problem IDs, prompt hashes and rewards are in completions.jsonl.
- Checkpoint inventory passed: adapter/head-only, 10,880,008 bytes, no base-model or optimizer weights.
- Combined launcher output is preserved in launcher_output.txt; runner-created stdout/stderr files remain empty and unchanged.
- Post-exit GPU release passed: 0 MiB and no compute process.
- All existing post-run CPU gates passed, including 270 tests.

## Backup

- Evidence-complete archive: /root/autodl-fs/math-rlvr-backups/ppo_single_update_qwen25_05b_20260714T051538Z.failure-evidence-complete.tar.gz
- SHA256 and inventory verification are recorded in backup_manifest.json.
- The earlier failure archive is retained as an intermediate preservation copy.
