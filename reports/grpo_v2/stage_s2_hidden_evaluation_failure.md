# Stage S.2 hidden-test suite stopped after Base finalization failure

## Status

- Suite timestamp: `20260728T073339Z`
- Base run: `base_hidden_grpo_v2_seed42_20260728T073339Z`
- Status: `engineering_failure_after_generation_during_metric_finalization`
- Automatic retries: 0
- Old GRPO-v1, warm-start-only, selected GRPO-v2 commands: `not_executed_suite_stopped`
- Training/resume/config changes: none

The Base worker generated and persisted all 1,300 frozen completion rows. Metric
finalization then called the dev-only aggregator, which requires exactly 128 rows, on
the 400 candidate-0 rows and raised `dev aggregate requires 128 rows`. Therefore the
role was not marked scientific success and the suite stopped under the preregistered
failure policy. Base must not be regenerated or joined with a new run.

## Preserved primary evidence

Independent CPU validation of immutable `completions.jsonl` passed:

- Candidate-0: 400/400
- Shared subset: 100/100 problems
- Shared candidates: exactly 10/problem, 1,000 rows
- Total completions: 1,300/1,300
- Generated tokens: 152,567
- Duplicate/missing problem-candidate keys: 0
- Candidate-0 provisional evidence counts: 6 canonical pass, 31 format-valid,
  28 valid-answer/parseable, 357 EOS, 43 truncated

These counts demonstrate prefix completeness but are not published as a finalized
scientific Base result because the frozen metric/report artifacts were not produced.

## Resources and recovery evidence

- Wall time: 1,288.526 seconds
- Peak nvidia-smi VRAM: 5,297 MiB
- Mean sampled GPU utilization: 38.577%
- GPU-hours: 0.357924
- Cost at CNY 8.88/GPU-hour: CNY 3.1784
- Post-process GPU: 0 MiB, no compute process
- Failure archive: `/root/autodl-fs/math-rlvr-backups/base_hidden_grpo_v2_seed42_20260728T073339Z.failure.tar.gz`
- Archive SHA256: `532ac2854ade3374c3725410f509f6092e2508453fbd68522cf1b85c9660e215`
- Archive entries: 18; SHA sidecar validation passed

The unique blocker is the dev-only 128-row aggregate assumption in hidden evaluation
finalization. A separately authorized CPU-only repair may derive Base metrics from the
immutable 1,300-row evidence and prepare the remaining three commands. It must not
rerun Base, modify sampling/checkpoints, or access additional hidden completions.
