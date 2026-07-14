# Matched PPO pilot seed 42 failure assessment

`Matched 0.5B pilot - not the final benchmark`

## Status

- Run ID: `ppo_matched_0p5b_seed42_20260714T073357Z`
- Algorithm/seed: PPO / 42
- Initial commit: `10fcc2173e45b5eab438b0712b9aa9562abdf214`
- Attempt: 1 of 1; automatic retries: 0
- Status: `failure_before_generation/no_update`
- Suite decision: stop before runs 2–6
- Blocker category: execution-contract prompt-stage routing
- Process exit code: 1

The frozen pilot identity and expected 16-key prompt-major contract passed preflight.
During the delayed real execution path, the value-model scalar head was initialized,
then PPO dataset prompt rendering called the general training prompt selector. That
selector recognizes only experiment names beginning with `smoke-` as eligible for
`prompt_v1_strict_concise`; the valid pilot name therefore fell into the main/formal
branch and raised:

`ValueError: main/formal configs must not activate a smoke prompt`

The runtime caller is `ppo_runtime.py` through `render_training_prompt`; the rejecting
branch is `prompt.py:88-99`. The CPU config resolver already recognizes the pilot
family, so dry-run preflight did not expose this delayed runtime-path mismatch.

## Frozen evidence and counters

- Config: `configs/pilot/resolved/ppo_seed_42.json`
- Config SHA256: `1daeba7e6cd5e0af43c7f7cb9db87b46d44608adf9fdf432dc7b2c34ea059fdd`
- Pilot contract SHA256: `0235210e038bc27ebf2e7218691f36f09c8e11f0bbc743f46a5318a279f6bc1f`
- Expected comparison keys: all 16 frozen `problem_id::generation_index` keys, recorded in prompt-major order
- Actual completions/generated tokens: 0 / 0
- Actual updates/optimizer steps/global steps: 0 / 0 / 0
- Reward and training metrics: unavailable because generation and training never began
- Checkpoint: absent; no checkpoint or model weights were written
- Full base-model weights in artifacts: none

No scientific PPO result exists for this attempt. It must not be aggregated with a
successful run and must not be described as evidence for or against PPO.

## Resource and release evidence

- Measured resource window: 4.899377426 seconds
- `nvidia-smi` peak: 4 MiB
- GPU-hours: 0.001360938173925711
- Estimated cost: CNY 0.012085130984460315
- Post-process GPU: 0 MiB, no compute process
- Worker PID: exited

## Artifact and backup evidence

- Full artifacts: `/root/autodl-tmp/runs/math_rlvr/ppo_matched_0p5b_seed42_20260714T073357Z/`
- Git-safe report: `reports/runs/ppo_matched_0p5b_seed42_20260714T073357Z/`
- Backup: `/root/autodl-fs/math-rlvr-backups/ppo_matched_0p5b_seed42_20260714T073357Z.failure.tar.gz`
- Backup SHA256: `21a64fb02f8522901eea92f4f027ba143b8b04f2a8c08292b75d0b6e9ec8f7a2`
- Archive listing: passed
- `sha256sum -c`: passed

The archive contains the failed run evidence and an empty checkpoint directory only;
it contains no Hugging Face cache, checkpoint weights, complete base model, credential,
token, or auth material.

## Continuation decision

Runs 2–6 were not executed. Repairing the pilot-aware prompt selection requires a
separate CPU-only source change, regression tests, new commit, and new explicit GPU
authorization. This failed command must never be retried under the present suite
authorization.
