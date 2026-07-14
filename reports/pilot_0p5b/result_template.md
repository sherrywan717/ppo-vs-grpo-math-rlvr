# Matched 0.5B pilot run result

`Matched 0.5B pilot - not the final benchmark`

- Run ID: `<required>`
- Algorithm / seed: `<required>`
- Resolved config path / SHA256: `<required>`
- Pilot manifest SHA256: `0235210e038bc27ebf2e7218691f36f09c8e11f0bbc743f46a5318a279f6bc1f`
- Status / stop reason / warning list: `<required>`
- Automatic retries: `0`

## Identity and counters

Record model revision, prompt/reward/parser/verifier identities, policy LoRA, sampling, 16 pair keys, 16 completions, exact generated tokens (≤2,048), and update/optimizer/global/checkpoint counters (all exactly one).

## Completion and group evidence

Link the 16 raw completion/token records and four problem groups. Report each canonical `RewardStatus`, staged components/scalar, group variance, zero-advantage state, pass@1, pass@4, format accuracy, valid-expression rate, and number-usage accuracy.

## Training and resources

Report reward mean/std, loss, policy loss, PPO value loss, grad norm, entropy, KL availability/value, wall time, allocator and nvidia-smi peaks separately, GPU-hours, and RMB cost. Unavailable values must be `null`, `available=false`, with a reason.

## Checkpoint and artifacts

Record inventory SHA256, adapter/head-only result, absence of full base/optimizer weights, archive/checksum paths, and verification. Confirm this run did not inherit or overwrite any other run.

## Interpretation

This single matched pilot run is execution evidence, not a final benchmark or an algorithm-superiority claim.
