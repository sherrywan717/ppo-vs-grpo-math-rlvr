# Formal PPO seed 42 — second failed attempt evidence

- Run ID: `ppo_formal_1p5b_seed42_20260719T131800Z`
- Status: engineering failure; excluded from scientific aggregation
- Failure: `FormalRuntimeError: formal checkpoint cadence mismatch`
- Retry count: 0; no GRPO, seed-123, baseline, checkpoint validation, or final-test command ran.

## Evidence boundary

The real trainer completed 32/32 updates, 32 optimizer/global steps, 512 ordered
training completions, and 51,369 rollout tokens, all supported by the incrementally
persisted `metrics.jsonl`, `completions.jsonl`, and checkpoint prefixes. The token total
is below the frozen 131,072 training cap. This proves the training loop executed but
does not make the attempt a scientific success: no frozen 64-problem checkpoint
validation ran.

After training returned, `CompletedTrainerBackend` replayed the scheduled checkpoint
sequence starting at step 8 while the incremental observer already held update 32.
`FormalProgressGuard.record_checkpoint(8)` correctly rejected `8 != 32`. The failure
occurred after checkpoint directories 8/16/24/32 had been written and before any
validation. No retry was attempted and the run is immutable.

The generic `final_summary.json` reports zero counters because exception finalization
received no success result object. For this failure, the authoritative counters are the
32 metric rows, 512 completion rows, `failure_report.json` counters, and the four
checkpoint prefix manifests; they agree at 32/512/51,369.

## Training evidence (diagnostic only)

Across 32 updates, mean reward was 0.23652, canonical pass 0.16797, format accuracy
0.48633, policy/value/total loss 0.00996/4.47417/0.45737, approximate KL 0.000607,
ratio 0.99485, clip fraction 0.00461, and TRL native policy entropy mean 1.28051.
Canonical statuses over 512 completions were 86 verified pass, 132 wrong answer, 31
parse error, and 263 format error. Fourteen completions reached the 256-token cap
(2.7344%); EOS rate was 97.2656%. Ninety-six of 128 reward groups had nonzero
variance. These are preserved failure diagnostics and are not included in the formal
PPO-versus-GRPO aggregate.

TRL 0.24.0 did not expose policy/value grad norm, entropy standard deviation,
response-mask-weighted entropy, advantage, or return. Each remains
`null`/`available=false` with its actual reason; none was coerced to zero. The available
entropy is TRL `policy/entropy_avg`: response-axis logits, prompt excluded, but no PAD,
EOS, or response-token mask in its reduction.

## Checkpoints and validation

Checkpoint directories 8/16/24/32 each contain policy/value adapters, scalar head,
optimizer, scheduler, Python/NumPy/PyTorch CPU/CUDA RNG evidence, runtime/trainer
prefixes, identity, and an artifact SHA256 inventory. Every manifest says
`base_weights_included=false`; no full base-model weights were saved. Because the run
failed before run-level checkpoint inventory finalization and before validation, none
of these checkpoints is authorized for resume or evaluation in the current state.

Validation completions/tokens are 0/0 and remain separate from the 512/51,369 training
ledger. The existing seed-42 baseline is an independent formal-test result, not a
matching 64-problem validation baseline and not a delta.

## Resources, release, and backup

- Wall time: 751.268899 seconds
- GPU-hours: 0.208685805
- Cost at CNY 8.88/GPU-hour: CNY 1.853130
- Peak nvidia-smi VRAM: 53,151 MiB; mean utilization: 34.8984%
- Peak PyTorch allocated/reserved: 49,360.812 / 52,472 MiB
- After process exit: 0 MiB, 0% utilization, no compute process
- Verified failure backup: `/root/autodl-fs/math-rlvr-backups/ppo_formal_1p5b_seed42_20260719T131800Z.failure.tar.gz`
- Backup SHA256: `f63812afed44cdc9f0fcafdf0931454548da1a4ce145840ebf91bb6fa5a6d7c5`

## Unique blocker

The sole next task is a bounded CPU-only repair that aligns checkpoint/validation
recording with the incremental observer state, so checkpoint 8 is not replayed after
the observer has reached update 32. This report does not authorize that repair, a PPO
rerun, GRPO, seed 123, validation, or final test.
