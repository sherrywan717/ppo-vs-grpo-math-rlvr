# Qwen 0.5B GRPO v1 single-update assessment

Run: `grpo_single_update_qwen25_05b_20260713T112100Z`

This is a single-update smoke diagnostic, not a training-quality conclusion.

## Outcome

- Infrastructure smoke: **PASS**. Eight complete completion-evidence records, 276 generated tokens, four microsteps, exactly one optimizer/global step, finite exposed metrics, one compliant `checkpoint-1`, finalized artifacts, a verified persistent backup, and post-process GPU release all passed.
- Learning-signal smoke: **FAIL**. Both four-completion problem groups have rewards `[0, 0, 0, 0]`; both group variances are zero and the zero-advantage fraction is 1.0. No claim of learning is supported.
- PPO: not started and not authorized.

## Frozen identity and counters

- Prompt: `prompt_v1_strict_concise`
- Prompt SHA256: `6842002e4591630a4105a4ca8fdf4cab91676b3902708ec0cb7f7b458864ecd7`
- Renderer: `math_rlvr.prompt.chat_template.v1`
- Unique prompts/completions/tokens: 2 / 8 / 276
- Microsteps/optimizer steps/global step: 4 / 1 / 1

## Completion evidence

All records contain completion IDs, masks, exact mask-derived counts, problem and generation indices, decoded text, identical reward/verifier input, and ordered reward results.

1. `countdown:train:0`, generation 0, 17 tokens, `FORMAT_ERROR`:

   ```text
   <answer>
   13 + 13 / 6
   </answer>
   ```

2. `countdown:train:0`, generation 1, 16 tokens, `FORMAT_ERROR`:

   ```text
   <answer>13 * 19 / 5</answer>
   ```

3. `countdown:train:0`, generation 2, 20 tokens, `FORMAT_ERROR`:

   ```text
   <answer>
   13 * (6 - 9) + 5
   </answer>
   ```

4. `countdown:train:0`, generation 3, 128 tokens, `FORMAT_ERROR` (clipped):

   ```text
   <reasoning>
   - Use 5, 9, 13, 6 in any order
   - Apply operations: +, -, *, /
   - Make 15/26
   - Possible combinations: 5 + 13 - 6, 5 * 6 + 13, (5 * 6) - 13
   - Evaluate combinations:
     5 + 13 - 6 = 12
     5 * 6 + 13 = 37
     (5 * 6) - 13 = 23
   - Determine the
   ```

5. `countdown:train:1`, generation 0, 16 tokens, `FORMAT_ERROR`:

   ```text
   <answer>4 - 11 * 13</answer>
   ```

6. `countdown:train:1`, generation 1, 22 tokens, `FORMAT_ERROR`:

   ```text
   <answer>
   -4 = -(16 + 4) - 11
   </answer>
   ```

7. `countdown:train:1`, generation 2, 16 tokens, `FORMAT_ERROR`:

   ```text
   <answer>11 - 13 + 4</answer>
   ```

8. `countdown:train:1`, generation 3, 41 tokens, `FORMAT_ERROR`:

   ```text
   <reasoning>
   -4 = -(4 - 11) * 13
   <answer>
   -4 = -(4 - 11) * 13
   </answer>
   ```

Every verifier detail is `each tag must appear exactly once`; RewardStatus distribution is `FORMAT_ERROR: 8`.

## Learning-signal evidence

| Problem | Rewards in generation order | Variance | Zero advantage |
|---|---:|---:|---:|
| `countdown:train:0` | 0, 0, 0, 0 | 0 | yes |
| `countdown:train:1` | 0, 0, 0, 0 | 0 | yes |

- Loss: 0.0
- Grad norm: 0.0
- Entropy: 0.5070152431726456
- Learning rate: 1e-5
- Reward mean/std: 0.0 / 0.0
- Zero-advantage fraction: 1.0
- KL: unavailable (`null`) because frozen `beta=0.0`; TRL 0.24.0 does not compute reference-model KL.

The values exposed by Trainer are finite, but zero within-group variance means this update had no GRPO learning signal.

## LoRA and checkpoint

- LoRA: r=16, alpha=32, dropout=0, targets q/k/v/o projections.
- Adapter tensor metadata contains 2,162,688 trainable parameters. Against the previously verified 496,195,456 parameters including adapter, the ratio is 0.4358540518%.
- The only checkpoint root is `checkpoint-1`: 12 files, 24,570,864 bytes, no duplicate checkpoint and no complete base-model weights.
- `adapter_model.safetensors`: 8,676,008 bytes, SHA256 `2ea8925f28f6e723493e8b1ca8fd866375d34665c7ffc0914560733f3383cc3c`.
- `adapter_config.json`: 1,152 bytes, SHA256 `6a341d35da0161ff19cb3f682e3a6986e267ee55910a784d096370e379e5a334`.
- `training_args.bin`: 7,441 bytes, SHA256 `409bb1e558e886a7ce69d5028fb1eba858660e69d559ab484b8f2e03127a38c0`; trainer metadata only and not deserialized.

The full filename/size/SHA256 inventory is in `checkpoint_inventory.json`.

## Resources and release

- PyTorch peak allocated/reserved: 1,501.549 / 1,936 MiB.
- Worker pre-exit current allocated/reserved: 64 / 108 MiB. This is a warning, not a leak verdict.
- After command exit, independent `nvidia-smi` observation returned to 0 MiB and showed no compute process; GPU release is verified.
- Runtime nvidia-smi peak: unavailable because finalized `gpu_metrics.csv` is empty. No value is inferred.
- BudgetGuard wall time: 10.567161 seconds.
- GPU-hours: 0.002935323.
- Cost at CNY 8.88/GPU-hour: CNY 0.026066.

## Persistent backup

- Archive: `/root/autodl-fs/math-rlvr-backups/grpo_single_update_qwen25_05b_20260713T112100Z.tar.gz`
- SHA256: `583ae58d892fe7f743531cddbd1eaf6685d809c9607f6532f2471f875a44180c`
- Archive listing, checksum, cache/base-model/credential exclusion: verified.

Runner-finalized files and their original checksum manifest remain unchanged. Supplemental files and figures were derived only from saved JSON/JSONL and have a separate checksum manifest.
