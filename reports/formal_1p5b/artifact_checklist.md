# Formal 1.5B artifact checklist

This checklist applies to every future baseline, training, validation, and final-test
run. Large artifacts live under `/root/autodl-tmp/runs/math_rlvr/<run_id>/`; adapters,
heads, and verified archives are copied to `/root/autodl-fs/math-rlvr-backups/`.
Git receives only small reports, CSV/JSON/JSONL evidence, plot scripts, and figures.

## Identity and lifecycle

- Exact command, run ID, start/end UTC, Git commit, dirty-state decision, and process
  exit code.
- Resolved config plus raw config/template SHA256; model repo/revision/local snapshot
  evidence; offline flags; dependency versions.
- Data registry, train schedule, prompt, reward, parser, verifier-router, domain
  verifier, policy LoRA, sampling, seed, and budget identities.
- Explicit model roles: policy, PPO value, reference, and parameter-free reward;
  trainable counts, ratios, disjointness, and exact optimizer union.
- Stop reason and scientific/execution status. Missing metrics remain `null` with an
  availability flag and reason; required non-finite evidence fails closed.

## Training evidence

- `completions.jsonl`: prompt/problem/generation key, raw text, token IDs/mask/count,
  truncation flag, canonical status/detail, reward scalar/components, checkpoint/update.
- `training_metrics.csv` and JSONL: update, reward mean/std/variance, zero-advantage
  fraction, loss, policy loss, PPO value loss, KL availability/value, entropy, grad
  norm, learning rate, response length, format/valid-answer/canonical pass rates, and
  verifier-status counts.
- Optimizer/global/update counts and PPO epoch/minibatch/backward evidence or GRPO
  group/microstep evidence. Counts agree across summary, manifest, metrics, and
  completion records.
- Validation metrics at steps 8, 16, 24, and 32. Test metrics are absent during
  training and checkpoint selection.

## Evaluation evidence

- `completions.jsonl` with every pass@1/pass@4 sample and pairing key.
- `per_problem_metrics.csv`, `aggregate_metrics.csv`, `aggregate_metrics.json`, and
  `verifier_status.csv`.
- Per-domain GSM8K/MATH500 results; MATH500 Level 1–5; format/valid-answer/canonical
  correctness; completion length/truncation; shaped reward distribution.
- Every seed's raw value, mean, sample SD, paired pre/post deltas, and problem-level
  bootstrap 95% interval. Two seeds never support a statistical-significance claim.

## Resource and checkpoint evidence

- `resource_metrics.csv` and JSONL with wall time, GPU utilization, PyTorch allocated/
  reserved current and peak, nvidia-smi memory, GPU-hours, and CNY cost.
- Checkpoints exactly at 8/16/24/32. Inventory records path, size, SHA256, role, and
  validation result.
- PPO: policy adapter, value adapter, scalar head, and safe metadata only. GRPO: policy
  adapter and safe metadata only. Full base-model and optimizer weights are forbidden.
- Parent-process PID exit, no compute process, memory restoration, archive listing,
  `sha256sum -c`, secret scan, and large-file scan.

## Reports and figures

- `report.md`, assessment/summary JSON, case-study and error-analysis Markdown.
- Figures regenerated only from persisted CSV/JSON via
  `scripts/plot_formal_results.py`: reward/update, validation pass/update,
  format/valid-answer, PPO policy/value loss, GRPO loss, KL/entropy/grad norm,
  completion length/truncation, verifier status, VRAM/utilization, wall time/cost,
  algorithm-by-seed, baseline versus post pass@1/pass@4, domain results, MATH500
  Levels 1–5, and confidence intervals.
- Git-safe artifact manifest with file size/SHA256. No cache, base weight, credential,
  auth file, environment dump, checkpoint, or unreviewed large binary enters Git.
