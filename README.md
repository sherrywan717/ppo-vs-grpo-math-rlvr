# PPO vs GRPO for Few-Shot Math RLVR at 1.5B

A reproducible comparison of PPO and GRPO sample efficiency, stability, and generalization on `Qwen/Qwen2.5-1.5B-Instruct`. Formal training uses GSM8K and MATH Level 1–3; evaluation uses GSM8K test and MATH500. Countdown is smoke/verifier-only and is never a headline benchmark.

## Frozen design

The active formal suite uses PPO/GRPO at seeds `42` and `123`; seed `2026` remains `reserved_not_scheduled`. Both algorithms share the frozen manifests and 2-GSM8K/2-MATH update order, `prompt_v2_formal_math`, BF16 policy LoRA (`r=16`, alpha 32, dropout 0; q/k/v/o projections), four responses per prompt, temperature 0.8, top-p 0.95, amended prompt capacity 832, completion length 256, and one domain-aware reward policy. Comparisons align actual completions and generated tokens, not trainer steps.

Formal reward `shaped_v3_domain` is fixed: answer block 0.05, strict protocol 0.05, domain-valid answer 0.10, and canonical correctness 0.80. Countdown exact-number-usage is not applied to GSM8K or MATH. Formal pass metrics use only canonical verifier status; infrastructure errors abort.

PPO uses a separate sequence-classification value model from the policy checkpoint,
value LoRA `r=8`, alpha 16 on q/v projections, and a trainable scalar head. The
guarded runner completed the accepted one-update Qwen 0.5B smoke
`ppo_single_update_qwen25_05b_20260714T051538Z`. Its scientific/execution status is
`execution_success/nonessential_telemetry_warning`; it must not be rerun.

## Data and layout

Frozen manifests live under `/root/autodl-tmp/datasets/math_rlvr/manifests`; Hugging Face cache is `/root/autodl-tmp/cache/huggingface`. `src/math_rlvr/` contains schema, prompt, verifiers, rewards, rollout accounting, metrics, and preflight entry points. `src/math_rlvr/execution/` is retained only as legacy/out-of-scope history and is not imported by the math pipeline.

The output contract is exactly one `<reasoning>...</reasoning>` block followed by exactly one terminal `<answer>...</answer>` block. Only answer content is verified. Verifiers never call `eval`, `exec`, dynamic imports, subprocesses, or generated code.

## GRPO single-update smoke contract

The checked-in GRPO smoke YAML is the configuration source of truth: two unique prompts, four generations per prompt, generation batch 8, micro-batch 2, gradient accumulation 4, one iteration, one global/optimizer step, eight completions, and a 1,024 generated-token hard cap. TRL 0.24.0 must infer `steps_per_generation=4`; never configure both that field and `generation_batch_size`. This is an integration smoke contract, not a formal experiment result.

## Shared smoke prompt

The Qwen 0.5B PPO and GRPO smoke configs select the same frozen
`prompt_v1_strict_concise` renderer. Its candidate status is `approved_for_smoke`, but
its production status is `not_approved`: the matched generation-only diagnostic raised
complete-envelope compliance from 0% to 25% and created nonzero reward variance in both
Countdown groups, while valid-expression, number-usage, pass@1, and pass@4 remained 0.
`prompt_v0_grpo_smoke` remains unchanged for historical replay, and main/formal 1.5B
configs do not activate v1.

PPO and GRPO must resolve and report the same `prompt_version`, `prompt_sha256`, and
`renderer_version`; rendering the same `MathProblem` must be byte-identical. See
`docs/smoke-prompt-fairness.md`. Both current one-update smoke paths have now used
this identity; any new GPU execution still requires separate explicit authorization.

## Guarded GRPO execution

The default GRPO CLI is dry-run only. `--execute` by itself still fails closed. The only real smoke path requires the frozen smoke config and both flags:

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m math_rlvr.training.grpo --config configs/smoke/grpo.yaml --execute --confirm-single-update
```

Before delayed model imports, the CLI requires a clean `pivot/math-rlvr` worktree, the fixed local revision, and the complete batching/budget contract. `trl_compat.py` is the sole TRL 0.24.0 private-hook shim and exact token accounting uses completion IDs/masks, not decode/re-tokenize. The artifact state is fail-closed: success requires complete artifacts, adapter-only checkpoint inventory, tar backup, and verified SHA256. It never advances to PPO automatically.

## Guarded PPO single-update contract

The PPO smoke YAML resolves under TRL 0.24.0 to four fixed Countdown dataset rows,
`total_episodes=4`, rollout batch 4, one response per row, one PPO epoch, one
minibatch, one optimizer/update/global step, four total completions, and a 512-token
hard cap. The shared-schema `generation.num_generations=4` is explicitly recorded as
ignored for PPO: TRL PPO does not consume it and does not multiply the rollout into 16
responses. The shim applies the configured `top_p=0.95`, because TRL PPO otherwise
constructs its internal generation config with top-p 1.0.

Policy and value backbones are distinct objects loaded local-only from the same
validated Qwen 0.5B snapshot. The optimizer must exactly contain policy LoRA plus value
LoRA/scalar-head parameters. The frozen reference is the policy base with its PEFT
adapter disabled; the verifier reward model has zero parameters. The sole
`checkpoint-1` contains separate policy adapter, value adapter, and scalar-head
safetensors plus JSON metadata, never base-model or optimizer weights.

The default PPO CLI remains a dry-run. The accepted historical smoke used the frozen
config, clean branch, both offline variables, fixed snapshot, and both flags:

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m math_rlvr.training.ppo --config configs/smoke/ppo.yaml --execute --confirm-single-update
```

This exact smoke was run once and must not be retried. The command remains documentation
of the guarded entry point, not authorization for another GPU run.

## CPU-only checks

```bash
python -m compileall src tests
ruff check .
pytest -q
python scripts/check_env.py
python scripts/validate_manifests.py
make smoke-ppo
make smoke-grpo
make main-ppo
make main-grpo
```

All four commands shown above are static preflights and do not train. The guarded real
entry points require separate dual-confirmation authorization; the CPU gates do not
load a model or initialize CUDA.

## Metrics

Both runs record pass@1/pass@4; GSM8K, MATH500, and per-Level accuracy; format, parse,
expression, and number-usage validity; reward, completions, generated tokens,
completion length, wall time, KL, entropy, peak VRAM, GPU-hours, and CNY cost. The
guarded PPO smoke normalizes only metrics exposed by reviewed TRL 0.24.0 keys,
including policy/value loss, KL, entropy, clip fraction, ratio, reward mean, and
learning rate. The sole nullable nonessential telemetry allowlist currently contains
`val/ratio_var`: a non-finite value becomes `null`, `available=false`, and retains its
raw key, classification, and reason. It is never fabricated as zero. Any non-finite
required or unreviewed metric still fails closed.

## GRPO evidence and checkpoint safety

The single-update runner uses the Trainer-created top-level `checkpoint-1` as its sole authoritative checkpoint. It never performs a second manual `save_model`. The exact `training_args.bin` basename is accepted only as non-symlink regular trainer metadata directly under that checkpoint, capped at 1 MiB and hashed without deserialization; arbitrary `.bin` files remain forbidden.

The sole TRL 0.24.0 shim binds completion IDs/masks, exact mask-derived token counts, Unicode decoded text, exact verifier input, and ordered reward results into eight JSONL records. Missing or reordered evidence fails closed. The frozen config resolves to `beta=0.0`, so KL is represented as unavailable with `null` and an explicit reason. PyTorch allocator peaks are recorded separately from nvidia-smi. See `docs/artifact-schema.md` and `docs/checkpoint-safety.md`.

## Guarded generation-only prompt diagnostic

The independent v0/v1 diagnostic defaults to a CPU-only static preflight:

```bash
PYTHONPATH=src python -m math_rlvr.evaluation.prompt_ab --config configs/diagnostics/prompt_ab.yaml
```

Real generation is not authorized by the training flags. It requires both
`--generate-only --confirm-prompt-diagnostic`, a clean worktree, offline mode, and the
exact local Qwen 0.5B snapshot. It uses the BF16 base model in eval/inference mode with
all parameters frozen, matched seeds across prompt variants, 16 completions, and a 2,048
token cap. Trainer, LoRA, train, backward, optimizer, checkpoint/model writes, retries,
and automatic v1 activation are fail-closed.

The candidate decision is diagnostic only: v1 must improve complete-envelope rate, yield
at least one envelope, avoid higher truncation, and create nonzero within-problem reward

`docs/prompt-diagnostic-artifact-schema.md`. A versioned capability manifest must prove
paired artifacts, per-problem rewards, allocator evidence, failure backup, post-worker
GPU verification, and cross-file consistency before the fixed worker may start. The
non-CUDA parent launches one fixed spawned worker, then verifies PID exit, absence from
the nvidia-smi compute list, and restoration to baseline before final backup/publication.

## Staged smoke reward v2

After the first v1 GRPO smoke demonstrated a successful execution pipeline but zero
within-group reward variance, the two 0.5B smoke configs now select the shared
`shaped_v2_staged` policy. This is a public post-smoke intervention, not a rewrite of
the historical run and not an activation for main/formal 1.5B experiments.

The staged scalar components are answer block 0.05, strict protocol 0.05, safe valid
expression 0.05, exact Countdown number use 0.05, and canonical correctness 0.80.
Canonical `RewardStatus`, strict format metrics, pass@1/pass@4, expression validity,
number-usage accuracy, and the sparse policy remain unchanged. `RESOURCE_LIMIT` gets
no partial score; `INFRA_ERROR` aborts. Only an original strict canonical
`VERIFIED_PASS` reaches 1.0.

PPO and GRPO smoke configs resolve the same reward version, component weights, and
policy SHA256. Both accepted current technical smokes used this identity; this does not
authorize another GPU execution.


## Stage D technical-smoke conclusion

The accepted PPO run `ppo_single_update_qwen25_05b_20260714T051538Z` and GRPO run
`grpo_single_update_qwen25_05b_20260713T122258Z` both completed a real single update
with the fixed Qwen 0.5B revision, `prompt_v1_strict_concise`,
`shaped_v2_staged`, nonzero reward variation, and safe adapter-only checkpoints.
Stage D technical smoke is complete.

These single-update smokes validate execution, evidence, reward integration, and
checkpoint paths only. They do not prove task learning or that PPO is better than
GRPO. The runs are not an algorithm-effect comparison because PPO used four prompts
with one response each while GRPO used two prompts with four responses each, and their
completion/token budgets differ. See `reports/stage_d/smoke_readiness_matrix.md` for
the evidence matrix and the planning-only matched 0.5B pilot proposal.


## Matched 0.5B pilot freeze

`Matched 0.5B pilot - not the final benchmark`

The CPU-only pilot freeze selects the first four Countdown train records in original
order and matches PPO/GRPO at four responses per prompt, 16 completions, a 2,048-token
hard cap, one optimizer/global update, policy LoRA, sampling, and seeds 42/123/2026.
The ordered manifest SHA256 is
`0235210e038bc27ebf2e7218691f36f09c8e11f0bbc743f46a5318a279f6bc1f`.
Six committed resolved JSON configs are authorized by exact path and SHA; temporary CLI
seed overrides are forbidden. Parser and Countdown verifier identities are derived
from canonical semantic JSON, not comments or Markdown.

The two execution-contract blockers are resolved CPU-only. PPO now replaces the TRL
0.24.0 shuffled loader immediately after Trainer construction with an explicit
single-device `SequentialSampler` loader, prepared by the existing Trainer Accelerator.
The prepared batch and the iterator consumed by `train()` must both match the 16
prompt-major episode identities. PPO and GRPO finalization select immutable 4/8/16
evidence profiles only from exact config path/SHA256 allowlists; online overflow and
final under/over counts fail closed without weakening Stage D.

This makes the six pilot commands technically ready for a future separately authorized
GPU suite; it does not authorize or execute them. The existing `--execute
--confirm-single-update` controls, clean/offline/snapshot gates, fixed ordering and zero
retry policy remain. See `reports/pilot_0p5b/execution_contract_fix.md` and
`reports/pilot_0p5b/plan.md`. A matched pilot is execution/aggregation evidence only;
it cannot prove learning or PPO/GRPO superiority.

### Matched 0.5B pilot result

The six-run matched PPO/GRPO pilot is complete. It validates matched execution,
single-update budgets, checkpoint safety, and aggregatable artifacts; it is not the
final benchmark and does not establish learning or algorithm superiority. All six
runs had zero canonical pass@1/pass@4. See
`reports/pilot_0p5b/final_report.md` for the three-seed results and limitations.

## Stage E formal 1.5B freeze

Stage E freezes the CPU-only formal experiment at
`Qwen/Qwen2.5-1.5B-Instruct` revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. No weights or tokenizer were downloaded or
loaded, CUDA was not initialized, and no generation or training occurred. The six
resolved descriptors under `configs/formal_1p5b/resolved/` bind seeds 42/123/2026,
the same prompt/reward/parser/verifier/data identities, 32 updates, 512 completions,
a 131,072-token cap, and checkpoints at steps 8/16/24/32.

PPO derives rollout batch 16 from microbatch 4 and GA4, with one PPO epoch and one
minibatch per update. GRPO uses generation batch 16, four generations, microbatch 4,
GA4, and no dataset shuffle. Both have exactly 32 optimizer/global steps. PPO's
separate value base, rank-8 q/v adapter, and scalar head are an explicit algorithmic
cost difference; they are never presented as a matched model architecture.

The frozen baseline/final protocol uses GSM8K test 200 and MATH500 200 for pass@1 plus
fixed 50+50 subsets for pass@4. Test data is used only for the shared base baseline and
fixed step-32 final evaluation, never for prompt, reward, hyperparameter, or checkpoint
selection. See `reports/formal_1p5b/experiment_plan.md`. Model download, CUDA sanity, and the
two-seed baseline are now complete; every training run remains separately authorized.

### Formal 1.5B four-run amendment

The original six-run Stage E decision remains preserved at commit `499fea9f`. The
active portfolio-scale comparison now uses PPO/GRPO for seeds 42 and 123 only, with a
seed-42 review before seed 123 and four fixed step-32 final evaluations. The two
seed-2026 descriptors remain frozen as `reserved_not_scheduled` and are excluded from
the active queue, costs, and statistics. Two seeds support transparent raw results,
mean, sample SD, paired deltas, and problem-level bootstrap intervals—not a claim of
statistical significance or general algorithm superiority.

The formal CPU runtime is frozen in `math_rlvr.training.formal_runtime` and
`math_rlvr.evaluation.formal_runtime`. Fake PPO and GRPO runs exercise all 32 updates,
512 ordered completions, four checkpoints/validations, same-run resume continuity,
overflow failure, backup, and baseline/final artifact finalization. This does not
authorize or claim a 1.5B model load or training result.


## Current formal 1.5B status

The pinned Qwen 1.5B snapshot download and local-only BF16 CUDA/model-load sanity are
complete. The frozen base baseline also completed successfully for seeds 42 and 123;
results and CSV-derived figures are in
[`reports/formal_1p5b/01_baseline_results.md`](reports/formal_1p5b/01_baseline_results.md).

Two earlier seed-42 engineering attempts are preserved unchanged: one failed reward-
evidence serialization at 0/800 and one hit the historical 512-token prompt cap at
642/800. Both are explicitly excluded from scientific statistics. The public
post-freeze capacity amendment and full audit are documented in
[`reports/formal_1p5b/prompt_length_amendment.md`](reports/formal_1p5b/prompt_length_amendment.md).

Two formal PPO seed-42 attempts remain immutable engineering failures as individual
runs. Stage H.3 has now recovered all four checkpoint validations from the second
attempt's trusted checkpoints without rerunning or resuming training. The transparent
composite is `scientifically_complete_with_recovered_validation`; its 32-update
training and checkpoint curves are in
[`reports/formal_1p5b/03_ppo_training.md`](reports/formal_1p5b/03_ppo_training.md).
Stage H.4 corrected the prospective native `valid_answer_rate` mapping to the existing
flat reward-component evidence, with explicit definition and null/unavailable handling.
It is reporting-only; historical PPO artifacts remain immutable and the recovered
composite remains scientifically complete. The sole next task in
[`docs/NEXT_TASK.md`](docs/NEXT_TASK.md) is separately authorized formal GRPO seed 42.
No GRPO, seed 123, baseline, or final test starts automatically.
