# Next task: CPU-only formal 1.5B model-bound CLI wiring

Status: CPU-only implementation complete; final repository verification, commits, and
static backup are in progress. Do not download/load a model, initialize CUDA,
generate, train, run backward, or step an optimizer.

## Exact same-run resume decision

Git-safe/final exports remain policy/value adapters plus the PPO scalar head only.
Training-recovery checkpoints in the full run and AutoDL backup additionally contain
trusted optimizer, scheduler, Trainer/global-step, Python/PyTorch CPU/CUDA RNG,
sampler/comparison-key prefix, runtime counters, generated-token total, and exact file
size/SHA256 inventory. GRPO omits the value adapter/head. Neither form contains full
Qwen base weights, and training state never enters Git.

Before any torch/pickle state load, resume validates a canonical direct-child path in
the same project run, exact filename allowlist and size ceilings, artifact-manifest
SHA256 inventory, run/algorithm/seed/checkpoint step, and suite/config/model/data/
prompt/reward/parser/verifier identities. Only steps 8/16/24 continue training; step
32 remains a final/evaluation checkpoint. CPU fake tests establish counter, key, and
token continuity plus bit-identical float64 final fake parameters for continuous 32
steps versus save at 16 and resume to 32.

## Goal

Connect the already frozen formal multi-update and evaluation contracts to explicit,
model-bound PPO, GRPO, and evaluation CLIs without changing the scientific contract.
Reuse the GPU-validated Stage D/0.5B runtime boundaries; do not create a new training
framework or artifact schema.

## Portfolio evidence must not be reduced

Implementation must preserve the future evidence and delivery contract in
`docs/PORTFOLIO_DELIVERABLES.md`. Stage E.1 must not remove, narrow, rename away, or
make impossible the required training metrics, baseline/final evaluation metrics,
per-problem evidence, resource/cost evidence, reproducible figures, error analysis,
case studies, Markdown analysis, or exact reproduction commands. Missing or
unreliable metrics remain nullable with reasons; this is not permission to omit the
artifact fields or fabricate zeros.

Do not create empty final-result files or placeholder scientific metrics during CLI
wiring. Implement only the boundaries needed to preserve future real evidence.

## Immutable inputs

- Branch: `pivot/math-rlvr`
- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Revision: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Active suite SHA256:
  `f6de8c555a70837d08c1e34e13a738a21ce247b3a09531df6244a2f1d3ef53bd`
- Four active config path/SHA pairs:
  - `configs/formal_1p5b/resolved/ppo_seed_42.json` —
    `717502aa665e9d5ef967e04a5ab27aa53329ccb061bda228db3c715f4dab967b`
  - `configs/formal_1p5b/resolved/grpo_seed_42.json` —
    `6776f8894e9ac725a39748b06b57b62782cea2dab61faf51fd3cc3ceb5ae58bf`
  - `configs/formal_1p5b/resolved/grpo_seed_123.json` —
    `4ce0918f7284220c36555b9f23db181354168ebe252d7244ac3ac9587be236fa`
  - `configs/formal_1p5b/resolved/ppo_seed_123.json` —
    `a68524e85e427e335abf6447aa2cc391686fd3aa4da6d42efb0e522beec1a0b3`

Seed 2026 is `reserved_not_scheduled`. Both 2026 configs must be rejected by an
execute path even if a caller supplies a plausible experiment name or seed override.
Do not change any frozen config, manifest, prompt, reward, parser, verifier, sampling,
LoRA, data order, budget, or artifact contract.

## CLI authorization boundary

Provide separate formal PPO, GRPO, and evaluation CLI execute paths. Each real path
must require the existing two-part explicit confirmation pattern:

- `--execute`
- a formal operation-specific confirmation flag that cannot be satisfied by dry-run
  defaults or a partial/misspelled argument

Dry-run remains the default. Missing either confirmation must stop before snapshot,
Transformers, TRL, CUDA, or model imports. An evaluation confirmation does not
authorize training; PPO confirmation does not authorize GRPO; and one completed stage
never authorizes the next paid/GPU stage.

Execution profile selection must come only from the exact repository-relative config
path, raw file SHA256, active-suite membership, and validated formal scope. Arbitrary
CLI numeric overrides, experiment-name prefixes, absolute-path aliases, symlinked
configs, reserved configs, main configs, pilot configs, and unknown hashes fail closed.

## Pinned local snapshot boundary

All real paths must bind the exact repo/revision above and require:

- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`
- `local_files_only=true`
- a validated canonical local snapshot for the pinned revision
- no network fallback

If the snapshot is absent, incomplete, drifted, symlink-unsafe, or the wrong Qwen2
causal-LM identity, stop before CUDA initialization and before any model/tokenizer
load. Stage E.1 tests must exercise this missing-snapshot failure with fakes or an
explicit nonexistent path; they must not download the snapshot.

## Formal PPO assembly contract

Wire the active PPO configs to the existing validated model-source, prompt-scope,
ordered-data, reward, guard, resource, and artifact boundaries.

- Policy: pinned 1.5B BF16 base plus policy LoRA r16/alpha32/dropout0 on q/k/v/o.
- Value: a distinct object from the same pinned base, value LoRA r8/alpha16 on q/v,
  plus a trainable scalar head.
- Reference: frozen policy reference/adapter-disabled role, with no optimizer params.
- Reward: parameter-free `shaped_v3_domain` using canonical GSM8K/MATH verifiers.
- Optimizer parameters: exactly the disjoint union of policy LoRA, value LoRA, and
  scalar-head trainables; no reference, reward, or base-model parameter.
- No full policy/value base weights in checkpoints.
- Resume only from a compliant checkpoint belonging to the same run/config/suite,
  with exact counter and comparison-key prefix continuity.

The execute path must preserve the frozen prompt-major data schedule and verify the
actual trainer-consumed mapping/keys. PPO never receives `num_generations`.

## Formal GRPO assembly contract

- Policy: the same pinned BF16 base and identical policy LoRA as PPO.
- Reward: the exact same parameter-free domain-aware reward/parser/verifiers.
- No value model or scalar head.
- Optimizer: policy LoRA parameters only.
- Git-safe export is adapter-only; the non-Git same-run recovery checkpoint contains
  only trusted continuation state, never base weights.
- Exact ordered four-generation grouping for every prompt and stable
  `problem_id::generation_index` keys.
- Same-run-only resume with exact identity and counter continuity.

Do not alter GRPO batching semantics to resemble PPO. Preserve the matched completion,
token, prompt order, sampling, checkpoint, and validation budgets.

## Multi-update execution contract

For each active training run enforce and persist:

- 32 outer/optimizer/global steps
- 4 ordered prompts x 4 responses per step
- 16 completions per step and exactly 512 at finalization
- maximum completion length 256
- online and final generated-token cap 131,072 from actual token IDs/masks
- checkpoint steps 8, 16, 24, 32
- validation steps 8, 16, 24, 32
- automatic retries 0

PPO remains rollout16, batch4/GA4, one PPO epoch, one minibatch, four backward
microbatches per update. GRPO remains generation batch16, four generations, batch4/
GA4, one iteration. Under/overflow or counter drift fails closed; nullable noncritical
telemetry does not.

## Baseline and adapter evaluation

Wire one formal evaluation CLI that selects a validated mode rather than inferring
from names:

- Base baseline: pinned untrained 1.5B model, no adapter.
- PPO evaluation: load only the selected policy adapter from a compliant step-32 PPO
  checkpoint; never load value adapter/head into generation.
- GRPO evaluation: load only the selected policy adapter from a compliant step-32
  GRPO checkpoint.

The frozen protocol is two baseline seeds with 800 completions each, sixteen
64-completion checkpoint validations, and four step-32 final evaluations with 800
completions each. It covers GSM8K test 200 + MATH500 200 for pass@1 and the fixed
50+50 pass@4 subset. Baseline and post-training paths must share prompt, sampling,
seeds, token cap, manifests, parser, and canonical verifier. Test results cannot tune
the prompt, reward, hyperparameters, or checkpoint.

Persist the existing evaluation artifact contract: raw completions/tokens,
per-problem and aggregate metrics, verifier statuses, domain/level slices, truncation,
resources, report, and figures reproducible from CSV/JSON. Never invent a missing
metric or coerce unavailable telemetry to zero.

## CPU-only end-to-end acceptance tests

Use injected fake model/tokenizer/trainer/evaluator boundaries only. Tests must prove:

- Dry-run and missing-confirmation paths import/load/initialize nothing expensive.
- Exact four-config path/SHA allowlist accepts only PPO42, GRPO42, GRPO123, PPO123.
- PPO/GRPO seed 2026 execute paths are rejected as `reserved_not_scheduled`.
- Wrong, missing, or drifted snapshot stops before CUDA/model load.
- CPU resolver, ExpectedRunContract, delayed runtime, prompt selector, model roles,
  and artifact manifest agree on scope and identity.
- Fake PPO and GRPO each finalize 32 steps, 512 exact keys/completions, checkpoint and
  validation milestones 8/16/24/32, and correct resume continuity.
- Token 131,072 is accepted and 131,073 fails online/finalization.
- PPO optimizer-role union and GRPO policy-only optimizer are exact and disjoint from
  frozen/reference/reward/base parameters.
- Baseline uses no adapter; PPO/GRPO evaluation loads only the correct policy adapter.
- Fake baseline, checkpoint validation, and final evaluation produce the existing
  artifact schema and frozen quantities.
- Checkpoint isolation, no-full-model inventory, failure backup, and historical
  artifact immutability hold.
- Stage D and matched-pilot dry-run/evidence behavior remains unchanged.

Run the relevant targeted tests plus full `compileall`, Ruff, pytest, environment
check, manifest validation, four formal dry-runs, baseline/final evaluation dry-runs,
fake 32-step PPO/GRPO finalization, fake evaluation finalization, `git diff --check`,
and secret/large-file audit. Record explicitly:

- `cuda_initialized=false`
- real model/tokenizer loads = 0
- generation/real Trainer/backward/optimizer = 0

Tiny fake CPU code must not claim to validate GPU memory or model compatibility.

## Stop condition and handoff

Only training-result correctness, PPO/GRPO fairness, safety, recoverability, or report
truthfulness issues may block. Do not add gates for optional telemetry, plots, PTY
capture, SVG whitespace, CSV line endings, or other presentation details.

After Stage E.1 passes, commit the wiring and create a static backup without model,
cache, checkpoint, run artifact, or credential content. Stop. Do not download 1.5B,
initialize CUDA, run model-load sanity, baseline evaluation, PPO, or GRPO without the
next explicit authorization.
