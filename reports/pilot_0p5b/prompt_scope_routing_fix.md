# Pilot-aware prompt routing fix

`Matched 0.5B pilot - not the final benchmark`

## Outcome

The confirmed delayed-runtime routing blocker is fixed CPU-only. PPO and GRPO no
longer infer prompt eligibility from an experiment-name prefix. No CUDA, real
model/tokenizer load, generation, Trainer, backward, optimizer step, pilot GPU run, or
1.5B action occurred.

## Root cause

The old `prompt_version_from_config` treated only names beginning with `smoke-` as
eligible for `prompt_v1_strict_concise`. CPU pilot resolution separately special-cased
the pilot family, but the delayed PPO and GRPO dataset builders called the old selector
again after model loading. The valid PPO seed-42 pilot therefore reached the
main/formal rejection branch before generation.

## Validated-scope routing

`ValidatedExperimentScope` is selected centrally from the canonical repository path
and the raw config SHA256:

- `STAGE_D_SMOKE`: comes from a protected Stage D `ExpectedRunContract`; v1 allowed.
- `MATCHED_0P5B_PILOT`: comes from a protected pilot `ExpectedRunContract`; v1 allowed.
- `MAIN_FORMAL`: comes from the exact checked-in main config path/SHA allowlist, has
  no executable evidence profile, and still rejects v1.
- Unknown paths, SHA drift, algorithm mismatch, serialized scope drift and plain string
  scopes fail closed.

The resolved config records scope, algorithm, canonical config path/SHA and expected
profile. Experiment names do not participate in scope selection.

## Runtime consistency and pre-model rendering

Both execute paths now run `prepare_runtime_prompt_preflight` before snapshot/model
handling. It checks that CPU-resolved, ExpectedRunContract, delayed-runtime and prompt
selector scopes are identical.

- PPO renders 16 Python message rows, verifies prompt-major episode metadata, frozen
  rendered hashes and all 16 comparison keys.
- GRPO renders four Python prompt rows and verifies their frozen hashes plus the same
  16 comparison keys.
- The actual delayed PPO/GRPO dataset-builder functions receive the validated scope
  explicitly and repeat the scope/hash checks.
- Future runs persist this evidence as `prompt_scope_preflight.json`.

## CPU regressions

The real delayed dataset-builder functions were exercised for PPO and GRPO seeds 42,
123 and 2026 with a fake tokenizer/model boundary. PPO produced exactly 16 ordered
rows per seed; GRPO produced four prompts and 16 unique comparison keys per seed. The
seed-42 path no longer raises `main/formal configs must not activate a smoke prompt`.

Main configs renamed in memory to `pilot-*` still retain `MAIN_FORMAL` scope and
reject v1. Unknown path/hash and arbitrary string scope tests fail closed. Stage D
continues to select v1, while the historical v0 replay contract is unchanged.

Validation passed: 168 targeted tests, 337 full tests, Ruff, compileall, check_env,
manifest validation, six pilot dry-runs, two Stage D dry-runs, and fake PPO/GRPO
16-completion finalization. `check_env` reported `cuda_initialized=false` and zero
model/tokenizer loads.

## Frozen identities

- Pilot manifest: `0235210e038bc27ebf2e7218691f36f09c8e11f0bbc743f46a5318a279f6bc1f`
- Prompt v1: `6842002e4591630a4105a4ca8fdf4cab91676b3902708ec0cb7f7b458864ecd7`
- Reward: `90af0614676279eb8a47636acfdbeaded6d92237d3b16f027d79557057ca0e14`
- Parser: `655c30f20c677ead5728b402a1b6d5a4d4cefe54e4c1b34abebdafe41f3ba0ad`
- Verifier: `593fa4f1f12702411248b77d8059b4df84a182334a8f9923a2283d04a3fb0c74`
- All six resolved config raw SHA256 values remain unchanged; see the JSON report.

## Historical failure preservation

`ppo_matched_0p5b_seed42_20260714T073357Z` remains immutable
`failure_before_generation/no_update`: 0 completions/tokens, 0
update/optimizer/global steps, no checkpoint and no PPO scientific result.

- Full-run tree fingerprint: `5a6aa625d9a5860a1525080b9f7cae103b81cd6bdcf4a4d754e7d0883ffabb67`
- Git-safe tree: `b37527cacc1e85ec90219e6cc7429d3d283c097d`
- Backup SHA256: `21a64fb02f8522901eea92f4f027ba143b8b04f2a8c08292b75d0b6e9ec8f7a2`

## Decision

No prompt-routing correctness blocker remains in the reviewed scope. GPU execution is
not authorized by this repair. A future matched suite needs a new explicit
authorization and the standard clean/offline/snapshot/idle-GPU preflight. It must
start as a new suite from PPO seed 42; the historical failed attempt stays excluded
and must never be rewritten.

PTY capture, SVG whitespace, CSV CRLF, nullable telemetry, scalar-head notices,
allocator residue and optional plots were intentionally not changed.
