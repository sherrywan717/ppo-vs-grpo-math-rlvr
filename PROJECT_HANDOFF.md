# Math RLVR formal 1.5B handoff

## Purpose and authority

This repository is an artifact-first, portfolio-grade fair comparison of PPO and GRPO
for mathematical RLVR on Qwen 1.5B. Authority order is Git/configs/manifests/original
artifacts, this handoff, `docs/PORTFOLIO_DELIVERABLES.md`, `docs/NEXT_TASK.md`,
`AGENTS.md`/`memory.md`, then historical chat. Never rewrite primary evidence to make
a derived document agree.

## Verified repository state

- Active branch: `improve/grpo-v2`
- Stage P authorized base HEAD: `6895fa0a00c82ed0fcef12ba8514b1fc9c14b53e`; the result commit is the commit containing the Stage P handoff below.
- Portfolio `main` and peeled tag `v0.1.0-formal-rlvr`: `f744b7866f1bdd4380a5597957359b0a953dd686`; both remain unchanged.
- Worktree: clean at Stage P start; expected clean after the result commit.
- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Revision: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Canonical local snapshot:
  `/root/autodl-tmp/cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306`

## Current formal identity

- Evaluation raw SHA256:
  `85100dd0f613f295a7219a45a42a03e3ad4a45e24893c7f296e1d8da9a1f4a35`
- Evaluation canonical SHA256:
  `d8ba5ab80ab0553d2ec7246fb4876956dcbc5dd0bcf8642fd33c4ec19da6fe44`
- Active-suite raw SHA256:
  `11869c63f4365aee5d4bf8e13fe263c9d0397164a18a88b419da07218f6a2017`
- Active-suite canonical SHA256:
  `1d7c29f76d9bfbf11e1838cd6b0bc8f3da6d0133e0605420e9ed838de729d600`

Active training runs, in frozen order:

1. PPO seed 42 — `configs/formal_1p5b/resolved/ppo_seed_42.json` —
   `1093e87a8363a0a2a6ab640a6f723c04cb6cfb22edef2e38a8c3a0062693ec43`
2. GRPO seed 42 — `configs/formal_1p5b/resolved/grpo_seed_42.json` —
   `3371d23166d01834c67830eb8dfd51a02d4af483687b0b29e941194174099199`
3. GRPO seed 123 — `configs/formal_1p5b/resolved/grpo_seed_123.json` —
   `cc95138f50f37fafa76766d3a08b0995ffd5e0bf87cd7b9050acedb5e0bbc75e`
4. PPO seed 123 — `configs/formal_1p5b/resolved/ppo_seed_123.json` —
   `3d6cc1f30f7b72bfadb5191613298ac3f64a1ba3c699cc8d1e30ce147218c15e`

Seed 2026 remains `reserved_not_scheduled`. Prompt, reward, parser, verifier, data,
sampling, LoRA, completion/token/update budgets, and fixed step-32 selection remain
frozen. Test data is never used for tuning or checkpoint selection.

## Completed execution stages

Stage E.1 formal model-bound CLI wiring and trusted same-run resume are complete.
Stage F is complete:

- Snapshot download commit: `5bbf913358f018e413ea70ef3ce34fa38afcfa1d`
- CUDA sanity commit: `2a86af7572d4f6b1419b1012b9b19f50cf9cbade`
- Sanity run: `cuda_load_sanity_qwen25_1p5b_20260718T113620Z`
- BF16 local-only load and two finite forwards passed; generation/training/backward/
  optimizer/checkpoint counters were zero; post-process GPU release passed.

## Immutable failed baseline attempts

Both attempts are engineering failures and `included_in_scientific_aggregate=false`:

- `baseline_formal_1p5b_seed42_20260718T114907Z`: reward-evidence serialization
  attempted `evaluation.to_dict()["components"]`; 0/800 persisted completion rows.
  Repair commit: `cce3a212f7f5a60edbf43ffd6eef4794850173f6`.
- `baseline_formal_1p5b_seed42_20260718T120909Z`: old 512-token prompt cap rejected
  `math:HuggingFaceH4/MATH-500:test:219` at exactly 800 tokens after 642/800 rows.
  The 642 rows and 74,968 exact tokens remain immutable and are never reused.

Their run directories, checksums, and verified failure backups remain primary evidence.
They are excluded from all baseline metrics and figures.

## Post-freeze prompt-length capacity amendment

A pinned-tokenizer audit covered 1,192 actual-mode rows / 592 unique frozen problems.
Maximum prompt lengths were train 713, validation 339, and test/overall 800. Three
unique problems exceeded the old 512 cap. Commit
`edecfcf503cff8ee8aef3c7ef2136dae04e192b7` publicly amends the shared evaluation,
PPO, and GRPO cap from 512 to 832; 832 + unchanged completion 256 = 1,088, below the
32,768 model context.

This is capacity-only. No prompt text/token IDs, data/order, sampling, reward,
parser/verifier, LoRA, optimizer, or completion/token/update budget changed. Previously
fitting prompt token IDs are unchanged, and PPO/GRPO retain the same protocol. See
`reports/formal_1p5b/prompt_length_amendment.{md,json}`.

## Scientific baseline result

Only these successful post-amendment runs enter the scientific aggregate:

- Seed 42: `baseline_formal_1p5b_seed42_20260718T125833Z` — 800/800,
  96,150 tokens, sampled pass@1 0.040, pass@4 0.100.
- Seed 123: `baseline_formal_1p5b_seed123_20260718T133624Z` — 800/800,
  91,651 tokens, sampled pass@1 0.025, pass@4 0.060.

Greedy accuracy is `null/unavailable` because the frozen protocol has no separate
greedy completion. Aggregate report: `reports/formal_1p5b/01_baseline_results.md`.
Commit: `287f7d313c5ad8ac1500eb416eeacd605c3298f3`.

## Stage H PPO seed-42 failed attempt

The one authorized command ran exactly once as `ppo_formal_1p5b_seed42_20260718T150510Z` and failed at the
step-8 checkpoint callback: `FormalRuntimeError: formal Trainer did not expose required
grad norm`. The missing TRL field was optional under the authorization contract and
should have been persisted as null/unavailable, not treated as a training blocker.

The finalized completion/metric/verifier JSONL files are empty and formal counters are
zero. Live stdout reached the displayed 8/32 boundary and `episode=128`, but those are
not persisted primary evidence; generated tokens and scientific training metrics are
therefore unavailable. No validation ran. The run is an immutable engineering failure
and `included_in_scientific_aggregate=false`.

The partial `checkpoint-8` contains policy/value adapters plus scalar head, but no
optimizer, scheduler, RNG, runtime/counter, comparison-prefix, or trusted inventory
state. It is not resume-capable and must not be evaluated. No full base-model weights
were saved. The verified failure archive is
`/root/autodl-fs/math-rlvr-backups/ppo_formal_1p5b_seed42_20260718T150510Z.failure.tar.gz`, SHA256
`76896c5b3db3ee4439566b8b68c0cad798af5b5610f393138aa23eba6c40debb`. Measured usage was 181.951470 seconds, 0.050542075 GPU-hours,
CNY 0.448814, peak 28,655 MiB nvidia-smi VRAM; post-process GPU state was 0 MiB and no
compute process.

## Stage H.1 repair complete

From pre-repair HEAD `108aa260481710ceb90080200af348f7a0ec0765`, the bounded CPU
repair made missing aggregate/policy/value grad norms nullable with raw-key evidence and
moved validated atomic completion/metric prefix persistence to every PPO update before
the checkpoint callback. A fake step-8 callback failure retains 128 completions, eight
metric rows, and matching counters. Existing trusted checkpoint/resume structure is
unchanged. Twenty-three related pytest cases, affected Ruff/compileall, and formal PPO
dry-run passed; full pytest was not run under the experiment-first limit.

The historical failed run remains immutable and excluded. Its checksums and every
frozen SHA remain unchanged. No CUDA, model/tokenizer, generation, real Trainer,
backward, optimizer, baseline, validation, or final test ran in Stage H.1.

## Stage H PPO seed-42 second failed attempt

The newly authorized command ran exactly once as
`ppo_formal_1p5b_seed42_20260719T131800Z`; automatic retries were zero. Incremental
primary evidence confirms 32 updates, 32 optimizer/global steps, 512 training
completions, and 51,369 training rollout tokens. Checkpoint directories 8/16/24/32
were written with role-separated adapters/head, trusted training state, artifact
manifests, and `base_weights_included=false`.

After training returned, the backend replayed scheduled checkpoint 8 while the
incremental observer was already at update 32. The cadence guard rejected the mismatch
before any frozen 64-problem validation ran. The attempt is therefore an immutable
engineering failure and `included_in_scientific_aggregate=false`; none of its
checkpoints was authorized for resume or evaluation at failure time. Stage H.2 later
made the four trusted checkpoints eligible for the separately authorized validation-only
recovery recorded below; training resume remains unauthorized. The generic exception
`final_summary.json` has zero counters because no success result object reached
finalization, while the authoritative JSONL prefixes, failure report, and checkpoint
manifests agree at 32/512/51,369.

Measured usage was 751.268899 seconds, 0.208685805 GPU-hours, CNY 1.853130,
53,151 MiB peak nvidia-smi VRAM, and 34.8984% mean utilization. After worker exit the
GPU was 0 MiB with no compute process. The verified failure archive is
`/root/autodl-fs/math-rlvr-backups/ppo_formal_1p5b_seed42_20260719T131800Z.failure.tar.gz`,
SHA256 `f63812afed44cdc9f0fcafdf0931454548da1a4ce145840ebf91bb6fa5a6d7c5`.

## Stage H.2 cadence repair and recovery eligibility

Training updates remain a monotonic 1..32 cursor with unchanged optimizer/global,
completion-key, and token-budget checks. Checkpoint and validation use independent
ordered cursors `[8,16,24,32]`; online and deferred execution are both supported.
A validation step requires its same-position trusted checkpoint, while validation
rows/tokens never modify training counters or budgets.

Read-only replay validated 32 metric rows, 512 completion rows, 51,369 tokens, and all
comparison keys. Checkpoints 8/16/24/32 passed actual file SHA256, inventory, prefix,
same-run and frozen-identity validation, and existing formal evaluation selection. No
base weights are present. Therefore `validation_only_eligible=true`,
`training_rerun_required=false`, and `training_resume_authorized=false`. The
original run remains `engineering_failure_after_training / validation_pending` and
is not yet a scientific success.

Nineteen targeted CPU/fake tests, affected Ruff/compileall, PPO dry-run, and validation
protocol dry-run passed. CUDA/model/tokenizer/generation/Trainer/backward/optimizer and
real validation counts were zero. See
`reports/formal_1p5b/ppo_seed42_validation_recovery_eligibility.{md,json}`.

## Stage H.3 validation-only recovery complete

Four independently backed-up runs evaluated checkpoints 8/16/24/32 in order against
the frozen 64-problem validation manifest. They completed 64 rows each and generated
7,533 / 7,848 / 7,663 / 7,497 tokens. Sampled pass@1 was 4.6875% / 3.125% /
3.125% / 3.125%; pass@4 is null/unavailable because validation has one candidate per
problem. Total recovery usage was 0.271514 GPU-hours and CNY 2.411047; all artifact
checksums/backups verified and the GPU returned to 0 MiB with no compute process.

The original training run remains unchanged as
`engineering_failure_after_training / validation_pending` and excluded as an
individual run. The transparent composite linking its complete 32-update training
evidence, four checkpoint SHAs, and four validation artifact SHAs is
`scientifically_complete_with_recovered_validation`. Training was not rerun or
resumed, final test was not run, and checkpoint selection was not changed. See
`reports/formal_1p5b/03_ppo_training.md` and
`reports/formal_1p5b/ppo_seed42_composite_result.json`.

## Stage H.4 valid-answer telemetry repair

The stale `components.valid_answer` lookup was replaced by the existing flat
`valid_answer_component` evidence through one shared PPO/GRPO definition. It counts
positive extracted-answer verifier components over all update completions; it is not
canonical parseable rate. Numerator, denominator, raw source, definition version,
availability, and reason are persisted. Missing/zero-denominator values remain null,
and contradictory aggregates fail finalization. Reward scalar and all optimization
inputs are unchanged.

Forty-four targeted CPU tests, affected Ruff/compileall, and PPO/GRPO formal dry-runs
passed. Historical PPO artifacts/checksums remain unchanged; the recovered composite
stays `scientifically_complete_with_recovered_validation` and needs no rerun.

## Stage I formal GRPO seed-42 success

`grpo_formal_1p5b_seed42_20260720T031006Z` ran once with no retry and completed
32/32 updates, 512/512 training completions, 50,773 rollout tokens, checkpoints and
frozen validation at 8/16/24/32. The four validations produced 256 completions and
29,113 separate tokens; pass@1 rose 3.125% / 4.6875% / 6.25% / 7.8125%, while
pass@4 remained null/unavailable under the one-candidate protocol. All checksums,
checkpoint inventories and the backup SHA
`b584363595f99c1d3b61a7b6cc088cdda7ac38a29169058df7b30cd38bea5023` verified.
GPU release passed and final test was not run.

The seed-42 PPO/GRPO matched review is complete and descriptive only. GRPO ended with
higher validation pass@1 and lower measured resource use, but one seed and 64 problems
do not establish statistical significance or general superiority. Native loss and
entropy magnitudes are not directly comparable across algorithms.

## Stage J formal GRPO seed-123 success

`grpo_formal_1p5b_seed123_20260720T035927Z` ran once with no retry and completed 32/32 updates, 128 microsteps,
512/512 training completions, 52,284 rollout tokens, and checkpoints plus frozen
validation at 8/16/24/32. The four validations produced 256 completions and 27,513
separate tokens; pass@1 was 4.6875% / 7.8125% / 9.375% / 9.375%, while pass@4
remained null/unavailable under the one-candidate protocol. All run checksums,
checkpoint inventories and backup SHA `e78eb0719bc93c1076bd06e50037cc453cbaa5103cf1e1fbfc9e8151212e521a` verified. GPU release passed;
no final test ran.

The GRPO seed-42/123 stability review is descriptive only. Both runs preserved group
learning signal and showed improving but seed-variable validation curves. Two seeds do
not establish statistical significance or general algorithm superiority.

## Stage K formal PPO seed-123 success

`ppo_formal_1p5b_seed123_20260720T043732Z` ran once from execution-base commit
`f6a62eeb8ce59c438f355b675db493552044de18`, with zero retries. It completed 32/32
updates and optimizer/global steps, 512 training completions, 51,969 training tokens,
trusted PPO checkpoints at 8/16/24/32, and 256 independent validation completions /
26,859 validation tokens. Pass@1 was 3.1250% / 4.6875% / 4.6875% / 4.6875%;
pass@4 is null/unavailable under the one-candidate protocol. All checksums and backup
SHA `689924eaa4392a4806f9d1adaa2bbf890b76d6813a6edfeafc2ca50213bc63c0` verified;
GPU release passed. No final test ran.

All four active formal training runs are scientifically complete (PPO42 via its
transparent recovered-validation composite). The two-seed/four-run aggregate is
descriptive only and does not establish significance or general algorithm superiority.

## Stage L1 PPO seed-42 final evaluation

The first process, `ppo_final_formal_1p5b_seed42_20260720T052931Z`, was terminated by
a host power/network outage after 429 persisted rows and 41,144 tokens. It remains
immutable and excluded; no row was resumed, copied, or mixed into a scientific run.
Its verified failure-evidence backup SHA256 is
`498f0a33696cd3aed77a6d2e9f7fc02e1515fcf624f51aef023b5b12dcc65e21`.

After new explicit authorization, `ppo_final_formal_1p5b_seed42_20260721T022152Z`
ran once from update-free checkpoint-32 evaluation and succeeded with 800/800
completions and 98,018 tokens. Sampled pass@1 is 15/400 (3.75%) and independent-pool
pass@4 is 9/100 (9.0%); greedy is null/unavailable. The matching seed-42 base values
are 4.0% and 10.0%. Paired pass@1 had 8 improvements and 9 regressions; paired pass@4
had 2 and 3. No test result changed the fixed checkpoint or any scientific setting.

All 20 run checksums and the trusted checkpoint manifest SHA
`18534747eb6bb1c0945676c7490fce29c90e1f67bff939bd9318ee1101ee1952`
verified. Usage was 0.911251 GPU-hours / CNY 8.091905, peak nvidia-smi VRAM 3,933
MiB, and GPU release was 0 MiB/no process. The complete backup SHA256 is
`04fcb03b22ab74e865e2627c0e02460b62c6c731e2245d054aefe5ff6b562fc1`.

## Stage L2 GRPO seed-42 final evaluation

Stage L2 completed the fixed GRPO seed-42 checkpoint-32 held-out evaluation as
`grpo_final_formal_1p5b_seed42_20260721T034104Z`: 800/800 candidates, 94,288 exact
tokens, sampled pass@1 28/400 (7.0%), and independent pass@4 14/100 (14.0%). Its
checkpoint manifest SHA is
`c0c1a3dc04b28d8d42463009b728b3ffd144e677c35e5d3ace2749fc925fa65a`;
all 20 runtime checksums and persistent backup SHA
`97be0c2f1931690fb631dec557eb17201df12b177810c0b270192c46e6920e48`
verified. GPU release was 0 MiB/no compute process.

All Base/PPO/GRPO candidate identities match. GRPO minus Base pass@1 is +3.0 points
(bootstrap 95% CI +1.0 to +5.0) and GRPO minus PPO is +3.25 points (+1.25 to +5.5).
Both independent-pass@4 intervals span zero. This one-seed result is not a general
algorithm-superiority claim and did not affect any scientific setting.

## Stage M portfolio freeze

Portfolio v1 publishes the frozen four-run training/validation evidence and the complete seed-42 Base/PPO/GRPO final comparison. The headline sampled pass@1 is Base 4.0%, PPO 3.75%, and GRPO 7.0%. GRPO versus Base is +3.0 points with paired bootstrap 95% CI [+1.0,+5.0] and exact McNemar p=0.00754; GRPO versus PPO is +3.25 points with CI [+1.25,+5.5] and p=0.00443. Positive pass@4 deltas remain trends because their intervals cross zero.

GRPO seed-123 and PPO seed-123 final evaluations are `deferred_not_executed`. Their completed training and 64-problem checkpoint-validation evidence remains public, but no final-test value is inferred. Model caches, complete checkpoints/optimizer state, credentials, and large runtime archives remain outside GitHub and are indexed in `release/remote_artifacts.md`.

## Unique next task

After publication, the only next task is a separately authorized CPU-only Stage N GRPO-v2 design freeze. It must create a new versioned identity and may not tune against the published held-out test. No GPU task is authorized. See `docs/NEXT_TASK.md` and `docs/grpo_v2_roadmap.md`.

## Stage N GRPO-v2 CPU-only design freeze

- Active improvement branch: `improve/grpo-v2`; base HEAD `f744b7866f1bdd4380a5597957359b0a953dd686`; the Stage N commit is the commit containing this handoff. Portfolio v1 main/tag and all v1 artifacts are unchanged.
- No CUDA, model/tokenizer load, generation, Trainer, backward, optimizer, warm-start, GRPO, dev, or hidden-test execution occurred.
- New manifests: train/warmstart/dev/test = 512/256/128/400. Core split and all-v1 content/source overlap counts are zero. Public execution manifests omit gold; trusted verifier/target records live under `/root/autodl-tmp/datasets/math_rlvr/grpo_v2/trusted`.
- Strictly unseen MATH500 capacity is 3/50/65/88/94; frozen hidden-test allocation is 3/33/43/59/62 and nested subset is 3/8/10/14/15. The three Level-1 rows are included and marked diagnostic-only small-n.
- Warm-start: seed 42, 256 examples, one epoch, effective batch 16, 16 optimizer/scheduler steps, adapter-only checkpoint, then one separate 128-problem dev evaluation.
- GRPO-v2: warmstart adapter initialization, 128 updates, 512 expected microsteps, 2,048 completions, 524,288-token cap; checkpoints/dev at 32/64/96/128. Only dev selects via canonical pass, parseable, format, truncation, then earlier step.
- Hidden test remains unexecuted. Future four-model comparison is Base / old v1 GRPO42 / warmstart-only / selected v2, 700 completions each with genuine nested pass@4.
- Unique next task is Stage O: explicit GPU authorization plus the model-bound warm-start entrypoint/token-length preflight. Do not start it automatically.

## Stage O.2 warm-start capacity/runtime freeze

- Stage O.1 failure evidence is preserved: 48/256 targets exceeded 256 and one prompt exceeded 832; no truncation occurred.
- Authorized capacity-only amendment: prompt 928, active target including EOS 640, actual sequence 1,088 unchanged. Full retokenization passed 256/256; observed maxima remain 914/609/1,019.
- Guarded CLI/runtime is frozen at `python -m math_rlvr.training.warmstart`: seed42, 256 samples, one epoch, microbatch4, GA4, effective batch16, 64 microsteps, and 16 optimizer/global/scheduler steps. Static policy LoRA trainables are 4,358,144.
- Checkpoint-16 is policy-adapter-only plus optimizer/scheduler/Python/NumPy/PyTorch CPU/CUDA RNG, trainer/runtime/data cursor, full identities, and SHA inventory. GRPO receives only the adapter and source SHA and starts a fresh GRPO optimizer.
- Secondary nested pass@10 freezes 50 problems inside pass@4 (GSM8K25; MATH L1–5 2/4/5/7/7), candidates0–9, 1,000 completions/model and 4,000/four models. It is not a selection/tuning metric.
- No CUDA, model weight load, generation, Trainer, backward, optimizer, warm-start, dev, GRPO, or hidden test ran in Stage O.2. The next task requires explicit authorization for exactly one real warm-start run.

## Stage O.3 shared unbiased pass@k freeze

- The O.2 50-problem pass@10 contract is `superseded_before_any_evaluation`; its manifest moved unchanged to `configs/grpo_v2/manifests/legacy/`. No hidden-test model generation or result existed.
- Active subset is the unchanged 100-problem Stage N pass@4 manifest (GSM8K 50; MATH 3/8/10/14/15), now sampled once with n=10. Exact unbiased estimates use `1-C(10-c,k)/C(10,k)` for k=1/4/10.
- `candidate0_accuracy_all_400` is separate from 100-problem unbiased pass@1. Ledger is 1,300 completions/model and 5,200/four models.
- Only evaluation/pass@k and propagated registry identities changed. Warm-start/GRPO configs, manifests, curriculum, model, LoRA, prompt/reward/parser/verifier, and capacity remain unchanged. Stage O.3 ran no CUDA/model/generation/training/evaluation.
- Unique next task remains one separately authorized Stage O warm-start run from `docs/NEXT_TASK.md`; do not start it automatically.

## Stage P warm-start execution

- Run `warmstart_grpo_v2_seed42_20260722T051218Z` executed exactly once from `6895fa0a00c82ed0fcef12ba8514b1fc9c14b53e` and is `scientific_training_success`. Counters are 256 unique samples, one epoch, 64 batches/microsteps, 16 optimizer/global/scheduler steps, and 46,058 supervised tokens.
- All loss and grad-norm values are finite. Policy LoRA has 4,358,144 trainables; trusted optimizer inspection matched exactly 224 adapter tensors and 4,358,144 elements. No value model, trainable reward model, or base weight checkpoint exists.
- Checkpoint-16 artifact SHA is `507749d393f38690915a76228b4c53a8b5c8927d40aada9f2768a90334d892f0`; adapter SHA is `44066dd13d8cfa4f5c40f10cad705eea617c37ce2e2f85ff5407751fb5a972b9`. Runtime/postprocess backups are `6515c49d...1592` and `5f0287e4...b97e`.
- Full-command wall time was 15.324s, peak nvidia-smi VRAM 23,443 MiB, GPU-hours 0.004257, and CNY 0.0378. GPU returned to 0 MiB with no compute process.
- Base and warm-start dev-v2 were not executed: no frozen model-bound dev evaluator/CLI exists at this HEAD. Their values are unavailable, not zero. Training is not downgraded or rerun. The sole next task is the CPU-only evaluator freeze in `docs/NEXT_TASK.md`; GRPO-v2 remains unauthorized.

## Stage P.1 matched dev evaluator freeze

The CPU-only shared evaluator is frozen at config SHA `8501bfb945f85dda895d9278bb5d1d74a5d9c2c0791f9daa7cb0152d25e02528`; runtime registry canonical SHA is `fc1cbf10698528a084406adf7a88f9f64cd02141f63d5c91cb9b025d07997db2`. Fifteen evaluator/safety tests plus two directly affected registry regressions passed. The pinned tokenizer rendered all 128 dev prompts at 112--453 tokens under cap 832 without CUDA. Warm-start training/checkpoint remain immutable. The two separately authorized matched runs are now pending in strict Base then warm-start order; GRPO-v2 and hidden test remain unexecuted.

## Stage P.1 matched dev execution result

Base `base_dev_grpo_v2_seed42_20260722T060500Z` and warm-start `warmstart_dev_grpo_v2_seed42_20260722T060500Z` both completed 128/128 candidate-0 generations with matched IDs/hashes/seeds/prompts. Pass@1 was 6/128 versus 8/128 (+1.5625 pp); format 17/128 versus 23/128; parseable/valid-answer 16/128 versus 20/128. Paired transitions were 5 improved, 3 regressed, 3 unchanged-correct, 117 unchanged-wrong; bootstrap 95% CI [-2.34375,+6.25] pp and McNemar p=0.7265625. The result supports better protocol adherence and only an uncertain positive dev ability change. Both runs/checksums/backups passed and GPU returned to 0 MiB. Warm-start training remains `scientific_training_success`. The unique next task is CPU-only guarded GRPO-v2 runtime freeze; no GRPO-v2 or hidden test is authorized.

## Stage Q GRPO-v2 runtime freeze

Stage Q completed from clean implementation base `b5d52ce158fa2208fc1ee00cdbf9254b39c37bdc` on `improve/grpo-v2`. The result commit is the commit containing this handoff. Runtime registry canonical SHA is `43ef900265e37a355d7edf271384a5f7c84166a17b378034349c344228dab3fa`; the frozen GRPO config remains `059553888fdc997a5b9f214fde526d4be8c309ca84abe212c243fd74305b1b66`. Train/dev/hidden-test manifests, curriculum, model, prompt, reward, parser/verifier, LoRA, capacity, budgets and shared n=10 pass@k contract are byte-identical.

The guarded `math_rlvr.training.grpo_v2` entrypoint now binds the immutable warm-start checkpoint and adapter, creates a fresh GRPO optimizer/scheduler, enforces 128 updates/512 microsteps/2,048 completions/524,288 tokens, writes primary evidence per update before checkpoint callbacks, saves trusted adapter-only resume checkpoints at 32/64/96/128, and runs matched dev with separate ledgers. Dev-only lexicographic selection remains frozen; hidden test remains sealed. Forty-nine targeted tests plus Ruff, compileall, dry-run, environment/manifest and safety audits passed with Stage Q CUDA/model/tokenizer/generation/train/backward/optimizer counts all zero. The only next task is a separately authorized real seed-42 GRPO-v2 run using `docs/NEXT_TASK.md`.

## Stage R GRPO-v2 first attempt

Run `grpo_v2_seed42_20260726T030733Z` executed exactly once from `b738221a2b6df90ac5b2b895b27e9fac2e12525e` and failed before training because frozen prompt `math:DigitalLearningGmbH/MATH-lighteval:train:4567` is 914 tokens versus GRPO-v2 cap 832. It has 0 updates/microsteps/optimizer/global steps, 0 completions/tokens, no checkpoint/dev, and no hidden-test access. The immutable failure archive SHA is `3fa2cbb730c5a72faa83cd35172873ce367537e19fc705d890c8d9bce4748fb8`; all raw checksums passed and GPU ended at 0 MiB/no compute process. Warm-start and matched dev successes remain unchanged. The unique next task is the CPU-only capacity reconciliation in `docs/NEXT_TASK.md`; no retry is authorized.

## Stage R.1 GRPO-v2 capacity reconciliation

Stage R.1 completed CPU-only on `improve/grpo-v2`; the result commit is the commit
containing this handoff. The pinned Qwen tokenizer and exact runtime renderer replayed
512 training plus 128 dev prompts with CUDA uninitialized and zero model-weight loads,
generation, Trainer constructions or optimizer steps. Training prompt statistics are
min 109, mean 155.793, median 146, p90 184, p95 209, p99 315 and max 918; dev max is
453. The old 832 cap overflowed for two training prompts, including the historical
914-token failure row. The actual maximum is
`math:DigitalLearningGmbH/MATH-lighteval:train:4207` at curriculum 23/update 6/slot 2.

The deterministic amendment is prompt 928, completion 256 unchanged, and explicit
sequence ceiling 1,184. Final overflow and truncation counts are zero; maximum potential
combined length is 1,174 versus Qwen context 32,768. GRPO config SHA is
`ce3883b0326492b9109963e8d95496936aa3b3b8670cb9d3b4e9346f65c8cc93`, dev config SHA is `cafd9f4945a31a9befcf90ae1524107e086f0820178447e8b5767cf19c2ffa59`, runtime registry canonical SHA is
`fad035928e6fdc285ec290d295f4d481700c04ac7f5639f41d3e3ac8a0451beb` (raw `32d83b2ac2e7bb64cbab3d09cec3f2834baca0e46df416e9c407ebf7bcf3fd3b`), and audit identity is `0868361e0c79e11a7e70f267927ac27e341e48a9c08fbecdec6995da47854e31`. All data,
curriculum, hidden-test, warm-start, model/prompt/reward/parser/verifier/LoRA/sampling,
budget and pass@k identities remain unchanged. The failed Stage R run and backup remain
immutable/excluded. The unique next task is a separately authorized fresh Stage R GRPO-v2
run from update 0; no GPU work is authorized by this handoff.

## Stage R.2 GRPO-v2 fresh attempt

- Authorized start: branch `improve/grpo-v2`, HEAD
  `999faa507fbca3bbb97e3bd37253e0f2a972f45b`, clean worktree.
- Run `grpo_v2_seed42_20260726T034649Z` executed once with no retry. The reconciled
  512-training/128-dev capacity preflight passed at 928/256/1,184 with no overflow,
  truncation or hidden-test access.
- The run then stopped before training: Trainer construction returned with
  `trainer.optimizer` unset, and the fresh-optimizer audit attempted to read its
  `.state`. Status is `engineering_failure_before_training_optimizer_state_uninitialized`.
- Counters are zero updates/microsteps/optimizer/global steps, zero completions/tokens,
  and zero checkpoints/dev. No scientific result exists and this run is excluded.
- All raw checksums and failure archive SHA
  `7c4a7c367723c47c13d0b3d4f4810478196716f69a39f4c27761ef88a28d1f50`
  passed. Archive contains no checkpoint/model weights. GPU ended at 0 MiB/no process.
- Frozen config/data/curriculum/model/prompt/reward/parser/verifier/LoRA/sampling and
  budgets remain unchanged. The unique next task is CPU-only diagnosis and minimal
  repair of this initialization boundary. No retry or hidden-test execution is
  authorized.

## Stage R.3 lazy optimizer lifecycle repair

- Repair base: `improve/grpo-v2` at
  `5f40bd159c9015ecc76df61213df4d8d37e452da`; result commit is the commit
  containing this handoff.
- The old post-constructor `.state` read is removed. `optimizer=None` is valid and
  recorded as `lazy_not_initialized`.
- Exact policy-LoRA/fresh-state/scheduler auditing now runs at native
  `on_train_begin`; first state materialization, scheduler advancement and
  optimizer/global/update=1 are verified at native `on_step_end`.
- TRL/Transformers remains the sole optimizer lifecycle. Fresh runs load no SFT
  optimizer/scheduler; Accelerate wrapper identity is not a gate.
- Four focused tests, affected Ruff/compileall, dry-run, manifest validation,
  check_env and diff check passed. The only real optimizer step was one synthetic CPU
  Trainer step; CUDA/model/generation/real GRPO/dev/hidden-test counts stayed zero.
- Config, dev, registry, curriculum/data and all scientific SHA values are unchanged.
  Both failed Stage R runs remain immutable/excluded.
- 下一步直接重新授权真实GRPO-v2训练。No additional CPU preparation stage is
  required; this handoff does not itself authorize GPU execution.

## Stage R.4 GRPO-v2 scientific result

Run `grpo_v2_seed42_20260726T044303Z` completed 512/512 unique curriculum prompts,
128/128 updates, 512/512 microsteps, 128/128 optimizer/global steps, 2,048/2,048
completions and 230,675/524,288 training tokens. Dev remained separate at four
checkpoints with 512 completions and 53,609 tokens; hidden-test accesses were zero.

Dev canonical pass was 23/128, 27/128, 33/128 and 28/128 at steps 32/64/96/128.
The preregistered lexicographic rule therefore selected checkpoint-96. The warm-start
adapter SHA remained `44066dd13d8cfa4f5c40f10cad705eea617c37ce2e2f85ff5407751fb5a972b9`;
the GRPO optimizer/scheduler were newly initialized and exact policy-LoRA role checks
passed. All checkpoints are adapter-only with trusted optimizer/scheduler/RNG/counter/
cursor/prefix inventories.

Scientific artifacts and backup SHA
`af88cd652ef1ff1a23ff34a728fafb24e62c55ac68b83472575dea90c3d2a6f2`; the non-overwriting post-release archive SHA is `52ccacdccfd07259993ae3075301fdbb50ab00e628954871809ac8319e239fcb` finalized
successfully. Afterwards, the worker blocked transporting an oversized finalized
result through multiprocessing IPC. It and the stuck parent were terminated without
retry; GPU release is 0 MiB/no compute process. The scientific status is
`scientific_training_and_dev_success`; launcher status is separately
`launcher_ipc_manual_termination_after_scientific_finalization`.

The sole next task is CPU-only implementation/freeze of the four-model hidden-test
evaluator. Hidden-test generation is still unauthorized and has never run.

## Stage S.1 four-model hidden evaluator freeze

The narrow evaluator is frozen at `configs/grpo_v2/hidden_test_evaluation.json` (raw SHA `ff588378a5a6bf1331d08ad95d7311648373eb6e28cae763447d9d67941b7d22`). Roles are Base, old GRPO-v1 checkpoint-32, warmstart checkpoint-16, and selected GRPO-v2 checkpoint-96; checkpoint-128 is rejected. Their adapter/checkpoint identities were verified metadata-only. The unchanged ledger is 1,300 completions/model and 5,200 total with matched problem/candidate seeds. Primary evidence is file-backed and IPC is primitive-only and capped at 4 KiB. Nine targeted tests and four role dry-runs passed with CUDA/model/tokenizer/generation counts zero and trusted hidden gold unopened. A separately authorized one-time GPU evaluation is the only next task.

## Stage S.2 suite stop after Base finalization failure

Base run `base_hidden_grpo_v2_seed42_20260728T073339Z` completed all 1,300 frozen generations and 152,567 tokens, with exact 400 candidate-0 plus 100x10 shared-key coverage. Metric finalization then failed at `dev aggregate requires 128 rows`; no scientific Base summary was produced. The verified failure archive SHA is `532ac2854ade3374c3725410f509f6092e2508453fbd68522cf1b85c9660e215`; GPU returned to 0 MiB/no process. No retry occurred and the remaining three roles were not executed. The unique blocker is a CPU-only aggregate row-count repair and Base report recovery from immutable evidence.
