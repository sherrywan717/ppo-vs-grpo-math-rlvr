# Stage E formal 1.5B PPO versus GRPO experiment plan

Status: CPU-only configuration frozen. No 1.5B weights, tokenizer, CUDA context,
generation, Trainer, backward pass, or optimizer step was used in this stage.

The completed 0.5B matched pilot validates execution, checkpoint safety, evidence
alignment, and rough resource cost. It is not the final benchmark and supplies no
algorithm-effect conclusion.

## Approved four-run amendment

The original six-run decision remains preserved at commit `499fea9f6de6f991229f15b949e23e63c496e6cc`. The approved amendment activates only seeds 42 and 123: PPO42, GRPO42, seed-42 review, GRPO123, PPO123, four fixed step-32 final evaluations, then CPU aggregation. Seed-2026 descriptors remain byte-identical and are `reserved_not_scheduled`; they are excluded from the active queue, costs, and statistics. The active-suite canonical SHA256 is `f6de8c555a70837d08c1e34e13a738a21ce247b3a09531df6244a2f1d3ef53bd`. Two seeds permit raw values, mean, sample SD, paired delta, and problem-level bootstrap intervals, but not statistical-significance or general-superiority claims.

## Frozen identities

The formal model is `Qwen/Qwen2.5-1.5B-Instruct` at exact revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Stage E queried only the official model
API and pinned `config.json` metadata; it downloaded no weights. Every future process
must resolve this revision into a local snapshot and use `local_files_only=true` with
both Hugging Face and Transformers offline modes.

| Identity | Frozen value |
|---|---|
| Prompt | `prompt_v2_formal_math` |
| Prompt SHA256 | `89e459da827474d9bcc66e4407b06b5f8a968ce10d0be92e830c59fd9830a994` |
| Renderer | `math_rlvr.prompt.chat_template.v1` |
| Reward | `shaped_v3_domain` |
| Reward SHA256 | `b9eda9520bb0271e28f6c209db85a408cdc0a65c2d403871b2b0fcc06e06a463` |
| Parser | `strict_completion_parser_v1` / `655c30f2…ba0ad` |
| Verifier router | `gsm8k_math_domain_router_v1` / `ac360315…886fd` |
| GSM8K verifier | `gsm8k_exact_numeric_v1` / `91f9de47…4b50` |
| MATH verifier | `math_verify_equivalence_v1` / `0a4fb547…efa7` |
| Data registry | `d7c53f61…e7393` |
| Training schedule | `a4b3745e…8b6ee` |

Policy LoRA is identical for PPO and GRPO: rank 16, alpha 32, dropout 0, and
q/k/v/o projections. Pinned Qwen metadata gives hidden size 1,536, 28 layers, and a
256-wide K/V projection. Static calculation therefore gives 4,358,144 policy-LoRA
parameters, 0.282315% of the published 1,543,714,304 base parameters.

PPO necessarily has a second base snapshot for the value role. Its rank-8 q/v LoRA
has 1,089,536 trainable parameters and its scalar head has 1,537, for 1,091,073 value
trainables. PPO's optimizer union is 5,449,217 parameters; GRPO's is 4,358,144.
Policy/value trainables are disjoint. Reference and verifier reward roles are frozen
and parameter-free from the optimizer. Checkpoints must remain adapter/head-only; the
PPO value memory and files are reported as an algorithm-required difference, never as
a matched architecture.

## Data protocol and leakage audit

Training is GSM8K train 64 plus MATH train 64. Validation is 32 plus 32. Fixed test is
GSM8K test 200 plus MATH500 200, with deterministic 50+50 pass@4 subsets. File,
ordered-ID, ordered-content, source revision, category, and difficulty evidence lives
in `configs/formal_1p5b/data_registry.json`.

No normalized problem hash overlaps train, validation, GSM8K test, or MATH500 test.
Train and validation also have zero overlap with all 500 MATH500 records. Pass@4
manifests are intentional subsets of their parent tests. MATH500 test is stratified
40 per Level 1–5; its pass@4 subset is 10 per level. The selected MATH training slice
contains Levels 1–3 only (16/24/24).

The historical validation manifest records `source_split=validation`, although its
rows physically originate from the source datasets' train splits. Stage E preserves
that file and discloses the derived provenance as `physical_source_split=train` and
`selection_split=validation`. Local Arrow SHA and row-index checks reproduce all 592
frozen manifest records. This provenance wording is a documented warning, not hidden
or rewritten evidence.

To avoid a domain-order confound, each of 32 updates consumes two GSM8K then two MATH
problems, preserving original order within each domain. Both algorithms bind the same
128 ordered IDs and the same 512 `problem_id::generation_index` keys.

Test outputs are never used to change prompt, reward, sampling, hyperparameters, or
checkpoint choice. Step 32 is fixed for final evaluation before any execution.

## Domain-aware reward audit

The Countdown reward was not reused blindly. Formal scalar reward is:

- 0.05 for one usable answer block;
- 0.05 for the strict two-block protocol;
- 0.10 when the domain canonical verifier accepts a semantically valid answer;
- 0.80 only for canonical `VERIFIED_PASS`.

Countdown exact-number-usage is absent. GSM8K uses exact numeric comparison and MATH
uses `math-verify==0.9.0` equivalence. PPO and GRPO share one reward selector, weights,
and SHA. Training may optimize the shaped scalar, while pass@1, pass@4, and formal
correctness are computed only from canonical verifier status. `INFRA_ERROR` remains
fail-closed; missing nonessential telemetry remains nullable with a reason.

## Matched 32-update training contract

Every active algorithm/seed run uses seed 42 or 123; 128 unique prompts; four
responses per prompt; 512 completions; maximum completion length 256; 131,072 generated
tokens; temperature 0.8; top-p 0.95; 32 optimizer/global steps; and checkpoints at
steps 8, 16, 24, and 32. Automatic retries are zero.

| Derived TRL 0.24.0 value | PPO | GRPO |
|---|---:|---:|
| Dataset records | 512 prompt-major episodes | 128 unique prompts |
| Rollout/generation batch | 16 | 16 |
| Per-device microbatch | 4 | 4 |
| Gradient accumulation | 4 | 4 |
| Responses per prompt | 4 by deterministic episode repetition | `num_generations=4` |
| PPO epochs / minibatches | 1 / 1 | not applicable |
| Microsteps | 4 per update | 4 per update, 128 total |
| Outer/optimizer/global steps | 32 / 32 / 32 | 32 / 32 / 32 |
| Completions / token cap | 512 / 131,072 | 512 / 131,072 |
| Checkpoints | 8, 16, 24, 32 | 8, 16, 24, 32 |

PPO never receives `num_generations`; one episode produces one response. GRPO freezes
`shuffle_dataset=false`, `drop_last=true`, and worker count zero so its four-completion
groups follow the same problem order. The original 32-update proposal therefore
requires no scientific-budget revision.

## Baseline, validation, and final evaluation

The untrained pinned base model is evaluated once for each seed and shared as the PPO
and GRPO baseline. For each seed, pass@1 covers 400 test problems and pass@4 covers
the frozen 100-problem subset, for 800 completions. Each step-32 PPO/GRPO checkpoint
uses the identical prompt, temperature, top-p, seed, max length, and manifests.

Validation evaluates all 64 validation problems at steps 8, 16, 24, and 32. It is a
learning-signal diagnostic and cannot switch the final checkpoint. The final test is run only after all four active training runs and always evaluates step 32.

Saved evaluation metrics include pass@1/pass@4, format and valid-answer rates,
canonical correctness, status distribution, per-domain results, MATH500 Levels 1–5,
completion length, truncation, reward distribution, each problem/seed result, and
paired pre/post deltas. Aggregation reports every seed, mean, sample SD, and a paired
problem-level 10,000-resample bootstrap 95% interval. Two seeds are deliberately modest and do not justify a statistical-significance claim.

## Staged future execution order

Each numbered GPU stage requires a new explicit authorization and stops afterward.

1. Pinned 1.5B CUDA/model-load sanity.
2. Shared untrained baseline evaluation.
3. PPO seed 42 training.
4. GRPO seed 42 training.
5. CPU/validation review of seed 42 learning signal and checkpoint validity.
6. GRPO seed 123 then PPO seed 123, only if both seed-42 paths produced valid updates and checkpoints.
7. Frozen step-32 final test for all four active checkpoints.
8. CPU-only aggregation, error analysis, case studies, and final report.

No stage inherits another run's checkpoint. A run has one attempt, independent run ID,
four independent checkpoints, and its own full persistent backup.

## Static resource plan

These are planning estimates, not measured 1.5B results. They scale the measured 0.5B
pilot by 32 updates and the pinned 1.5B policy/value structure. Resource-only ceilings
may be revised after the separately authorized sanity run; completions, tokens, seeds,
and updates may not.

| Run | Expected time | Ceiling | Expected VRAM | VRAM ceiling | Expected GPU-h / CNY | Ceiling GPU-h / CNY |
|---|---:|---:|---:|---:|---:|---:|
| Each PPO seed | 52.7 min | 110 min | 34 GiB | 55 GiB | 0.8783 / 7.80 | 1.8333 / 16.28 |
| Each GRPO seed | 36.1 min | 75 min | 11 GiB | 24 GiB | 0.6017 / 5.34 | 1.2500 / 11.10 |
| Four active training runs | 2.96 h | 6.1667 h | — | — | 2.96 / 26.28 | 6.1667 / 54.76 |

Evaluation planning uses 8 GiB expected / 16 GiB ceiling because it loads one model
without training state. A conservative initial plan is 25/50 minutes per 800-completion
baseline or final-evaluation seed, and 5/10 minutes per 64-problem validation pass.
Across one sanity, two baseline seeds, 16 checkpoint-validation passes, and four final evaluations, the all-stage expected total is approximately 6.8767 GPU-hours / CNY 61.06; the planning ceiling is 14.0 GPU-hours / CNY 124.32 at CNY 8.88/GPU-hour.
These evaluation estimates must be replaced by measured sanity/baseline throughput
before authorizing the four-run training suite.

## Evidence and claim boundary

Every run stores resolved identities, full completion evidence, per-update training and
validation metrics, required finite losses/counters, resource timelines, adapter/head
checkpoint inventories and hashes, and reproducible plots. Exact file requirements are
in `artifact_checklist.md`; plot regeneration is in `scripts/plot_formal_results.py`.

Stage E freezes a fair, auditable experiment capable of supporting a portfolio result.
It does not itself show that PPO or GRPO learns, generalizes, or outperforms the other.

## Resolved training descriptors

| Run | SHA256 |
|---|---|
| PPO seed 42 | `717502aa665e9d5ef967e04a5ab27aa53329ccb061bda228db3c715f4dab967b` |
| PPO seed 123 | `a68524e85e427e335abf6447aa2cc391686fd3aa4da6d42efb0e522beec1a0b3` |
| PPO seed 2026 (reserved_not_scheduled) | `b270b594fb8463fb7a0f62875840ea4a574e9359c986e5b51c26aafae07428db` |
| GRPO seed 42 | `6776f8894e9ac725a39748b06b57b62782cea2dab61faf51fd3cc3ceb5ae58bf` |
| GRPO seed 123 | `4ce0918f7284220c36555b9f23db181354168ebe252d7244ac3ac9587be236fa` |
| GRPO seed 2026 (reserved_not_scheduled) | `02479719cce9409cef89d162f36bedc20ca352c216804168d8fe7ae52545f5df` |

Each descriptor binds its seed, algorithm, template hash, schedule SHA, and ordered-ID
SHA. No runtime CLI seed override is accepted. The CPU-validated multi-update runtime is implemented in this amendment stage, but model download, CUDA sanity, evaluation, and training each remain separately unauthorized.
