# GRPO-v2 prompt/context capacity amendment

Status: **CPU validated; no GPU run authorized or executed.** This is a `post_freeze_prompt_context_capacity_reconciliation`.

## Cause and historical truth

The immutable Stage R attempt `grpo_v2_seed42_20260726T030733Z` failed before Trainer construction at curriculum position 83 / update 21 / slot 2: `math:DigitalLearningGmbH/MATH-lighteval:train:4567` rendered to 914 tokens, above the stale GRPO-v2 cap of 832. It produced zero updates, completions and training tokens and remains excluded from scientific statistics. Its verified failure backup SHA256 is `3fa2cbb730c5a72faa83cd35172873ce367537e19fc705d890c8d9bce4748fb8`.

## Full pinned-tokenizer audit

The exact runtime renderer and pinned Qwen tokenizer replayed all **512 training** and **128 dev** prompts without truncation. Training lengths were min 109, mean 155.793, median 146.0, p90 184, p95 209, p99 315, max 918. Dev lengths were min 112, mean 151.922, median 145.0, p90 180, p95 201, p99 276, max 453.

The actual maximum is `math:DigitalLearningGmbH/MATH-lighteval:train:4207` at curriculum position 23 / update 6 / slot 2 with **918 tokens**. The old 832 cap overflowed for 2 prompts; 928 overflowed for 0; the final cap overflows for 0. Maximum prompt-plus-256 potential length is 1,174.

## Deterministic amendment

`new_prompt_cap = max(928, ceil(max_observed / 32) * 32)` gives **928**. The completion cap remains **256**. The explicit generation sequence ceiling becomes **1,184 = 928 + 256**. Qwen's frozen context window is 32,768, leaving 31,584 tokens of ceiling margin. No prompt is truncated, deleted, replaced, shortened, filtered or reordered.

| Identity | Old | New |
|---|---:|---:|
| `max_prompt_length` | 832 | 928 |
| `max_completion_length` | 256 | 256 |
| `max_sequence_length` | implicit/stale | 1,184 |
| GRPO config SHA256 | `059553888fdc997a5b9f214fde526d4be8c309ca84abe212c243fd74305b1b66` | `ce3883b0326492b9109963e8d95496936aa3b3b8670cb9d3b4e9346f65c8cc93` |
| dev config SHA256 | `8501bfb945f85dda895d9278bb5d1d74a5d9c2c0791f9daa7cb0152d25e02528` | `cafd9f4945a31a9befcf90ae1524107e086f0820178447e8b5767cf19c2ffa59` |
| runtime registry canonical SHA256 | `43ef900265e37a355d7edf271384a5f7c84166a17b378034349c344228dab3fa` | `fad035928e6fdc285ec290d295f4d481700c04ac7f5639f41d3e3ac8a0451beb` |
| runtime registry raw SHA256 | `c8f26679c646a4637cbb053eccd28071388253f76fb3830ba1380b9b96dc9f87` | `32d83b2ac2e7bb64cbab3d09cec3f2834baca0e46df416e9c407ebf7bcf3fd3b` |

## Unchanged scientific identity

Train/dev/hidden-test manifests, curriculum bytes/order, warm-start checkpoint and adapter, model revision, prompt bytes/renderer semantics, reward, parser, verifier, LoRA, sampling, seed, completion cap, 128 updates, 512 microsteps, 2,048 completions, 524,288-token budget, checkpoint/dev cadence and shared unbiased n=10 pass@k contract are unchanged. Hidden-test accesses were 0.

## Runtime ordering

The guarded CLI now resolves the pinned local snapshot and runs the complete 512+128 tokenizer audit before the model-bound supervisor. Therefore it occurs before CUDA initialization, model/adapter loading, Trainer construction, optimizer initialization and generation. The delayed worker retains per-row prompt and sequence checks as defense in depth.

## Reproducible evidence

- [Audit JSON](prompt_capacity_audit.json)
- [Audit CSV](prompt_capacity_audit.csv)
- [By-slice CSV](prompt_capacity_by_slice.csv)
- [Length histogram](figures/prompt_capacity_histogram.png)
- [Domain/level maxima](figures/prompt_capacity_by_domain_level.png)
- [Curriculum-position plot](figures/curriculum_prompt_length.png)
