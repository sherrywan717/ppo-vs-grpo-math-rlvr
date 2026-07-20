# Next task: Stage H.4 formal valid-answer metric truth fix

Status: PPO seed-42 training plus recovered checkpoint validation is scientifically
complete as a transparent composite. Stage H.3 aggregation exposed one true blocker
before formal GRPO: native training `valid_answer_rate` reads the obsolete nested
`components.valid_answer` path and therefore persisted misleading zeros after reward
evidence serialization became flat.

The only next task is a bounded CPU-only correction of that existing field mapping.
Do not initialize CUDA, load a model/tokenizer, generate, train, run validation, run
baseline/final test, or execute PPO/GRPO.

Allowed scope:

- Confirm the current flat RewardResult evidence keys already persisted per completion.
- Make formal PPO/GRPO training `valid_answer_rate` use the existing flat
  `valid_answer_component` or the already canonical status mapping, preserving its
  frozen semantic definition.
- A genuinely unavailable value must be `null`, `available=false`, with a reason; it
  must never be fabricated as zero.
- Add/update only directly affected CPU/fake tests for format, parseable/wrong,
  canonical pass, and zero-denominator behavior.
- Run only those targeted tests, affected Ruff/compileall, and formal PPO/GRPO dry-runs.
- Preserve all configs, manifests, prompts, rewards, parsers, verifiers, sampling,
  budgets, historical runs, checkpoints, and checksums. Do not add a schema, guard,
  fallback, artifact type, or unrelated refactor.

After the minimal repair is committed and backed up with a clean worktree, the next
separately authorized GPU stage may be formal GRPO seed 42 using frozen config
`configs/formal_1p5b/resolved/grpo_seed_42.json` SHA256
`3371d23166d01834c67830eb8dfd51a02d4af483687b0b29e941194174099199`.

This file authorizes no GPU execution. PPO rerun/resume, seed 123, baseline, final
test, and automatic progression remain unauthorized.
