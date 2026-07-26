# Next task: CPU-only GRPO-v2 optimizer-initialization repair

Stage R.2 run `grpo_v2_seed42_20260726T034649Z` passed the complete capacity
preflight, then failed before training because Trainer construction returned with
`trainer.optimizer` unset and the runtime immediately inspected
`trainer.optimizer.state`. It has zero updates, completions, tokens, checkpoints and
dev evaluations and is excluded from science. The earlier capacity-failure run also
remains immutable/excluded.

The sole next task requires new explicit CPU-only authorization:

1. reproduce the Trainer optimizer lifecycle with a bounded fake/static test;
2. move fresh-optimizer creation or its audit to the correct lifecycle boundary;
3. prove no SFT optimizer/scheduler state is inherited;
4. prove the optimizer parameter set equals policy-LoRA trainables;
5. run only directly affected CPU tests/dry-run and preserve both failed runs.

Do not run GRPO-v2, dev evaluation or hidden test. A further GPU attempt requires a
separate authorization after a committed, backed-up, clean CPU repair.
