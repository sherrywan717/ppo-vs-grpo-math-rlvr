# GRPO single-update smoke diagnostic — FAILURE

- Run ID: `grpo_single_update_qwen25_05b_20260713T050407Z`
- Primary failure: `ValueError: unexpected model checkpoint` during guarded trainer construction.
- Secondary artifact failure: non-serializable BudgetGuard `clock` callable.
- Prompts/completions/tokens: 2 / 0 / 0
- Microsteps/optimizer/global step: 0 / 0 / 0
- GPU peak: 0 MiB; GPU released; no residual compute process.
- Wall time: 4.940 s; GPU-hours: 0.00137213; cost: ¥0.012184.
- Checkpoint: none.
- This is a failed single-update smoke diagnostic, not an experiment result.

## Follow-up

The CPU-only fix is recorded by the commit containing this note with subject `fix: validate pinned snapshot and serialize budget state`. It introduces a canonical pinned-snapshot validator and primitive-only `BudgetGuard.snapshot()`; it does not alter this run's failure status, metrics, or immutable full-run backup.
