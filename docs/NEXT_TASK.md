# Next task: freeze the four-model GRPO-v2 hidden-test evaluator

Stage R.4 scientific training and matched dev are complete. Run
`grpo_v2_seed42_20260726T044303Z` finished 128 updates, 2,048 completions and four
dev evaluations. The preregistered dev rule selected checkpoint-96 (33/128).
Training must not be rerun.

The only next task is CPU-only implementation and contract freeze for one narrow
model-bound hidden-test evaluator covering Base, old GRPO-v1 seed42, warmstart-only,
and selected GRPO-v2 checkpoint-96. It must implement the already frozen 400-problem
candidate-0 ledger and shared 100-problem n=10 unbiased pass@1/pass@4/pass@10
contract. No hidden-test generation is authorized by this document.

The evaluator must also avoid returning large primary evidence through a
join-before-read multiprocessing queue. Stage R.4 scientific artifacts finalized
successfully, but its launcher required manual termination after IPC transport
blocked; GPU release was independently verified.
