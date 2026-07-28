# Next task: execute the frozen four-model GRPO-v2 hidden test

Stage S.1 froze the only model-bound hidden-test evaluator at
`math_rlvr.evaluation.grpo_v2_hidden`, config
`configs/grpo_v2/hidden_test_evaluation.json`, raw SHA
`ff588378a5a6bf1331d08ad95d7311648373eb6e28cae763447d9d67941b7d22`.
CPU gates and all four role dry-runs passed without opening trusted hidden gold,
loading a model/tokenizer, initializing CUDA, or generating.

The sole next task requires a new explicit GPU authorization to execute Base, old
GRPO-v1 checkpoint-32, warmstart-only checkpoint-16, and selected GRPO-v2
checkpoint-96 exactly once each. Each role has 400 candidate-0 keys and a shared
100-problem n=10 batch, totaling 1,300 completions/model and 5,200 overall. The
candidate key/seed schedule is identical across roles. Candidate-0 accuracy over 400
problems is distinct from unbiased pass@1/@4/@10 over the shared 100 problems.

The worker writes completion payloads to files and returns only a primitive status,
paths, counts, and failure reason through IPC. Hidden test must not alter training,
checkpoint selection, prompt/reward/sampling, or trigger another attempt. Stage S.1
does not authorize GPU execution.
