# GRPO-v2 training status

The first authorized seed-42 attempt, [`grpo_v2_seed42_20260726T030733Z`](grpo_v2_training/grpo_v2_seed42_20260726T030733Z/report.md), stopped before training because one frozen 914-token prompt exceeded the frozen 832-token GRPO prompt cap. Counters are 0 updates, 0 completions and 0 generated tokens. It is an engineering failure, excluded from scientific analysis, and was not retried. Warm-start and matched dev successes remain valid; hidden test remains unrun.
Stage R.1 completed the authorized CPU-only reconciliation without changing that historical run. The pinned tokenizer and exact runtime renderer audited all 512 training and 128 dev prompts. The actual maximum is 918 tokens; prompt/completion/sequence limits are now 928/256/1,184, with zero overflow and zero truncation. The guarded execute path runs this full audit before CUDA, model/adapter loading, Trainer/optimizer construction or generation. A fresh GPU run remains not authorized.

The newly authorized fresh attempt
[`grpo_v2_seed42_20260726T034649Z`](grpo_v2_training/grpo_v2_seed42_20260726T034649Z/report.md)
passed the complete capacity preflight but stopped before training because the Trainer
optimizer was still unset when the fresh-optimizer state audit ran. It also has zero
updates/completions/tokens and no checkpoint/dev result, is excluded from science, and
was not retried. Its verified failure archive SHA is
`7c4a7c367723c47c13d0b3d4f4810478196716f69a39f4c27761ef88a28d1f50`;
GPU returned to 0 MiB/no process.

The third fresh attempt,
[`grpo_v2_seed42_20260726T044303Z`](grpo_v2_training/grpo_v2_seed42_20260726T044303Z/report.md),
is the scientific training and matched-dev success: 128 updates, 2,048 completions,
230,675 training tokens, four trusted checkpoints and 512 isolated dev completions.
Dev pass@1 at steps 32/64/96/128 was 23/128, 27/128, 33/128 and 28/128; the frozen
rule selected checkpoint-96. Scientific artifacts and the verified full backup
completed before an oversized multiprocessing result blocked launcher IPC. The
completed worker and stuck parent were terminated without retry; post-release GPU
state was 0 MiB/no process. This launcher issue does not alter the immutable
scientific result.
