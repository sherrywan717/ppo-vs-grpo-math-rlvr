# Stage S.1: four-model hidden-test evaluator freeze

Status: `cpu_validated_ready_for_separate_gpu_authorization`
Hidden-test generation/accesses: `0`
Training/resume/model loading/CUDA: `0`

## Frozen roles

| Role | Checkpoint | Checkpoint/artifact SHA256 | Adapter SHA256 |
|---|---|---|---|
| Base | none | unavailable (no adapter) | unavailable (no adapter) |
| old_grpo_v1 | checkpoint-32 | `c0c1a3dc04b28d8d42463009b728b3ffd144e677c35e5d3ace2749fc925fa65a` | `7d70a5b149cbb8c8382d6c3c5ba1b57d6da77c817147ff79ba7df2d9d7d86316` |
| warmstart_only | checkpoint-16 | `507749d393f38690915a76228b4c53a8b5c8927d40aada9f2768a90334d892f0` | `44066dd13d8cfa4f5c40f10cad705eea617c37ce2e2f85ff5407751fb5a972b9` |
| selected_grpo_v2 | checkpoint-96 | `73bb15a32911f490216be2a80eb0d112be0f79236a6d461fd81fbd0579639246` | `0ebfe5752fb066273692512bd8c3ef23bda4d58786bdfc017aa6aca75fa57080` |

All adapter roles resolve to Qwen2.5-1.5B revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, policy LoRA r16/alpha32/dropout0
on q/k/v/o, without base weights. Optimizer, scheduler, and RNG are not loaded for
evaluation. Selected GRPO-v2 rejects checkpoint-128.

## Ledger and metric contract

The unchanged public execution manifest contains 400 problems: GSM8K 200 and
MATH500 200 (levels 3/33/43/59/62). Candidate index 0 appears once for each problem.
The unchanged shared subset contains 100 problems: GSM8K 50 and MATH500 levels
3/8/10/14/15. Each shared problem has one exchangeable `generate(n=10)` batch with
candidate indices 0–9. The same problem ID, content hash, candidate index, batch seed,
sampling, prompt/parser/verifier, and token cap are used for all four roles.

Per model the ledger is 300 non-subset candidate-0 rows plus 1,000 shared candidate
rows, or 1,300 completions. The four-model total is 5,200. Candidate-0 accuracy over
all 400 problems is the primary paired binary metric. On the shared 100 problems,
pass@k reuses the frozen exact estimator
`1 - C(10-c,k)/C(10,k)` for k=1/4/10. Missing, duplicate, or fewer/more than ten
shared candidates fail closed. Hidden results cannot select a checkpoint or change
training.

## IPC and artifact contract

The worker atomically writes completion JSONL, per-problem/metric CSV/JSON, resource
summaries, reports, figures, and checksums in its run directory. It returns only
primitive `status`, `run_id`, `run_dir`, `summary_path`, counts, and failure reason;
the serialized IPC object is capped at 4 KiB. The parent reads the summary from disk.
A 1,300-row fake evidence test confirmed no completion text or token IDs travel through
the queue.

Per-model and four-model aggregate schemas are frozen in
`grpo_v2_hidden_runtime.artifact_schema()`. Real aggregate values remain unavailable
because no hidden generation ran.

## CPU verification

- Targeted hidden evaluator plus frozen O.3 pass@k tests: 20 passed.
- Warning: one unrelated legacy O.3 SHA assertion still names the pre-R.1 GRPO config SHA; it was excluded rather than changed in this narrow stage.
- Four role dry-runs: passed, 1,300 completions/model and no trusted gold open.
- Affected Ruff: passed.
- Affected compileall: passed.
- `scripts/check_env.py`: `cuda_initialized=false`, model/tokenizer not loaded.
- Existing manifest validation and `git diff --check`: passed.

Config raw SHA256:
`ff588378a5a6bf1331d08ad95d7311648373eb6e28cae763447d9d67941b7d22`.
The public hidden manifest, shared subset, pass@k contract, model, prompt, reward,
parser/verifier, sampling, training, curriculum, checkpoint selection, and capacity
identities are unchanged.

## Future commands (not executed)

Replace each `<UTC_TIMESTAMP>` once; each command requires a separate explicit GPU
execution authorization.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
python -m math_rlvr.evaluation.grpo_v2_hidden \
  --config configs/grpo_v2/hidden_test_evaluation.json \
  --role base \
  --run-dir /root/autodl-tmp/runs/math_rlvr/base_hidden_grpo_v2_seed42_<UTC_TIMESTAMP> \
  --execute --confirm-grpo-v2-hidden
```

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
python -m math_rlvr.evaluation.grpo_v2_hidden \
  --config configs/grpo_v2/hidden_test_evaluation.json \
  --role old_grpo_v1 \
  --checkpoint /root/autodl-tmp/runs/math_rlvr/grpo_formal_1p5b_seed42_20260720T031006Z/checkpoint-32 \
  --run-dir /root/autodl-tmp/runs/math_rlvr/old_grpo_v1_hidden_grpo_v2_seed42_<UTC_TIMESTAMP> \
  --execute --confirm-grpo-v2-hidden
```

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
python -m math_rlvr.evaluation.grpo_v2_hidden \
  --config configs/grpo_v2/hidden_test_evaluation.json \
  --role warmstart_only \
  --checkpoint /root/autodl-tmp/runs/math_rlvr/warmstart_grpo_v2_seed42_20260722T051218Z/checkpoint-16 \
  --run-dir /root/autodl-tmp/runs/math_rlvr/warmstart_only_hidden_grpo_v2_seed42_<UTC_TIMESTAMP> \
  --execute --confirm-grpo-v2-hidden
```

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
python -m math_rlvr.evaluation.grpo_v2_hidden \
  --config configs/grpo_v2/hidden_test_evaluation.json \
  --role selected_grpo_v2 \
  --checkpoint /root/autodl-tmp/runs/math_rlvr/grpo_v2_seed42_20260726T044303Z/checkpoint-96 \
  --run-dir /root/autodl-tmp/runs/math_rlvr/selected_grpo_v2_hidden_grpo_v2_seed42_<UTC_TIMESTAMP> \
  --execute --confirm-grpo-v2-hidden
```
