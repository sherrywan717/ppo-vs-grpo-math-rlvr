# Next task: CPU-only freeze the matched dev-v2 evaluator

Stage P warm-start training is complete and must not be rerun. The sole blocker is that commit `6895fa0a00c82ed0fcef12ba8514b1fc9c14b53e` has no frozen model-bound dev-v2 evaluator or CLI, so Base and warm-start matched dev evaluations are `not_executed_evaluator_unavailable`.

The next separately authorized task is CPU-only: implement and freeze one guarded evaluator for the unchanged 128-problem `dev_v2` manifest, with identical problem order, prompt/parser/verifier, single-candidate sampling and per-problem seeds for Base and checkpoint-16. It must dry-run without model/CUDA and must not run either evaluation. After that, a new explicit GPU authorization is required for exactly the two matched evaluations.

Authoritative warm-start input for that future evaluator:

- run: `warmstart_grpo_v2_seed42_20260722T051218Z`
- checkpoint: `/root/autodl-tmp/runs/math_rlvr/warmstart_grpo_v2_seed42_20260722T051218Z/checkpoint-16`
- checkpoint artifact SHA256: `507749d393f38690915a76228b4c53a8b5c8927d40aada9f2768a90334d892f0`
- policy adapter SHA256: `44066dd13d8cfa4f5c40f10cad705eea617c37ce2e2f85ff5407751fb5a972b9`

Do not start GRPO-v2, hidden test, Base dev, warm-start dev, another warm-start, or any other GPU task automatically.
