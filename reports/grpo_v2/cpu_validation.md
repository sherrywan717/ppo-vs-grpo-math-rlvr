# Stage N CPU validation

- Targeted pytest: `tests/test_grpo_v2_contract.py` — 7 passed.
- Ruff: all affected files passed.
- compileall: all affected modules passed.
- GRPO-v2 contract dry-run: passed; train/warmstart/dev/test = 512/256/128/400; nested = 100.
- Existing formal manifest validation: passed unchanged.
- Deterministic rebuild: two complete generated-tree SHA256 listings were byte-identical.
- Environment: `cuda_initialized=false`, `model_or_tokenizer_loaded=false`, generated-code execution false.
- Stage N runtime counters: tokenizer/model/CUDA/generation/Trainer/backward/optimizer/training updates = 0.
- Full pytest was intentionally not run; the authorized scope was targeted CPU validation only.
