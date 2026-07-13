.PHONY: check-env static test smoke-ppo smoke-grpo main-ppo main-grpo

check-env:
	PYTHONPATH=src python scripts/check_env.py

static:
	python -m compileall -q src scripts tests

test:
	PYTHONPATH=src python -m pytest

# Training targets are explicit and never part of setup/check/test.
smoke-ppo:
	PYTHONPATH=src python -m math_rlvr.training.ppo --config configs/smoke/ppo.yaml

smoke-grpo:
	PYTHONPATH=src python -m math_rlvr.training.grpo --config configs/smoke/grpo.yaml

main-ppo:
	PYTHONPATH=src python -m math_rlvr.training.ppo --config configs/main/ppo.yaml

main-grpo:
	PYTHONPATH=src python -m math_rlvr.training.grpo --config configs/main/grpo.yaml
