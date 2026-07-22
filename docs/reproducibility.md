# Reproducibility guide

The complete public procedure is in [REPRODUCIBILITY.md](../REPRODUCIBILITY.md). The core principle is identity-first execution: exact config and suite hashes, a pinned local-only model revision, separate training/validation/test ledgers, trusted same-run checkpoints, and primary evidence written before checkpoint callbacks.

For a quick CPU-only audit:

```bash
PYTHONPATH=src python scripts/check_env.py
PYTHONPATH=src python scripts/validate_manifests.py
python scripts/build_portfolio_v1.py
```

For real training/evaluation templates, safety restrictions, and public/private artifact boundaries, use the root guide. Running those templates is not implied authorization for paid GPU work.
