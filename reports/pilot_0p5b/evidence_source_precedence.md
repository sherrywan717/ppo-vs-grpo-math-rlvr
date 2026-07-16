# Matched pilot evidence-source precedence

For recovery and scientific-status decisions, evidence is authoritative in this order:

1. Complete-run raw manifests, counters, metrics, and completion evidence.
2. Checkpoint inventory and file SHA256 values.
3. AutoDL persistent archive and archive SHA256.
4. Git-safe run summary, assessment, completions, and metrics.
5. Committed run evidence.
6. `run_registry.csv`.
7. `suite_recovery_assessment.md`.

Derived indexes cannot override checksum-verified primary evidence. Missing values
are recorded as unavailable; they are never reconstructed from a derived index.
