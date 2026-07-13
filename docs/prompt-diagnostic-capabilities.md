# Prompt diagnostic capability manifest

Schema `prompt-ab-evidence-contract-v1` requires these booleans:

- `paired_artifacts_supported`
- `group_rewards_supported`
- `zero_advantage_groups_supported`
- `allocator_evidence_supported`
- `failure_backup_supported`
- `post_worker_gpu_verification_supported`
- `cross_file_consistency_supported`

The guarded CLI reads the checked-in manifest before local snapshot resolution or any
delayed runtime import. Missing, extra, false, or incorrectly typed capability fields
reject execution. This manifest describes implemented, tested evidence behavior; it is
not a constant bypass, execution authorization, or permission to activate v1.

Lifecycle: `preflight_rejected` creates no run; an accepted request creates a unique run
and one spawned worker; worker success becomes `worker_complete`; parent exit/GPU checks
lead to `pending_backup`; only verified backup plus cross-file consistency leads to
`success`. Any runtime, artifact, consistency, or backup error produces `failure`, a
primitive fallback record when needed, and a verified `-failure` archive whenever the
run directory exists.
