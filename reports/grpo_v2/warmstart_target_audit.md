# GRPO-v2 warm-start target audit — blocked

The offline pinned tokenizer audit covered all **256/256** frozen warm-start records. Trusted targets were rebuilt from official training rows and matched the stored target hashes exactly. Canonical gold verification, train-subset membership, zero dev/test/v1 overlap, one-pass chat templating, response boundary, completion-only labels, and EOS activation all passed. No truncation was performed.

## Blocking result

- **48/256** targets exceed the frozen 256-token target cap when active EOS is included.
- **1/256** prompts exceed the frozen 832-token prompt cap.
- Required observed minimum target cap: **609** tokens including EOS.
- Required observed minimum prompt cap: **914** tokens.
- Maximum actual combined sequence: **1019** tokens, still below the existing 1,088 combined ceiling and the tokenizer context window.

Target active-label statistics: min 55, mean 179.91, median 156.5, p90 323, p95 363, p99 497, max 609. Prompt max is 914; combined max is 1019.

Because the frozen separate caps are exceeded, the audit is a training-correctness blocker. The config was not changed, no target was truncated, and runtime/pass@10 implementation did not proceed. Exact affected records are in `warmstart_target_audit.json` and `warmstart_target_lengths.csv`.
