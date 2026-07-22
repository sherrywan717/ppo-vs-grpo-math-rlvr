# Warm-start target audit after capacity amendment

The original Stage O.1 failure remains preserved. After the authorized 928/640/1,088 capacity-only amendment, all **256/256** records were rebuilt and retokenized with the pinned local tokenizer. Prompt over-cap, target over-cap, combined over-cap, truncation, missing EOS, invalid envelope, gold mismatch, prompt-label leakage, target-label mismatch, dev/test overlap, and source/hash/order drift are all **0**.

Active target tokens including EOS: p95 **363**, p99 **497**, max **609**. Prompt max is **914** and actual combined max is **1019**. No input was truncated or rewritten.
