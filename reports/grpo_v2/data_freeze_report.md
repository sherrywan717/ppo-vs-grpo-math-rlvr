# GRPO-v2 data freeze

All selections were frozen before model execution with seed 42 and gold-independent SHA-256 ranking. Public execution manifests omit gold and solutions; trusted verifier records are stored outside Git at `/root/autodl-tmp/datasets/math_rlvr/grpo_v2/trusted`.

## Counts

- train_v2: 512 (GSM8K 256; MATH L1/L2/L3 64/96/96)
- warmstart_v2: 256 declared subset (GSM8K 128; MATH 32/48/48)
- dev_v2: 128 (GSM8K 64; MATH 16/24/24)
- test_v2_hidden: 400 (GSM8K 200; MATH500 3/33/43/59/62)
- nested pass@4 subset: 100 (GSM8K 50; MATH500 3/8/10/14/15)

## Capacity amendment

MATH500 has 500 records. V1 observed 200; after strict hash exclusion the available per-level counts are **3/50/65/88/94**. Equal 40-per-level allocation is impossible because only three unseen Level-1 records remain. The preregistered unequal allocation **3/33/43/59/62** includes every unseen Level-1 record and preserves strict decontamination; this is a pre-run capacity amendment, not result-driven selection. L2-L5 use only revision/split/source-ID/namespace/seed in their selection keys. The nested MATH allocation is **3/8/10/14/15** under an independent namespace.

Every core off-diagonal overlap in [the matrix](data_overlap_matrix.csv) is zero. Warmstart-within-train and pass4-within-test are intentional declared subset relations. Level 1 is `diagnostic_only_small_n`, reported as integer numerator/3 with a Wilson interval and never used as a headline or checkpoint-selection result.
