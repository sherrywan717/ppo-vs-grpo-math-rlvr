# Next task: recover Base hidden metrics without regeneration

Stage S.2 executed only the frozen Base command. Run
`base_hidden_grpo_v2_seed42_20260728T073339Z` persisted all 1,300 completion rows and
152,567 generated tokens, then failed metric finalization because the hidden evaluator
called a dev-only aggregator that requires 128 rows on the 400 candidate-0 rows.

Base is immutable and must not be regenerated, resumed, copied into a new run, or
mixed with new completion evidence. Old GRPO-v1, warmstart-only, and selected
GRPO-v2 were not executed because the suite stopped under the frozen failure policy.

The only next task is a separately authorized CPU-only minimal repair of the aggregate
row-count assumption plus recovery of all Base metric/report artifacts from its existing
1,300-row primary evidence. After that repair is committed and CPU-validated, new
explicit authorization is required for the remaining three frozen GPU commands. No
training, Base generation, checkpoint change, sampling change, or additional hidden
access is authorized by this document.
