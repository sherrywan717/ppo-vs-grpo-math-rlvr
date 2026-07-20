# Stage H.4 valid-answer metric mapping fix

## Outcome

The formal training `valid_answer_rate` is reporting telemetry only. It is persisted in
training metric JSONL/CSV and consumed by reports/plots; it is not read by scalar reward,
advantage/return, policy/value/total loss, optimizer, scheduler, checkpoint selection,
early stopping, or any training stop condition.

The obsolete `components.valid_answer` lookup was replaced with the existing flat
`RewardEvaluation.to_dict()["valid_answer_component"]` evidence. The metric definition
version is `formal_domain_valid_answer_component_v1`:

- numerator: completion rows whose flat `valid_answer_component` is greater than zero;
- denominator: all completion rows in the update;
- positive source states: the extracted-answer verifier probe returned `wrong_answer`
  or `verified_pass`;
- excluded probe states include `format_error`, `parse_error`, `invalid_expression`,
  `invalid_number_usage`, `resource_limit`, and `infra_error`;
- zero denominator is `null`, `available=false`, `reason=zero_denominator`;
- missing source evidence is `null`, `available=false`,
  `reason=valid_answer_component_missing`.

This metric is not the same as canonical `parseable_rate`: the formal reward probes an
extracted answer separately, so a canonical `format_error` completion can still have a
positive valid-answer component. The persisted definition records that status scope and
raw source field rather than conflating the two metrics.

## Truthfulness and training invariance

The runtime recomputes the metric and its numerator/denominator/definition metadata from
completion evidence and rejects a contradictory aggregate during finalization. It does
not change the reward policy or scalar, Trainer log inputs, loss, advantage/return,
optimizer/scheduler inputs, sampling, budgets, or any frozen scientific identity. PPO
and GRPO call the same aggregation function.

The historical native false zeros in
`ppo_formal_1p5b_seed42_20260719T131800Z` remain immutable. Its checksums file SHA256 is
`43295b905f4175a41de21cd41e71e1e42d687c80a411af0421f91ecc3133e372` before and after
this repair. The final PPO composite report was already derived from immutable canonical
primary evidence with an explicit disclosure, so no PPO rerun or artifact rewrite is
needed; its status remains `scientifically_complete_with_recovered_validation`.

## CPU verification

- 44 targeted metric, reward, and formal-runtime tests passed; 0 failed.
- Affected-file Ruff passed.
- Affected-module compileall passed.
- Formal PPO and GRPO dry-runs passed and both reported `no training started`.
- Full pytest was intentionally not run under the bounded Stage H.4 scope.
- CUDA/model/tokenizer/generation/Trainer/backward/optimizer counts were all zero.

Frozen PPO42 config, GRPO42 config, active-suite raw/canonical identities, and historical
PPO checksums did not change. No blocker remains in this field mapping; formal GRPO seed
42 still requires separate explicit GPU authorization.
