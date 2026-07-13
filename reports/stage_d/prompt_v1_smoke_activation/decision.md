# Prompt v1 smoke activation decision

Status: approved for the bounded 0.5B PPO/GRPO smoke path only.

`prompt_v1_strict_concise` is frozen byte-for-byte as the shared prompt for the
Qwen 0.5B PPO and GRPO smoke configurations. Its candidate status is
`approved_for_smoke`; its production status remains `not_approved`. This decision
does not activate v1 for main/formal 1.5B experiments, authorize a GPU run, or
authorize PPO. `prompt_v0_grpo_smoke` remains versioned and unchanged for historical
replay.

## Evidence

The matched-seed generation-only A/B run
`prompt_ab_qwen25_05b_20260713T105428Z` used the same local Qwen 0.5B revision,
problems, seeds, sampling, and 128-token completion cap for both variants. The run
completed 16 generations without training, backward, optimizer steps, checkpoints,
or model writes.

| Metric | v0 | v1 |
|---|---:|---:|
| RewardStatus | 8/8 `FORMAT_ERROR` | 6 `FORMAT_ERROR`, 2 `INVALID_EXPRESSION` |
| Complete-envelope rate | 0% | 25% |
| Nonzero within-problem reward-variance groups | 0/2 | 2/2 |
| Valid-expression rate | 0% | 0% |
| Number-usage accuracy | 0% | 0% |
| pass@1 / pass@4 | 0% / 0% | 0% / 0% |

v1 therefore improves strict-envelope compliance and supplies nonzero within-group
reward variance, which is sufficient to review the training integration in one
separately authorized smoke. It does not demonstrate correct mathematical output or
learning readiness, so it is not an approved final/production prompt.

## Frozen identity and selector scope

- v0 prompt SHA256: `20b54a2ae00ebc762a1a90a3221f5c2409c7e64d2b35fcf2c6dfaaff48a9ef4f`
- v1 prompt SHA256: `6842002e4591630a4105a4ca8fdf4cab91676b3902708ec0cb7f7b458864ecd7`
- Renderer version: `math_rlvr.prompt.chat_template.v1`
- GRPO smoke YAML SHA256 before/after:
  `3e6ea0f568c7d946a3023eb14b67988751e37b1cb692b52018faa9dbb622a398` /
  `5df5d72f71ada14a6ce903990b1b21bbd9d682ba8a05b1f77a91bc974c3872e0`
- PPO smoke YAML SHA256 before/after:
  `1db287f772f11da9fb6e69a304857b0055dde2bb0b74baec3bfb07d0d7f0b820` /
  `b888b12fb56fe356633b2d04f2c9713bb8d02c13be66fe349f60b5d40cbc1ee3`

### Authorized config diff

`configs/smoke/grpo.yaml`:

```diff
 experiment: {name: smoke-grpo-qwen-0.5b, algorithm: grpo, seed: 42}
+prompt: {version: prompt_v1_strict_concise}
```

`configs/smoke/ppo.yaml`:

```diff
 experiment: {name: smoke-ppo-qwen-0.5b, algorithm: ppo, seed: 42}
+prompt: {version: prompt_v1_strict_concise}
```

Only the prompt selector/version for `configs/smoke/grpo.yaml` and
`configs/smoke/ppo.yaml` changes. Model revision, data, reward/parser/verifier,
sampling, LoRA, seeds, batching, token/completion budgets, steps, and checkpoint
strategy remain unchanged. Main/formal configurations remain unactivated.

## Fairness and reporting

PPO and GRPO smoke runs must resolve the same `prompt_version`, `prompt_sha256`, and
`renderer_version`. For the same `MathProblem`, their shared renderer output must be
byte-identical. Resolved config, run manifest, and later reports must persist all
three identity fields. A mismatch fails preflight because it affects comparison
fairness and report truthfulness.

This is a smoke-only selector decision. The next executable stage is one newly and
separately authorized GRPO single-update smoke. PPO remains unauthorized and must not
start automatically.
