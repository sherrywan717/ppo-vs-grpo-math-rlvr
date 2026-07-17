# Inherited Stage E.1 worktree audit

Audit date: 2026-07-17 UTC. Scope: 14 tracked modifications and 7 untracked files inherited from the prior Codex window. No file was reset, deleted, overwritten, or recreated from scratch.

## Classification and disposition

| File(s) | Classification | Audit result |
|---|---|---|
| `PROJECT_HANDOFF.md`, `docs/NEXT_TASK.md` | handoff/portfolio/entropy documentation | In-scope status edits; they correctly disclosed the former resume blocker but require final Stage E.1 outcome updates. |
| `reports/formal_1p5b/cli_wiring_cpu_validation.{json,md}` | handoff/portfolio evidence | Reusable inherited CPU-validation evidence; must be refreshed after exact-resume and final gates. |
| `src/math_rlvr/artifacts/manager.py` | Stage E.1 model-bound CLI wiring | Complete, minimal evaluation support for omitting an empty checkpoint directory. |
| `src/math_rlvr/evaluation/formal.py`, `evaluation/formal_cli.py`, `evaluation/formal_model_runtime.py`, `evaluation/formal_runtime.py` | Stage E.1 model-bound CLI wiring | Dual confirmation, exact evaluation config binding, base/policy-adapter assembly, delayed local-only model handling, validation/final artifact wiring are substantially complete and reusable. Portfolio metric coverage still needs final audit. |
| `src/math_rlvr/training/common.py`, `training/execution_contract.py`, `training/formal.py`, `training/formal_cli.py`, `training/grpo.py`, `training/ppo.py`, `training/trl_compat.py` | Stage E.1 model-bound CLI wiring, with resume entry points | Exact active-config routing, reserved-seed rejection, formal confirmations, prompt preflight, delayed imports, and protected PPO/GRPO dispatch are substantially complete and reusable. The explicit resume rejection is now obsolete under the newly authorized trusted-state design. |
| `src/math_rlvr/training/formal_runtime.py` | exact-resume/checkpoint implementation plus artifact wiring | Reusable frozen counters, comparison-key prefixes, same-run check, checkpoint cadence, observer restoration, and adapter/head inventory exist. Resume safety/state persistence is incomplete. |
| `src/math_rlvr/training/formal_model_runtime.py` | Stage E.1 model assembly plus exact-resume/checkpoint implementation | Reusable PPO/GRPO role assembly, adapter/head checkpoint writing, resource monitoring, and validation callbacks exist. It currently saves no optimizer/scheduler/RNG state and does not restore model/trainer/runtime state. |
| `tests/test_formal_cli.py`, `tests/test_formal_evaluation.py`, `tests/test_formal_runtime.py` | tests | In-scope inherited CLI, routing, artifact, role, and fake-runtime tests; exact-state resume and portfolio availability tests remain to be added. |

No unrelated or suspicious file was found. There is no competing second resume implementation: the existing counter/inventory layer and model-bound checkpoint writer are complementary and should be extended in place.

## Completed and incomplete work

Completed and directly reusable: exact four-run path/SHA routing, seed-2026 rejection, formal dual confirmations, offline/local-only snapshot boundary, prompt preflight, delayed PPO/GRPO/base/adapter assembly, 32-step fake orchestration, checkpoint/validation cadence, adapter-only evaluation selection, artifact/resource placeholders, and safe-base-weight rejection.

Incomplete: trusted same-run training-state inventory and pre-deserialization validation; optimizer, scheduler, Python/PyTorch/CUDA RNG, trainer/global state and online-counter persistence; actual adapter/head/state restoration; resumed dataloader/comparison-prefix continuity; full entropy definition/availability and completion-diversity schema; final refreshed CPU evidence and documentation.

## Resume blocker: actual call chain

`training.ppo` / `training.grpo` parses `--resume-checkpoint` but currently rejects it before the paid boundary. If that rejection were removed, `_execute` selects the old run directory and `execute_formal_training` restores only `resume_manifest.json` counters plus finalized JSON/CSV evidence. `_assemble_backend` nevertheless creates fresh adapters, optimizer, evidence recorder, and `FormalOnlineGuard`; it does not load checkpoint state or pass the checkpoint into `CompletedTrainerBackend`. The backend therefore expects update 1 and cannot continue from update 9/17/25. `_write_checkpoint` stores policy adapter, PPO value adapter/head, `trainer_state.json`, and a counter manifest, explicitly records `optimizer_state_included=false`, and has no optimizer/scheduler/RNG state or signed payload inventory. This is the real same-run resume blocker.

## Safety and inherited test state

`git diff --check` passed. All dirty files are text and at most 44,473 bytes. Credential-pattern scanning returned no matches; no model/cache/weight/checkpoint binary is present. The active suite remains `f6de8c555a70837d08c1e34e13a738a21ce247b3a09531df6244a2f1d3ef53bd`; the four active config SHA256 values remain `717502...`, `6776f8...`, `4ce091...`, and `a68524...` exactly as frozen.

The inherited validation report records compileall/Ruff/check_env/manifest success and pytest `418 passed, 3 deselected`; this authorized turn has not yet rerun tests. Those results are provisional until the Phase D targeted and full CPU gates complete.
