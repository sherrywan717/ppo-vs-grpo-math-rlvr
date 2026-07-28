# Remote-only artifacts

GitHub contains the Git-safe evidence layer. The following large runtime artifacts remain on the AutoDL persistent volume and are deliberately excluded from Git and the static portfolio package. Paths are provenance records, not portable download URLs.

| Run ID | Remote archive | Archive SHA256 | Checkpoint-32 artifact manifest SHA256 | Why excluded |
|---|---|---|---|---|
| `baseline_formal_1p5b_seed42_20260718T125833Z` | `/root/autodl-fs/math-rlvr-backups/baseline_formal_1p5b_seed42_20260718T125833Z.tar.gz` | `77105f38c67ecb773edc54cedb63fe489df1298155d2eabbe0cf07e7b7cd5a13` | not_applicable | Full generation runtime archive |
| `baseline_formal_1p5b_seed123_20260718T133624Z` | `/root/autodl-fs/math-rlvr-backups/baseline_formal_1p5b_seed123_20260718T133624Z.tar.gz` | `e473075db8123664c13a3d77e8c9960be108fe881aa5deef88c04660bff2edf0` | not_applicable | Full generation runtime archive |
| `ppo_formal_1p5b_seed42_20260719T131800Z` | `/root/autodl-fs/math-rlvr-backups/ppo_formal_1p5b_seed42_20260719T131800Z.failure.tar.gz` | `f63812afed44cdc9f0fcafdf0931454548da1a4ce145840ebf91bb6fa5a6d7c5` | `18534747eb6bb1c0945676c7490fce29c90e1f67bff939bd9318ee1101ee1952` | Full checkpoint/optimizer/RNG runtime state; original post-training status remains immutable |
| `grpo_formal_1p5b_seed42_20260720T031006Z` | `/root/autodl-fs/math-rlvr-backups/grpo_formal_1p5b_seed42_20260720T031006Z.tar.gz` | `b584363595f99c1d3b61a7b6cc088cdda7ac38a29169058df7b30cd38bea5023` | `c0c1a3dc04b28d8d42463009b728b3ffd144e677c35e5d3ace2749fc925fa65a` | Full checkpoint/optimizer/RNG runtime state |
| `grpo_formal_1p5b_seed123_20260720T035927Z` | `/root/autodl-fs/math-rlvr-backups/grpo_formal_1p5b_seed123_20260720T035927Z.tar.gz` | `e78eb0719bc93c1076bd06e50037cc453cbaa5103cf1e1fbfc9e8151212e521a` | `25cfad2d234530e7d0e17ea7156d3c98fb8bd96ce2d09bbf49ab9a6d026b343a` | Full checkpoint/optimizer/RNG runtime state; final test deferred |
| `ppo_formal_1p5b_seed123_20260720T043732Z` | `/root/autodl-fs/math-rlvr-backups/ppo_formal_1p5b_seed123_20260720T043732Z.tar.gz` | `689924eaa4392a4806f9d1adaa2bbf890b76d6813a6edfeafc2ca50213bc63c0` | `61b0364c79317bec1391e1087d680c8eced17c7ec66190254534ef887ba650b1` | Full checkpoint/optimizer/RNG runtime state; final test deferred |
| `ppo_final_formal_1p5b_seed42_20260721T022152Z` | `/root/autodl-fs/math-rlvr-backups/ppo_final_formal_1p5b_seed42_20260721T022152Z.tar.gz` | `04fcb03b22ab74e865e2627c0e02460b62c6c731e2245d054aefe5ff6b562fc1` | `18534747eb6bb1c0945676c7490fce29c90e1f67bff939bd9318ee1101ee1952` | Full final-evaluation runtime archive |
| `grpo_final_formal_1p5b_seed42_20260721T034104Z` | `/root/autodl-fs/math-rlvr-backups/grpo_final_formal_1p5b_seed42_20260721T034104Z.tar.gz` | `97be0c2f1931690fb631dec557eb17201df12b177810c0b270192c46e6920e48` | `c0c1a3dc04b28d8d42463009b728b3ffd144e677c35e5d3ace2749fc925fa65a` | Full final-evaluation runtime archive |

The canonical Hugging Face snapshot at `/root/autodl-tmp/cache/huggingface` is also excluded. It contains the pinned Qwen base weights and is reproducible from the public repository ID plus revision; individual multi-gigabyte weight hashes are intentionally not duplicated in Git. No archive or model weight is copied into the GitHub repository or portfolio tarball.
## GRPO-v2 remote-only artifacts

These artifacts contain adapters, optimizer/scheduler/RNG state, complete completion
ledgers, or full runtime evidence and therefore remain outside GitHub.

| Run or aggregate | Remote archive | SHA256 | Key checkpoint/artifact identity | Why excluded |
|---|---|---|---|---|
| Warm-start seed42 | `/root/autodl-fs/math-rlvr-backups/warmstart_grpo_v2_seed42_20260722T051218Z.postprocess.tar.gz` | `5f0287e4e30f94cb45645fb7f12eec728ea74be68195e3fa8d6904adca85b97e` | checkpoint artifact `507749d393f38690915a76228b4c53a8b5c8927d40aada9f2768a90334d892f0` | Full adapter plus optimizer/RNG state |
| GRPO-v2 training | `/root/autodl-fs/math-rlvr-backups/grpo_v2_seed42_20260726T044303Z.postrelease.tar.gz` | `52ccacdccfd07259993ae3075301fdbb50ab00e628954871809ac8319e239fcb` | selected checkpoint-96 manifest `73bb15a32911f490216be2a80eb0d112be0f79236a6d461fd81fbd0579639246` | Four full trusted resume checkpoints and runtime evidence |
| Base hidden original | `/root/autodl-fs/math-rlvr-backups/base_hidden_grpo_v2_seed42_20260728T073339Z.failure.tar.gz` | `532ac2854ade3374c3725410f509f6092e2508453fbd68522cf1b85c9660e215` | immutable 1,300-row primary evidence | Original launcher-failure archive |
| Base hidden recovery | `/root/autodl-fs/math-rlvr-backups/base_hidden_grpo_v2_seed42_20260728T073339Z.metric_recovery.tar.gz` | `72dbb79ecf47549242a8c27229c01bebef5db3195e575ef489f148a7e641ae07` | recovered metric finalization only | Non-overwriting supplemental recovery |
| Old GRPO-v1 hidden | `/root/autodl-fs/math-rlvr-backups/old_grpo_v1_hidden_grpo_v2_seed42_20260728T083153Z.tar.gz` | `7b29b77362c39f16a04bd960025024bb70b4f63831732066b49e8e833d7f2bc7` | old checkpoint manifest `c0c1a3dc04b28d8d42463009b728b3ffd144e677c35e5d3ace2749fc925fa65a` | Full 1,300-completion runtime archive |
| Warm-start-only hidden | `/root/autodl-fs/math-rlvr-backups/warmstart_only_hidden_grpo_v2_seed42_20260728T083153Z.tar.gz` | `914d5e941e1db8f707efcac72c6699a72feaea958ca372fea1fa01460d603d7d` | warm-start artifact `507749d393f38690915a76228b4c53a8b5c8927d40aada9f2768a90334d892f0` | Full 1,300-completion runtime archive |
| Selected GRPO-v2 hidden | `/root/autodl-fs/math-rlvr-backups/selected_grpo_v2_hidden_grpo_v2_seed42_20260728T083153Z.tar.gz` | `b693178aa1a210a1f7dc555c8a4b0b5eea265a083cc655e863233cf9d3796ebc` | checkpoint-96 manifest `73bb15a32911f490216be2a80eb0d112be0f79236a6d461fd81fbd0579639246` | Full 1,300-completion runtime archive |
| Four-model aggregate | `/root/autodl-fs/math-rlvr-backups/grpo_v2_hidden_four_model_20260728T083153Z.final.tar.gz` | `8a9e411fde3fdf79cc28a604359ca8a556ee7dc5db52ff03471dba5bbfcde47a` | aggregate checksum manifest `dca6bb253156986bdfbf057950da1756db5285fbf8d787280026094142bb2a6d` | Independent static aggregate backup |

## Stage T.1 pre-destruction preservation

Seven audited public bundles are tracked in `release/stage_t1_release_assets.{json,csv}`.
Until the `v0.2.1-grpo-v2-archive` custom assets are uploaded and independently
re-downloaded, the project status is `disaster_recovery_incomplete`. AutoDL paths and
SHA indexes alone are not off-server backups. Full optimizer/scheduler/RNG checkpoints
remain private-only and are deliberately excluded from public Release assets.
