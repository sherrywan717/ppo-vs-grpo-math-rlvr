# Pre-destruction backup

Current status: **safe_to_destroy_server**

GitHub source publication alone was not considered disaster recovery. The project now
has seven custom off-server Release assets, and every asset was downloaded from GitHub
into a fresh temporary directory, checked against the committed SHA, safely extracted,
and verified with its internal checksum manifest.

## Verified Release assets

| Asset | Bytes | SHA256 | Independent verification |
|---|---:|---|---|
| [`old-grpo-v1-checkpoint-32-adapter-v0.2.1.tar.gz`](https://github.com/sherrywan717/ppo-vs-grpo-math-rlvr/releases/download/v0.2.1-grpo-v2-archive/old-grpo-v1-checkpoint-32-adapter-v0.2.1.tar.gz) | 16,061,659 | `799adff43a98d991624f3719973d42337f502ea678304ade0ccc3470317047e1` | verified |
| [`warmstart-checkpoint-16-adapter-v0.2.1.tar.gz`](https://github.com/sherrywan717/ppo-vs-grpo-math-rlvr/releases/download/v0.2.1-grpo-v2-archive/warmstart-checkpoint-16-adapter-v0.2.1.tar.gz) | 15,843,701 | `be45ffd2014b7e1acf8abc56277e0a7a57e1a03e3c5dcdc83205a6cc06256122` | verified |
| [`selected-grpo-v2-checkpoint-96-adapter-v0.2.1.tar.gz`](https://github.com/sherrywan717/ppo-vs-grpo-math-rlvr/releases/download/v0.2.1-grpo-v2-archive/selected-grpo-v2-checkpoint-96-adapter-v0.2.1.tar.gz) | 16,090,414 | `c9690ab2f4190a12568ad63cc75881b4cdaf8cb12575301c5f29e5e0a93c3ce5` | verified |
| [`grpo-v2-training-evidence-v0.2.1.tar.gz`](https://github.com/sherrywan717/ppo-vs-grpo-math-rlvr/releases/download/v0.2.1-grpo-v2-archive/grpo-v2-training-evidence-v0.2.1.tar.gz) | 2,103,748 | `5cc622ea90a63a386c27d30bc4378456be168c72aa2614af5f6c266d8076f74b` | verified |
| [`four-model-hidden-evidence-v0.2.1.tar.gz`](https://github.com/sherrywan717/ppo-vs-grpo-math-rlvr/releases/download/v0.2.1-grpo-v2-archive/four-model-hidden-evidence-v0.2.1.tar.gz) | 3,368,730 | `dd4a89701ba037b8d4d130e72049a756138b7128385d28de255994a698d697e2` | verified |
| [`portfolio-v1-evidence-v0.2.1.tar.gz`](https://github.com/sherrywan717/ppo-vs-grpo-math-rlvr/releases/download/v0.2.1-grpo-v2-archive/portfolio-v1-evidence-v0.2.1.tar.gz) | 8,229,078 | `78cb18670b1fa1345c6b78d63f7251aac854ba4ccaaca2de57f5f935447b9b69` | verified |
| [`reproducibility-v0.2.1.tar.gz`](https://github.com/sherrywan717/ppo-vs-grpo-math-rlvr/releases/download/v0.2.1-grpo-v2-archive/reproducibility-v0.2.1.tar.gz) | 775,367 | `0888a070ee8206d387249485fb3af30709474d1cd1427f0f90f6a6782b9cf52b` | verified |

Release: [https://github.com/sherrywan717/ppo-vs-grpo-math-rlvr/releases/tag/v0.2.1-grpo-v2-archive](https://github.com/sherrywan717/ppo-vs-grpo-math-rlvr/releases/tag/v0.2.1-grpo-v2-archive)

## What is recoverable

- Exact evaluated old GRPO-v1, warm-start and selected GRPO-v2 inference adapters.
- GRPO-v2 training/update/completion/reward/dev evidence and checkpoint selection.
- All four hidden-test primary evidence, Base metric recovery and paired aggregates.
- Portfolio-v1 evidence and the source/config/test reproducibility bundle.
- Repository source, 200-question guide, manifests, reports and checksums from GitHub.

The fixed Qwen base is reconstructed from public revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`; model/cache bytes are intentionally absent.
Full optimizer/scheduler/RNG resume state is not publicly preserved and is not needed to
reproduce final inference from the evaluated adapters. Copy it separately to private
Windows storage only if exact training resume is desired.

## Windows download and verification

```powershell
$repo = "https://github.com/sherrywan717/ppo-vs-grpo-math-rlvr.git"
git clone $repo
cd ppo-vs-grpo-math-rlvr
git checkout v0.2.1-grpo-v2-archive
# Download assets from the Release page or use each browser_download_url in
# release/stage_t1_release_assets.json. Then verify, for example:
Get-FileHash .\selected-grpo-v2-checkpoint-96-adapter-v0.2.1.tar.gz -Algorithm SHA256
Get-FileHash .\grpo-v2-training-evidence-v0.2.1.tar.gz -Algorithm SHA256
Get-FileHash .\four-model-hidden-evidence-v0.2.1.tar.gz -Algorithm SHA256
tar -tzf .\selected-grpo-v2-checkpoint-96-adapter-v0.2.1.tar.gz
tar -xzf .\selected-grpo-v2-checkpoint-96-adapter-v0.2.1.tar.gz
```

Compare all hashes with `release/stage_t1_release_assets.csv`. Machine-readable download
proof is in `release/independent_download_verification.json`.
