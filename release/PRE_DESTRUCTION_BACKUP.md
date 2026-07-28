# Pre-destruction backup plan

Current status: **disaster_recovery_incomplete**

The GitHub portfolio and automatic source archives are not sufficient disaster recovery.
The seven audited custom assets below have been built and locally extracted/checksummed,
but server destruction is not authorized until upload and independent re-download pass.

## Public release assets

| Asset | Bytes | SHA256 | State |
|---|---:|---|---|
| `old-grpo-v1-checkpoint-32-adapter-v0.2.1.tar.gz` | 16,061,659 | `799adff43a98d991624f3719973d42337f502ea678304ade0ccc3470317047e1` | built; upload pending |
| `warmstart-checkpoint-16-adapter-v0.2.1.tar.gz` | 15,843,701 | `be45ffd2014b7e1acf8abc56277e0a7a57e1a03e3c5dcdc83205a6cc06256122` | built; upload pending |
| `selected-grpo-v2-checkpoint-96-adapter-v0.2.1.tar.gz` | 16,090,414 | `c9690ab2f4190a12568ad63cc75881b4cdaf8cb12575301c5f29e5e0a93c3ce5` | built; upload pending |
| `grpo-v2-training-evidence-v0.2.1.tar.gz` | 2,103,748 | `5cc622ea90a63a386c27d30bc4378456be168c72aa2614af5f6c266d8076f74b` | built; upload pending |
| `four-model-hidden-evidence-v0.2.1.tar.gz` | 3,368,730 | `dd4a89701ba037b8d4d130e72049a756138b7128385d28de255994a698d697e2` | built; upload pending |
| `portfolio-v1-evidence-v0.2.1.tar.gz` | 8,229,078 | `78cb18670b1fa1345c6b78d63f7251aac854ba4ccaaca2de57f5f935447b9b69` | built; upload pending |
| `reproducibility-v0.2.1.tar.gz` | 775,367 | `0888a070ee8206d387249485fb3af30709474d1cd1427f0f90f6a6782b9cf52b` | built; upload pending |

Machine-readable source: [preservation_inventory.json](preservation_inventory.json).
Full training resume binaries are intentionally not public; copy them to Windows/private
persistent storage only if exact resume capability is required. Inference reconstruction
uses the fixed Qwen revision plus the three adapter-only assets.

## Windows download and verification

After `v0.2.1-grpo-v2-archive` is published:

```powershell
$repo = "https://github.com/sherrywan717/ppo-vs-grpo-math-rlvr"
git clone $repo
cd ppo-vs-grpo-math-rlvr
git checkout v0.2.1-grpo-v2-archive
# Download each custom asset from the GitHub Releases page, then verify, for example:
Get-FileHash .\selected-grpo-v2-checkpoint-96-adapter-v0.2.1.tar.gz -Algorithm SHA256
Get-FileHash .\grpo-v2-training-evidence-v0.2.1.tar.gz -Algorithm SHA256
Get-FileHash .\four-model-hidden-evidence-v0.2.1.tar.gz -Algorithm SHA256
tar -tzf .\selected-grpo-v2-checkpoint-96-adapter-v0.2.1.tar.gz
```

Compare every value with `release/stage_t1_release_assets.csv`; do not rely on filenames alone.
