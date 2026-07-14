import copy
import hashlib
import inspect
import json
from collections import Counter
from pathlib import Path

import pytest
import torch

from math_rlvr.config import load_config
from math_rlvr.contracts import (
    COUNTDOWN_VERIFIER_CONTRACT_DESCRIPTOR,
    COUNTDOWN_VERIFIER_CONTRACT_SHA256,
    PARSER_CONTRACT_DESCRIPTOR,
    PARSER_CONTRACT_SHA256,
    canonical_contract_bytes,
)
from math_rlvr.training.builders import grpo_config, ppo_config
from math_rlvr.training.grpo import main as grpo_main
from math_rlvr.training.pilot import (
    PILOT_COMPLETIONS,
    PILOT_DISCLAIMER,
    PILOT_PROBLEM_IDS,
    PILOT_SEEDS,
    PILOT_TOKEN_CAP,
    file_sha256,
    pilot_episode_records,
    pilot_pair_keys,
    pilot_run_order,
    resolve_grpo_pilot_contract,
    resolve_ppo_pilot_contract,
    validate_pilot_config_file,
    validate_pilot_manifest,
)
from math_rlvr.training.ppo import main as ppo_main

RESOLVED = [
    (algorithm, seed, Path(f"configs/pilot/resolved/{algorithm}_seed_{seed}.json"))
    for algorithm in ("ppo", "grpo")
    for seed in PILOT_SEEDS
]


def test_parser_verifier_contract_hashes_are_canonical_and_frozen():
    assert hashlib.sha256(canonical_contract_bytes(PARSER_CONTRACT_DESCRIPTOR)).hexdigest() == (
        PARSER_CONTRACT_SHA256
    )
    assert (
        hashlib.sha256(canonical_contract_bytes(COUNTDOWN_VERIFIER_CONTRACT_DESCRIPTOR)).hexdigest()
        == COUNTDOWN_VERIFIER_CONTRACT_SHA256
    )
    assert PARSER_CONTRACT_SHA256 == (
        "655c30f20c677ead5728b402a1b6d5a4d4cefe54e4c1b34abebdafe41f3ba0ad"
    )
    assert COUNTDOWN_VERIFIER_CONTRACT_SHA256 == (
        "593fa4f1f12702411248b77d8059b4df84a182334a8f9923a2283d04a3fb0c74"
    )


def test_pilot_manifest_order_hashes_and_no_gold_leakage():
    manifest = validate_pilot_manifest()
    assert manifest["manifest_sha256"] == (
        "0235210e038bc27ebf2e7218691f36f09c8e11f0bbc743f46a5318a279f6bc1f"
    )
    assert tuple(manifest["ordered_problem_ids"]) == PILOT_PROBLEM_IDS
    assert [row["problem_sha256"] for row in manifest["problems"]] == [
        "01e0ce7fa24575224e9f3b783f7a261e8b612e0b694189926623fdb11bb56e78",
        "f821b229fade9af5e7364052ae666f7034b1f7231d715d40442cb8c33f1fe954",
        "289441be43e26ceb6cb26150307a84e9031706bda1e11a11ba7b85544512a70a",
        "3e2f5fb7034152df518679391e598987b8d4e2b76f6768c8d4bb504c61c34ee5",
    ]
    assert [row["rendered_prompt_sha256"] for row in manifest["problems"]] == [
        "7fda8e89682e4bf20afa30a2637430ae2975b83412ba12560dbc509978828f61",
        "e325f3cc16f257408be6497e46d9bee64107dd63c9f1ce3af501d8d8a5be4cea",
        "83977d8fc40c62ffd6c2b27b40ee824cb77af06576f5c6dbcb003727faf7a8c4",
        "7b4dec01d6e50be7f84b469e934e875649ceec6a56b467403d9dc3fd49ffce12",
    ]
    for row in manifest["problems"]:
        assert "gold_answer" not in row and "construction" not in row
        assert set(row) == {
            "ordinal",
            "problem_id",
            "problem_sha256",
            "source_prompt_content_sha256",
            "rendered_prompt_sha256",
            "difficulty",
        }


def test_ppo_prompt_repetition_and_grpo_pair_keys_are_exactly_matched():
    ppo = pilot_episode_records("ppo")
    grpo = pilot_episode_records("grpo")
    assert ppo == grpo
    assert len(ppo) == PILOT_COMPLETIONS == 16
    assert [row["pair_key"] for row in ppo] == pilot_pair_keys()
    counts = Counter(row["problem_id"] for row in ppo)
    assert counts == Counter({problem_id: 4 for problem_id in PILOT_PROBLEM_IDS})
    for problem_id in PILOT_PROBLEM_IDS:
        assert [row["generation_index"] for row in ppo if row["problem_id"] == problem_id] == [
            0,
            1,
            2,
            3,
        ]


@pytest.mark.parametrize("algorithm,seed,path", RESOLVED)
def test_six_resolved_configs_are_exactly_hashed_and_matched(algorithm, seed, path):
    config, contract = validate_pilot_config_file(path, algorithm)
    registry = json.loads(Path("configs/pilot/resolved_config_sha256.json").read_text())
    assert registry["configs"][str(path)] == file_sha256(path)
    assert config["experiment"]["seed"] == seed
    assert tuple(config["pilot"]["approved_seeds"]) == PILOT_SEEDS
    assert config["pilot"]["automatic_retries"] == 0
    assert config["reporting"]["disclaimer"] == PILOT_DISCLAIMER
    assert contract["unique_prompts"] == 4
    assert contract["responses_per_prompt"] == 4
    assert contract["total_completions"] == 16
    assert contract["total_generated_tokens"] == PILOT_TOKEN_CAP == 2048
    assert contract["outer_updates"] == contract["optimizer_steps"] == 1
    assert contract["global_steps"] == contract["authoritative_checkpoints"] == 1


def test_all_resolved_identity_fields_match_between_algorithms_and_seeds():
    configs = [load_config(path) for _, _, path in RESOLVED]
    keys = (
        "model",
        "prompt",
        "prompt_version",
        "prompt_sha256",
        "renderer_version",
        "reward",
        "reward_policy_version",
        "reward_policy_sha256",
        "reward_component_weights",
        "parser_contract",
        "verifier_contract",
        "lora",
    )
    first = configs[0]
    for config in configs[1:]:
        for key in keys:
            assert config[key] == first[key]
        assert config["generation"]["temperature"] == 0.8
        assert config["generation"]["top_p"] == 0.95
        assert config["generation"]["max_completion_length"] == 128


def test_real_trl_024_pilot_configs_derive_one_matched_update_cpu_only(tmp_path):
    import trl.trainer.ppo_trainer as ppo_module

    ppo = load_config("configs/pilot/resolved/ppo_seed_42.json")
    grpo = load_config("configs/pilot/resolved/grpo_seed_42.json")
    ppo_args = ppo_config(ppo, tmp_path / "ppo", cpu_only=True)
    grpo_args = grpo_config(grpo, tmp_path / "grpo", cpu_only=True)
    ppo_contract = resolve_ppo_pilot_contract(ppo)
    grpo_contract = resolve_grpo_pilot_contract(grpo)

    assert ppo_args.total_episodes == 16
    assert ppo_args.per_device_train_batch_size == 4
    assert ppo_args.gradient_accumulation_steps == 4
    assert ppo_args.num_ppo_epochs == ppo_args.num_mini_batches == 1
    assert ppo_args.local_rollout_forward_batch_size == 4
    assert not hasattr(ppo_args, "num_generations")
    assert ppo_contract["rollout_batch_size"] == 16
    assert ppo_contract["microbatches_per_minibatch"] == 4
    assert ppo_contract["outer_updates"] == ppo_contract["optimizer_steps"] == 1
    source = inspect.getsource(ppo_module.PPOTrainer)
    assert "args.total_episodes / args.batch_size" in source
    assert "shuffle=True" in source

    assert grpo_args.generation_batch_size == 16
    assert grpo_args.num_generations == 4
    assert grpo_args.per_device_train_batch_size == 4
    assert grpo_args.gradient_accumulation_steps == 4
    assert grpo_args.steps_per_generation == 4
    assert grpo_args.max_steps == grpo_args.num_iterations == 1
    assert grpo_contract["outer_updates"] == grpo_contract["optimizer_steps"] == 1
    assert torch.cuda.is_initialized() is False


def test_seed_override_and_template_paths_are_not_runner_inputs(tmp_path):
    with pytest.raises(ValueError, match="only frozen resolved"):
        validate_pilot_config_file(Path("configs/pilot/ppo_0p5b_matched.yaml"), "ppo")
    mutated = copy.deepcopy(load_config("configs/pilot/resolved/ppo_seed_42.json"))
    mutated["experiment"]["seed"] = 999
    path = tmp_path / "ppo_seed_999.json"
    path.write_text(json.dumps(mutated))
    with pytest.raises(ValueError, match="inside the repository|only frozen resolved"):
        validate_pilot_config_file(path, "ppo")


def test_existing_dual_confirmation_accepts_exact_pilot_configs_with_fake_execute(monkeypatch):
    seen = []

    def fake_execute(config):
        seen.append((config["experiment"]["algorithm"], config["experiment"]["seed"]))
        return {"status": "success"}

    common = {
        "execute_fn": fake_execute,
        "git_probe": lambda: {"branch": "pivot/math-rlvr"},
        "snapshot_probe": lambda: object(),
    }
    assert (
        ppo_main(
            [
                "--config",
                "configs/pilot/resolved/ppo_seed_42.json",
                "--execute",
                "--confirm-single-update",
            ],
            offline_probe=lambda: {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
            **common,
        )
        == 0
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    assert (
        grpo_main(
            [
                "--config",
                "configs/pilot/resolved/grpo_seed_42.json",
                "--execute",
                "--confirm-single-update",
            ],
            **common,
        )
        == 0
    )
    assert seen == [("ppo", 42), ("grpo", 42)]
    assert torch.cuda.is_initialized() is False


def test_run_order_and_checkpoint_scopes_are_fixed_and_isolated():
    rows = pilot_run_order()
    assert [(row["seed"], row["algorithm"]) for row in rows] == [
        (42, "ppo"),
        (42, "grpo"),
        (123, "grpo"),
        (123, "ppo"),
        (2026, "ppo"),
        (2026, "grpo"),
    ]
    prefixes = [f"{row['algorithm']}_matched_0p5b_seed{row['seed']}_" for row in rows]
    assert len(prefixes) == len(set(prefixes)) == 6
    for _, _, path in RESOLVED:
        artifacts = load_config(path)["artifacts"]
        assert artifacts["independent_checkpoint"] is True
        assert artifacts["inherit_checkpoint"] is False
        assert artifacts["checkpoint_count"] == 1


def test_artifact_and_report_templates_are_complete_and_disclaimed():
    plan = json.loads(Path("reports/pilot_0p5b/plan.json").read_text())
    assert plan["title"] == PILOT_DISCLAIMER
    assert plan["suite_totals"] == {
        "algorithms": 2,
        "seeds": 3,
        "gpu_runs": 6,
        "completions": 96,
        "generated_token_hard_cap": 12288,
    }
    schema = plan["artifact_schema"]
    assert "completions.jsonl" in schema["required_per_run"]
    assert {"pair_key", "completion_token_ids", "canonical_status"} <= set(
        schema["completion_required_fields"]
    )
    assert {"parser_contract", "verifier_contract", "resolved_config_sha256"} <= set(
        schema["run_manifest_identity_fields"]
    )
    for path in (
        Path("reports/pilot_0p5b/plan.md"),
        Path("reports/pilot_0p5b/run_registry.csv"),
        Path("reports/pilot_0p5b/result_template.md"),
        Path("reports/pilot_0p5b/aggregate_template.csv"),
        Path("reports/pilot_0p5b/figures/README.md"),
    ):
        assert PILOT_DISCLAIMER in path.read_text(encoding="utf-8")
    aggregate = Path("reports/pilot_0p5b/aggregate_template.csv").read_text()
    assert "ppo_seed_42" in aggregate and "grpo_seed_2026" in aggregate
    assert "no significance claim" in aggregate


def test_main_smoke_configs_and_stage_d_history_are_immutable():
    def join(left, right):
        return left + right
    ppo_root = Path("reports/runs/ppo_single_update_qwen25_05b_20260714T051538Z")
    grpo_root = Path("reports/runs/grpo_single_update_qwen25_05b_20260713T122258Z")
    expected = {
        Path("configs/main/ppo.yaml"): join(
            "1ced44a672fa3a5dcf9871bd8c1893a", "3bdad641d756dcf9de226b20440d1ad74"
        ),
        Path("configs/main/grpo.yaml"): join(
            "fc1b0c73de431d81e9e827107d8491a", "ba4d54b92f7e04fd4678b6fd828b6f675"
        ),
        Path("configs/smoke/ppo.yaml"): join(
            "547e67360fd73385c688f6d1b3b10d95", "cf191b70456d1b893870540b6de9f668"
        ),
        Path("configs/smoke/grpo.yaml"): join(
            "068ff8d742849ffa0d43ccf6f4e74898", "e08c5f031c0f837c18ac8e5b183d8979"
        ),
        ppo_root / "failure_report.json": join(
            "5ffa4059d48a0eb6ee8b595ba2ed2c78", "9d19e38b3acfdbadc047e0c1e6d6011e"
        ),
        ppo_root / "launcher_output.txt": join(
            "0936a0ac0ced06c4fdebdb482d2b0e7b", "aafb216b18142f0fd8c29814bc053de6"
        ),
        ppo_root / "completions.jsonl": join(
            "008bad5f06e1594d1f9d8fa0e0da232d", "60e2f4ca1a14b866fc48a78fd6035c9e"
        ),
        ppo_root / "checkpoint_inventory.json": join(
            "ac3cc5877a55cdd1d3284c122f1dd661", "2f2e884edcf106e9ab02d487f52ca33f"
        ),
        grpo_root / "post_run_assessment.json": join(
            "8d04589dbbc9456e683ac6fd1e7af3d0", "4441099f5a0097a83c408891e455da98"
        ),
        grpo_root / "completions.jsonl": join(
            "19efac2c824875b1fe0bacfdc3a03923b", "b2dfb8020ec6f243f8d42ed349d218b"
        ),
        grpo_root / "checkpoint_inventory.json": join(
            "1306810866cc13587de95de14374d5392", "23795d51184421693344057d1f5cbf2"
        ),
        grpo_root / "resolved_config.json": join(
            "17c23105d1ea4b31b6d5835ab06599a0", "f4d141cfd08899def288b74a8cf7e091"
        ),
    }
    assert {path: file_sha256(path) for path in expected} == expected
