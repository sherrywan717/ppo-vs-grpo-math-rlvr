import hashlib
import inspect
from collections.abc import Mapping, MutableMapping
from pathlib import Path

import pytest
import torch
from accelerate import Accelerator
from datasets import Dataset
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers.tokenization_utils_base import BatchEncoding

from math_rlvr.training.common import preflight
from math_rlvr.training.execution_contract import (
    expected_run_contract_for_config,
    validated_scope_from_config,
)
from math_rlvr.training.guarded_ppo import ppo_execution_problems_and_episodes
from math_rlvr.training.ppo_runtime import build_ppo_runtime_dataset_rows
from math_rlvr.training.trl_compat import (
    ORDERED_EPISODE_FIELDS,
    OrderedMetadataCollator,
    TRLContractError,
    extract_ordered_episode_batch,
    validate_ordered_episode_batch,
)

SNAPSHOT = Path(
    "/root/autodl-tmp/cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/"
    "7ae557604adf67be50417f59c2c2f167def9a775"
)


def _runtime_rows(config_path: str, tokenizer):
    config = preflight(Path(config_path), "ppo")
    contract = expected_run_contract_for_config(config, "ppo")
    scope = validated_scope_from_config(config, "ppo")
    problems, records = ppo_execution_problems_and_episodes(config, contract)
    _lookup, rows = build_ppo_runtime_dataset_rows(config, tokenizer, problems, records, scope)
    return config, contract, records, Dataset.from_list(rows)


@pytest.fixture(scope="module")
def local_tokenizer():
    assert torch.cuda.is_initialized() is False
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    assert torch.cuda.is_initialized() is False
    return tokenizer


def _split_features(dataset):
    features = [dataset[index] for index in range(len(dataset))]
    model_features = [
        {key: value for key, value in feature.items() if key not in ORDERED_EPISODE_FIELDS}
        for feature in features
    ]
    return features, model_features


def test_frozen_seed42_reproduces_legacy_dict_only_failure(local_tokenizer):
    config, _contract, _records, dataset = _runtime_rows(
        "configs/pilot/resolved/ppo_seed_42.json", local_tokenizer
    )
    features, model_features = _split_features(dataset)
    base_collator = DataCollatorWithPadding(local_tokenizer)
    batch = base_collator(model_features)

    assert config["resolved_config_sha256"] == (
        "1daeba7e6cd5e0af43c7f7cb9db87b46d44608adf9fdf432dc7b2c34ea059fdd"
    )
    assert len(features) == 16
    assert type(batch) is BatchEncoding
    assert isinstance(batch, MutableMapping)
    assert not isinstance(batch, dict)
    assert list(batch) == ["input_ids", "attention_mask"]
    assert tuple(batch["input_ids"].shape) == (16, 161)
    assert tuple(batch["attention_mask"].shape) == (16, 161)
    assert batch["input_ids"].dtype == batch["attention_mask"].dtype == torch.int64

    def legacy_dict_only_contract():
        legacy_batch = base_collator(model_features)
        if not isinstance(legacy_batch, dict):
            raise TRLContractError("PPO data collator must return a mapping")

    with pytest.raises(TRLContractError, match="PPO data collator must return a mapping"):
        legacy_dict_only_contract()


def test_mapping_wrapper_preserves_batch_encoding_and_tensor_contract(local_tokenizer):
    _config, _contract, records, dataset = _runtime_rows(
        "configs/pilot/resolved/ppo_seed_42.json", local_tokenizer
    )
    features, _model_features = _split_features(dataset)

    class RecordingBaseCollator:
        def __init__(self):
            self.base = DataCollatorWithPadding(local_tokenizer)
            self.calls = 0
            self.last_batch = None
            self.last_features = None

        def __call__(self, model_features):
            self.calls += 1
            self.last_features = model_features
            self.last_batch = self.base(model_features)
            return self.last_batch

    recording = RecordingBaseCollator()
    batch = OrderedMetadataCollator(recording)(features)
    assert batch is recording.last_batch
    assert type(batch) is BatchEncoding
    assert isinstance(batch, MutableMapping)
    assert recording.calls == 1
    assert all(
        set(feature).isdisjoint(ORDERED_EPISODE_FIELDS) for feature in recording.last_features
    )
    assert tuple(batch["input_ids"].shape) == (16, 161)
    assert tuple(batch["attention_mask"].shape) == (16, 161)
    assert batch["input_ids"].dtype == batch["attention_mask"].dtype == torch.int64
    assert extract_ordered_episode_batch(batch) == records


def test_cpu_accelerator_prepared_batch_is_mapping_and_train_consumable(local_tokenizer):
    _config, contract, records, dataset = _runtime_rows(
        "configs/pilot/resolved/ppo_seed_42.json", local_tokenizer
    )
    loader = DataLoader(
        dataset,
        batch_size=16,
        sampler=SequentialSampler(dataset),
        collate_fn=OrderedMetadataCollator(DataCollatorWithPadding(local_tokenizer)),
        drop_last=True,
        num_workers=0,
    )
    assert isinstance(loader.sampler, SequentialSampler)
    assert loader.batch_size == 16
    assert loader.drop_last is True
    assert loader.num_workers == 0

    prepared = Accelerator(cpu=True).prepare_data_loader(loader)
    batch = next(iter(prepared))
    assert isinstance(batch, Mapping)
    assert type(batch) is BatchEncoding
    assert {"input_ids", "attention_mask"}.issubset(batch)
    assert tuple(batch["input_ids"].shape) == (16, 161)
    assert tuple(batch["attention_mask"].shape) == (16, 161)
    actual = validate_ordered_episode_batch(batch, records)
    assert [row["pair_key"] for row in actual] == list(contract.pair_keys)
    assert len({row["pair_key"] for row in actual}) == 16
    assert [row["episode_position"] for row in actual] == list(range(16))
    for problem_index in range(4):
        group = actual[problem_index * 4 : (problem_index + 1) * 4]
        assert [row["generation_index"] for row in group] == [0, 1, 2, 3]

    def ppo_train_consumption_proxy(prepared_batch):
        queries = prepared_batch["input_ids"]
        model_kwargs = {"input_ids": queries}
        return queries, model_kwargs

    queries, model_kwargs = ppo_train_consumption_proxy(batch)
    assert queries is batch["input_ids"]
    assert set(model_kwargs) == {"input_ids"}
    assert set(model_kwargs).isdisjoint(ORDERED_EPISODE_FIELDS)

    import trl.trainer.ppo_trainer as ppo_module

    train_source = inspect.getsource(ppo_module.PPOTrainer.train)
    assert 'queries = data["input_ids"]' in train_source
    assert "model(**data)" not in train_source
    assert torch.cuda.is_initialized() is False


def test_stage_d_four_row_mapping_collator_contract_is_unchanged(local_tokenizer):
    _config, contract, records, dataset = _runtime_rows("configs/smoke/ppo.yaml", local_tokenizer)
    assert contract.profile == "ppo_stage_d_smoke"
    assert len(dataset) == len(records) == 4
    loader = DataLoader(
        dataset,
        batch_size=4,
        sampler=SequentialSampler(dataset),
        collate_fn=OrderedMetadataCollator(DataCollatorWithPadding(local_tokenizer)),
        drop_last=True,
        num_workers=0,
    )
    batch = next(iter(loader))
    assert isinstance(batch, Mapping)
    assert tuple(batch["input_ids"].shape)[0] == 4
    assert tuple(batch["attention_mask"].shape) == tuple(batch["input_ids"].shape)
    assert extract_ordered_episode_batch(batch) == records


def test_both_historical_seed42_failures_remain_immutable():
    expected = {
        Path(
            "reports/runs/ppo_matched_0p5b_seed42_20260714T073357Z/failure_report.json"
        ): "9240b4d13b649ded6e27360965870705763cea09d8098743dd113c27b2e6d4d2",
        Path(
            "reports/runs/ppo_matched_0p5b_seed42_20260714T073357Z/summary.json"
        ): "80cb79dd1e6341e11b6bb06aceff376c400cdf197ccbe4094bae0c53e20a63d7",
        Path(
            "reports/runs/ppo_matched_0p5b_seed42_20260714T082003Z/failure_report.json"
        ): "53b7363b571bd32ef69cc653f8210b105398d34eb350a77e555cd45d7114cc57",
        Path(
            "reports/runs/ppo_matched_0p5b_seed42_20260714T082003Z/summary.json"
        ): "bf1b04eef3544c6f1c8cb49f9f50fdb683ba8ab9c283afe3c30ad638db441ef2",
        Path(
            "reports/runs/ppo_matched_0p5b_seed42_20260714T082003Z/assessment.md"
        ): "b40ffded39f6a9cb986f62a8c78f264530acd4a96452ea100992aa89f3f33338",
        Path(
            "/root/autodl-tmp/runs/math_rlvr/"
            "ppo_matched_0p5b_seed42_20260714T082003Z/final_summary.json"
        ): "d6384d40bc0aa75c20f85d80bd5b7a71ca219845b1e9b580a9336646d35162e4",
    }
    for path, digest in expected.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_frozen_pilot_files_are_unchanged():
    expected = {
        "configs/pilot/resolved/ppo_seed_42.json": (
            "1daeba7e6cd5e0af43c7f7cb9db87b46d44608adf9fdf432dc7b2c34ea059fdd"
        ),
        "configs/pilot/resolved/ppo_seed_123.json": (
            "9da0ad35e943cdeda2da410c20eec73e6d105f0ef66f7f67b1be22950a0e43c5"
        ),
        "configs/pilot/resolved/ppo_seed_2026.json": (
            "d3255ddb849224a4d87a069d981fcacf85cb98a7afa986ff0a9fb284b7698044"
        ),
        "configs/pilot/resolved/grpo_seed_42.json": (
            "83992a9c312b3ea6ab87f33dce1d4e9572a9647bbdb72bd67a6e98e90c182ac8"
        ),
        "configs/pilot/resolved/grpo_seed_123.json": (
            "edec9ce1265dfaec8c712b2c65046fe860cbd3e10aab52cf31b6d5e0350c2a28"
        ),
        "configs/pilot/resolved/grpo_seed_2026.json": (
            "1d558da6ea57cfa074fee30868f1772c76c617920e4e93a4897be2e2b48d6b00"
        ),
    }
    for raw_path, digest in expected.items():
        assert hashlib.sha256(Path(raw_path).read_bytes()).hexdigest() == digest
    manifest = Path("configs/pilot/matched_0p5b_manifest.json")
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == (
        "a79ea8ee9d8bdc8f3d6fba8307995cba0c4516b90331cb13ba18ef1b55fa1b0d"
    )
