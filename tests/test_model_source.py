import json
from pathlib import Path

import pytest

from math_rlvr.config import load_config
from math_rlvr.training.builders import build_grpo_trainer
from math_rlvr.training.model_source import (
    PINNED_REPO_ID,
    PINNED_REVISION,
    SnapshotValidationError,
    ValidatedModelSource,
)


def make_snapshot(
    cache_root: Path,
    repo_cache="models--Qwen--Qwen2.5-0.5B-Instruct",
    revision=PINNED_REVISION,
):
    snapshot = cache_root / "hub" / repo_cache / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(
        json.dumps({"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]})
    )
    (snapshot / "model.safetensors").write_bytes(b"fake")
    (snapshot / "tokenizer.json").write_text("{}")
    (snapshot / "tokenizer_config.json").write_text("{}")
    return snapshot


def resolve(cache_root: Path, returned: Path | None = None, **kwargs):
    snapshot = returned or (
        cache_root
        / "hub"
        / "models--Qwen--Qwen2.5-0.5B-Instruct"
        / "snapshots"
        / PINNED_REVISION
    )
    calls = []

    def resolver(**resolver_kwargs):
        calls.append(resolver_kwargs)
        return str(snapshot)

    source = ValidatedModelSource.resolve(
        kwargs.pop("repo_id", PINNED_REPO_ID),
        kwargs.pop("revision", PINNED_REVISION),
        cache_root=cache_root,
        snapshot_resolver=resolver,
    )
    return source, calls


def test_pinned_snapshot_resolves_canonically_and_offline(tmp_path):
    cache = tmp_path / "cache"
    snapshot = make_snapshot(cache)
    source, calls = resolve(cache)
    assert source.snapshot_path == snapshot.resolve(strict=True)
    assert source.repo_id == PINNED_REPO_ID
    assert source.revision == PINNED_REVISION
    assert source.local_files_only is True
    assert source.config_identity.model_type == "qwen2"
    assert calls == [
        {
            "repo_id": PINNED_REPO_ID,
            "revision": PINNED_REVISION,
            "cache_dir": str(cache.resolve() / "hub"),
            "local_files_only": True,
        }
    ]


def test_builder_receives_validated_local_path_not_repo_id(tmp_path):
    cache = tmp_path / "cache"
    snapshot = make_snapshot(cache)
    source, _ = resolve(cache)
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        return object()

    config = load_config("configs/smoke/grpo.yaml")
    build_grpo_trainer(
        config,
        [],
        lambda *_args, **_kwargs: [],
        tmp_path / "output",
        trainer_factory=factory,
        cpu_only=True,
        model_source=source,
    )
    assert calls[0]["model"] == str(snapshot.resolve())
    assert calls[0]["model"] != PINNED_REPO_ID


@pytest.mark.parametrize(
    ("repo_id", "revision", "message"),
    [
        ("Qwen/Qwen2.5-3B-Instruct", PINNED_REVISION, "repository"),
        (PINNED_REPO_ID, "bad-revision", "revision"),
    ],
)
def test_wrong_repo_or_revision_is_rejected(tmp_path, repo_id, revision, message):
    cache = tmp_path / "cache"
    make_snapshot(cache)
    with pytest.raises(SnapshotValidationError, match=message):
        resolve(cache, repo_id=repo_id, revision=revision)


@pytest.mark.parametrize(
    "repo_cache",
    [
        "models--Qwen--Qwen2.5-0.5B-Instruct-copy",
        "models--Qwen--Qwen2.5-1.5B-Instruct",
    ],
)
def test_other_or_similar_repo_cache_is_rejected(tmp_path, repo_cache):
    cache = tmp_path / "cache"
    make_snapshot(cache)
    other = make_snapshot(cache, repo_cache=repo_cache)
    with pytest.raises(SnapshotValidationError, match="unexpected snapshot"):
        resolve(cache, returned=other)


def test_dot_dot_traversal_is_rejected(tmp_path):
    cache = tmp_path / "cache"
    snapshot = make_snapshot(cache)
    traversed = snapshot.parent / ".." / "snapshots" / PINNED_REVISION
    with pytest.raises(SnapshotValidationError, match="traversal"):
        resolve(cache, returned=traversed)


def test_snapshot_level_symlink_escape_is_rejected(tmp_path):
    cache = tmp_path / "cache"
    target = tmp_path / "outside"
    target.mkdir()
    expected = (
        cache
        / "hub"
        / "models--Qwen--Qwen2.5-0.5B-Instruct"
        / "snapshots"
        / PINNED_REVISION
    )
    expected.parent.mkdir(parents=True)
    expected.symlink_to(target, target_is_directory=True)
    with pytest.raises(SnapshotValidationError, match="symlink"):
        resolve(cache, returned=expected)


def test_missing_snapshot_is_rejected(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    missing = cache / "hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots" / PINNED_REVISION
    with pytest.raises(SnapshotValidationError, match="does not exist"):
        resolve(cache, returned=missing)


def test_wrong_config_identity_is_rejected(tmp_path):
    cache = tmp_path / "cache"
    snapshot = make_snapshot(cache)
    (snapshot / "config.json").write_text(
        json.dumps({"model_type": "qwen2", "architectures": ["WrongForCausalLM"]})
    )
    with pytest.raises(SnapshotValidationError, match="identity"):
        resolve(cache)


def test_formal_1p5b_sharded_snapshot_is_exact_and_offline(tmp_path):
    from math_rlvr.training.model_source import FORMAL_REPO_ID, FORMAL_REVISION

    cache = tmp_path / "cache"
    snapshot = (
        cache
        / "hub"
        / "models--Qwen--Qwen2.5-1.5B-Instruct"
        / "snapshots"
        / FORMAL_REVISION
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(
        '{"model_type":"qwen2","architectures":["Qwen2ForCausalLM"]}\n'
    )
    (snapshot / "tokenizer.json").write_text("{}\n")
    (snapshot / "tokenizer_config.json").write_text("{}\n")
    (snapshot / "model-00001-of-00002.safetensors").write_bytes(b"first")
    (snapshot / "model-00002-of-00002.safetensors").write_bytes(b"second")
    (snapshot / "model.safetensors.index.json").write_text(
        '{"weight_map":{"a":"model-00001-of-00002.safetensors",'
        '"b":"model-00002-of-00002.safetensors"}}\n'
    )
    calls = []

    def resolver(**kwargs):
        calls.append(kwargs)
        return str(snapshot)

    source = ValidatedModelSource.resolve(
        FORMAL_REPO_ID,
        FORMAL_REVISION,
        cache_root=cache,
        snapshot_resolver=resolver,
    )
    assert source.snapshot_path == snapshot
    assert source.local_files_only is True
    assert calls == [
        {
            "repo_id": FORMAL_REPO_ID,
            "revision": FORMAL_REVISION,
            "cache_dir": str(cache.resolve() / "hub"),
            "local_files_only": True,
        }
    ]
