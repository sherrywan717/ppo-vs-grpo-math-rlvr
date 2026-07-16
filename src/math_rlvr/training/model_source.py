"""Fail-closed resolution of the one authorized local smoke-model snapshot."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

PINNED_REPO_ID = "Qwen/Qwen2.5-0.5B-Instruct"
PINNED_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
FORMAL_REPO_ID = "Qwen/Qwen2.5-1.5B-Instruct"
FORMAL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
DEFAULT_CACHE_ROOT = Path("/root/autodl-tmp/cache/huggingface")
EXPECTED_MODEL_TYPE = "qwen2"
EXPECTED_ARCHITECTURE = "Qwen2ForCausalLM"
_MODEL_CACHE_SLUGS = {
    (PINNED_REPO_ID, PINNED_REVISION): "models--Qwen--Qwen2.5-0.5B-Instruct",
    (FORMAL_REPO_ID, FORMAL_REVISION): "models--Qwen--Qwen2.5-1.5B-Instruct",
}


class SnapshotValidationError(RuntimeError):
    """The requested local snapshot is not the exact authorized model source."""


@dataclass(frozen=True)
class ConfigIdentity:
    model_type: str
    architectures: tuple[str, ...]


@dataclass(frozen=True)
class ValidatedModelSource:
    repo_id: str
    revision: str
    cache_root: Path
    snapshot_path: Path
    local_files_only: bool
    config_identity: ConfigIdentity

    @classmethod
    def resolve(
        cls,
        repo_id: str,
        revision: str,
        *,
        cache_root: Path = DEFAULT_CACHE_ROOT,
        snapshot_resolver: Callable[..., str] | None = None,
    ) -> ValidatedModelSource:
        profile = _MODEL_CACHE_SLUGS.get((repo_id, revision))
        if profile is None:
            if repo_id not in {key[0] for key in _MODEL_CACHE_SLUGS}:
                raise SnapshotValidationError("unexpected repository")
            raise SnapshotValidationError("unexpected revision")
        try:
            root = Path(cache_root).resolve(strict=True)
        except (FileNotFoundError, RuntimeError) as exc:
            raise SnapshotValidationError("cache root does not exist") from exc

        expected = (
            root
            / "hub"
            / profile
            / "snapshots"
            / revision
        )
        if expected.is_symlink():
            raise SnapshotValidationError("snapshot-level symlink is forbidden")
        try:
            expected_resolved = expected.resolve(strict=True)
        except (FileNotFoundError, RuntimeError) as exc:
            raise SnapshotValidationError("pinned snapshot does not exist") from exc

        if snapshot_resolver is None:
            from huggingface_hub import snapshot_download

            snapshot_resolver = snapshot_download
        returned = Path(
            snapshot_resolver(
                repo_id=repo_id,
                revision=revision,
                cache_dir=str(root / "hub"),
                local_files_only=True,
            )
        )
        if ".." in returned.parts:
            raise SnapshotValidationError("snapshot traversal is forbidden")
        if returned.is_symlink():
            raise SnapshotValidationError("snapshot-level symlink is forbidden")
        try:
            resolved = returned.resolve(strict=True)
        except (FileNotFoundError, RuntimeError) as exc:
            raise SnapshotValidationError("resolved snapshot does not exist") from exc
        if resolved != expected_resolved:
            raise SnapshotValidationError("resolver returned an unexpected snapshot")
        if (
            resolved.name != revision
            or resolved.parent.name != "snapshots"
            or resolved.parent.parent.name != profile
            or resolved.parent.parent.parent.name != "hub"
        ):
            raise SnapshotValidationError("snapshot cache structure mismatch")

        required = ("config.json", "tokenizer.json", "tokenizer_config.json")
        if not all((resolved / name).is_file() for name in required):
            raise SnapshotValidationError("pinned snapshot is incomplete")
        single_weight = resolved / "model.safetensors"
        index_path = resolved / "model.safetensors.index.json"
        if not single_weight.is_file() and not index_path.is_file():
            raise SnapshotValidationError("pinned snapshot has no safetensors weights")
        if index_path.is_file():
            try:
                weight_index = json.loads(index_path.read_text(encoding="utf-8"))
                shards = set(weight_index["weight_map"].values())
            except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
                raise SnapshotValidationError("invalid safetensors index") from exc
            if not shards or any(not (resolved / shard).is_file() for shard in shards):
                raise SnapshotValidationError("pinned snapshot weight shards are incomplete")
        try:
            raw_config = json.loads((resolved / "config.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SnapshotValidationError("invalid model config") from exc
        model_type = raw_config.get("model_type")
        architectures = tuple(raw_config.get("architectures") or ())
        if model_type != EXPECTED_MODEL_TYPE or EXPECTED_ARCHITECTURE not in architectures:
            raise SnapshotValidationError("model config identity mismatch")

        return cls(
            repo_id=repo_id,
            revision=revision,
            cache_root=root,
            snapshot_path=resolved,
            local_files_only=True,
            config_identity=ConfigIdentity(model_type, architectures),
        )
