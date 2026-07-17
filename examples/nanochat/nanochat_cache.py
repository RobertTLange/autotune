"""Trusted preparation manifest for the canonical autoresearch cache."""

import hashlib
import json
import os
import tempfile
from pathlib import Path


AUTORESEARCH_COMMIT = "228791fb499afffb54b46200aca536f79142f117"
EXPECTED_SHARDS = {f"shard_{index:05d}.parquet" for index in range(10)} | {"shard_06542.parquet"}
TOKENIZER_FILES = ("tokenizer.pkl", "token_bytes.pt")
MANIFEST_NAME = "autotune_data_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cache_root() -> Path:
    return Path.home() / ".cache" / "autoresearch"


def inspect_cache(include_data_hashes: bool) -> tuple[list[dict], dict[str, str]]:
    root = cache_root()
    data_dir = root / "data"
    actual = {path.name for path in data_dir.glob("*.parquet")}
    if actual != EXPECTED_SHARDS:
        raise SystemExit(
            f"canonical autoresearch cache mismatch; missing={sorted(EXPECTED_SHARDS - actual)}, "
            f"extra={sorted(actual - EXPECTED_SHARDS)}"
        )
    files = []
    for name in sorted(EXPECTED_SHARDS):
        path = data_dir / name
        if not path.is_file() or path.is_symlink():
            raise SystemExit(f"canonical data shard must be a regular, non-symlink file: {path}")
        info = path.stat()
        entry = {"name": name, "bytes": info.st_size, "mtime_ns": info.st_mtime_ns}
        if include_data_hashes:
            entry["sha256"] = sha256_file(path)
        files.append(entry)
    tokenizer = {}
    for name in TOKENIZER_FILES:
        path = root / "tokenizer" / name
        if not path.is_file() or path.is_symlink():
            raise SystemExit(f"missing regular canonical tokenizer artifact: {path}")
        tokenizer[name] = sha256_file(path)
    return files, tokenizer


def identity(manifest: dict) -> dict:
    return {
        "autoresearch_commit": AUTORESEARCH_COMMIT,
        "data_files": [
            {"name": entry["name"], "bytes": entry["bytes"], "sha256": entry["sha256"]}
            for entry in manifest["files"]
        ],
        "tokenizer_sha256": manifest["tokenizer_sha256"],
    }


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def create_manifest() -> str:
    path = cache_root() / MANIFEST_NAME
    if path.exists():
        raise SystemExit(f"manifest already exists; remove it only when intentionally rebuilding the cache: {path}")
    files, tokenizer = inspect_cache(include_data_hashes=True)
    manifest = {
        "schema_version": 1,
        "autoresearch_commit": AUTORESEARCH_COMMIT,
        "files": files,
        "tokenizer_sha256": tokenizer,
    }
    write_manifest(path, manifest)
    return canonical_hash(identity(manifest))


def verify_manifest(expected_identity: str | None = None, rehash_data: bool = False) -> dict:
    path = cache_root() / MANIFEST_NAME
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"missing trusted cache manifest; run prepare_nanochat_cache.py create: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid canonical data manifest: {path}") from exc
    if manifest.get("schema_version") != 1 or manifest.get("autoresearch_commit") != AUTORESEARCH_COMMIT:
        raise SystemExit("canonical data manifest has an unexpected schema or commit")
    current_files, tokenizer = inspect_cache(include_data_hashes=rehash_data)
    recorded_files = manifest.get("files")
    if not isinstance(recorded_files, list) or len(recorded_files) != len(current_files):
        raise SystemExit("canonical data manifest has an invalid file list")
    for recorded, current in zip(recorded_files, current_files, strict=True):
        fields = ("name", "bytes", "mtime_ns") + (("sha256",) if rehash_data else ())
        if any(recorded.get(field) != current.get(field) for field in fields):
            raise SystemExit(f"canonical data shard differs from manifest: {current['name']}")
        if not isinstance(recorded.get("sha256"), str) or len(recorded["sha256"]) != 64:
            raise SystemExit(f"canonical data manifest has an invalid digest: {current['name']}")
    if manifest.get("tokenizer_sha256") != tokenizer:
        raise SystemExit("canonical tokenizer differs from manifest")
    content_identity = identity(manifest)
    identity_sha256 = canonical_hash(content_identity)
    if expected_identity is not None and identity_sha256 != expected_identity:
        raise SystemExit("canonical data identity does not match the verified launcher identity")
    return {
        "cache_dir": str(cache_root()),
        "identity_sha256": identity_sha256,
        **content_identity,
    }
