"""Trusted preparation manifest for the canonical autoresearch cache."""

import errno
import fcntl
import hashlib
import json
import os
import shutil
import stat
import tempfile
from pathlib import Path


AUTORESEARCH_COMMIT = "228791fb499afffb54b46200aca536f79142f117"
EXPECTED_SHARDS = {f"shard_{index:05d}.parquet" for index in range(10)} | {"shard_06542.parquet"}
TOKENIZER_FILES = ("tokenizer.pkl", "token_bytes.pt")
MANIFEST_NAME = "autotune_data_manifest.json"
SNAPSHOT_MANIFEST_NAME = "snapshot.json"


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


def data_snapshot_path(dataset: dict) -> Path:
    return Path.home() / ".cache" / "autotune" / f"nanochat-data-{dataset['identity_sha256']}"


def snapshot_manifest(dataset: dict) -> dict:
    return {
        "schema_version": 1,
        "identity_sha256": dataset["identity_sha256"],
        "data_files": dataset["data_files"],
    }


def is_owned_and_nonwritable_by_others(metadata: os.stat_result) -> bool:
    return metadata.st_uid == os.getuid() and not metadata.st_mode & 0o022


def verified_data_snapshot(dataset: dict, rehash_data: bool = False) -> Path:
    snapshot = data_snapshot_path(dataset)
    data_dir = snapshot / "data"
    marker = snapshot / SNAPSHOT_MANIFEST_NAME
    try:
        snapshot_metadata = snapshot.lstat()
        data_metadata = data_dir.lstat()
        marker_metadata = marker.lstat()
    except OSError as exc:
        raise SystemExit(f"missing immutable data snapshot; run prepare_nanochat_cache.py verify: {snapshot}") from exc
    if (
        not stat.S_ISDIR(snapshot_metadata.st_mode)
        or stat.S_IMODE(snapshot_metadata.st_mode) != 0o500
        or not is_owned_and_nonwritable_by_others(snapshot_metadata)
        or not stat.S_ISDIR(data_metadata.st_mode)
        or stat.S_IMODE(data_metadata.st_mode) != 0o500
        or not is_owned_and_nonwritable_by_others(data_metadata)
        or not stat.S_ISREG(marker_metadata.st_mode)
        or stat.S_IMODE(marker_metadata.st_mode) != 0o400
        or not is_owned_and_nonwritable_by_others(marker_metadata)
    ):
        raise SystemExit(f"immutable data snapshot contains a symlink or unexpected file type: {snapshot}")
    try:
        recorded = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid immutable data snapshot manifest: {marker}") from exc
    if recorded != snapshot_manifest(dataset):
        raise SystemExit(f"immutable data snapshot identity differs from the cache manifest: {snapshot}")
    expected_names = {entry["name"] for entry in dataset["data_files"]}
    actual_names = {path.name for path in data_dir.iterdir()}
    if actual_names != expected_names:
        raise SystemExit(f"immutable data snapshot has an unexpected file set: {snapshot}")
    for entry in dataset["data_files"]:
        path = data_dir / entry["name"]
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise SystemExit(f"missing immutable data snapshot shard: {path}") from exc
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o400
            or not is_owned_and_nonwritable_by_others(metadata)
            or metadata.st_size != entry["bytes"]
        ):
            raise SystemExit(f"immutable data snapshot shard metadata differs: {path}")
        if rehash_data and sha256_file(path) != entry["sha256"]:
            raise SystemExit(f"immutable data snapshot shard content differs: {path}")
    return data_dir


def copy_snapshot_file(source: Path, destination: Path, expected_size: int, expected_sha256: str) -> None:
    source_flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0)
    try:
        source_descriptor = os.open(source, source_flags)
    except OSError as exc:
        raise SystemExit(f"could not safely open canonical data shard: {source}") from exc
    try:
        source_metadata = os.fstat(source_descriptor)
        if not stat.S_ISREG(source_metadata.st_mode) or source_metadata.st_size != expected_size:
            raise SystemExit(f"canonical data shard metadata changed while snapshotting: {source}")
        destination_descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            digest = hashlib.sha256()
            remaining = expected_size
            while remaining:
                chunk = os.read(source_descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    raise SystemExit(f"canonical data shard ended while snapshotting: {source}")
                digest.update(chunk)
                view = memoryview(chunk)
                while view:
                    written = os.write(destination_descriptor, view)
                    view = view[written:]
                remaining -= len(chunk)
            if os.read(source_descriptor, 1) or digest.hexdigest() != expected_sha256:
                raise SystemExit(f"canonical data shard changed while snapshotting: {source}")
            os.fsync(destination_descriptor)
            os.fchmod(destination_descriptor, 0o400)
        finally:
            os.close(destination_descriptor)
    finally:
        os.close(source_descriptor)


def prepare_data_snapshot(dataset: dict) -> Path:
    snapshot = data_snapshot_path(dataset)
    parent = snapshot.parent
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    parent_metadata = parent.lstat()
    if not stat.S_ISDIR(parent_metadata.st_mode):
        raise SystemExit(f"immutable data snapshot parent must be a non-symlink directory: {parent}")
    if not is_owned_and_nonwritable_by_others(parent_metadata):
        raise SystemExit(
            f"immutable data snapshot parent must be owned and non-writable by group or world: {parent}"
        )
    lock_flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        lock_descriptor = os.open(parent / f".{snapshot.name}.lock", lock_flags, 0o600)
    except OSError as exc:
        raise SystemExit(f"could not safely open immutable data snapshot lock under {parent}") from exc
    lock_metadata = os.fstat(lock_descriptor)
    if not stat.S_ISREG(lock_metadata.st_mode) or not is_owned_and_nonwritable_by_others(lock_metadata):
        os.close(lock_descriptor)
        raise SystemExit(f"immutable data snapshot lock must be an owned, private regular file under {parent}")
    try:
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        if snapshot.exists():
            return verified_data_snapshot(dataset, rehash_data=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{snapshot.name}.", suffix=".tmp", dir=parent))
        try:
            data_dir = temporary / "data"
            data_dir.mkdir(mode=0o700)
            for entry in dataset["data_files"]:
                copy_snapshot_file(
                    cache_root() / "data" / entry["name"],
                    data_dir / entry["name"],
                    entry["bytes"],
                    entry["sha256"],
                )
            write_manifest(temporary / SNAPSHOT_MANIFEST_NAME, snapshot_manifest(dataset))
            (temporary / SNAPSHOT_MANIFEST_NAME).chmod(0o400)
            data_dir.chmod(0o500)
            temporary.chmod(0o500)
            published = False
            try:
                os.replace(temporary, snapshot)
                published = True
            except OSError as exc:
                if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                    raise
        finally:
            if temporary.exists():
                temporary.chmod(0o700)
                data_dir = temporary / "data"
                if data_dir.exists():
                    data_dir.chmod(0o700)
                shutil.rmtree(temporary)
        return verified_data_snapshot(dataset, rehash_data=not published)
    finally:
        fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        os.close(lock_descriptor)
