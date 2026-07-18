#!/usr/bin/env python3
"""Validate and materialize CIFAR-10 tensors before parallel GPU trials."""

import argparse
import fcntl
import os
import stat
import tempfile
from pathlib import Path


CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
SPLITS = {"train.pt": 50_000, "test.pt": 10_000}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    return parser.parse_args()


def secure_owned_directory(path: Path) -> Path:
    path = path.expanduser()
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink():
        raise SystemExit(f"CIFAR-10 data path must not be a symlink: {path}")
    metadata = path.stat()
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid():
        raise SystemExit(f"CIFAR-10 data path must be an owned directory: {path}")
    if metadata.st_mode & 0o077:
        path.chmod(0o700)
    resolved = path.resolve()
    validate_private_parent_chain(resolved)
    return resolved


def validate_private_parent_chain(path: Path) -> None:
    current = path.parent
    while current != current.parent:
        metadata = current.stat()
        if metadata.st_mode & 0o022:
            trusted_sticky = metadata.st_mode & stat.S_ISVTX and metadata.st_uid in {0, os.getuid()}
            if not trusted_sticky:
                raise SystemExit(f"CIFAR-10 data path has an unsafe writable parent: {current}")
        current = current.parent


def open_lock(data_dir: Path):
    lock_path = data_dir / ".prepare.lock"
    try:
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    except OSError as exc:
        raise SystemExit(f"could not acquire CIFAR-10 preparation lock: {lock_path}") from exc
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid():
        os.close(descriptor)
        raise SystemExit(f"unsafe CIFAR-10 preparation lock: {lock_path}")
    os.fchmod(descriptor, 0o600)
    return os.fdopen(descriptor, "r+")


def secure_cache_file(path: Path) -> bool:
    if path.is_symlink():
        raise SystemExit(f"refusing symlinked CIFAR-10 tensor cache: {path}")
    if not path.exists():
        return False
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid():
        raise SystemExit(f"CIFAR-10 tensor cache must be an owned regular file: {path}")
    if metadata.st_mode & 0o077:
        path.chmod(0o600)
    return True


def valid_tensor_cache(path: Path, expected_examples: int) -> bool:
    if not secure_cache_file(path):
        return False
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return False
    if not isinstance(payload, dict) or set(payload) != {"images", "labels", "classes"}:
        return False
    images = payload["images"]
    labels = payload["labels"]
    return (
        getattr(images, "shape", None) == (expected_examples, 32, 32, 3)
        and getattr(images, "dtype", None) == torch.uint8
        and getattr(labels, "shape", None) == (expected_examples,)
        and getattr(labels, "dtype", None) == torch.int64
        and payload["classes"] == CLASSES
    )


def write_tensor_cache(path: Path, dataset: object) -> None:
    import torch

    payload = {
        "images": torch.tensor(dataset.data),
        "labels": torch.tensor(dataset.targets),
        "classes": dataset.classes,
    }
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        temporary_path.chmod(0o600)
        torch.save(payload, temporary_path)
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        path.chmod(0o600)
        directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary_path.unlink(missing_ok=True)


def prepare_data(data_dir: Path) -> None:
    data_dir = secure_owned_directory(data_dir)
    with open_lock(data_dir) as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for filename in SPLITS:
            secure_cache_file(data_dir / filename)
        from torchvision.datasets import CIFAR10

        write_tensor_cache(data_dir / "train.pt", CIFAR10(data_dir, download=True, train=True))
        write_tensor_cache(data_dir / "test.pt", CIFAR10(data_dir, download=True, train=False))
        if not all(valid_tensor_cache(data_dir / filename, count) for filename, count in SPLITS.items()):
            raise SystemExit("prepared CIFAR-10 tensor cache failed validation")
        print(f"Prepared validated CIFAR-10 tensor cache at {data_dir}")


def main() -> None:
    prepare_data(parse_args().data_dir)


if __name__ == "__main__":
    main()
