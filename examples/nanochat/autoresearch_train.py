#!/usr/bin/env python3
"""Run pinned autoresearch train.py with a fixed, auditable HPO configuration."""

import ast
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
import tempfile
from pathlib import Path

from nanochat_validation_support import PINNED_KERNEL_REVISIONS


CONFIG_TO_CONSTANT = {
    "aspect_ratio": "ASPECT_RATIO",
    "head_dim": "HEAD_DIM",
    "window_pattern": "WINDOW_PATTERN",
    "total_batch_size": "TOTAL_BATCH_SIZE",
    "embedding_lr": "EMBEDDING_LR",
    "unembedding_lr": "UNEMBEDDING_LR",
    "matrix_lr": "MATRIX_LR",
    "scalar_lr": "SCALAR_LR",
    "weight_decay": "WEIGHT_DECAY",
    "warmup_ratio": "WARMUP_RATIO",
    "warmdown_ratio": "WARMDOWN_RATIO",
    "final_lr_frac": "FINAL_LR_FRAC",
    "depth": "DEPTH",
    "device_batch_size": "DEVICE_BATCH_SIZE",
}
REQUIRED_KEYS = set(CONFIG_TO_CONSTANT) | {"seed"}


def load_config() -> dict[str, int | float | str]:
    raw = os.environ.get("AUTOTUNE_NANOCHAT_CONFIG")
    if not raw:
        raise SystemExit("AUTOTUNE_NANOCHAT_CONFIG is required")
    try:
        config = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid AUTOTUNE_NANOCHAT_CONFIG: {exc}") from exc
    if not isinstance(config, dict) or set(config) != REQUIRED_KEYS:
        raise SystemExit("AUTOTUNE_NANOCHAT_CONFIG does not match the HPO parameter schema")
    return config


def autoresearch_root() -> Path:
    raw = os.environ.get("AUTORESEARCH_DIR")
    if not raw:
        raise SystemExit("AUTORESEARCH_DIR is required")
    root = Path(raw).expanduser().resolve()
    if not (root / "train.py").is_file():
        raise SystemExit(f"missing canonical train.py under {root}")
    return root


def replace_once(source: str, pattern: str, replacement: str, label: str) -> str:
    updated, count = re.subn(pattern, replacement, source, flags=re.MULTILINE)
    if count != 1:
        raise SystemExit(f"canonical train.py drifted: expected one {label}, found {count}")
    return updated


def materialize_source(source: str, config: dict[str, int | float | str]) -> str:
    updated = source
    for key, constant in CONFIG_TO_CONSTANT.items():
        updated = replace_once(
            updated,
            rf"^{constant}\s*=.*$",
            f"{constant} = {config[key]!r}",
            constant,
        )
    seed = config["seed"]
    updated = replace_once(updated, r"^torch\.manual_seed\(42\)\s*$", f"torch.manual_seed({seed!r})", "CPU seed")
    updated = replace_once(
        updated,
        r"^torch\.cuda\.manual_seed\(42\)\s*$",
        f"torch.cuda.manual_seed({seed!r})",
        "CUDA seed",
    )
    pinned_revisions = repr(PINNED_KERNEL_REVISIONS)
    updated = replace_once(
        updated,
        r"^fa3 = get_kernel\(repo\)\.flash_attn_interface\s*$",
        f"fa3 = get_kernel(repo, revision={pinned_revisions}[repo]).flash_attn_interface",
        "kernel revision",
    )
    ast.parse(updated, filename="train.py")
    return updated


def verify_source(source: str, expected_name: str, label: str) -> None:
    expected = os.environ.get(expected_name)
    if expected and hashlib.sha256(source.encode("utf-8")).hexdigest() != expected:
        raise SystemExit(f"canonical {label} changed after checkout verification")


def loaded_kernel_provenance(globals_dict: dict) -> list[dict[str, str]]:
    repo_id = globals_dict.get("repo")
    kernel_file = str(getattr(globals_dict.get("fa3"), "__file__", ""))
    snapshot_match = re.search(r"/snapshots/([0-9a-f]{40,64})/", kernel_file)
    if not isinstance(repo_id, str) or repo_id not in PINNED_KERNEL_REVISIONS:
        raise RuntimeError("canonical kernel repository is unavailable")
    expected_revision = PINNED_KERNEL_REVISIONS[repo_id]
    if snapshot_match is None or snapshot_match.group(1) != expected_revision:
        raise RuntimeError("canonical kernel did not load from the pinned kernel snapshot")
    return [{
        "repo_id": repo_id,
        "revision": expected_revision,
        "snapshot_commit": expected_revision,
        "metadata_id": Path(kernel_file).stem,
    }]


def emit_execution_provenance(materialized: str, globals_dict: dict) -> None:
    if not os.environ.get("AUTOTUNE_AUTORESEARCH_TRAIN_SHA256"):
        return

    torch = globals_dict["torch"]
    runtime = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "kernels_package": importlib.metadata.version("kernels"),
        "loaded_kernels": loaded_kernel_provenance(globals_dict),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
    }
    print(f"autotune_materialized_sha256={hashlib.sha256(materialized.encode('utf-8')).hexdigest()}")
    print(f"autotune_runtime={json.dumps(runtime, sort_keys=True, separators=(',', ':'))}")


def snapshot_cache(snapshot: Path) -> None:
    live_home = Path.home()
    live_cache = live_home / ".cache" / "autoresearch"
    private_home = snapshot / "home"
    private_cache = private_home / ".cache" / "autoresearch"
    tokenizer_dir = private_cache / "tokenizer"
    tokenizer_dir.mkdir(mode=0o700, parents=True)
    expected = {
        "tokenizer.pkl": os.environ.get("AUTOTUNE_TOKENIZER_PKL_SHA256"),
        "token_bytes.pt": os.environ.get("AUTOTUNE_TOKEN_BYTES_SHA256"),
    }
    for name, digest in expected.items():
        source = (live_cache / "tokenizer" / name).read_bytes()
        if not digest or hashlib.sha256(source).hexdigest() != digest:
            raise SystemExit(f"canonical {name} changed after cache verification")
        destination = tokenizer_dir / name
        destination.write_bytes(source)
        destination.chmod(0o600)
    (private_cache / "data").symlink_to(live_cache / "data", target_is_directory=True)
    os.environ.setdefault("HF_HOME", str(live_home / ".cache" / "huggingface"))
    os.environ["HOME"] = str(private_home)


def main() -> None:
    root = autoresearch_root()
    config = load_config()
    source = (root / "train.py").read_text(encoding="utf-8")
    prepare_source = (root / "prepare.py").read_text(encoding="utf-8")
    verify_source(source, "AUTOTUNE_AUTORESEARCH_TRAIN_SHA256", "train.py")
    verify_source(prepare_source, "AUTOTUNE_AUTORESEARCH_PREPARE_SHA256", "prepare.py")
    materialized = materialize_source(source, config)
    with tempfile.TemporaryDirectory(prefix="autotune-autoresearch-") as temporary:
        snapshot = Path(temporary)
        train_path = snapshot / "train.py"
        prepare_path = snapshot / "prepare.py"
        train_path.write_text(materialized, encoding="utf-8")
        prepare_path.write_text(prepare_source, encoding="utf-8")
        train_path.chmod(0o600)
        prepare_path.chmod(0o600)
        sys.path.insert(0, str(snapshot))
        if os.environ.get("AUTOTUNE_AUTORESEARCH_TRAIN_SHA256"):
            snapshot_cache(snapshot)
        os.chdir(snapshot)
        globals_dict = {"__name__": "__main__", "__file__": str(train_path)}
        exec(compile(materialized, str(train_path), "exec"), globals_dict)
        emit_execution_provenance(materialized, globals_dict)


if __name__ == "__main__":
    main()
