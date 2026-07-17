"""Shared schema and identity helpers for nanochat finalist validation."""

import hashlib
import json
import re
from pathlib import Path


DISCOVERY_SEED = 42
FAILURE_VALUE = 100.0
DEFAULT_DEVICE_BATCH_SIZE = 128
DEFAULT_TOTAL_BATCH_SIZE = 524288
PINNED_KERNEL_REVISIONS = {
    "varunneal/flash-attention-3": "de87b9b5af06dd9984df595bef90b2eba44b181a",
    "kernels-community/flash-attn3": "9542c462013476380ce4b395b9ddc0e8118161ee",
}
PARAM_FLAGS = {
    "depth": "--depth",
    "aspect_ratio": "--aspect-ratio",
    "head_dim": "--head-dim",
    "batch_config": "--batch-config",
    "embedding_lr": "--embedding-lr",
    "unembedding_lr": "--unembedding-lr",
    "matrix_lr": "--matrix-lr",
    "scalar_lr": "--scalar-lr",
    "weight_decay": "--weight-decay",
    "warmup_ratio": "--warmup-ratio",
    "warmdown_ratio": "--warmdown-ratio",
    "final_lr_frac": "--final-lr-frac",
    "window_pattern": "--window-pattern",
}
REFINEMENT_PARAM_FLAGS = {
    **PARAM_FLAGS,
    "device_batch_size": "--device-batch-size",
    "total_batch_size": "--total-batch-size",
}
DEFAULT_PARAMS = {
    "depth": 8,
    "aspect_ratio": 64,
    "head_dim": 128,
    "batch_config": f"{DEFAULT_DEVICE_BATCH_SIZE}x{DEFAULT_TOTAL_BATCH_SIZE}",
    "embedding_lr": 0.6,
    "unembedding_lr": 0.004,
    "matrix_lr": 0.04,
    "scalar_lr": 0.5,
    "weight_decay": 0.2,
    "warmup_ratio": 0.0,
    "warmdown_ratio": 0.5,
    "final_lr_frac": 0.0,
    "window_pattern": "SSSL",
}
T_CRITICAL_95 = (
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
)


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_value(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def effective_params(params: dict) -> dict:
    unknown = params.keys() - REFINEMENT_PARAM_FLAGS.keys()
    if unknown:
        raise SystemExit(f"unknown finalist parameters: {sorted(unknown)}")
    active_params = {name: value for name, value in params.items() if name in PARAM_FLAGS}
    effective = {**DEFAULT_PARAMS, **active_params}
    if "batch_config" not in params:
        device_batch_size = params.get("device_batch_size", DEFAULT_DEVICE_BATCH_SIZE)
        total_batch_size = params.get("total_batch_size", DEFAULT_TOTAL_BATCH_SIZE)
        effective["batch_config"] = f"{device_batch_size}x{total_batch_size}"
    return effective


def normalized_config(params: dict) -> dict:
    config = effective_params(params)
    batch = config.pop("batch_config")
    match = re.fullmatch(r"([1-9][0-9]*)x([1-9][0-9]*)", str(batch))
    if not match:
        raise SystemExit(f"invalid finalist batch_config: {batch}")
    config["device_batch_size"] = int(match.group(1))
    config["total_batch_size"] = int(match.group(2))
    return config


def protocol_matches_benchmark(protocol: object, benchmark_sha256: str, data_identity_sha256: str | None = None) -> bool:
    if not isinstance(protocol, dict):
        return False
    return (
        protocol.get("time_budget_seconds") == 300
        and protocol.get("eval_tokens") == 40 * 524288
        and protocol.get("max_seq_len") == 2048
        and protocol.get("vocab_size") == 8192
        and protocol.get("gpu_count") == 1
        and protocol.get("autoresearch_commit") == "228791fb499afffb54b46200aca536f79142f117"
        and protocol.get("benchmark_sha256") == benchmark_sha256
        and re.fullmatch(r"[0-9a-f]{64}", str(protocol.get("adapter_sha256", ""))) is not None
        and re.fullmatch(r"[0-9a-f]{64}", str(protocol.get("data_identity_sha256", ""))) is not None
        and (data_identity_sha256 is None or protocol.get("data_identity_sha256") == data_identity_sha256)
        and isinstance(protocol.get("source_sha256"), dict)
        and isinstance(protocol.get("runtime"), dict)
    )
