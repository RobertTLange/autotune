#!/usr/bin/env python3
"""Autotune HPO wrapper for the pinned canonical autoresearch evaluator."""

import argparse
import hashlib
import json
import math
import os
import re
import secrets
import stat
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from nanochat_cache import verify_manifest
from nanochat_validation_support import (
    DEFAULT_DEVICE_BATCH_SIZE,
    DEFAULT_PARAMS,
    DEFAULT_TOTAL_BATCH_SIZE,
    PINNED_KERNEL_REVISIONS,
)


OOM_PENALTY = 100.0
AUTORESEARCH_COMMIT = "228791fb499afffb54b46200aca536f79142f117"
MAX_SEQ_LEN = 2048
TIME_BUDGET_SECONDS = 300
EVAL_TOKENS = 40 * 524288
VOCAB_SIZE = 8192


def env_nonnegative_int(name: str, default: int) -> int:
    value = os.environ.get(name, str(default))
    try:
        parsed = int(value)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer") from exc
    if parsed < 0:
        raise SystemExit(f"{name} must be non-negative")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one canonical autoresearch-compatible HPO trial.")
    parser.add_argument("--depth", type=int, default=DEFAULT_PARAMS["depth"])
    parser.add_argument("--aspect-ratio", type=int, default=DEFAULT_PARAMS["aspect_ratio"])
    parser.add_argument("--head-dim", type=int, default=DEFAULT_PARAMS["head_dim"])
    parser.add_argument("--batch-config", type=str)
    parser.add_argument("--device-batch-size", type=int, default=DEFAULT_DEVICE_BATCH_SIZE)
    parser.add_argument("--total-batch-size", type=int, default=DEFAULT_TOTAL_BATCH_SIZE)
    parser.add_argument("--embedding-lr", type=float, default=DEFAULT_PARAMS["embedding_lr"])
    parser.add_argument("--unembedding-lr", type=float, default=DEFAULT_PARAMS["unembedding_lr"])
    parser.add_argument("--matrix-lr", type=float, default=DEFAULT_PARAMS["matrix_lr"])
    parser.add_argument("--scalar-lr", type=float, default=DEFAULT_PARAMS["scalar_lr"])
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_PARAMS["weight_decay"])
    parser.add_argument("--warmup-ratio", type=float, default=DEFAULT_PARAMS["warmup_ratio"])
    parser.add_argument("--warmdown-ratio", type=float, default=DEFAULT_PARAMS["warmdown_ratio"])
    parser.add_argument("--final-lr-frac", type=float, default=DEFAULT_PARAMS["final_lr_frac"])
    parser.add_argument("--window-pattern", type=str, default=DEFAULT_PARAMS["window_pattern"])
    return parser.parse_args()


def autoresearch_dir() -> Path:
    raw_dir = os.environ.get("AUTORESEARCH_DIR")
    if not raw_dir:
        raise SystemExit("AUTORESEARCH_DIR must point at a pinned karpathy/autoresearch checkout")
    resolved = Path(raw_dir).expanduser().resolve()
    required = ("train.py", "prepare.py", "pyproject.toml", "uv.lock", ".python-version", "README.md")
    for name in required:
        path = resolved / name
        try:
            mode = path.lstat().st_mode
        except OSError as exc:
            raise SystemExit(f"missing canonical {name} under {resolved}") from exc
        if not stat.S_ISREG(mode) or path.is_symlink():
            raise SystemExit(f"canonical {name} must be a regular, non-symlink file")
    return resolved


def run_git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def git_blob(root: Path, name: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(root), "cat-file", "blob", f"{AUTORESEARCH_COMMIT}:{name}"],
        check=True,
        capture_output=True,
    ).stdout


def verify_checkout(root: Path) -> tuple[str, dict[str, str], dict[str, bytes]]:
    try:
        toplevel = Path(run_git(root, "rev-parse", "--show-toplevel")).resolve()
        commit = run_git(root, "rev-parse", "HEAD")
        tracked = {}
        sources = {}
        for name in ("train.py", "prepare.py", "pyproject.toml", "uv.lock", ".python-version", "README.md"):
            expected = git_blob(root, name)
            actual = (root / name).read_bytes()
            if actual != expected:
                raise SystemExit(f"canonical {name} differs from the pinned commit")
            tracked[name] = hashlib.sha256(actual).hexdigest()
            sources[name] = actual
        if run_git(root, "rev-parse", "HEAD") != commit:
            raise SystemExit("autoresearch HEAD changed during checkout verification")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"could not verify clean canonical autoresearch files under {root}") from exc
    if toplevel != root:
        raise SystemExit(f"AUTORESEARCH_DIR must be the git toplevel, found {toplevel}")
    if commit != AUTORESEARCH_COMMIT:
        raise SystemExit(f"expected pinned autoresearch commit {AUTORESEARCH_COMMIT}, found {commit}")
    return commit, tracked, sources


def prepare_uv_project(sources: dict[str, bytes], source_hashes: dict[str, str]) -> Path:
    project_hash = canonical_hash({name: source_hashes[name] for name in ("pyproject.toml", "uv.lock", ".python-version")})
    project = Path.home() / ".cache" / "autotune" / f"nanochat-uv-{project_hash[:16]}"
    project.mkdir(mode=0o700, parents=True, exist_ok=True)
    if project.is_symlink():
        raise SystemExit(f"refusing symlink uv project directory: {project}")
    for name in ("pyproject.toml", "uv.lock", ".python-version", "README.md"):
        path = project / name
        if path.exists():
            if path.is_symlink() or path.read_bytes() != sources[name]:
                raise SystemExit(f"cached uv project differs from pinned {name}: {path}")
            continue
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(sources[name])
    return project


def build_training_command(project: Path, adapter: Path) -> list[str]:
    uv = os.environ.get("AUTORESEARCH_UV", "uv")
    return [uv, "run", "--project", str(project), "--frozen", "python", str(adapter)]


def build_config(args: argparse.Namespace) -> dict[str, int | float | str]:
    device_batch_size, total_batch_size = resolve_batch_config(args)
    tokens_per_microbatch = device_batch_size * MAX_SEQ_LEN
    if total_batch_size < tokens_per_microbatch or total_batch_size % tokens_per_microbatch != 0:
        raise InfeasibleConfig(
            f"total_batch_size={total_batch_size} is incompatible with "
            f"device_batch_size={device_batch_size} and max_seq_len={MAX_SEQ_LEN}"
        )
    eval_tokens_per_batch = device_batch_size * MAX_SEQ_LEN
    if EVAL_TOKENS % eval_tokens_per_batch != 0:
        raise InfeasibleConfig(f"device_batch_size={device_batch_size} does not divide the fixed evaluation token budget")
    config: dict[str, int | float | str] = {
        "depth": args.depth,
        "aspect_ratio": args.aspect_ratio,
        "head_dim": args.head_dim,
        "window_pattern": args.window_pattern,
        "device_batch_size": device_batch_size,
        "total_batch_size": total_batch_size,
        "embedding_lr": args.embedding_lr,
        "unembedding_lr": args.unembedding_lr,
        "matrix_lr": args.matrix_lr,
        "scalar_lr": args.scalar_lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "warmdown_ratio": args.warmdown_ratio,
        "final_lr_frac": args.final_lr_frac,
        "seed": env_nonnegative_int("NANOCHAT_BENCHMARK_SEED", 42),
    }
    validate_config(config)
    return config


def validate_config(config: dict[str, int | float | str]) -> None:
    integer_ranges = {
        "depth": (1, 128),
        "aspect_ratio": (1, 1024),
        "head_dim": (1, 1024),
        "device_batch_size": (1, 65536),
        "total_batch_size": (1, 2**31),
        "seed": (0, 2**63 - 1),
    }
    for name, (low, high) in integer_ranges.items():
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise InfeasibleConfig(f"{name} must be an integer in [{low}, {high}]")
    float_ranges = {
        "embedding_lr": (0.0, 10.0),
        "unembedding_lr": (0.0, 10.0),
        "matrix_lr": (0.0, 10.0),
        "scalar_lr": (0.0, 10.0),
        "weight_decay": (0.0, 1.0),
        "warmup_ratio": (0.0, 1.0),
        "warmdown_ratio": (0.0, 1.0),
        "final_lr_frac": (0.0, 1.0),
    }
    for name, (low, high) in float_ranges.items():
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
            raise InfeasibleConfig(f"{name} must be finite and in [{low}, {high}]")
    if config["window_pattern"] not in {"SSSL", "LLLL", "SLSL", "SSSS"}:
        raise InfeasibleConfig("window_pattern is not an allowed canonical pattern")


def resolve_batch_config(args: argparse.Namespace) -> tuple[int, int]:
    if not args.batch_config:
        return args.device_batch_size, args.total_batch_size
    match = re.fullmatch(r"([1-9][0-9]*)x([1-9][0-9]*)", args.batch_config)
    if not match:
        raise InfeasibleConfig("batch_config must be formatted as <device_batch_size>x<total_batch_size>")
    return int(match.group(1)), int(match.group(2))


def parse_val_bpb(output: str) -> float:
    matches = re.findall(r"(?m)^val_bpb:\s*([0-9]+(?:\.[0-9]+)?)\s*$", output)
    if not matches:
        raise ValueError("could not parse canonical val_bpb summary")
    return float(matches[-1])


def is_oom(output: str) -> bool:
    lowered = output.lower()
    return "outofmemoryerror" in lowered or "out of memory" in lowered or "cuda error: out of memory" in lowered


class InfeasibleConfig(Exception):
    pass


class StreamCapture:
    def __init__(self, max_chars: int = 262144):
        self.max_chars = max_chars
        self.text = ""
        self._lock = threading.Lock()

    def append(self, chunk: str) -> None:
        with self._lock:
            self.text = (self.text + chunk)[-self.max_chars:]


def mirror_stream(stream, target, capture: StreamCapture) -> None:
    for chunk in iter(lambda: stream.readline(), ""):
        capture.append(chunk)
        print(chunk, end="", file=target, flush=True)


def run_nanochat(command: list[str], cwd: Path, env: dict[str, str]) -> tuple[int, str]:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout_capture = StreamCapture()
    stderr_capture = StreamCapture()
    stdout_thread = threading.Thread(target=mirror_stream, args=(process.stdout, sys.stdout, stdout_capture))
    stderr_thread = threading.Thread(target=mirror_stream, args=(process.stderr, sys.stderr, stderr_capture))
    stdout_thread.start()
    stderr_thread.start()
    returncode = process.wait()
    stdout_thread.join()
    stderr_thread.join()
    return returncode, f"{stdout_capture.text}\n{stderr_capture.text}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_training_summary(output: str) -> dict[str, int | float]:
    fields = {
        "training_seconds": (r"(?m)^training_seconds:\s*([0-9.]+)$", float),
        "total_tokens_m": (r"(?m)^total_tokens_M:\s*([0-9.]+)$", float),
        "num_steps": (r"(?m)^num_steps:\s*([0-9]+)$", int),
    }
    summary: dict[str, int | float] = {}
    for name, (pattern, parser) in fields.items():
        match = re.search(pattern, output)
        if match:
            summary[name] = parser(match.group(1))
    missing = fields.keys() - summary.keys()
    if missing:
        raise ValueError(f"canonical summary is missing fields: {sorted(missing)}")
    if summary["training_seconds"] < TIME_BUDGET_SECONDS:
        raise ValueError(f"canonical training stopped before {TIME_BUDGET_SECONDS} measured seconds")
    if summary["total_tokens_m"] <= 0 or summary["num_steps"] <= 10:
        raise ValueError("canonical training summary has invalid token or step totals")
    return summary


def parse_execution_provenance(output: str) -> tuple[str, dict]:
    materialized = re.findall(r"(?m)^autotune_materialized_sha256=([0-9a-f]{64})$", output)
    runtime_matches = re.findall(r"(?m)^autotune_runtime=(\{.*\})$", output)
    if len(materialized) != 1 or len(runtime_matches) != 1:
        raise ValueError("canonical adapter did not emit unique execution provenance")
    try:
        runtime = json.loads(runtime_matches[0])
    except json.JSONDecodeError as exc:
        raise ValueError("canonical adapter emitted invalid runtime provenance") from exc
    required = {"python", "torch", "cuda", "kernels_package", "loaded_kernels", "gpu_name", "gpu_capability"}
    if not isinstance(runtime, dict) or set(runtime) != required:
        raise ValueError("canonical adapter runtime provenance has an unexpected schema")
    validate_kernel_provenance(runtime["loaded_kernels"])
    return materialized[0], runtime


def validate_kernel_provenance(loaded_kernels: object) -> None:
    if not isinstance(loaded_kernels, list) or len(loaded_kernels) != 1:
        raise ValueError("canonical adapter did not attest one pinned kernel")
    loaded_kernel = loaded_kernels[0]
    if not isinstance(loaded_kernel, dict):
        raise ValueError("canonical adapter emitted invalid kernel provenance")
    expected_revision = PINNED_KERNEL_REVISIONS.get(loaded_kernel.get("repo_id"))
    observed_revision = loaded_kernel.get("revision")
    observed_snapshot = loaded_kernel.get("snapshot_commit")
    if expected_revision is None or observed_revision != expected_revision or observed_snapshot != expected_revision:
        raise ValueError("canonical adapter kernel provenance does not match a pinned revision")


def write_result(path: Path, result: dict) -> None:
    if path.is_symlink():
        raise SystemExit(f"refusing to replace symlink result path: {path}")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.parent.is_symlink():
        raise SystemExit(f"refusing symlink result directory: {path.parent}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o600)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def child_environment(config: dict, root: Path, source_hashes: dict[str, str], dataset: dict) -> dict[str, str]:
    allowed = {
        "PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TZ",
        "CUDA_VISIBLE_DEVICES", "CUDA_HOME", "LD_LIBRARY_PATH", "LIBRARY_PATH", "CPATH",
        "TMPDIR", "XDG_CACHE_HOME", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR",
        "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE",
    }
    env = {name: value for name, value in os.environ.items() if name in allowed}
    env["AUTORESEARCH_DIR"] = str(root)
    env["AUTOTUNE_NANOCHAT_CONFIG"] = json.dumps(config, sort_keys=True, separators=(",", ":"))
    env["AUTOTUNE_AUTORESEARCH_TRAIN_SHA256"] = source_hashes["train.py"]
    env["AUTOTUNE_AUTORESEARCH_PREPARE_SHA256"] = source_hashes["prepare.py"]
    env["AUTOTUNE_TOKENIZER_PKL_SHA256"] = dataset["tokenizer_sha256"]["tokenizer.pkl"]
    env["AUTOTUNE_TOKEN_BYTES_SHA256"] = dataset["tokenizer_sha256"]["token_bytes.pt"]
    return env


def main() -> int:
    args = parse_args()
    try:
        config = build_config(args)
    except InfeasibleConfig as exc:
        print(str(exc), file=sys.stderr)
        print(f"autotune_metric={OOM_PENALTY}")
        return 1
    root = autoresearch_dir()
    commit, source_hashes, sources = verify_checkout(root)
    expected_data_identity = os.environ.get("NANOCHAT_DATA_IDENTITY_SHA256")
    if not expected_data_identity:
        raise SystemExit("NANOCHAT_DATA_IDENTITY_SHA256 is required; run prepare_nanochat_cache.py verify")
    dataset = verify_manifest(expected_identity=expected_data_identity)
    adapter = Path(__file__).with_name("autoresearch_train.py").resolve()
    uv_project = prepare_uv_project(sources, source_hashes)
    command = build_training_command(uv_project, adapter)
    env = child_environment(config, root, source_hashes, dataset)
    returncode, output = run_nanochat(command, adapter.parent, env)
    if returncode != 0:
        if is_oom(output):
            print(f"autotune_metric={OOM_PENALTY}")
            return returncode
        return returncode

    try:
        metric = parse_val_bpb(output)
        training = parse_training_summary(output)
        materialized_sha256, runtime = parse_execution_provenance(output)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    protocol = {
        "time_budget_seconds": TIME_BUDGET_SECONDS,
        "eval_tokens": EVAL_TOKENS,
        "max_seq_len": MAX_SEQ_LEN,
        "vocab_size": VOCAB_SIZE,
        "gpu_count": 1,
        "autoresearch_commit": commit,
        "source_sha256": source_hashes,
        "benchmark_sha256": sha256_file(Path(__file__).resolve()),
        "adapter_sha256": sha256_file(adapter),
        "data_identity_sha256": dataset["identity_sha256"],
        "runtime": runtime,
    }
    result = {
        "schema_version": 1,
        "status": "complete",
        "metric": metric,
        "seed": config["seed"],
        "config": {key: value for key, value in config.items() if key != "seed"},
        "protocol": protocol,
        "protocol_sha256": canonical_hash(protocol),
        "autoresearch_commit": commit,
        "source_sha256": source_hashes,
        "adapter_sha256": protocol["adapter_sha256"],
        "materialized_sha256": materialized_sha256,
        "dataset": dataset,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "training": training,
    }
    result_path = os.environ.get("NANOCHAT_BENCHMARK_RESULT_JSON")
    if not result_path and (results_dir := os.environ.get("NANOCHAT_BENCHMARK_RESULTS_DIR")):
        run_id = f"{canonical_hash(config)[:12]}-{os.getpid()}-{secrets.token_hex(4)}"
        result_path = str(Path(results_dir).expanduser() / f"{run_id}.json")
    if result_path:
        result_file = Path(result_path).expanduser()
        if not result_file.is_absolute():
            result_file = Path.cwd() / result_file
        write_result(result_file, result)
    print(f"nanochat_protocol={json.dumps(result, sort_keys=True, separators=(',', ':'))}")
    print(f"autotune_metric={metric}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
