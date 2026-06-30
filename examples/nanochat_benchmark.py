#!/usr/bin/env python3
"""Autotune wrapper for the nanochat/autoresearch val_bpb benchmark."""

import argparse
import os
import re
import subprocess
import sys
import threading
from pathlib import Path


OOM_PENALTY = 100.0


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer") from exc


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one paper-inspired nanochat benchmark trial.")
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--aspect-ratio", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--device-batch-size", type=int, default=128)
    parser.add_argument("--total-batch-size", type=int, default=524288)
    parser.add_argument("--embedding-lr", type=float, default=0.6)
    parser.add_argument("--unembedding-lr", type=float, default=0.004)
    parser.add_argument("--matrix-lr", type=float, default=0.04)
    parser.add_argument("--scalar-lr", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=0.2)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--warmdown-ratio", type=float, default=0.5)
    parser.add_argument("--final-lr-frac", type=float, default=0.0)
    parser.add_argument("--window-pattern", type=str, default="SSSL")
    return parser.parse_args()


def nanochat_dir() -> Path:
    raw_dir = os.environ.get("NANOCHAT_DIR")
    if not raw_dir:
        raise SystemExit("NANOCHAT_DIR must point at a local karpathy/nanochat checkout")
    resolved = Path(raw_dir).expanduser().resolve()
    if not (resolved / "scripts" / "base_train.py").is_file():
        raise SystemExit(f"NANOCHAT_DIR does not look like nanochat: {resolved}")
    return resolved


def nanochat_python(root: Path) -> str:
    return os.environ.get("NANOCHAT_PYTHON", str(root / ".venv" / "bin" / "python"))


def build_command(args: argparse.Namespace) -> tuple[list[str], Path]:
    root = nanochat_dir()
    num_iterations = env_int("NANOCHAT_BENCHMARK_NUM_ITERATIONS", 500)
    warmup_steps = round(args.warmup_ratio * num_iterations)
    max_seq_len = env_int("NANOCHAT_BENCHMARK_MAX_SEQ_LEN", 2048)
    tokens_per_microbatch = args.device_batch_size * max_seq_len
    if args.total_batch_size < tokens_per_microbatch or args.total_batch_size % tokens_per_microbatch != 0:
        raise InfeasibleConfig(
            f"total_batch_size={args.total_batch_size} is incompatible with "
            f"device_batch_size={args.device_batch_size} and max_seq_len={max_seq_len}"
        )
    eval_every = env_int("NANOCHAT_BENCHMARK_EVAL_EVERY", max(1, num_iterations))
    eval_tokens = env_int("NANOCHAT_BENCHMARK_EVAL_TOKENS", 524288)
    run_name = env_str("NANOCHAT_BENCHMARK_RUN", "dummy")
    model_tag = env_str("NANOCHAT_BENCHMARK_MODEL_TAG", f"autotune-nanochat-{os.getpid()}")
    device_type = os.environ.get("NANOCHAT_BENCHMARK_DEVICE_TYPE")

    command = [
        nanochat_python(root),
        "-m",
        "scripts.base_train",
        "--run",
        run_name,
        "--depth",
        str(args.depth),
        "--aspect-ratio",
        str(args.aspect_ratio),
        "--head-dim",
        str(args.head_dim),
        "--max-seq-len",
        str(max_seq_len),
        "--window-pattern",
        args.window_pattern,
        "--num-iterations",
        str(num_iterations),
        "--device-batch-size",
        str(args.device_batch_size),
        "--total-batch-size",
        str(args.total_batch_size),
        "--embedding-lr",
        str(args.embedding_lr),
        "--unembedding-lr",
        str(args.unembedding_lr),
        "--weight-decay",
        str(args.weight_decay),
        "--matrix-lr",
        str(args.matrix_lr),
        "--scalar-lr",
        str(args.scalar_lr),
        "--warmup-steps",
        str(warmup_steps),
        "--warmdown-ratio",
        str(args.warmdown_ratio),
        "--final-lr-frac",
        str(args.final_lr_frac),
        "--eval-every",
        str(eval_every),
        "--eval-tokens",
        str(eval_tokens),
        "--core-metric-every",
        "-1",
        "--sample-every",
        "-1",
        "--save-every",
        "-1",
        "--model-tag",
        model_tag,
    ]
    if device_type:
        command.extend(["--device-type", device_type])
    return command, root


def parse_val_bpb(output: str) -> float:
    patterns = [
        r"Minimum validation bpb:\s*([0-9.]+)",
        r"Final validation bpb:\s*([0-9.]+)",
        r"Validation bpb:\s*([0-9.]+)",
        r"val_bpb:\s*([0-9.]+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, output)
        if matches:
            return float(matches[-1])
    raise ValueError("could not parse nanochat validation bpb")


def is_oom(output: str) -> bool:
    lowered = output.lower()
    return "outofmemoryerror" in output or "out of memory" in lowered or "cuda error: out of memory" in lowered


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


def run_nanochat(command: list[str], root: Path, env: dict[str, str]) -> tuple[int, str]:
    process = subprocess.Popen(
        command,
        cwd=root,
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


def main() -> int:
    args = parse_args()
    try:
        command, root = build_command(args)
    except InfeasibleConfig as exc:
        print(str(exc), file=sys.stderr)
        print(f"autotune_metric={OOM_PENALTY}")
        return 0
    env = dict(os.environ)
    env.setdefault("NANOCHAT_BASE_DIR", "/tmp/autotune-nanochat")
    returncode, output = run_nanochat(command, root, env)
    if returncode != 0:
        if is_oom(output):
            print(f"autotune_metric={OOM_PENALTY}")
            return 0
        return returncode

    try:
        print(f"autotune_metric={parse_val_bpb(output)}")
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
