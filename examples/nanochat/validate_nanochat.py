#!/usr/bin/env python3
"""Select discovery finalists and evaluate each over a fixed seed panel."""

import argparse
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from nanochat_validation_support import (
    DISCOVERY_SEED,
    FAILURE_VALUE,
    PARAM_FLAGS,
    T_CRITICAL_95,
    normalized_config,
    protocol_matches_benchmark,
    sha256_bytes,
    sha256_file,
    sha256_value,
)


class InterruptedRun(BaseException):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, default=Path(__file__).with_name("nanochat_benchmark.py"))
    parser.add_argument("--finalists", type=int, default=3)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in range(10)))
    parser.add_argument("--timeout-seconds", type=int, default=int(os.environ.get("NANOCHAT_TRIAL_TIMEOUT_SECONDS", "1200")))
    parser.add_argument("--max-attempts", type=int, default=2)
    return parser.parse_args()


def parse_seeds(raw: str) -> list[int]:
    try:
        seeds = [int(value) for value in raw.split(",") if value != ""]
    except ValueError as exc:
        raise SystemExit("--seeds must be a comma-separated integer list") from exc
    if not seeds or len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise SystemExit("--seeds must contain unique non-negative integers")
    if DISCOVERY_SEED in seeds:
        raise SystemExit(f"validation seeds must not include discovery seed {DISCOVERY_SEED}")
    return seeds


def parse_results(values: list[str]) -> dict[str, list[Path]]:
    results: dict[str, list[Path]] = {}
    for value in values:
        label, separator, raw_path = value.partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z0-9_-]+", label):
            raise SystemExit(f"invalid --result value: {value}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"missing discovery results: {path}")
        paths = results.setdefault(label, [])
        if path in paths:
            raise SystemExit(f"duplicate --result path for {label}: {path}")
        paths.append(path)
    return results


def valid_trial(trial: object) -> bool:
    if not isinstance(trial, dict) or trial.get("state") != "COMPLETE":
        return False
    value = trial.get("value")
    number = trial.get("number")
    params = trial.get("params")
    attrs = trial.get("user_attrs", {})
    if isinstance(number, bool) or not isinstance(number, int) or number < 0:
        return False
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 or value >= FAILURE_VALUE:
        return False
    if not isinstance(attrs, dict) or attrs.get("autotune_failure_reason") or attrs.get("autotune_transfer") is True:
        return False
    if not isinstance(params, dict) or set(params) != set(PARAM_FLAGS):
        return False
    return all(
        isinstance(item, (str, int, float))
        and not isinstance(item, bool)
        and (not isinstance(item, float) or math.isfinite(item))
        for item in params.values()
    )


def select_finalists(paths: list[Path], count: int) -> tuple[list[dict], list[dict]]:
    all_trials = []
    sources = []
    for path in paths:
        try:
            source = path.read_bytes()
            result = json.loads(source)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SystemExit(f"invalid discovery results: {path}") from exc
        if result.get("direction") != "minimize" or not isinstance(result.get("all_trials"), list):
            raise SystemExit(f"discovery results must contain a minimizing all_trials list: {path}")
        all_trials.extend({**trial, "_source_path": str(path)} for trial in result["all_trials"] if isinstance(trial, dict))
        sources.append({"path": str(path), "sha256": sha256_bytes(source)})
    trials = sorted(
        (trial for trial in all_trials if valid_trial(trial)),
        key=lambda trial: (float(trial["value"]), int(trial["number"]), sha256_value(trial["params"])),
    )
    selected = []
    seen = set()
    for trial in trials:
        candidate_id = sha256_value(trial["params"])
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        selected.append({
            "candidate_id": candidate_id,
            "source_trial": int(trial["number"]),
            "source_result": trial["_source_path"],
            "discovery_value": float(trial["value"]),
            "params": trial["params"],
        })
        if len(selected) == count:
            break
    if len(selected) != count:
        raise SystemExit(f"needed {count} unique valid finalists from {paths}, found {len(selected)}")
    return selected, sources


def atomic_write(path: Path, value: object) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_immutable(path: Path, value: object) -> None:
    if path.exists():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"invalid existing selection manifest: {path}") from exc
        if current != value:
            raise SystemExit(f"selection manifest drift at {path}; use a new output directory")
        return
    atomic_write(path, value)


def benchmark_argv(benchmark: Path, params: dict) -> list[str]:
    command = [sys.executable, str(benchmark.resolve())]
    for name, flag in PARAM_FLAGS.items():
        command.extend([flag, str(params[name])])
    return command


def attest_discovery(methods: dict, benchmark_sha256: str, data_identity_sha256: str) -> str:
    cache: dict[Path, list[tuple[Path, dict, str]]] = {}
    protocol_hashes = set()
    for candidates in methods.values():
        for candidate in candidates:
            provenance_dir = Path(candidate["source_result"]).parent / "trial_results"
            if provenance_dir not in cache:
                records = []
                for path in sorted(provenance_dir.glob("*.json")):
                    try:
                        source = path.read_bytes()
                        records.append((path, json.loads(source), sha256_bytes(source)))
                    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                        continue
                cache[provenance_dir] = records
            expected_config = normalized_config(candidate["params"])
            attestation = None
            for path, result, source_hash in cache[provenance_dir]:
                protocol = result.get("protocol")
                if (
                    result.get("schema_version") == 1
                    and result.get("status") == "complete"
                    and result.get("seed") == DISCOVERY_SEED
                    and result.get("config") == expected_config
                    and isinstance(result.get("metric"), (int, float))
                    and not isinstance(result.get("metric"), bool)
                    and math.isclose(float(result["metric"]), candidate["discovery_value"], rel_tol=1e-12, abs_tol=1e-12)
                    and protocol_matches_benchmark(protocol, benchmark_sha256, data_identity_sha256)
                    and sha256_value(protocol) == result.get("protocol_sha256")
                ):
                    attestation = {"path": str(path), "sha256": source_hash}
                    candidate["discovery_protocol_sha256"] = result["protocol_sha256"]
                    protocol_hashes.add(result["protocol_sha256"])
                    break
            if attestation is None:
                raise SystemExit(
                    f"no seed-{DISCOVERY_SEED} provenance record attests finalist "
                    f"{candidate['candidate_id']} under {provenance_dir}"
                )
            candidate["discovery_attestation"] = attestation
    if len(protocol_hashes) != 1:
        raise SystemExit(f"discovery trials used different evaluation protocols: {sorted(protocol_hashes)}")
    return protocol_hashes.pop()


def load_valid_attempt(
    job_dir: Path,
    seed: int,
    params: dict,
    benchmark_sha256: str,
    data_identity_sha256: str | None,
) -> dict | None:
    expected_config = normalized_config(params)
    for path in sorted(job_dir.glob("attempt_*/result.json")):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        metric = result.get("metric")
        protocol = result.get("protocol")
        protocol_hash = result.get("protocol_sha256")
        if (
            result.get("schema_version") == 1
            and result.get("status") == "complete"
            and result.get("seed") == seed
            and result.get("config") == expected_config
            and isinstance(metric, (int, float))
            and not isinstance(metric, bool)
            and math.isfinite(metric)
            and metric > 0
            and metric < FAILURE_VALUE
            and protocol_matches_benchmark(protocol, benchmark_sha256, data_identity_sha256)
            and re.fullmatch(r"[0-9a-f]{64}", str(protocol_hash or ""))
            and sha256_value(protocol) == protocol_hash
        ):
            result["result_path"] = str(path)
            return result
    return None


def terminate_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def tee_output(stream, log) -> None:
    for line in iter(stream.readline, ""):
        print(line, end="", flush=True)
        log.write(line)
        log.flush()


def interrupt_run(_signum, _frame) -> None:
    raise InterruptedRun()


def run_attempt(
    benchmark: Path,
    benchmark_sha256: str,
    job_dir: Path,
    seed: int,
    params: dict,
    timeout_seconds: int,
    data_identity_sha256: str | None,
) -> dict | None:
    attempts = [int(path.name.removeprefix("attempt_")) for path in job_dir.glob("attempt_[0-9][0-9][0-9]")]
    attempt_dir = job_dir / f"attempt_{max(attempts, default=0) + 1:03d}"
    attempt_dir.mkdir(mode=0o700, parents=True)
    result_path = attempt_dir / "result.json"
    log_path = attempt_dir / "run.log"
    env = dict(os.environ)
    env["NANOCHAT_BENCHMARK_SEED"] = str(seed)
    env["NANOCHAT_BENCHMARK_RESULT_JSON"] = str(result_path)
    command = benchmark_argv(benchmark, params)
    atomic_write(attempt_dir / "job.json", {"command": command, "seed": seed, "params": params})
    with log_path.open("w", encoding="utf-8") as log:
        previous_sigterm = signal.signal(signal.SIGTERM, interrupt_run)
        process = None
        output_thread = None
        try:
            process = subprocess.Popen(
                command,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            output_thread = threading.Thread(target=tee_output, args=(process.stdout, log))
            output_thread.start()
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            print(f"validation timed out after {timeout_seconds}s for seed {seed}", file=sys.stderr)
            terminate_process_group(process)
            returncode = process.returncode
        except BaseException:
            if process is not None:
                terminate_process_group(process)
            raise
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)
            if output_thread is not None:
                output_thread.join()
    if returncode != 0:
        print(f"validation failed for seed {seed}; see {log_path}", file=sys.stderr)
        return None
    return load_valid_attempt(job_dir, seed, params, benchmark_sha256, data_identity_sha256)


def load_frozen_protocol(output_dir: Path) -> dict | None:
    path = output_dir / "evaluator_protocol.json"
    if not path.exists():
        return None
    try:
        frozen = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid frozen evaluator protocol: {path}") from exc
    if (
        not isinstance(frozen, dict)
        or not isinstance(frozen.get("protocol"), dict)
        or sha256_value(frozen["protocol"]) != frozen.get("protocol_sha256")
    ):
        raise SystemExit(f"invalid frozen evaluator protocol: {path}")
    return frozen


def freeze_or_check_protocol(output_dir: Path, frozen: dict | None, result: dict) -> dict:
    observed = {"protocol": result["protocol"], "protocol_sha256": result["protocol_sha256"]}
    if frozen is None:
        write_immutable(output_dir / "evaluator_protocol.json", observed)
        return observed
    if frozen != observed:
        raise SystemExit(
            f"finalist evaluator protocol drift: expected {frozen['protocol_sha256']}, "
            f"observed {observed['protocol_sha256']}; use a new output directory"
        )
    return frozen


def collect_results(
    selection: dict,
    output_dir: Path,
    benchmark: Path,
    seeds: list[int],
    timeout_seconds: int,
    max_attempts: int,
) -> list[dict]:
    records = []
    frozen_protocol = load_frozen_protocol(output_dir)
    benchmark_sha256 = selection["benchmark"]["sha256"]
    data_identity_sha256 = selection["data_identity_sha256"]
    discovery_protocol_sha256 = selection["discovery_protocol_sha256"]
    for label, candidates in selection["methods"].items():
        for candidate in candidates:
            for seed in seeds:
                job_dir = output_dir / "jobs" / candidate["candidate_id"] / f"seed_{seed}"
                result = load_valid_attempt(
                    job_dir,
                    seed,
                    candidate["params"],
                    benchmark_sha256,
                    data_identity_sha256,
                )
                attempts = len(list(job_dir.glob("attempt_[0-9][0-9][0-9]")))
                while result is None and attempts < max_attempts:
                    result = run_attempt(
                        benchmark,
                        benchmark_sha256,
                        job_dir,
                        seed,
                        candidate["params"],
                        timeout_seconds,
                        data_identity_sha256,
                    )
                    attempts += 1
                if result is None:
                    raise SystemExit(
                        f"validation exhausted {max_attempts} attempts for "
                        f"candidate {candidate['candidate_id']} seed {seed}"
                    )
                if result is not None:
                    if result["protocol_sha256"] != discovery_protocol_sha256:
                        raise SystemExit(
                            f"validation protocol {result['protocol_sha256']} differs from attested discovery "
                            f"protocol {discovery_protocol_sha256}"
                        )
                    frozen_protocol = freeze_or_check_protocol(output_dir, frozen_protocol, result)
                    records.append({"method": label, "candidate_id": candidate["candidate_id"], "seed": seed, **result})
    return records


def summarize(selection: dict, records: list[dict], seeds: list[int]) -> dict:
    expected_jobs = sum(len(candidates) for candidates in selection["methods"].values()) * len(seeds)
    if len(records) != expected_jobs:
        raise SystemExit(f"finalist validation incomplete: expected {expected_jobs} valid jobs, found {len(records)}")
    protocol_hashes = {record["protocol_sha256"] for record in records}
    if len(protocol_hashes) != 1:
        raise SystemExit(f"finalist jobs used different evaluation protocols: {sorted(protocol_hashes)}")
    methods = {}
    for label, candidates in selection["methods"].items():
        summaries = []
        for candidate in candidates:
            candidate_records = [
                record
                for record in records
                if record["method"] == label and record["candidate_id"] == candidate["candidate_id"]
            ]
            metrics = {str(record["seed"]): float(record["metric"]) for record in candidate_records}
            values = [metrics[str(seed)] for seed in seeds]
            stdev = statistics.stdev(values) if len(values) > 1 else 0.0
            sem = stdev / math.sqrt(len(values))
            degrees_freedom = len(values) - 1
            critical = T_CRITICAL_95[degrees_freedom - 1] if 1 <= degrees_freedom <= 30 else 1.96
            summaries.append({
                **candidate,
                "n": len(values),
                "mean": statistics.mean(values),
                "stdev": stdev,
                "sem": sem,
                "ci95": critical * sem,
                "metrics_by_seed": metrics,
            })
        methods[label] = sorted(summaries, key=lambda item: (item["mean"], item["candidate_id"]))
    return {"schema_version": 1, "protocol_sha256": protocol_hashes.pop(), "seeds": seeds, "methods": methods}


def main() -> None:
    args = parse_args()
    if args.finalists < 1:
        raise SystemExit("--finalists must be positive")
    if args.timeout_seconds < 1:
        raise SystemExit("--timeout-seconds must be positive")
    if args.max_attempts < 1:
        raise SystemExit("--max-attempts must be positive")
    seeds = parse_seeds(args.seeds)
    result_paths = parse_results(args.result)
    output_dir = args.output_dir.expanduser().resolve()
    benchmark = args.benchmark.expanduser().resolve()
    if not benchmark.is_file():
        raise SystemExit(f"missing benchmark script: {benchmark}")
    data_identity_sha256 = os.environ.get("NANOCHAT_DATA_IDENTITY_SHA256")
    if not re.fullmatch(r"[0-9a-f]{64}", str(data_identity_sha256 or "")):
        raise SystemExit("NANOCHAT_DATA_IDENTITY_SHA256 must contain the verified discovery data identity")
    methods = {}
    sources = {}
    for label, paths in result_paths.items():
        methods[label], sources[label] = select_finalists(paths, args.finalists)
    benchmark_sha256 = sha256_file(benchmark)
    discovery_protocol_sha256 = attest_discovery(methods, benchmark_sha256, data_identity_sha256)
    selection = {
        "schema_version": 1,
        "discovery_seed": DISCOVERY_SEED,
        "discovery_protocol_sha256": discovery_protocol_sha256,
        "data_identity_sha256": data_identity_sha256,
        "validation_seeds": seeds,
        "timeout_seconds": args.timeout_seconds,
        "max_attempts": args.max_attempts,
        "finalists_per_method": args.finalists,
        "benchmark": {"path": str(benchmark), "sha256": benchmark_sha256},
        "validator_sha256": sha256_file(Path(__file__).resolve()),
        "validator_support_sha256": sha256_file(Path(__file__).with_name("nanochat_validation_support.py")),
        "sources": sources,
        "methods": methods,
    }
    write_immutable(output_dir / "selected_finalists.json", selection)
    records = collect_results(selection, output_dir, benchmark, seeds, args.timeout_seconds, args.max_attempts)
    summary = summarize(selection, records, seeds)
    atomic_write(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
