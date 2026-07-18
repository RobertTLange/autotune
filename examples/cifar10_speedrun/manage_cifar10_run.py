#!/usr/bin/env python3
"""Check CIFAR-10 discovery results and attest their immutable completion."""

import argparse
import hashlib
import json
import os
import re
import stat
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check")
    check.add_argument("--root", type=Path, required=True)
    check.add_argument("--result", action="append", default=[], metavar="LABEL=EXPECTED:PATH")

    create = commands.add_parser("create")
    create.add_argument("--manifest", type=Path, required=True)
    create.add_argument("--refine-rounds", type=int, required=True)
    create.add_argument("--result", action="append", default=[], metavar="LABEL=EXPECTED:PATH")

    verify = commands.add_parser("verify")
    verify.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def read_result(path: Path) -> tuple[bytes, int, tuple[int, int]]:
    payload, identity = read_safe_file(path, "discovery result")
    try:
        result = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid discovery result: {path}") from exc

    trials = result.get("all_trials") if isinstance(result, dict) else None
    if not isinstance(trials, list) or not all(isinstance(trial, dict) for trial in trials):
        raise SystemExit(f"invalid discovery trials: {path}")
    attempted = sum(
        1
        for trial in trials
        if not isinstance(trial.get("user_attrs"), dict)
        or trial["user_attrs"].get("autotune_transfer") is not True
    )
    return payload, attempted, identity


def parse_result_spec(raw: str, root: Path) -> dict[str, object]:
    label, label_separator, expected_path = raw.partition("=")
    expected_raw, count_separator, raw_path = expected_path.partition(":")
    if (
        not label_separator
        or not count_separator
        or not re.fullmatch(r"[A-Za-z0-9_-]+", label)
        or not re.fullmatch(r"(0|[1-9][0-9]*)", expected_raw)
    ):
        raise SystemExit(f"invalid --result value: {raw}")

    expected = int(expected_raw)
    unresolved_path = Path(raw_path).expanduser().absolute()
    if unresolved_path.is_symlink():
        raise SystemExit(f"discovery result must not be a symlink: {unresolved_path}")
    path = unresolved_path.resolve()
    try:
        relative_path = path.relative_to(root)
    except ValueError as exc:
        raise SystemExit(f"discovery result must be under {root}: {path}") from exc

    payload, attempted, identity = read_result(path)
    if attempted != expected:
        raise SystemExit(f"{label} attempted {attempted} new trials; expected {expected}")
    return {
        "label": label,
        "path": str(relative_path),
        "expected_new_trials": expected,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "_identity": identity,
    }


def parse_results(raw_results: list[str], root: Path) -> list[dict[str, object]]:
    if not raw_results:
        raise SystemExit("at least one --result is required")
    results = [parse_result_spec(raw, root) for raw in raw_results]
    labels = [result["label"] for result in results]
    if len(labels) != len(set(labels)):
        raise SystemExit("discovery result labels must be unique")
    paths = [result["path"] for result in results]
    identities = [result.pop("_identity") for result in results]
    if len(paths) != len(set(paths)) or len(identities) != len(set(identities)):
        raise SystemExit("discovery result files must be unique")
    return results


def check_results(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    parse_results(args.result, root)


def create_manifest(args: argparse.Namespace) -> None:
    if not 0 <= args.refine_rounds <= 1000:
        raise SystemExit("--refine-rounds must be between 0 and 1000")
    unresolved_path = args.manifest.expanduser().absolute()
    if unresolved_path.is_symlink():
        raise SystemExit(f"discovery manifest must not be a symlink: {unresolved_path}")
    manifest_path = unresolved_path.parent.resolve() / unresolved_path.name
    results = parse_results(args.result, manifest_path.parent)
    expected_labels = completion_labels(args.refine_rounds)
    if {result["label"] for result in results} != expected_labels:
        raise SystemExit("discovery results do not match the expected four-arm topology")
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "refine_rounds": args.refine_rounds,
        "results": results,
    }
    payload = f"{json.dumps(manifest, indent=2, sort_keys=True)}\n"
    publish_manifest(manifest_path, payload)


def verify_manifest(args: argparse.Namespace) -> None:
    unresolved_path = args.manifest.expanduser().absolute()
    if unresolved_path.is_symlink():
        raise SystemExit(f"discovery manifest must not be a symlink: {unresolved_path}")
    manifest_path = unresolved_path.resolve()
    try:
        manifest_payload, _ = read_safe_file(manifest_path, "discovery manifest")
        manifest = json.loads(manifest_payload)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid discovery manifest: {manifest_path}") from exc
    if not valid_manifest(manifest):
        raise SystemExit(f"invalid discovery manifest: {manifest_path}")

    root = manifest_path.parent
    result_identities = set()
    for result in manifest["results"]:
        unresolved_result = root / result["path"]
        if unresolved_result.is_symlink():
            raise SystemExit(f"discovery result must not be a symlink: {unresolved_result}")
        path = unresolved_result.resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise SystemExit(f"invalid discovery result path: {path}") from exc
        payload, attempted, identity = read_result(path)
        if identity in result_identities:
            raise SystemExit(f"discovery result files must be unique: {path}")
        result_identities.add(identity)
        if hashlib.sha256(payload).hexdigest() != result.get("sha256"):
            raise SystemExit(f"discovery result changed after discovery completed: {path}")
        if attempted != result.get("expected_new_trials"):
            raise SystemExit(f"discovery trial count changed after discovery completed: {path}")
    print(manifest["refine_rounds"])


def valid_manifest(manifest: object) -> bool:
    if not isinstance(manifest, dict):
        return False
    rounds = manifest.get("refine_rounds")
    results = manifest.get("results")
    if not (
        set(manifest) == {"schema_version", "status", "refine_rounds", "results"}
        and manifest.get("schema_version") == 1
        and manifest.get("status") == "complete"
        and isinstance(rounds, int)
        and not isinstance(rounds, bool)
        and 0 <= rounds <= 1000
        and isinstance(results, list)
        and bool(results)
    ):
        return False
    expected_labels = completion_labels(rounds)
    if len(results) != len(expected_labels) or not all(valid_manifest_result(result) for result in results):
        return False
    return {result["label"] for result in results} == expected_labels


def valid_manifest_result(result: object) -> bool:
    if not isinstance(result, dict) or set(result) != {
        "label", "path", "expected_new_trials", "sha256"
    }:
        return False
    count = result["expected_new_trials"]
    raw_path = result["path"]
    return (
        isinstance(result["label"], str)
        and isinstance(raw_path, str)
        and bool(raw_path)
        and not Path(raw_path).is_absolute()
        and ".." not in Path(raw_path).parts
        and isinstance(count, int)
        and not isinstance(count, bool)
        and count > 0
        and isinstance(result["sha256"], str)
        and re.fullmatch(r"[0-9a-f]{64}", result["sha256"]) is not None
    )


def completion_labels(refine_rounds: int) -> set[str]:
    labels = {"optuna_baseline", "centaur"}
    for round_index in range(refine_rounds + 1):
        labels.add(f"resets_no_transfer_round_{round_index}")
        labels.add(f"resets_trial_transfer_round_{round_index}")
    return labels


def read_safe_file(path: Path, description: str) -> tuple[bytes, tuple[int, int]]:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        raise SystemExit(f"invalid {description}: {path}") from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_mode & 0o022
        ):
            raise SystemExit(f"unsafe {description}: {path}")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            return handle.read(), (metadata.st_dev, metadata.st_ino)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def publish_manifest(manifest_path: Path, payload: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{manifest_path.name}.", dir=manifest_path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, manifest_path)
        except FileExistsError as exc:
            raise SystemExit(f"discovery manifest already exists: {manifest_path}") from exc
        directory_descriptor = os.open(manifest_path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    if args.command == "check":
        check_results(args)
    elif args.command == "create":
        create_manifest(args)
    else:
        verify_manifest(args)


if __name__ == "__main__":
    main()
