#!/usr/bin/env python3
"""Create and verify the immutable Nanochat discovery-completion manifest."""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--manifest", type=Path, required=True)
    create.add_argument("--refine-rounds", type=int, required=True)
    create.add_argument("--result", action="append", default=[], metavar="LABEL=EXPECTED:PATH")
    check = commands.add_parser("check")
    check.add_argument("--root", type=Path, required=True)
    check.add_argument("--result", action="append", default=[], metavar="LABEL=EXPECTED:PATH")
    verify = commands.add_parser("verify")
    verify.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def read_result(path: Path) -> tuple[bytes, int]:
    try:
        payload = path.read_bytes()
        result = json.loads(payload)
    except (OSError, json.JSONDecodeError) as exc:
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
    return payload, attempted


def parse_result_spec(raw: str, root: Path) -> dict:
    label, separator, expected_path = raw.partition("=")
    expected_raw, count_separator, raw_path = expected_path.partition(":")
    if not separator or not count_separator or not re.fullmatch(r"[A-Za-z0-9_-]+", label):
        raise SystemExit(f"invalid --result value: {raw}")
    try:
        expected = int(expected_raw)
    except ValueError as exc:
        raise SystemExit(f"invalid expected trial count: {raw}") from exc
    if expected < 0:
        raise SystemExit(f"invalid expected trial count: {raw}")
    path = Path(raw_path).expanduser().resolve()
    try:
        relative_path = path.relative_to(root)
    except ValueError as exc:
        raise SystemExit(f"discovery result must be under {root}: {path}") from exc
    payload, attempted = read_result(path)
    if attempted != expected:
        raise SystemExit(f"{label} attempted {attempted} new trials; expected {expected}")
    return {
        "label": label,
        "path": str(relative_path),
        "expected_new_trials": expected,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def create_manifest(args: argparse.Namespace) -> None:
    if args.refine_rounds < 0 or not args.result:
        raise SystemExit("create requires non-negative --refine-rounds and at least one --result")
    manifest_path = args.manifest.expanduser().resolve()
    root = manifest_path.parent
    results = [parse_result_spec(raw, root) for raw in args.result]
    labels = [result["label"] for result in results]
    if len(labels) != len(set(labels)):
        raise SystemExit("discovery result labels must be unique")
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "refine_rounds": args.refine_rounds,
        "results": results,
    }
    payload = f"{json.dumps(manifest, indent=2, sort_keys=True)}\n"
    try:
        descriptor = os.open(manifest_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise SystemExit(f"discovery manifest already exists: {manifest_path}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def check_results(args: argparse.Namespace) -> None:
    if not args.result:
        raise SystemExit("check requires at least one --result")
    root = args.root.expanduser().resolve()
    for raw in args.result:
        parse_result_spec(raw, root)


def verify_manifest(args: argparse.Namespace) -> None:
    manifest_path = args.manifest.expanduser().resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid discovery manifest: {manifest_path}") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or manifest.get("status") != "complete"
        or isinstance(manifest.get("refine_rounds"), bool)
        or not isinstance(manifest.get("refine_rounds"), int)
        or manifest["refine_rounds"] < 0
        or not isinstance(manifest.get("results"), list)
        or not manifest["results"]
    ):
        raise SystemExit(f"invalid discovery manifest: {manifest_path}")
    root = manifest_path.parent
    for result in manifest["results"]:
        if not isinstance(result, dict) or not isinstance(result.get("path"), str):
            raise SystemExit(f"invalid discovery manifest: {manifest_path}")
        path = (root / result["path"]).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise SystemExit(f"invalid discovery result path: {path}") from exc
        payload, attempted = read_result(path)
        if hashlib.sha256(payload).hexdigest() != result.get("sha256"):
            raise SystemExit(f"discovery result changed after discovery completed: {path}")
        if attempted != result.get("expected_new_trials"):
            raise SystemExit(f"discovery trial count changed after discovery completed: {path}")
    print(manifest["refine_rounds"])


def main() -> None:
    args = parse_args()
    if args.command == "create":
        create_manifest(args)
    elif args.command == "verify":
        verify_manifest(args)
    else:
        check_results(args)


if __name__ == "__main__":
    main()
