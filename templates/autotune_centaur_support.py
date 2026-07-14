"""Validation, parsing, process, and artifact support for Centaur."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import signal
import subprocess
import tempfile
import threading
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import cmaes
import numpy as np
import optuna
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)


MAX_CAPTURE_CHARS = 256 * 1024
MAX_PROMPT_BYTES = 1024 * 1024
HEADLESS_TIMEOUT_SECONDS = 600
MAX_JSON_CANDIDATES = 64
MAX_JSON_DEPTH = 20
SUPPORTED_OPTUNA = (4, 8)
SUPPORTED_CMAES = (0, 12)
HEADLESS_ENV_ALLOWLIST = {
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "TMPDIR",
    "TMP",
    "TEMP",
    "LANG",
    "LC_ALL",
    "TERM",
    "NO_COLOR",
    "FORCE_COLOR",
    "CI",
    "XDG_CONFIG_HOME",
    "XDG_CACHE_HOME",
    "CODEX_HOME",
    "CLAUDE_CONFIG_DIR",
    "HEADLESS_CONFIG",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "OPENROUTER_API_KEY",
    "CURSOR_API_KEY",
}


def build_distributions(
    parameters: Sequence[Mapping[str, Any]],
) -> Dict[str, BaseDistribution]:
    distributions: Dict[str, BaseDistribution] = {}
    for parameter in parameters:
        name = nonempty("parameter name", parameter.get("name"))
        if name in distributions:
            raise ValueError("duplicate Centaur parameter: " + name)
        kind = parameter.get("type")
        if kind == "float":
            distributions[name] = FloatDistribution(
                float(parameter["low"]),
                float(parameter["high"]),
                log=bool(parameter.get("log", False)),
            )
        elif kind == "int":
            distributions[name] = IntDistribution(
                int(parameter["low"]),
                int(parameter["high"]),
            )
        elif kind == "categorical":
            distributions[name] = CategoricalDistribution(parameter["choices"])
        else:
            raise ValueError("unsupported Centaur parameter type: " + str(kind))
    return distributions


def extract_proposal(
    output: str, distributions: Mapping[str, BaseDistribution]
) -> Dict[str, Any]:
    candidates: List[Mapping[str, Any]] = []
    expected = set(distributions)
    for text in _text_candidates(output):
        for candidate in _json_objects(text):
            if set(candidate) & expected:
                candidates.append(candidate)
    if not candidates:
        raise ValueError("headless output contained no proposal JSON object")
    candidate = candidates[-1]
    if set(candidate) != expected:
        missing = sorted(expected - set(candidate))
        extra = sorted(set(candidate) - expected)
        raise ValueError(f"proposal keys mismatch; missing={missing}, extra={extra}")
    return {
        name: _validate_value(name, candidate[name], distribution)
        for name, distribution in distributions.items()
    }


def _text_candidates(output: str) -> List[str]:
    texts = [output]
    for line in output.splitlines()[-128:]:
        try:
            value = json.loads(line, object_pairs_hook=_unique_object)
        except Exception:
            continue
        texts.extend(_string_leaves(value, MAX_JSON_CANDIDATES - len(texts)))
        if len(texts) >= MAX_JSON_CANDIDATES:
            break
    return texts


def _string_leaves(value: Any, limit: int) -> List[str]:
    texts: List[str] = []
    pending = [(value, 0)]
    while pending and len(texts) < limit:
        current, depth = pending.pop()
        if depth > MAX_JSON_DEPTH:
            raise ValueError("headless JSON nesting exceeds safety limit")
        if isinstance(current, str):
            texts.append(current)
        elif isinstance(current, Mapping):
            pending.extend((child, depth + 1) for child in current.values())
        elif isinstance(current, list):
            pending.extend((child, depth + 1) for child in current)
    return texts


def _json_objects(text: str) -> List[Mapping[str, Any]]:
    snippets = [text]
    snippets.extend(
        re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.I | re.S)[:16]
    )
    objects: List[Mapping[str, Any]] = []
    decoder = json.JSONDecoder(object_pairs_hook=_unique_object)
    attempts = 0
    for snippet in snippets:
        index = snippet.find("{")
        while index >= 0 and attempts < MAX_JSON_CANDIDATES:
            attempts += 1
            try:
                value, _ = decoder.raw_decode(snippet[index:])
            except (ValueError, json.JSONDecodeError):
                pass
            else:
                if isinstance(value, Mapping):
                    objects.append(value)
            index = snippet.find("{", index + 1)
        if attempts >= MAX_JSON_CANDIDATES:
            break
    return objects


def _unique_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("duplicate JSON key: " + name)
        result[name] = value
    return result


def _validate_value(name: str, value: Any, distribution: BaseDistribution) -> Any:
    if isinstance(distribution, FloatDistribution):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(name + " must be a JSON number")
        numeric = float(value)
        if not math.isfinite(numeric) or not distribution._contains(
            distribution.to_internal_repr(numeric)
        ):
            raise ValueError(name + " is outside its finite float bounds")
        return numeric
    if isinstance(distribution, IntDistribution):
        if type(value) is not int or not distribution._contains(
            distribution.to_internal_repr(value)
        ):
            raise ValueError(name + " must be an in-range JSON integer")
        return value
    if isinstance(distribution, CategoricalDistribution):
        for choice in distribution.choices:
            if type(value) is type(choice) and value == choice:
                return value
        raise ValueError(name + " must exactly match a categorical choice")
    raise ValueError("unsupported distribution for " + name)


class _TailCapture:
    def __init__(self) -> None:
        self._chunks: deque[str] = deque()
        self._size = 0

    def drain(self, stream: Any) -> None:
        for chunk in iter(lambda: stream.read(8192), ""):
            self._chunks.append(chunk)
            self._size += len(chunk)
            self._trim()

    def _trim(self) -> None:
        excess = self._size - MAX_CAPTURE_CHARS
        while excess > 0 and self._chunks:
            first = self._chunks[0]
            if len(first) <= excess:
                self._chunks.popleft()
                self._size -= len(first)
                excess -= len(first)
            else:
                self._chunks[0] = first[excess:]
                self._size -= excess
                excess = 0

    @property
    def value(self) -> str:
        return "".join(self._chunks)


def bounded_process(argv: Sequence[str], *, cwd: Path) -> str:
    process = subprocess.Popen(
        list(argv),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        start_new_session=True,
        cwd=cwd,
        env={
            name: os.environ[name]
            for name in HEADLESS_ENV_ALLOWLIST
            if name in os.environ
        },
    )
    stdout = _TailCapture()
    stderr = _TailCapture()
    threads = [
        threading.Thread(target=stdout.drain, args=(process.stdout,), daemon=True),
        threading.Thread(target=stderr.drain, args=(process.stderr,), daemon=True),
    ]
    for thread in threads:
        thread.start()
    try:
        returncode = process.wait(timeout=HEADLESS_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise RuntimeError("headless proposal timed out")
    finally:
        for thread in threads:
            thread.join(timeout=1)
        if any(thread.is_alive() for thread in threads):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if process.stdout:
                process.stdout.close()
            if process.stderr:
                process.stderr.close()
            for thread in threads:
                thread.join(timeout=1)
    if returncode != 0:
        raise RuntimeError(f"headless proposal exited with status {returncode}")
    return stdout.value


def prepare_artifact_root(work_dir: Path) -> Path:
    root = work_dir / "centaur"
    if root.is_symlink():
        raise ValueError("Centaur artifact directory cannot be a symlink")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if root.is_symlink():
        raise ValueError("Centaur artifact directory cannot be a symlink")
    resolved = root.resolve()
    if resolved.parent != work_dir:
        raise ValueError("Centaur artifact directory must remain inside work_dir")
    os.chmod(resolved, 0o700)
    return resolved


def write_private(path: Path, content: str) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def native(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(name): native(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _keyed_hash(domain: str, *parts: Any) -> bytes:
    payload = "\0".join([domain, *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode("utf-8")).digest()


def unit_hash(domain: str, *parts: Any) -> float:
    return int.from_bytes(_keyed_hash(domain, *parts)[:8], "big") / float(2**64)


def integer_hash(domain: str, *parts: Any) -> int:
    return 1 + int.from_bytes(_keyed_hash(domain, *parts)[:8], "big") % (2**31 - 2)


def probability(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("llm_probability must be a number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("llm_probability must be between 0 and 1")
    return result


def nonnegative_int(name: str, value: Any) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(name + " must be a non-negative integer")
    return value


def nonempty(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(name + " must be a non-empty string")
    return value


def safe_error(error: Exception) -> str:
    return re.sub(r"[^a-zA-Z0-9 .,;:_=+\-\[\]]", "?", str(error))[-500:]


def _version_tuple(version: str) -> Tuple[int, int]:
    match = re.match(r"(\d+)\.(\d+)", version)
    return (int(match.group(1)), int(match.group(2))) if match else (0, 0)


def require_supported_versions() -> None:
    optuna_version = _version_tuple(optuna.__version__)
    cmaes_version = _version_tuple(cmaes.__version__)
    if optuna_version[0] != 4 or optuna_version < SUPPORTED_OPTUNA:
        raise RuntimeError("Centaur requires Optuna >=4.8 and <5")
    if cmaes_version < SUPPORTED_CMAES:
        raise RuntimeError("Centaur requires cmaes >=0.12")
