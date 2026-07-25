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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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
HEADLESS_BASE_ENV_ALLOWLIST = set(
    "PATH HOME USER LOGNAME SHELL TMPDIR TMP TEMP LANG LC_ALL TERM NO_COLOR "
    "FORCE_COLOR CI XDG_CONFIG_HOME XDG_CACHE_HOME XDG_DATA_HOME HEADLESS_CONFIG".split()
)
NPM_ENV_ALLOWLIST = set(
    "SystemRoot ComSpec PATHEXT USERPROFILE APPDATA LOCALAPPDATA PROGRAMDATA "
    "HTTP_PROXY HTTPS_PROXY NO_PROXY ALL_PROXY http_proxy https_proxy no_proxy all_proxy "
    "NODE_EXTRA_CA_CERTS NPM_CONFIG_REGISTRY NPM_CONFIG_PROXY NPM_CONFIG_HTTPS_PROXY "
    "NPM_CONFIG_NOPROXY NPM_CONFIG_CAFILE NPM_CONFIG_CACHE NPM_CONFIG_STRICT_SSL "
    "NPM_CONFIG_USERCONFIG NPM_CONFIG_GLOBALCONFIG "
    "npm_config_registry npm_config_proxy npm_config_https_proxy npm_config_noproxy "
    "npm_config_cafile npm_config_cache npm_config_strict_ssl "
    "npm_config_userconfig npm_config_globalconfig".split()
)
HEADLESS_AGENT_ENV_ALLOWLIST = {
    "codex": {"CODEX_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL", "CODEX_HOME"},
    "cursor": {"CURSOR_API_KEY"},
    "pi": set(
        "PI_CODING_AGENT_API_KEY PI_CODING_AGENT_MODEL PI_CODING_AGENT_MODELS "
        "PI_CODING_AGENT_PROVIDER".split()
    ),
}
CLAUDE_COMMON_ENV_ALLOWLIST = {
    "CLAUDE_CONFIG_DIR",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_SMALL_FAST_MODEL",
}
CLAUDE_PROVIDER_ENV_ALLOWLIST = {
    "anthropic": set(
        "ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN ANTHROPIC_BASE_URL ANTHROPIC_CUSTOM_HEADERS "
        "CLAUDE_CODE_OAUTH_TOKEN".split()
    ),
    "aws": set(
        "CLAUDE_CODE_USE_BEDROCK CLAUDE_CODE_SKIP_BEDROCK_AUTH "
        "ANTHROPIC_BEDROCK_BASE_URL ANTHROPIC_SMALL_FAST_MODEL_AWS_REGION".split()
    ),
    "vertex": set(
        "CLAUDE_CODE_USE_VERTEX CLAUDE_CODE_SKIP_VERTEX_AUTH "
        "ANTHROPIC_VERTEX_BASE_URL ANTHROPIC_VERTEX_PROJECT_ID CLOUD_ML_REGION".split()
    ),
}
GEMINI_COMMON_ENV_ALLOWLIST = {"GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_PROJECT_ID"}
GEMINI_PROVIDER_ENV_ALLOWLIST = {
    "google": {"GEMINI_API_KEY", "GOOGLE_API_KEY"},
    "vertex": set(
        "GOOGLE_API_KEY GOOGLE_APPLICATION_CREDENTIALS GOOGLE_GENAI_USE_VERTEXAI "
        "GOOGLE_CLOUD_LOCATION".split()
    ),
}
HEADLESS_PROVIDER_ENV_ALLOWLIST = {
    "anthropic": {"ANTHROPIC_API_KEY"},
    "aws": set(
        "AWS_ACCESS_KEY_ID AWS_BEARER_TOKEN_BEDROCK AWS_CONTAINER_AUTHORIZATION_TOKEN "
        "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE AWS_CONTAINER_CREDENTIALS_FULL_URI "
        "AWS_CONFIG_FILE AWS_CONTAINER_CREDENTIALS_RELATIVE_URI AWS_DEFAULT_REGION AWS_PROFILE "
        "AWS_REGION AWS_ROLE_ARN AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN "
        "AWS_SHARED_CREDENTIALS_FILE AWS_WEB_IDENTITY_TOKEN_FILE".split()
    ),
    "azure": set(
        "AZURE_OPENAI_API_KEY AZURE_OPENAI_API_VERSION AZURE_OPENAI_BASE_URL "
        "AZURE_OPENAI_DEPLOYMENT_NAME_MAP AZURE_OPENAI_ENDPOINT "
        "AZURE_OPENAI_RESOURCE_NAME AZURE_RESOURCE_NAME".split()
    ),
    "google": {"GEMINI_API_KEY", "GOOGLE_API_KEY"},
    "vertex": set(
        "GCLOUD_PROJECT GOOGLE_APPLICATION_CREDENTIALS GOOGLE_CLOUD_API_KEY "
        "GOOGLE_CLOUD_LOCATION GOOGLE_CLOUD_PROJECT".split()
    ),
    "openai": {"OPENAI_API_KEY", "OPENAI_BASE_URL"},
    "openrouter": {"OPENROUTER_API_KEY"},
}
HEADLESS_PROVIDER_FAMILY = {
    "amazon-bedrock": "aws",
    "anthropic": "anthropic",
    "google": "google",
    "google-vertex": "vertex",
    "openai": "openai",
    "openai-codex": "openai",
    "openrouter": "openrouter",
}
HEADLESS_AGENT_PROVIDER_FAMILY = {
    "opencode": {"azure": "azure"},
    "pi": {"aws": "aws", "azure-openai-responses": "azure"},
}
EXPLICIT_HEADLESS_ENV = "AUTOTUNE_HEADLESS_ENV"
LEGACY_EXPLICIT_HEADLESS_ENV = "AUTOTUNE_CENTAUR_HEADLESS_ENV"
ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
EXPLICIT_CREDENTIAL_OR_CONFIG = re.compile(
    r"(?:^|_)(?:API_KEY|TOKEN|SECRET|PASSWORD|CREDENTIALS(?:_FILE)?|"
    r"CONFIG(?:_DIR|_FILE)?|PROFILE|PROJECT|REGION|LOCATION|ENDPOINT|BASE_URL|"
    r"URL|DEPLOYMENT|RESOURCE|ACCOUNT(?:_ID)?|TENANT(?:_ID)?|CLIENT_ID|"
    r"ORG(?:ANIZATION)?|MODEL|AUTH)$"
)


def headless_environment(agent: str, model: Optional[str]) -> Dict[str, str]:
    names = set(HEADLESS_BASE_ENV_ALLOWLIST)
    normalized_agent = agent.strip().lower()
    explicit_names = _explicit_headless_env_names()
    names.update(HEADLESS_AGENT_ENV_ALLOWLIST.get(normalized_agent, set()))
    provider = _headless_provider(agent, model, bool(explicit_names))
    names.update(HEADLESS_PROVIDER_ENV_ALLOWLIST.get(provider, set()))
    if normalized_agent == "claude":
        names.update(CLAUDE_COMMON_ENV_ALLOWLIST)
        names.update(CLAUDE_PROVIDER_ENV_ALLOWLIST.get(provider, set()))
    if normalized_agent == "gemini":
        names.update(GEMINI_COMMON_ENV_ALLOWLIST)
        names.update(GEMINI_PROVIDER_ENV_ALLOWLIST.get(provider, set()))
    names.update(explicit_names)
    return {name: os.environ[name] for name in names if name in os.environ}


def npm_environment(
    base: Mapping[str, str], node_executable: Optional[str] = None
) -> Dict[str, str]:
    environment = dict(base)
    environment.update(
        {name: os.environ[name] for name in NPM_ENV_ALLOWLIST if name in os.environ}
    )
    if node_executable:
        path_entries = [str(Path(node_executable).resolve().parent)]
        path_entries.extend(
            entry
            for entry in environment.get("PATH", "").split(os.pathsep)
            if Path(entry).is_absolute()
        )
        if os.name != "nt":
            path_entries.extend(os.defpath.split(os.pathsep))
        environment["PATH"] = os.pathsep.join(
            dict.fromkeys(entry for entry in path_entries if entry)
        )
    return environment


def _headless_provider(
    agent: str, model: Optional[str], has_explicit_environment: bool = False
) -> str:
    normalized_agent = agent.strip().lower()
    if normalized_agent == "claude":
        bedrock = _enabled(os.environ.get("CLAUDE_CODE_USE_BEDROCK"))
        vertex = _enabled(os.environ.get("CLAUDE_CODE_USE_VERTEX"))
        if bedrock and vertex:
            raise ValueError(
                "CLAUDE_CODE_USE_BEDROCK and CLAUDE_CODE_USE_VERTEX cannot both be enabled"
            )
        return "aws" if bedrock else "vertex" if vertex else "anthropic"
    if normalized_agent == "gemini":
        return "vertex" if _enabled(os.environ.get("GOOGLE_GENAI_USE_VERTEXAI")) else "google"
    if normalized_agent not in ("opencode", "pi"):
        return ""
    configured_model = model
    if normalized_agent == "pi" and not configured_model:
        configured_model = os.environ.get("PI_CODING_AGENT_MODEL")
    if not configured_model:
        if has_explicit_environment:
            return ""
        raise ValueError(
            f"Centaur agent {normalized_agent} requires a provider-qualified model or "
            f"{EXPLICIT_HEADLESS_ENV}"
        )
    model_parts = configured_model.split("/", 1)
    if len(model_parts) == 2 and all(model_parts):
        provider = model_parts[0].lower()
    elif normalized_agent == "pi":
        provider = os.environ.get("PI_CODING_AGENT_PROVIDER", "").lower()
        if not provider:
            if has_explicit_environment:
                return ""
            raise ValueError(
                "Centaur agent pi requires a provider-qualified model or "
                f"PI_CODING_AGENT_PROVIDER or {EXPLICIT_HEADLESS_ENV}"
            )
    else:
        if has_explicit_environment:
            return ""
        raise ValueError(
            "Centaur agent opencode requires a provider-qualified model or "
            f"{EXPLICIT_HEADLESS_ENV}"
        )
    agent_provider = HEADLESS_AGENT_PROVIDER_FAMILY.get(normalized_agent, {})
    return agent_provider.get(provider, HEADLESS_PROVIDER_FAMILY.get(provider, ""))


def _explicit_headless_env_names() -> List[str]:
    configured = os.environ.get(EXPLICIT_HEADLESS_ENV)
    legacy = os.environ.get(LEGACY_EXPLICIT_HEADLESS_ENV)
    if configured is not None and legacy is not None and configured != legacy:
        raise ValueError(
            f"{EXPLICIT_HEADLESS_ENV} and {LEGACY_EXPLICIT_HEADLESS_ENV} conflict"
        )
    configured = configured if configured is not None else legacy or ""
    names = [name.strip() for name in configured.split(",") if name.strip()]
    if len(names) > 32 or any(
        not ENVIRONMENT_NAME.fullmatch(name)
        or not EXPLICIT_CREDENTIAL_OR_CONFIG.search(name)
        for name in names
    ):
        raise ValueError(
            f"{EXPLICIT_HEADLESS_ENV} must list at most 32 credential or config "
            "environment variable names"
        )
    return names


def _enabled(value: Optional[str]) -> bool:
    return value is not None and value.strip().lower() not in ("", "0", "false", "no")


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


def bounded_process(
    argv: Sequence[str], *, cwd: Path, env: Mapping[str, str]
) -> str:
    process = subprocess.Popen(
        list(argv),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        start_new_session=True,
        cwd=cwd,
        env=dict(env),
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
        _terminate_process_tree(process)
        process.wait()
        raise RuntimeError("headless proposal timed out")
    finally:
        for thread in threads:
            thread.join(timeout=1)
        if any(thread.is_alive() for thread in threads):
            _terminate_process_tree(process)
            if process.stdout:
                process.stdout.close()
            if process.stderr:
                process.stderr.close()
            for thread in threads:
                thread.join(timeout=1)
    if returncode != 0:
        raise RuntimeError(f"headless proposal exited with status {returncode}")
    return stdout.value


def _terminate_process_tree(process: subprocess.Popen[Any]) -> None:
    if os.name == "nt":
        system_root = _windows_system_root()
        taskkill = system_root / "System32" / "taskkill.exe"
        try:
            completed = subprocess.run(
                [str(taskkill), "/PID", str(process.pid), "/T", "/F"],
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                env={"SystemRoot": str(system_root), "windir": str(system_root)},
            )
            if completed.returncode != 0 and process.poll() is None:
                process.kill()
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _windows_system_root() -> Path:
    standard = Path("C:/Windows")
    if (standard / "System32" / "taskkill.exe").is_file():
        return standard
    configured = os.environ.get("SystemRoot")
    if configured:
        candidate = Path(configured)
        taskkill = candidate / "System32" / "taskkill.exe"
        if (
            candidate.is_absolute()
            and re.fullmatch(r"[A-Za-z]:", candidate.drive)
            and candidate.parent == Path(candidate.anchor)
            and candidate.name.lower() in ("windows", "winnt")
            and taskkill.is_file()
        ):
            return candidate.resolve()
    return standard


def prepare_artifact_root(work_dir: Path, study_name: str) -> Path:
    base = work_dir / "centaur"
    if base.is_symlink():
        raise ValueError("Centaur artifact directory cannot be a symlink")
    base.mkdir(parents=True, exist_ok=True, mode=0o700)
    if base.is_symlink():
        raise ValueError("Centaur artifact directory cannot be a symlink")
    resolved_base = base.resolve()
    if resolved_base.parent != work_dir:
        raise ValueError("Centaur artifact directory must remain inside work_dir")
    os.chmod(resolved_base, 0o700)
    prefix = f"study-{sha256(study_name)[:16]}-"
    run_root = Path(tempfile.mkdtemp(prefix=prefix, dir=resolved_base))
    resolved_run_root = run_root.resolve()
    if resolved_run_root.parent != resolved_base:
        raise ValueError("Centaur study artifacts must remain inside artifact directory")
    os.chmod(resolved_run_root, 0o700)
    return resolved_run_root


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


def acquire_study_lock(storage: Optional[str], study_name: str) -> Optional[int]:
    if not storage:
        return None
    lock_path = _sqlite_study_lock_path(storage, study_name)
    if lock_path is None:
        return None
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, 0o600)
        _lock_descriptor(descriptor)
    except BlockingIOError:
        os.close(descriptor)
        raise RuntimeError(
            "Centaur study is already being optimized by another process"
        ) from None
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _sqlite_study_lock_path(storage: str, study_name: str) -> Optional[Path]:
    match = re.match(r"^sqlite(?:\+[A-Za-z0-9_.-]+)?:///(.*)$", storage)
    if match is None:
        raise ValueError("Centaur persistent storage currently requires a SQLite URI")
    database = match.group(1).split("?", 1)[0]
    if database == ":memory:":
        return None
    if not database or database.startswith("file:"):
        raise ValueError("Centaur requires a file-backed SQLite storage URI")
    database_path = Path(database).resolve()
    study_hash = sha256(study_name)[:16]
    return database_path.with_name(f".{database_path.name}.centaur-{study_hash}.lock")


def _lock_descriptor(descriptor: int) -> None:
    try:
        import fcntl
    except ImportError:
        import msvcrt

        if os.fstat(descriptor).st_size == 0:
            os.write(descriptor, b"\0")
        os.lseek(descriptor, 0, os.SEEK_SET)
        try:
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        except OSError as error:
            raise BlockingIOError from error
        return
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)


def release_study_lock(descriptor: Optional[int]) -> None:
    if descriptor is not None:
        os.close(descriptor)


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
