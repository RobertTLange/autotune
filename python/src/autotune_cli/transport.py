from __future__ import annotations

import os
import shutil
import signal
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from os import PathLike
from typing import IO, Any, BinaryIO

from .errors import AutotuneError, AutotuneNotFoundError
from .models import CommandResult

DEFAULT_MAX_OUTPUT_BYTES = 32 * 1024 * 1024
TERMINATE_GRACE_SECONDS = 1.0


def discover_binary(
    explicit: str | PathLike[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> str:
    if explicit is not None:
        return os.fspath(explicit)
    environment = os.environ if env is None else env
    configured = environment.get("AUTOTUNE_CLI_BIN") or environment.get("AUTOTUNE_BIN")
    if configured:
        return configured
    found = shutil.which("autotune", path=environment.get("PATH"))
    if found:
        return found
    result = CommandResult(127, "", "", ("autotune",))
    raise AutotuneNotFoundError(
        "autotune executable not found; install it, set AUTOTUNE_CLI_BIN, or pass binary=...",
        result,
    )


def merged_environment(
    base: Mapping[str, str] | None, extra: Mapping[str, str] | None
) -> dict[str, str]:
    merged = dict(os.environ)
    if base is not None:
        merged.update(base)
    if extra is not None:
        merged.update(extra)
    return merged


class SubprocessTransport:
    def __init__(
        self,
        binary: str | PathLike[str] | None = None,
        *,
        env: Mapping[str, str] | None = None,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    ) -> None:
        if max_output_bytes < 1:
            raise ValueError("max_output_bytes must be positive")
        effective_env = merged_environment(env, None)
        self.binary = discover_binary(binary, env=effective_env)
        self._env = dict(env) if env is not None else {}
        self._max_output_bytes = max_output_bytes

    def invoke(
        self,
        args: Sequence[str],
        *,
        cwd: str | PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        check: bool = True,
    ) -> CommandResult:
        argv = (self.binary, *(str(value) for value in args))
        try:
            process = subprocess.Popen(
                argv,
                cwd=cwd,
                env=merged_environment(self._env, env),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=os.name == "posix",
                creationflags=_windows_creation_flags(),
            )
        except FileNotFoundError as error:
            result = CommandResult(127, "", "", argv)
            raise AutotuneNotFoundError(f"autotune executable not found: {self.binary}", result) from error

        stdout = _BoundedBytes(self._max_output_bytes)
        stderr = _BoundedBytes(self._max_output_bytes, tail=True)
        threads = [
            threading.Thread(target=_read_pipe, args=(process.stdout, stdout), daemon=True),
            threading.Thread(target=_read_pipe, args=(process.stderr, stderr), daemon=True),
        ]
        for thread in threads:
            thread.start()
        timed_out = False
        started_at = time.monotonic()
        try:
            while process.poll() is None:
                if stdout.overflowed or stderr.overflowed:
                    _terminate_process_tree(process)
                    break
                if timeout is not None and time.monotonic() - started_at >= timeout:
                    timed_out = True
                    _terminate_process_tree(process)
                    break
                time.sleep(0.01)
        except BaseException:
            _terminate_process_tree(process)
            raise
        finally:
            process.wait()
            for thread in threads:
                thread.join(timeout=TERMINATE_GRACE_SECONDS)
            _close_pipe(process.stdout)
            _close_pipe(process.stderr)

        result = CommandResult(process.returncode, stdout.text(), stderr.text(), argv)
        if timed_out:
            raise subprocess.TimeoutExpired(argv, timeout or 0.0, output=result.stdout, stderr=result.stderr)
        if stdout.overflowed:
            raise AutotuneError(
                f"autotune stdout exceeded the {self._max_output_bytes} byte capture limit", result
            )
        if stderr.overflowed:
            raise AutotuneError(
                f"autotune stderr exceeded the {self._max_output_bytes} byte capture limit", result
            )
        if check and result.returncode != 0:
            raise AutotuneError("autotune command failed", result)
        return result


class _BoundedBytes:
    def __init__(self, limit: int, *, tail: bool = False) -> None:
        self._limit = limit
        self._tail = tail
        self._data = bytearray()
        self.overflowed = False
        self._lock = threading.Lock()

    def append(self, chunk: bytes) -> None:
        with self._lock:
            if self._tail:
                self._data.extend(chunk)
                if len(self._data) > self._limit:
                    del self._data[:-self._limit]
                    self.overflowed = True
                return
            remaining = self._limit - len(self._data)
            if remaining <= 0:
                self.overflowed = True
                return
            self._data.extend(chunk[:remaining])
            if len(chunk) > remaining:
                self.overflowed = True

    def text(self) -> str:
        with self._lock:
            return bytes(self._data).decode("utf-8", errors="replace")


def _read_pipe(pipe: BinaryIO | None, destination: _BoundedBytes) -> None:
    if pipe is None:
        return
    try:
        while chunk := os.read(pipe.fileno(), 65536):
            destination.append(chunk)
    finally:
        _close_pipe(pipe)


def _terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.wait(timeout=TERMINATE_GRACE_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGKILL)
    else:
        process.kill()


def _close_pipe(pipe: IO[Any] | None) -> None:
    if pipe is not None:
        pipe.close()


def _windows_creation_flags() -> int:
    return int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)) if os.name == "nt" else 0
