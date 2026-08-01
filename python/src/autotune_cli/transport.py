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
from .pipe_capture import OutputPipeCapture
from .processes import (
    READER_CLEANUP_SECONDS,
    OutputPipes,
    capture_process_group_members,
    signal_processes,
    terminate_output_holders,
    terminate_process_group_members,
)

DEFAULT_MAX_OUTPUT_BYTES = 32 * 1024 * 1024
TERMINATE_GRACE_SECONDS = 4.0


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
        return self._invoke(
            args,
            cwd=cwd,
            env=env,
            timeout=timeout,
            check=check,
            cancel_event=None,
        )

    def _invoke(
        self,
        args: Sequence[str],
        *,
        cwd: str | PathLike[str] | None,
        env: Mapping[str, str] | None,
        timeout: float | None,
        check: bool,
        cancel_event: threading.Event | None,
    ) -> CommandResult:
        if cancel_event is not None and cancel_event.is_set():
            raise _InvocationCancelled
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
            if error.filename != self.binary:
                raise
            result = CommandResult(127, "", "", argv)
            raise AutotuneNotFoundError(f"autotune executable not found: {self.binary}", result) from error

        started_at = time.monotonic()
        output_pipes = OutputPipes()
        stdout = _BoundedBytes(self._max_output_bytes)
        stderr = _BoundedBytes(self._max_output_bytes, tail=True)
        reader_stop = threading.Event()
        threads: list[threading.Thread] = []
        pipe_capture = OutputPipeCapture(())
        timed_out = False
        cancelled = False
        try:
            pipe_fds = tuple(
                pipe.fileno()
                for pipe in (process.stdout, process.stderr)
                if pipe is not None
            )
            pipe_capture = OutputPipeCapture(pipe_fds)
            pipe_capture.start()
            for pipe, destination in (
                (process.stdout, stdout),
                (process.stderr, stderr),
            ):
                thread = threading.Thread(
                    target=_read_pipe,
                    args=(pipe, destination, reader_stop),
                    daemon=True,
                )
                thread.start()
                threads.append(thread)
            output_pipes = pipe_capture.result(READER_CLEANUP_SECONDS, cancel_event)
            deadline = started_at + timeout if timeout is not None else None
            timed_out = deadline is not None and time.monotonic() >= deadline
            while process.poll() is None and not timed_out:
                if cancel_event is not None and cancel_event.is_set():
                    cancelled = True
                    break
                if stdout.overflowed or stderr.overflowed:
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    timed_out = True
                    break
                time.sleep(0.01)
        except BaseException:
            output_pipes = _captured_output_pipes(pipe_capture)
            _terminate_process_tree(process, output_pipes)
            raise
        finally:
            if cancelled or timed_out or stdout.overflowed or stderr.overflowed:
                output_pipes = _captured_output_pipes(pipe_capture)
                _terminate_process_tree(process, output_pipes)
            process.wait()
            drain_deadline = (
                started_at + timeout
                if timeout is not None
                else time.monotonic() + TERMINATE_GRACE_SECONDS
            )
            cancelled = cancelled or _join_threads_until(
                threads, drain_deadline, cancel_event
            )
            if any(thread.is_alive() for thread in threads):
                timed_out = timeout is not None
                _terminate_process_tree(process, output_pipes)
                _join_threads_until(threads, time.monotonic() + READER_CLEANUP_SECONDS)
            if any(thread.is_alive() for thread in threads):
                reader_stop.set()
                _join_threads_until(threads, time.monotonic() + READER_CLEANUP_SECONDS)
            _close_pipe(process.stdout)
            _close_pipe(process.stderr)

        result = CommandResult(process.returncode, stdout.text(), stderr.text(), argv)
        if cancelled:
            raise _InvocationCancelled
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


class _InvocationCancelled(Exception):
    pass


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


def _read_pipe(
    pipe: BinaryIO | None, destination: _BoundedBytes, stop: threading.Event
) -> None:
    if pipe is None:
        return
    descriptor = pipe.fileno()
    if os.name == "posix":
        os.set_blocking(descriptor, False)
    try:
        while not stop.is_set():
            try:
                chunk = os.read(descriptor, 65536)
            except BlockingIOError:
                stop.wait(0.01)
                continue
            if not chunk:
                break
            destination.append(chunk)
    finally:
        _close_pipe(pipe)


def _terminate_process_tree(
    process: subprocess.Popen[bytes], output_pipes: OutputPipes | None = None
) -> None:
    if os.name != "posix":
        if process.poll() is not None:
            return
        _windows_terminate_tree(process.pid)
        return

    leader_running = process.poll() is None
    group_members = (
        capture_process_group_members(process.pid, process.pid)
        if leader_running
        else ()
    )
    leader_running = process.poll() is None
    if not leader_running:
        group_members = ()
    if leader_running:
        signal_processes(group_members, signal.SIGTERM)
        try:
            process.terminate()
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=TERMINATE_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
    terminate_process_group_members(group_members)
    if output_pipes is not None:
        terminate_output_holders(output_pipes, process.pid)


def _captured_output_pipes(capture: OutputPipeCapture) -> OutputPipes:
    try:
        return capture.result(READER_CLEANUP_SECONDS)
    except BaseException:  # noqa: BLE001 - cleanup must preserve interrupts
        return OutputPipes()


def _join_threads_until(
    threads: Sequence[threading.Thread],
    deadline: float,
    cancel_event: threading.Event | None = None,
) -> bool:
    for thread in threads:
        while thread.is_alive() and time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            thread.join(timeout=min(0.05, remaining))
            if cancel_event is not None and cancel_event.is_set():
                return True
    return False


def _close_pipe(pipe: IO[Any] | None) -> None:
    if pipe is not None:
        pipe.close()


def _windows_creation_flags() -> int:
    return int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)) if os.name == "nt" else 0


def _windows_terminate_tree(pid: int) -> None:
    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False, capture_output=True)
