from __future__ import annotations

import asyncio
import os
import signal
import subprocess
from collections.abc import Mapping, Sequence
from os import PathLike

from .errors import AutotuneError, AutotuneNotFoundError
from .models import CommandResult
from .transport import (
    DEFAULT_MAX_OUTPUT_BYTES,
    TERMINATE_GRACE_SECONDS,
    _BoundedBytes,
    discover_binary,
    merged_environment,
)


class AsyncSubprocessTransport:
    def __init__(self, binary: str | PathLike[str] | None = None, *, env: Mapping[str, str] | None = None, max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES) -> None:
        if max_output_bytes < 1:
            raise ValueError("max_output_bytes must be positive")
        self.binary = discover_binary(binary, env=merged_environment(env, None))
        self._env = dict(env) if env is not None else {}
        self._max_output_bytes = max_output_bytes

    async def invoke(self, args: Sequence[str], *, cwd: str | PathLike[str] | None = None, env: Mapping[str, str] | None = None, timeout: float | None = None, check: bool = True) -> CommandResult:
        argv = (self.binary, *(str(value) for value in args))
        try:
            process = await asyncio.create_subprocess_exec(
                *argv, cwd=cwd, env=merged_environment(self._env, env), stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=os.name == "posix",
            )
        except FileNotFoundError as error:
            raise AutotuneNotFoundError(f"autotune executable not found: {self.binary}", CommandResult(127, "", "", argv)) from error
        stdout = _BoundedBytes(self._max_output_bytes)
        stderr = _BoundedBytes(self._max_output_bytes, tail=True)
        overflow = asyncio.Event()
        readers = [asyncio.create_task(_read_stream(process.stdout, stdout, overflow)), asyncio.create_task(_read_stream(process.stderr, stderr, overflow))]
        waiter = asyncio.create_task(process.wait())
        overflow_waiter = asyncio.create_task(overflow.wait())
        timed_out = False
        try:
            done, _ = await asyncio.wait([waiter, overflow_waiter], timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
            if not done:
                timed_out = True
                await _terminate_process_tree(process)
            elif overflow.is_set():
                await _terminate_process_tree(process)
            await waiter
        except asyncio.CancelledError:
            await _terminate_process_tree(process)
            raise
        finally:
            overflow_waiter.cancel()
            await asyncio.gather(*readers, return_exceptions=True)
        result = CommandResult(process.returncode or 0, stdout.text(), stderr.text(), argv)
        if timed_out:
            raise subprocess.TimeoutExpired(argv, timeout or 0.0, output=result.stdout, stderr=result.stderr)
        if stdout.overflowed:
            raise AutotuneError(f"autotune stdout exceeded the {self._max_output_bytes} byte capture limit", result)
        if stderr.overflowed:
            raise AutotuneError(f"autotune stderr exceeded the {self._max_output_bytes} byte capture limit", result)
        if check and result.returncode != 0:
            raise AutotuneError("autotune command failed", result)
        return result


async def _read_stream(stream: asyncio.StreamReader | None, destination: _BoundedBytes, overflow: asyncio.Event) -> None:
    if stream is None:
        return
    while chunk := await stream.read(65536):
        destination.append(chunk)
        if destination.overflowed:
            overflow.set()


async def _terminate_process_tree(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=TERMINATE_GRACE_SECONDS)
    except TimeoutError:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        await process.wait()
