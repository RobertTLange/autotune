from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Mapping, Sequence
from os import PathLike
from typing import Any, TypeVar

from .models import CommandResult
from .transport import (
    DEFAULT_MAX_OUTPUT_BYTES,
    SubprocessTransport,
    discover_binary,
    merged_environment,
)

ResultT = TypeVar("ResultT")


class AsyncSubprocessTransport:
    def __init__(self, binary: str | PathLike[str] | None = None, *, env: Mapping[str, str] | None = None, max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES) -> None:
        if max_output_bytes < 1:
            raise ValueError("max_output_bytes must be positive")
        self.binary = discover_binary(binary, env=merged_environment(env, None))
        self._env = dict(env) if env is not None else {}
        self._max_output_bytes = max_output_bytes

    async def invoke(self, args: Sequence[str], *, cwd: str | PathLike[str] | None = None, env: Mapping[str, str] | None = None, timeout: float | None = None, check: bool = True) -> CommandResult:
        cancel_event = threading.Event()
        transport = SubprocessTransport(
            self.binary,
            env=self._env,
            max_output_bytes=self._max_output_bytes,
        )
        future = _run_in_daemon_thread(
            _invoke_sync,
            transport,
            tuple(args),
            cwd,
            env,
            timeout,
            check,
            cancel_event,
        )
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            cancel_event.set()
            await _finish_cancelled_invocation(future)
            raise


def _invoke_sync(
    transport: SubprocessTransport,
    args: Sequence[str],
    cwd: str | PathLike[str] | None,
    env: Mapping[str, str] | None,
    timeout: float | None,
    check: bool,
    cancel_event: threading.Event,
) -> CommandResult:
    return transport._invoke(
        args,
        cwd=cwd,
        env=env,
        timeout=timeout,
        check=check,
        cancel_event=cancel_event,
    )


async def _finish_cancelled_invocation(
    future: asyncio.Future[CommandResult],
) -> None:
    while not future.done():
        try:
            await asyncio.shield(future)
        except asyncio.CancelledError:
            continue
        except BaseException:
            return
    try:
        future.result()
    except (BaseException, asyncio.InvalidStateError):
        pass


def _run_in_daemon_thread(
    function: Callable[..., ResultT], *args: object
) -> asyncio.Future[ResultT]:
    loop = asyncio.get_running_loop()
    future: asyncio.Future[ResultT] = loop.create_future()

    def run() -> None:
        try:
            result = function(*args)
        except BaseException as error:
            try:
                loop.call_soon_threadsafe(_set_future_exception, future, error)
            except RuntimeError:
                pass
        else:
            try:
                loop.call_soon_threadsafe(_set_future_result, future, result)
            except RuntimeError:
                pass

    threading.Thread(target=run, daemon=True).start()
    return future


def _set_future_result(future: asyncio.Future[ResultT], result: ResultT) -> None:
    if not future.done():
        future.set_result(result)


def _set_future_exception(future: asyncio.Future[Any], error: BaseException) -> None:
    if not future.done():
        future.set_exception(error)
