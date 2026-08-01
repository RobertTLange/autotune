from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

import pytest

import autotune_cli.async_transport as async_transport_module
import autotune_cli.processes as processes_module
import autotune_cli.transport as transport_module
from autotune_cli.async_transport import AsyncSubprocessTransport
from autotune_cli.errors import AutotuneError
from autotune_cli.transport import SubprocessTransport


def test_transport_terminates_the_process_group_on_timeout(fake_binary: Path, tmp_path: Path) -> None:
    marker = tmp_path / "child-survived"

    with pytest.raises(subprocess.TimeoutExpired):
        SubprocessTransport(fake_binary).invoke(["child", str(marker)], timeout=0.1)

    time.sleep(0.7)
    assert not marker.exists()


def test_transport_timeout_redacts_sensitive_arguments(
    fake_binary: Path, tmp_path: Path
) -> None:
    marker = tmp_path / "redacted-timeout"

    with pytest.raises(subprocess.TimeoutExpired) as raised:
        SubprocessTransport(fake_binary).invoke(
            ["leader", str(marker), "--storage=postgres://user:secret@db"],
            timeout=0.1,
        )

    assert "secret" not in str(raised.value)
    assert "[REDACTED]" in str(raised.value.cmd)


@pytest.mark.skipif(os.name != "posix", reason="POSIX process cleanup")
def test_transport_timeout_bounds_post_termination_wait(
    fake_binary: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "bounded-wait-leader"
    wait_timeouts: list[float | None] = []

    def fail_reap(
        _process: subprocess.Popen[bytes], *, timeout: float | None
    ) -> None:
        wait_timeouts.append(timeout)
        raise RuntimeError("process did not exit")

    monkeypatch.setattr(transport_module, "_wait_for_process_exit", fail_reap)

    with pytest.raises(subprocess.TimeoutExpired):
        SubprocessTransport(fake_binary).invoke(
            ["leader", str(marker)], timeout=0.1
        )

    assert wait_timeouts == [transport_module.TERMINATE_GRACE_SECONDS]


def test_transport_rejects_stdout_overflow(fake_binary: Path) -> None:
    with pytest.raises(AutotuneError, match="stdout exceeded"):
        SubprocessTransport(fake_binary, max_output_bytes=128).invoke(["overflow"])


@pytest.mark.skipif(os.name != "posix", reason="POSIX interrupt behavior")
def test_transport_interrupt_during_pipe_capture_terminates_process(
    fake_binary: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "interrupt-holder.pid"
    capture = processes_module.capture_output_pipes

    def delayed_capture(pipe_fds: object) -> object:
        time.sleep(0.3)
        return capture(pipe_fds)

    monkeypatch.setattr(processes_module, "capture_output_pipes", delayed_capture)
    def interrupt_after_marker() -> None:
        deadline = time.monotonic() + 1
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        os.kill(os.getpid(), signal.SIGINT)

    interrupt = threading.Timer(0.01, interrupt_after_marker)
    interrupt.start()
    try:
        with pytest.raises(KeyboardInterrupt):
            SubprocessTransport(fake_binary).invoke(["holder", str(marker)])
    finally:
        interrupt.cancel()

    assert_process_stopped(int(marker.read_text()))


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group behavior")
def test_transport_times_out_after_launcher_exits_with_output_holder(
    fake_binary: Path, tmp_path: Path
) -> None:
    marker = tmp_path / "holder.pid"

    with pytest.raises(subprocess.TimeoutExpired):
        SubprocessTransport(fake_binary).invoke(["holder", str(marker)], timeout=0.75)

    assert_process_stopped(int(marker.read_text()))


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group behavior")
def test_transport_kills_sigterm_resistant_group_member_without_output(
    fake_binary: Path, tmp_path: Path
) -> None:
    ready = tmp_path / "quiet-child-ready"
    marker = tmp_path / "quiet-child-survived"

    with pytest.raises(subprocess.TimeoutExpired):
        SubprocessTransport(fake_binary).invoke(
            ["quiet-child", str(ready), str(marker)], timeout=0.5
        )

    time.sleep(1.2)
    assert not marker.exists()


def test_transport_cleans_up_after_partial_reader_thread_start(
    fake_binary: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "thread-start-child-survived"
    start = threading.Thread.start
    starts = 0

    def fail_third_start(thread: threading.Thread) -> None:
        nonlocal starts
        starts += 1
        if starts == 3:
            raise RuntimeError("thread unavailable")
        start(thread)

    monkeypatch.setattr(threading.Thread, "start", fail_third_start)
    with pytest.raises(RuntimeError, match="thread unavailable"):
        SubprocessTransport(fake_binary).invoke(["child", str(marker)])

    time.sleep(0.7)
    assert not marker.exists()


def test_transport_terminates_leader_when_group_capture_fails(
    fake_binary: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "capture-failure-leader-survived"
    monkeypatch.setattr(
        transport_module, "capture_process_group_members", lambda _pid, _excluded: ()
    )

    with pytest.raises(subprocess.TimeoutExpired):
        SubprocessTransport(fake_binary).invoke(["leader", str(marker)], timeout=0.1)

    time.sleep(1.2)
    assert not marker.exists()


def test_async_transport_cancellation_terminates_the_process_group(fake_binary: Path, tmp_path: Path) -> None:
    marker = tmp_path / "async-child-survived"

    async def exercise() -> None:
        task = asyncio.create_task(AsyncSubprocessTransport(fake_binary).invoke(["child", str(marker)]))
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())
    time.sleep(0.7)
    assert not marker.exists()


def test_async_transport_cancellation_during_pipe_capture_terminates_process(
    fake_binary: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "async-capture-child-survived"
    capture = processes_module.capture_output_pipes

    def delayed_capture(pipe_fds: object) -> object:
        time.sleep(0.3)
        return capture(pipe_fds)

    monkeypatch.setattr(processes_module, "capture_output_pipes", delayed_capture)

    async def exercise() -> None:
        task = asyncio.create_task(
            AsyncSubprocessTransport(fake_binary).invoke(["child", str(marker)])
        )
        await asyncio.sleep(0.05)
        task.cancel()
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())
    time.sleep(0.7)
    assert not marker.exists()


def test_async_transport_cancellation_wins_over_worker_failure(
    fake_binary: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker_started = threading.Event()

    def fail_after_cancellation(*args: object) -> object:
        cancel_event = args[-1]
        assert isinstance(cancel_event, threading.Event)
        worker_started.set()
        cancel_event.wait()
        raise RuntimeError("concurrent worker failure")

    monkeypatch.setattr(async_transport_module, "_invoke_sync", fail_after_cancellation)

    async def exercise() -> None:
        task = asyncio.create_task(AsyncSubprocessTransport(fake_binary).invoke([]))
        while not worker_started.is_set():
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())


@pytest.mark.skipif(os.name != "posix", reason="POSIX output-holder behavior")
def test_async_transport_cancellation_interrupts_output_drain(
    fake_binary: Path, tmp_path: Path
) -> None:
    marker = tmp_path / "cancelled-holder.pid"

    async def exercise() -> float:
        task = asyncio.create_task(
            AsyncSubprocessTransport(fake_binary).invoke(["holder", str(marker)])
        )
        while not marker.exists():
            await asyncio.sleep(0.01)
        started_at = time.monotonic()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return time.monotonic() - started_at

    elapsed = asyncio.run(exercise())
    assert elapsed < 2
    assert_process_stopped(int(marker.read_text()))


def test_async_transport_cancelled_before_worker_start_does_not_launch(
    fake_binary: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "cancelled-before-start"
    invoke_sync = async_transport_module._invoke_sync

    def delayed_invoke(*args: object) -> object:
        time.sleep(0.2)
        return invoke_sync(*args)

    monkeypatch.setattr(async_transport_module, "_invoke_sync", delayed_invoke)

    async def exercise() -> None:
        task = asyncio.create_task(
            AsyncSubprocessTransport(fake_binary).invoke(["touch", str(marker)])
        )
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())
    time.sleep(0.3)
    assert not marker.exists()


def test_async_transport_rejects_stdout_overflow(fake_binary: Path) -> None:
    async def exercise() -> None:
        with pytest.raises(AutotuneError, match="stdout exceeded"):
            await AsyncSubprocessTransport(fake_binary, max_output_bytes=128).invoke(["overflow"])

    asyncio.run(exercise())


def test_async_transport_timeout_redacts_sensitive_arguments(
    fake_binary: Path, tmp_path: Path
) -> None:
    marker = tmp_path / "async-redacted-timeout"

    async def exercise() -> None:
        with pytest.raises(subprocess.TimeoutExpired) as raised:
            await AsyncSubprocessTransport(fake_binary).invoke(
                ["leader", str(marker), "--agent-guidance", "api-key"],
                timeout=0.1,
            )
        assert "api-key" not in str(raised.value)
        assert "[REDACTED]" in str(raised.value.cmd)

    asyncio.run(exercise())


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group behavior")
def test_async_transport_times_out_after_launcher_exits_with_output_holder(
    fake_binary: Path, tmp_path: Path
) -> None:
    marker = tmp_path / "async-holder.pid"

    async def exercise() -> None:
        with pytest.raises(subprocess.TimeoutExpired):
            await AsyncSubprocessTransport(fake_binary).invoke(
                ["holder", str(marker)], timeout=0.75
            )

    asyncio.run(asyncio.wait_for(exercise(), timeout=5))
    assert_process_stopped(int(marker.read_text()))


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group behavior")
def test_async_transport_kills_sigterm_resistant_group_member_without_output(
    fake_binary: Path, tmp_path: Path
) -> None:
    ready = tmp_path / "async-quiet-child-ready"
    marker = tmp_path / "async-quiet-child-survived"

    async def exercise() -> None:
        with pytest.raises(subprocess.TimeoutExpired):
            await AsyncSubprocessTransport(fake_binary).invoke(
                ["quiet-child", str(ready), str(marker)], timeout=0.5
            )

    asyncio.run(exercise())
    time.sleep(1.2)
    assert not marker.exists()


def test_async_transport_terminates_leader_when_group_capture_fails(
    fake_binary: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "async-capture-failure-leader-survived"
    monkeypatch.setattr(
        transport_module,
        "capture_process_group_members",
        lambda _pid, _excluded: (),
    )

    async def exercise() -> None:
        with pytest.raises(subprocess.TimeoutExpired):
            await AsyncSubprocessTransport(fake_binary).invoke(
                ["leader", str(marker)], timeout=0.1
            )

    asyncio.run(exercise())
    time.sleep(1.2)
    assert not marker.exists()


def test_transport_preserves_invalid_working_directory_errors(fake_binary: Path, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        SubprocessTransport(fake_binary).invoke([], cwd=tmp_path / "missing")


@pytest.fixture
def fake_binary(tmp_path: Path) -> Path:
    binary = tmp_path / "noisy-cli"
    binary.write_text(
        """#!/usr/bin/env python3
import pathlib
import subprocess
import sys
import time

if sys.argv[1] == "child":
    marker = sys.argv[2]
    subprocess.Popen([sys.executable, "-c", "import pathlib,time,sys; time.sleep(0.5); pathlib.Path(sys.argv[1]).write_text('survived')", marker])
    time.sleep(30)
if sys.argv[1] == "overflow":
    sys.stdout.write("x" * 4096)
    sys.stdout.flush()
    time.sleep(30)
if sys.argv[1] == "holder":
    holder = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        start_new_session=True,
    )
    pathlib.Path(sys.argv[2]).write_text(str(holder.pid))
if sys.argv[1] == "leader":
    time.sleep(1)
    pathlib.Path(sys.argv[2]).touch()
if sys.argv[1] == "touch":
    pathlib.Path(sys.argv[2]).touch()
if sys.argv[1] == "quiet-child":
    subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import pathlib,signal,sys,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); pathlib.Path(sys.argv[1]).touch(); time.sleep(1); pathlib.Path(sys.argv[2]).touch()",
            sys.argv[2],
            sys.argv[3],
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    while not pathlib.Path(sys.argv[2]).exists():
        time.sleep(0.01)
    time.sleep(30)
""",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    return binary


def assert_process_stopped(pid: int) -> None:
    ps = next(path for path in ("/bin/ps", "/usr/bin/ps") if Path(path).is_file())
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        result = subprocess.run(
            [ps, "-o", "stat=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=0.2,
            check=False,
        )
        status = result.stdout.strip()
        if result.returncode != 0 or not status or status.startswith("Z"):
            return
        time.sleep(0.01)
    pytest.fail(f"process {pid} survived cleanup")
