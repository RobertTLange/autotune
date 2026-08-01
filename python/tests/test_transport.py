from __future__ import annotations

import asyncio
import subprocess
import time
from pathlib import Path

import pytest

from autotune_cli.async_transport import AsyncSubprocessTransport
from autotune_cli.errors import AutotuneError
from autotune_cli.transport import SubprocessTransport


def test_transport_terminates_the_process_group_on_timeout(fake_binary: Path, tmp_path: Path) -> None:
    marker = tmp_path / "child-survived"

    with pytest.raises(subprocess.TimeoutExpired):
        SubprocessTransport(fake_binary).invoke(["child", str(marker)], timeout=0.1)

    time.sleep(0.7)
    assert not marker.exists()


def test_transport_rejects_stdout_overflow(fake_binary: Path) -> None:
    with pytest.raises(AutotuneError, match="stdout exceeded"):
        SubprocessTransport(fake_binary, max_output_bytes=128).invoke(["overflow"])


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


def test_async_transport_rejects_stdout_overflow(fake_binary: Path) -> None:
    async def exercise() -> None:
        with pytest.raises(AutotuneError, match="stdout exceeded"):
            await AsyncSubprocessTransport(fake_binary, max_output_bytes=128).invoke(["overflow"])

    asyncio.run(exercise())


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
""",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    return binary
