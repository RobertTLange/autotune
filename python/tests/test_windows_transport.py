from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import autotune_cli.transport as transport_module
import autotune_cli.windows_job as windows_job_module
from autotune_cli.transport import SubprocessTransport


def test_windows_tree_cleanup_uses_trusted_bounded_taskkill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = MagicMock(pid=1234)
    run = MagicMock(return_value=subprocess.CompletedProcess([], 0))
    monkeypatch.setattr(
        windows_job_module, "_windows_system_directory", lambda: r"C:\Windows\System32"
    )
    monkeypatch.setattr(windows_job_module.os.path, "isfile", lambda _path: True)
    monkeypatch.setattr(windows_job_module.subprocess, "run", run)

    windows_job_module._windows_terminate_tree(process)

    run.assert_called_once_with(
        [r"C:\Windows\System32\taskkill.exe", "/PID", "1234", "/T", "/F"],
        check=False,
        env={"SystemRoot": r"C:\Windows", "WINDIR": r"C:\Windows"},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=5.0,
    )
    process.kill.assert_not_called()


def test_windows_tree_cleanup_uses_job_when_taskkill_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = MagicMock(pid=1234)
    job = MagicMock()
    job.terminate.return_value = True
    monkeypatch.setattr(
        windows_job_module, "_windows_system_directory", lambda: r"C:\Windows\System32"
    )
    monkeypatch.setattr(windows_job_module.os.path, "isfile", lambda _path: True)
    monkeypatch.setattr(
        windows_job_module.subprocess,
        "run",
        MagicMock(side_effect=subprocess.TimeoutExpired("taskkill", 5.0)),
    )

    windows_job_module._windows_terminate_tree(process, job)

    job.terminate.assert_called_once_with()
    process.kill.assert_not_called()


def test_windows_tree_cleanup_terminates_job_after_launcher_exits() -> None:
    process = MagicMock(pid=1234)
    process.poll.return_value = 0
    job = MagicMock()

    windows_job_module.terminate_windows_process_tree(process, job)

    job.terminate.assert_called_once_with()
    process.kill.assert_not_called()


def test_windows_process_is_assigned_before_it_is_resumed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = MagicMock(pid=1234)
    job = MagicMock()
    events: list[str] = []
    monkeypatch.setattr(windows_job_module, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(
        windows_job_module,
        "create_windows_job",
        lambda _process: events.append("assigned") or job,
    )
    monkeypatch.setattr(
        windows_job_module,
        "resume_windows_process",
        lambda _pid: events.append("resumed") or True,
    )

    assert windows_job_module.prepare_windows_process(process) is job
    assert events == ["assigned", "resumed"]


def test_windows_process_does_not_resume_without_job_containment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = MagicMock(pid=1234, stdout=None, stderr=None)
    process.wait.return_value = -9
    resume = MagicMock()
    monkeypatch.setattr(windows_job_module, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(windows_job_module, "create_windows_job", lambda _process: None)
    monkeypatch.setattr(windows_job_module, "resume_windows_process", resume)

    with pytest.raises(RuntimeError, match="contain and resume"):
        windows_job_module.prepare_windows_process(process)

    resume.assert_not_called()
    process.kill.assert_called_once_with()


def test_windows_creation_suspends_process_before_job_assignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(transport_module, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(
        transport_module.subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200, raising=False
    )
    monkeypatch.setattr(
        transport_module.subprocess, "CREATE_SUSPENDED", 0x004, raising=False
    )

    assert transport_module._windows_creation_flags() == 0x204


def test_windows_process_exit_wait_is_bounded() -> None:
    process = MagicMock(pid=1234)
    process.wait.side_effect = subprocess.TimeoutExpired("autotune", 5.0)

    with pytest.raises(RuntimeError, match="did not exit"):
        transport_module._wait_for_process_exit(process, timeout=5.0)

    process.wait.assert_called_once_with(timeout=5.0)


def test_cleanup_failure_preserves_primary_error() -> None:
    primary = KeyboardInterrupt()

    transport_module._raise_cleanup_error(RuntimeError("cleanup failed"), primary)

    notes = getattr(primary, "__notes__", [])
    if notes:
        assert notes == ["Cleanup also failed: cleanup failed"]


def test_cleanup_failure_does_not_replace_timeout() -> None:
    timeout = subprocess.TimeoutExpired(["autotune"], 1.0)

    with pytest.raises(subprocess.TimeoutExpired) as raised:
        transport_module._raise_invocation_error(
            timeout, RuntimeError("cleanup failed")
        )

    assert raised.value is timeout


def test_windows_setup_interrupt_reclaims_suspended_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = MagicMock(pid=1234)
    process.wait.return_value = -9
    monkeypatch.setattr(windows_job_module, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(
        windows_job_module,
        "create_windows_job",
        MagicMock(side_effect=KeyboardInterrupt),
    )

    with pytest.raises(KeyboardInterrupt):
        windows_job_module.prepare_windows_process(process)

    process.kill.assert_called_once_with()
    process.stdout.close.assert_called_once_with()
    process.stderr.close.assert_called_once_with()


def test_transport_closes_job_when_post_prepare_initialization_interrupts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = MagicMock()
    job = MagicMock()
    monkeypatch.setattr(transport_module.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(transport_module, "prepare_windows_process", lambda _process: job)
    monkeypatch.setattr(
        transport_module.time, "monotonic", MagicMock(side_effect=KeyboardInterrupt)
    )

    with pytest.raises(KeyboardInterrupt):
        SubprocessTransport("autotune").invoke([])

    job.close.assert_called_once_with()


def test_windows_failed_start_closes_job_when_termination_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = MagicMock(pid=1234)
    process.wait.return_value = -9
    job = MagicMock()
    job.terminate.side_effect = OSError("terminate failed")
    monkeypatch.setattr(windows_job_module, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(windows_job_module, "create_windows_job", lambda _process: job)
    monkeypatch.setattr(windows_job_module, "resume_windows_process", lambda _pid: False)

    with pytest.raises(RuntimeError, match="contain and resume"):
        windows_job_module.prepare_windows_process(process)

    job.close.assert_called_once_with()
    process.stdout.close.assert_called_once_with()
    process.stderr.close.assert_called_once_with()


def test_windows_job_configures_assigns_terminates_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel32 = MagicMock()
    kernel32.CreateJobObjectW.return_value = 100
    kernel32.SetInformationJobObject.return_value = 1
    kernel32.AssignProcessToJobObject.return_value = 1
    kernel32.TerminateJobObject.return_value = 1
    process = MagicMock()
    process._handle = 200
    monkeypatch.setattr(windows_job_module, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(
        windows_job_module.ctypes,
        "WinDLL",
        lambda *_args, **_kwargs: kernel32,
        raising=False,
    )

    job = windows_job_module.create_windows_job(process)

    assert job is not None
    assert job.terminate()
    job.close()
    job.close()
    kernel32.AssignProcessToJobObject.assert_called_once_with(100, 200)
    kernel32.TerminateJobObject.assert_called_once_with(100, 1)
    kernel32.CloseHandle.assert_called_once_with(100)


def test_windows_job_creation_interrupt_closes_untransferred_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel32 = MagicMock()
    kernel32.CreateJobObjectW.return_value = 100
    kernel32.SetInformationJobObject.side_effect = KeyboardInterrupt
    process = MagicMock()
    process._handle = 200
    monkeypatch.setattr(windows_job_module, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(
        windows_job_module.ctypes,
        "WinDLL",
        lambda *_args, **_kwargs: kernel32,
        raising=False,
    )

    with pytest.raises(KeyboardInterrupt):
        windows_job_module.create_windows_job(process)

    kernel32.CloseHandle.assert_called_once_with(100)


def test_windows_process_resume_uses_suspended_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel32 = MagicMock()
    kernel32.CreateToolhelp32Snapshot.return_value = 100

    def first_thread(_snapshot: int, entry_pointer: object) -> int:
        entry = entry_pointer._obj  # type: ignore[attr-defined]
        entry.th32OwnerProcessID = 1234
        entry.th32ThreadID = 5678
        return 1

    kernel32.Thread32First.side_effect = first_thread
    kernel32.Thread32Next.return_value = 0
    kernel32.OpenThread.return_value = 200
    kernel32.ResumeThread.return_value = 1
    monkeypatch.setattr(windows_job_module, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(
        windows_job_module.ctypes,
        "WinDLL",
        lambda *_args, **_kwargs: kernel32,
        raising=False,
    )

    assert windows_job_module.resume_windows_process(1234)
    kernel32.OpenThread.assert_called_once_with(0x0002, False, 5678)
    kernel32.ResumeThread.assert_called_once_with(200)
    assert kernel32.CloseHandle.call_args_list == [((200,),), ((100,),)]


@pytest.mark.parametrize(
    "directory",
    [r"System32", r"\\server\share\System32", r"C:\Windows\SysWOW64"],
)
def test_windows_system_directory_validation_rejects_untrusted_paths(
    directory: str,
) -> None:
    assert not windows_job_module._is_valid_windows_system_directory(directory)


@pytest.mark.skipif(os.name != "nt", reason="Windows process-tree behavior")
def test_windows_job_fallback_terminates_descendant_after_launcher_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "windows-child-survived"
    monkeypatch.setattr(windows_job_module, "_windows_system_directory", lambda: None)
    child = (
        "import pathlib,sys,time; time.sleep(1); "
        "pathlib.Path(sys.argv[1]).touch()"
    )
    launcher = (
        "import subprocess,sys; "
        "subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]])"
    )

    with pytest.raises(subprocess.TimeoutExpired):
        SubprocessTransport(sys.executable).invoke(
            ["-c", launcher, child, str(marker)], timeout=0.5
        )

    time.sleep(1.2)
    assert not marker.exists()
