from __future__ import annotations

import ctypes
import ntpath
import os
import subprocess
from collections.abc import Callable
from typing import IO, Any

_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_TH32CS_SNAPTHREAD = 0x00000004
_THREAD_SUSPEND_RESUME = 0x0002
_INVALID_DWORD = 0xFFFFFFFF
WINDOWS_TERMINATE_TIMEOUT_SECONDS = 5.0


class _IoCounters(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _BasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _ExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _ThreadEntry(ctypes.Structure):
    _fields_ = [
        ("dwSize", ctypes.c_uint32),
        ("cntUsage", ctypes.c_uint32),
        ("th32ThreadID", ctypes.c_uint32),
        ("th32OwnerProcessID", ctypes.c_uint32),
        ("tpBasePri", ctypes.c_long),
        ("tpDeltaPri", ctypes.c_long),
        ("dwFlags", ctypes.c_uint32),
    ]


class WindowsJob:
    def __init__(self, kernel32: Any, handle: int) -> None:
        self._kernel32 = kernel32
        self._handle = handle

    def terminate(self) -> bool:
        return bool(self._kernel32.TerminateJobObject(self._handle, 1))

    def close(self) -> None:
        if self._handle:
            self._kernel32.CloseHandle(self._handle)
            self._handle = 0


def create_windows_job(process: subprocess.Popen[bytes]) -> WindowsJob | None:
    if os.name != "nt":
        return None
    loader = getattr(ctypes, "WinDLL", None)
    process_handle = getattr(process, "_handle", None)
    if loader is None or process_handle is None:
        return None
    kernel32: Any | None = None
    handle = 0
    transferred = False
    try:
        kernel32 = loader("kernel32", use_last_error=True)
        _configure_kernel32(kernel32)
        handle = int(kernel32.CreateJobObjectW(None, None) or 0)
        if not handle:
            return None
        limits = _ExtendedLimitInformation()
        limits.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        configured = kernel32.SetInformationJobObject(
            handle,
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        )
        assigned = configured and kernel32.AssignProcessToJobObject(
            handle, int(process_handle)
        )
        if not assigned:
            return None
        job = WindowsJob(kernel32, handle)
        transferred = True
        return job
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    finally:
        if handle and not transferred and kernel32 is not None:
            try:
                kernel32.CloseHandle(handle)
            except (AttributeError, OSError):
                pass


def resume_windows_process(pid: int) -> bool:
    if os.name != "nt":
        return True
    loader = getattr(ctypes, "WinDLL", None)
    if loader is None:
        return False
    try:
        kernel32 = loader("kernel32", use_last_error=True)
        _configure_kernel32(kernel32)
        snapshot = int(
            kernel32.CreateToolhelp32Snapshot(_TH32CS_SNAPTHREAD, 0) or 0
        )
    except (AttributeError, OSError, TypeError, ValueError):
        return False
    try:
        if snapshot in {0, int(ctypes.c_void_p(-1).value or -1)}:
            return False
        return _resume_process_threads(kernel32, snapshot, pid)
    finally:
        if snapshot not in {0, int(ctypes.c_void_p(-1).value or -1)}:
            kernel32.CloseHandle(snapshot)


def prepare_windows_process(
    process: subprocess.Popen[bytes],
) -> WindowsJob | None:
    if os.name != "nt":
        return None
    windows_job: WindowsJob | None = None
    resumed = False
    primary_error: BaseException | None = None
    try:
        windows_job = create_windows_job(process)
        if windows_job is None or not resume_windows_process(process.pid):
            raise RuntimeError(
                f"failed to contain and resume autotune process {process.pid}"
            )
        resumed = True
        return windows_job
    except BaseException as error:
        primary_error = error
        raise
    finally:
        if not resumed:
            cleanup_actions: list[Callable[[], None]] = []
            if windows_job is None:
                cleanup_actions.append(lambda: _kill_process(process))
            else:
                cleanup_actions.extend(
                    (
                        lambda: _fallback_windows_termination(process, windows_job),
                        windows_job.close,
                    )
                )
            cleanup_actions.extend(
                (
                    lambda: _close_pipe(process.stdout),
                    lambda: _close_pipe(process.stderr),
                    lambda: _wait_for_process_exit(process),
                )
            )
            cleanup_errors: list[BaseException] = []
            for action in cleanup_actions:
                try:
                    action()
                except BaseException as cleanup_exception:  # noqa: BLE001
                    cleanup_errors.append(cleanup_exception)
            if primary_error is None and cleanup_errors:
                raise cleanup_errors[0]
            if primary_error is not None:
                for stored_cleanup_error in cleanup_errors:
                    _annotate_cleanup_error(primary_error, stored_cleanup_error)


def terminate_windows_process_tree(
    process: subprocess.Popen[bytes], windows_job: WindowsJob | None
) -> None:
    if process.poll() is not None:
        if windows_job is not None:
            windows_job.terminate()
        return
    _windows_terminate_tree(process, windows_job)


def _windows_terminate_tree(
    process: subprocess.Popen[bytes], windows_job: WindowsJob | None = None
) -> None:
    system_directory = _windows_system_directory()
    if system_directory is None:
        _fallback_windows_termination(process, windows_job)
        return
    taskkill = ntpath.join(system_directory, "taskkill.exe")
    if not os.path.isfile(taskkill):
        _fallback_windows_termination(process, windows_job)
        return
    system_root = ntpath.dirname(system_directory)
    try:
        result = subprocess.run(
            [taskkill, "/PID", str(process.pid), "/T", "/F"],
            check=False,
            env={"SystemRoot": system_root, "WINDIR": system_root},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=WINDOWS_TERMINATE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        _fallback_windows_termination(process, windows_job)
        return
    if result.returncode != 0 and process.poll() is None:
        _fallback_windows_termination(process, windows_job)


def _fallback_windows_termination(
    process: subprocess.Popen[bytes], windows_job: WindowsJob | None
) -> None:
    if windows_job is not None and windows_job.terminate():
        return
    _kill_process(process)


def _windows_system_directory() -> str | None:
    if os.name != "nt":
        return None
    loader = getattr(ctypes, "WinDLL", None)
    if loader is None:
        return None
    try:
        get_system_directory = loader(
            "kernel32", use_last_error=True
        ).GetSystemDirectoryW
        get_system_directory.argtypes = [ctypes.c_wchar_p, ctypes.c_uint]
        get_system_directory.restype = ctypes.c_uint
        buffer = ctypes.create_unicode_buffer(32768)
        length = get_system_directory(buffer, len(buffer))
    except (AttributeError, OSError):
        return None
    if length == 0 or length >= len(buffer):
        return None
    directory = ntpath.normpath(buffer.value)
    return directory if _is_valid_windows_system_directory(directory) else None


def _is_valid_windows_system_directory(directory: str) -> bool:
    normalized = ntpath.normpath(directory)
    drive, _ = ntpath.splitdrive(normalized)
    return (
        len(drive) == 2
        and drive[1] == ":"
        and ntpath.isabs(normalized)
        and ntpath.basename(normalized).lower() == "system32"
    )


def _kill_process(process: subprocess.Popen[bytes]) -> None:
    try:
        process.kill()
    except OSError:
        pass


def _wait_for_process_exit(process: subprocess.Popen[bytes]) -> None:
    try:
        process.wait(timeout=WINDOWS_TERMINATE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"autotune process {process.pid} did not exit after termination"
        ) from error


def _close_pipe(pipe: IO[Any] | None) -> None:
    if pipe is not None:
        pipe.close()


def _annotate_cleanup_error(
    primary_error: BaseException, cleanup_error: BaseException
) -> None:
    add_note = getattr(primary_error, "add_note", None)
    if add_note is not None:
        add_note(f"Cleanup also failed: {cleanup_error}")


def _resume_process_threads(kernel32: Any, snapshot: int, pid: int) -> bool:
    entry = _ThreadEntry()
    entry.dwSize = ctypes.sizeof(entry)
    has_entry = bool(kernel32.Thread32First(snapshot, ctypes.byref(entry)))
    resumed = False
    while has_entry:
        if entry.th32OwnerProcessID == pid:
            thread = int(
                kernel32.OpenThread(
                    _THREAD_SUSPEND_RESUME, False, entry.th32ThreadID
                )
                or 0
            )
            if not thread:
                return False
            try:
                if kernel32.ResumeThread(thread) == _INVALID_DWORD:
                    return False
                resumed = True
            finally:
                kernel32.CloseHandle(thread)
        has_entry = bool(kernel32.Thread32Next(snapshot, ctypes.byref(entry)))
    return resumed


def _configure_kernel32(kernel32: Any) -> None:
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
    kernel32.CreateJobObjectW.restype = ctypes.c_void_p
    kernel32.SetInformationJobObject.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    kernel32.SetInformationJobObject.restype = ctypes.c_int
    kernel32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    kernel32.AssignProcessToJobObject.restype = ctypes.c_int
    kernel32.TerminateJobObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel32.TerminateJobObject.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int
    kernel32.CreateToolhelp32Snapshot.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
    kernel32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
    kernel32.Thread32First.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    kernel32.Thread32First.restype = ctypes.c_int
    kernel32.Thread32Next.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    kernel32.Thread32Next.restype = ctypes.c_int
    kernel32.OpenThread.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel32.OpenThread.restype = ctypes.c_void_p
    kernel32.ResumeThread.argtypes = [ctypes.c_void_p]
    kernel32.ResumeThread.restype = ctypes.c_uint32
