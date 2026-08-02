from __future__ import annotations

import ctypes
import os
import time

UInt32 = ctypes.c_uint32
UInt64 = ctypes.c_uint64
Int32 = ctypes.c_int32
Int64 = ctypes.c_int64

_LIBPROC = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
_PROC_PIDINFO = _LIBPROC.proc_pidinfo
_PROC_PIDINFO.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_uint64,
    ctypes.c_void_p,
    ctypes.c_int,
]
_PROC_PIDINFO.restype = ctypes.c_int
_PROC_PIDFDINFO = _LIBPROC.proc_pidfdinfo
_PROC_PIDFDINFO.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
]
_PROC_PIDFDINFO.restype = ctypes.c_int
_PROC_LISTALLPIDS = _LIBPROC.proc_listallpids
_PROC_LISTALLPIDS.argtypes = [ctypes.c_void_p, ctypes.c_int]
_PROC_LISTALLPIDS.restype = ctypes.c_int


class _ProcessInfo(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint32),
        ("status", ctypes.c_uint32),
        ("exit_status", ctypes.c_uint32),
        ("pid", ctypes.c_uint32),
        ("parent_pid", ctypes.c_uint32),
        ("uid", ctypes.c_uint32),
        ("gid", ctypes.c_uint32),
        ("real_uid", ctypes.c_uint32),
        ("real_gid", ctypes.c_uint32),
        ("saved_uid", ctypes.c_uint32),
        ("saved_gid", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("command", ctypes.c_char * 16),
        ("name", ctypes.c_char * 32),
        ("file_count", ctypes.c_uint32),
        ("process_group", ctypes.c_uint32),
        ("job_control_count", ctypes.c_uint32),
        ("terminal_device", ctypes.c_uint32),
        ("terminal_process_group", ctypes.c_uint32),
        ("nice", ctypes.c_int32),
        ("start_seconds", ctypes.c_uint64),
        ("start_microseconds", ctypes.c_uint64),
    ]


class _ProcessFileInfo(ctypes.Structure):
    _fields_ = [
        ("open_flags", UInt32),
        ("status", UInt32),
        ("offset", Int64),
        ("file_type", Int32),
        ("guard_flags", UInt32),
    ]


class _FileDescriptorInfo(ctypes.Structure):
    _fields_ = [("descriptor", Int32), ("file_type", UInt32)]


class _VnodeStat(ctypes.Structure):
    _fields_ = [
        ("device", UInt32),
        ("mode", ctypes.c_uint16),
        ("link_count", ctypes.c_uint16),
        ("inode", UInt64),
        ("uid", UInt32),
        ("gid", UInt32),
        *( (name, Int64) for name in ("access_time", "access_nanoseconds", "modify_time", "modify_nanoseconds", "change_time", "change_nanoseconds", "birth_time", "birth_nanoseconds", "size", "blocks") ),
        ("block_size", Int32),
        ("flags", UInt32),
        ("generation", UInt32),
        ("raw_device", UInt32),
        ("spare", Int64 * 2),
    ]


class _PipeInfo(ctypes.Structure):
    _fields_ = [
        ("stat", _VnodeStat),
        ("handle", UInt64),
        ("peer_handle", UInt64),
        ("status", Int32),
        ("reserved", Int32),
    ]


class _PipeDescriptorInfo(ctypes.Structure):
    _fields_ = [("file", _ProcessFileInfo), ("pipe", _PipeInfo)]


def process_identity(pid: int) -> tuple[int, str] | None:
    info = _ProcessInfo()
    size = ctypes.sizeof(info)
    written = _PROC_PIDINFO(pid, 3, 0, ctypes.byref(info), size)
    if written != size or info.pid != pid:
        return None
    identity = f"{info.start_seconds}:{info.start_microseconds}:{info.process_group}"
    return info.uid, identity


def pipe_handles(pid: int, descriptors: tuple[int, ...]) -> frozenset[int]:
    handles: set[int] = set()
    for descriptor in descriptors:
        pipe = _pipe_descriptor_info(pid, descriptor)
        if pipe is not None:
            handles.update((pipe.handle, pipe.peer_handle))
    handles.discard(0)
    return frozenset(handles)


def pipe_holder_pids(handles: frozenset[int], deadline: float) -> list[int]:
    holders: list[int] = []
    for pid in _all_pids():
        if time.monotonic() >= deadline:
            break
        identity = process_identity(pid)
        if identity is None or identity[0] != os.getuid():
            continue
        if process_holds_pipe(pid, handles, deadline):
            holders.append(pid)
    return holders


def process_holds_pipe(pid: int, handles: frozenset[int], deadline: float) -> bool:
    for descriptor in _pipe_descriptors(pid):
        if time.monotonic() >= deadline:
            break
        pipe = _pipe_descriptor_info(pid, descriptor)
        if pipe is not None and handles.intersection((pipe.handle, pipe.peer_handle)):
            return True
    return False


def _all_pids() -> list[int]:
    count = _PROC_LISTALLPIDS(None, 0)
    if count <= 0:
        return []
    processes = (ctypes.c_int * (count + 128))()
    found = _PROC_LISTALLPIDS(processes, ctypes.sizeof(processes))
    return list(processes[: max(0, found)])


def _pipe_descriptors(pid: int) -> list[int]:
    needed = _PROC_PIDINFO(pid, 1, 0, None, 0)
    if needed <= 0:
        return []
    capacity = needed // ctypes.sizeof(_FileDescriptorInfo) + 16
    descriptors = (_FileDescriptorInfo * capacity)()
    written = _PROC_PIDINFO(pid, 1, 0, descriptors, ctypes.sizeof(descriptors))
    count = max(0, written) // ctypes.sizeof(_FileDescriptorInfo)
    return [item.descriptor for item in descriptors[:count] if item.file_type == 6]


def _pipe_descriptor_info(pid: int, descriptor: int) -> _PipeInfo | None:
    info = _PipeDescriptorInfo()
    size = ctypes.sizeof(info)
    written = _PROC_PIDFDINFO(pid, descriptor, 6, ctypes.byref(info), size)
    return info.pipe if written == size else None
