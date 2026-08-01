from __future__ import annotations

import heapq
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

READER_CLEANUP_SECONDS = 1.0


@dataclass(frozen=True)
class OutputPipes:
    proc_identities: frozenset[tuple[int, int]] = frozenset()
    darwin_handles: frozenset[int] = frozenset()
    lsof_devices: frozenset[str] = frozenset()


def capture_output_pipes(pipe_fds: Sequence[int]) -> OutputPipes:
    if os.name != "posix":
        return OutputPipes()
    if sys.platform.startswith("linux"):
        identities: set[tuple[int, int]] = set()
        for descriptor in pipe_fds:
            try:
                metadata = os.fstat(descriptor)
            except OSError:
                continue
            identities.add((metadata.st_dev, metadata.st_ino))
        return OutputPipes(proc_identities=frozenset(identities))
    if sys.platform == "darwin":
        from .darwin_process import pipe_handles

        return OutputPipes(darwin_handles=pipe_handles(os.getpid(), tuple(pipe_fds)))

    lsof = _trusted_command("/usr/sbin/lsof", "/usr/bin/lsof")
    if lsof is None:
        return OutputPipes()
    descriptors = ",".join(str(descriptor) for descriptor in pipe_fds)
    if not descriptors:
        return OutputPipes()
    try:
        result = subprocess.run(
            [
                lsof,
                *_lsof_fast_flags(),
                "-n",
                "-P",
                "-F",
                "pfdn",
                "-a",
                "-p",
                str(os.getpid()),
                "-d",
                descriptors,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=0.5,
            env={"PATH": str(Path(lsof).parent), "LC_ALL": "C"},
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return OutputPipes()
    devices = frozenset(
        line[3:] for line in result.stdout.splitlines() if line.startswith("n->")
    )
    return OutputPipes(lsof_devices=devices)


def terminate_output_holders(output_pipes: OutputPipes, excluded_pid: int) -> None:
    if os.name != "posix" or not (
        output_pipes.proc_identities
        or output_pipes.darwin_handles
        or output_pipes.lsof_devices
    ):
        return
    deadline = time.monotonic() + READER_CLEANUP_SECONDS
    for pid in _output_holder_pids(output_pipes, deadline):
        if pid in {os.getpid(), excluded_pid} or time.monotonic() >= deadline:
            continue
        expected_identity = _process_identity(pid, deadline)
        if expected_identity is None or expected_identity[0] != os.getuid():
            continue
        if not _process_holds_output(pid, output_pipes, deadline):
            continue
        _kill_matching_process(pid, expected_identity, deadline)


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    value: tuple[int, str]


def capture_process_group_members(
    pgid: int, excluded_pid: int
) -> tuple[ProcessIdentity, ...]:
    if os.name != "posix":
        return ()
    try:
        if os.getpgid(pgid) != pgid:
            return ()
    except OSError:
        return ()
    deadline = time.monotonic() + READER_CLEANUP_SECONDS
    leader_identity = _process_identity(pgid, deadline)
    if leader_identity is None or leader_identity[0] != os.getuid():
        return ()
    members: list[ProcessIdentity] = []
    for pid in _process_group_pids(pgid, deadline):
        if pid in {os.getpid(), excluded_pid} or time.monotonic() >= deadline:
            continue
        expected_identity = _process_identity(pid, deadline)
        if expected_identity is None or expected_identity[0] != os.getuid():
            continue
        try:
            current_pgid = os.getpgid(pid)
        except OSError:
            continue
        if current_pgid == pgid:
            members.append(ProcessIdentity(pid, expected_identity))
    if _process_identity(pgid, deadline) != leader_identity:
        return ()
    return tuple(members)


def terminate_process_group_members(members: Sequence[ProcessIdentity]) -> None:
    signal_processes(members, signal.SIGKILL)


def signal_processes(
    members: Sequence[ProcessIdentity], process_signal: signal.Signals
) -> None:
    deadline = time.monotonic() + READER_CLEANUP_SECONDS
    for member in members:
        if time.monotonic() >= deadline:
            break
        _signal_matching_process(member.pid, member.value, process_signal, deadline)


def _process_group_pids(pgid: int, deadline: float) -> list[int]:
    if sys.platform.startswith("linux"):
        try:
            entries = os.scandir("/proc")
        except OSError:
            return []
        members: list[int] = []
        with entries:
            for entry in entries:
                if time.monotonic() >= deadline:
                    break
                if entry.name.isdigit() and _process_group_matches(
                    int(entry.name), pgid
                ):
                    members.append(int(entry.name))
        return members

    ps = _trusted_command("/bin/ps", "/usr/bin/ps")
    if ps is None:
        return []
    try:
        result = subprocess.run(
            [ps, "-axo", "pid=,uid=,pgid="],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=max(0.01, deadline - time.monotonic()),
            env={"PATH": str(Path(ps).parent), "LC_ALL": "C"},
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    members: list[int] = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        try:
            pid, uid, process_group = (int(field) for field in fields)
        except ValueError:
            continue
        if uid == os.getuid() and process_group == pgid:
            members.append(pid)
    return members


def _process_group_matches(pid: int, pgid: int) -> bool:
    try:
        return os.getpgid(pid) == pgid
    except OSError:
        return False


def _output_holder_pids(output_pipes: OutputPipes, deadline: float) -> list[int]:
    if output_pipes.proc_identities:
        return _proc_output_holder_pids(output_pipes, deadline)
    if output_pipes.darwin_handles:
        from .darwin_process import pipe_holder_pids

        return pipe_holder_pids(output_pipes.darwin_handles, deadline)
    lsof = _trusted_command("/usr/sbin/lsof", "/usr/bin/lsof")
    if lsof is None:
        return []
    try:
        result = subprocess.run(
            [
                lsof,
                *_lsof_fast_flags(),
                "-n",
                "-P",
                "-F",
                "pfd",
                "-a",
                "-u",
                str(os.getuid()),
                "-d",
                "0-1023",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=max(0.01, deadline - time.monotonic()),
            env={"PATH": str(Path(lsof).parent), "LC_ALL": "C"},
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    holders: set[int] = set()
    pid: int | None = None
    for line in result.stdout.splitlines():
        if line.startswith("p"):
            try:
                pid = int(line[1:])
            except ValueError:
                pid = None
        elif (
            line.startswith("d")
            and pid is not None
            and line[1:] in output_pipes.lsof_devices
        ):
            holders.add(pid)
    return list(holders)


def _proc_output_holder_pids(output_pipes: OutputPipes, deadline: float) -> list[int]:
    holders: list[int] = []
    descriptor_budget = [65536]
    try:
        entries = os.scandir("/proc")
    except OSError:
        return holders
    with entries:
        discovery_deadline = time.monotonic() + max(
            0.001, (deadline - time.monotonic()) / 4
        )
        candidates = _recent_process_pids(entries, discovery_deadline)
    for pid in candidates:
        if time.monotonic() >= deadline or descriptor_budget[0] <= 0:
            break
        if _process_holds_output(pid, output_pipes, deadline, descriptor_budget):
            holders.append(pid)
    return holders


def _process_holds_output(
    pid: int,
    output_pipes: OutputPipes,
    deadline: float,
    descriptor_budget: list[int] | None = None,
) -> bool:
    if time.monotonic() >= deadline:
        return False
    if output_pipes.proc_identities:
        try:
            descriptors = os.scandir(f"/proc/{pid}/fd")
        except OSError:
            return False
        with descriptors:
            for index, descriptor in enumerate(descriptors):
                if index >= 1024 or time.monotonic() >= deadline:
                    break
                if descriptor_budget is not None:
                    if descriptor_budget[0] <= 0:
                        break
                    descriptor_budget[0] -= 1
                try:
                    metadata = os.stat(descriptor.path)
                except OSError:
                    continue
                if (metadata.st_dev, metadata.st_ino) in output_pipes.proc_identities:
                    return True
        return False
    if output_pipes.darwin_handles:
        from .darwin_process import process_holds_pipe

        return process_holds_pipe(pid, output_pipes.darwin_handles, deadline)

    lsof = _trusted_command("/usr/sbin/lsof", "/usr/bin/lsof")
    if lsof is None:
        return False
    try:
        result = subprocess.run(
            [
                lsof,
                *_lsof_fast_flags(),
                "-n",
                "-P",
                "-F",
                "pfd",
                "-a",
                "-p",
                str(pid),
                "-d",
                "0-1023",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=max(0.01, deadline - time.monotonic()),
            env={"PATH": str(Path(lsof).parent), "LC_ALL": "C"},
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return any(
        line.startswith("d") and line[1:] in output_pipes.lsof_devices
        for line in result.stdout.splitlines()
    )


def _trusted_command(*candidates: str) -> str | None:
    return next(
        (candidate for candidate in candidates if os.access(candidate, os.X_OK)), None
    )


def _lsof_fast_flags() -> list[str]:
    return ["-b", "-X"] if sys.platform == "darwin" else []


def _recent_process_pids(
    entries: Iterator[os.DirEntry[str]], deadline: float, limit: int = 4096
) -> list[int]:
    try:
        last_pid = int(Path("/proc/sys/kernel/ns_last_pid").read_text())
        pid_max = int(Path("/proc/sys/kernel/pid_max").read_text())
    except (OSError, ValueError):
        last_pid = 0
        pid_max = 0
    recent: list[tuple[int, int]] = []
    for entry in entries:
        if time.monotonic() >= deadline:
            break
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        score = (
            pid_max - ((last_pid - pid + pid_max) % pid_max)
            if last_pid > 0 and pid_max >= last_pid
            else pid
        )
        if len(recent) < limit:
            heapq.heappush(recent, (score, pid))
        elif score > recent[0][0]:
            heapq.heapreplace(recent, (score, pid))
    return [pid for _, pid in sorted(recent, reverse=True)]


def _process_identity(pid: int, deadline: float) -> tuple[int, str] | None:
    if time.monotonic() >= deadline:
        return None
    if sys.platform.startswith("linux"):
        try:
            stat_text = Path(f"/proc/{pid}/stat").read_text()
            status = Path(f"/proc/{pid}/status").read_text()
            start_time = stat_text[stat_text.rfind(")") + 2 :].split()[19]
            uid = int(
                next(
                    line.split()[1]
                    for line in status.splitlines()
                    if line.startswith("Uid:")
                )
            )
            return uid, start_time
        except (OSError, ValueError, IndexError, StopIteration):
            return None
    if sys.platform == "darwin":
        from .darwin_process import process_identity

        return process_identity(pid)
    ps = _trusted_command("/bin/ps", "/usr/bin/ps")
    if ps is None:
        return None
    try:
        result = subprocess.run(
            [ps, "-o", "pid=,uid=,lstart=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=max(0.01, deadline - time.monotonic()),
            env={"PATH": str(Path(ps).parent), "LC_ALL": "C"},
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    fields = result.stdout.strip().split()
    if len(fields) < 7 or fields[0] != str(pid):
        return None
    try:
        return int(fields[1]), " ".join(fields[2:7])
    except ValueError:
        return None


def _kill_matching_process(
    pid: int, expected_identity: tuple[int, str], deadline: float
) -> None:
    _signal_matching_process(pid, expected_identity, signal.SIGKILL, deadline)


def _signal_matching_process(
    pid: int,
    expected_identity: tuple[int, str],
    process_signal: signal.Signals,
    deadline: float,
) -> None:
    if sys.platform.startswith("linux") and hasattr(os, "pidfd_open"):
        try:
            pidfd = os.pidfd_open(pid)
        except ProcessLookupError:
            return
        except OSError:
            pidfd = None
        if pidfd is not None:
            try:
                if _process_identity(pid, deadline) != expected_identity:
                    return
                pidfd_send_signal = getattr(signal, "pidfd_send_signal", None)
                if pidfd_send_signal is not None:
                    try:
                        pidfd_send_signal(pidfd, process_signal)
                    except OSError:
                        pass
                    else:
                        return
            finally:
                os.close(pidfd)
    if _process_identity(pid, deadline) != expected_identity:
        return
    try:
        os.kill(pid, process_signal)
    except OSError:
        pass
