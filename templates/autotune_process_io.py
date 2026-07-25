"""Bounded output-holder discovery and termination for generated runtimes."""

from __future__ import annotations

import heapq
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

def _output_pipe_identities(process: subprocess.Popen[Any]) -> Tuple[str, Tuple[Any, ...]]:
    streams = (process.stdout, process.stderr)
    if sys.platform.startswith("linux"):
        identities = []
        for stream in streams:
            if stream:
                metadata = os.fstat(stream.fileno())
                identities.append((metadata.st_dev, metadata.st_ino))
        return ("proc", tuple(identities))
    lsof = _fixed_command(("/usr/sbin/lsof", "/usr/bin/lsof"))
    if not lsof:
        return ("none", ())
    try:
        result = subprocess.run(
            [lsof, "-n", "-P", "-F", "pfn", "-a", "-p", str(process.pid), "-d", "1,2"],
            capture_output=True,
            text=True,
            timeout=0.5,
            env={"PATH": str(Path(lsof).parent), "LC_ALL": "C"},
        )
    except (OSError, subprocess.TimeoutExpired):
        return ("none", ())
    return ("lsof", tuple(line[1:] for line in result.stdout.splitlines() if line.startswith("n")))


def _terminate_output_holders(
    pipe_identities: Tuple[str, Tuple[Any, ...]], excluded_pid: int
) -> None:
    kind, identities = pipe_identities
    if not identities or os.name == "nt":
        return
    deadline = time.monotonic() + 1
    for pid in _output_holder_pids(kind, identities, deadline):
        if pid in (os.getpid(), excluded_pid) or time.monotonic() >= deadline:
            continue
        expected = _process_identity(pid, deadline)
        if not expected or expected[0] != os.getuid():
            continue
        if not _process_holds_output(pid, kind, identities, deadline):
            continue
        if _process_identity(pid, deadline) != expected:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _output_holder_pids(kind: str, identities: Tuple[Any, ...], deadline: float) -> List[int]:
    if kind == "proc":
        holders = []
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
            if _process_holds_output(
                pid, kind, identities, deadline, descriptor_budget
            ):
                holders.append(pid)
        return holders
    lsof = _fixed_command(("/usr/sbin/lsof", "/usr/bin/lsof"))
    if not lsof:
        return []
    try:
        result = subprocess.run(
            [lsof, "-n", "-P", "-F", "pfn", "-a", "-u", str(os.getuid()), "-d", "0-1023"],
            capture_output=True,
            text=True,
            timeout=max(0.01, deadline - time.monotonic()),
            env={"PATH": str(Path(lsof).parent), "LC_ALL": "C"},
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    holders = set()
    pid: Optional[int] = None
    for line in result.stdout.splitlines():
        if line.startswith("p"):
            pid = int(line[1:])
        elif line.startswith("n") and pid and line[1:] in identities:
            holders.add(pid)
    return list(holders)


def _recent_process_pids(
    entries: Any,
    deadline: float,
    limit: int = 4096,
    last_pid: Optional[int] = None,
    pid_max: Optional[int] = None,
) -> List[int]:
    if last_pid is None or pid_max is None:
        try:
            last_pid = int(Path("/proc/sys/kernel/ns_last_pid").read_text())
            pid_max = int(Path("/proc/sys/kernel/pid_max").read_text())
        except (OSError, ValueError):
            last_pid = 0
            pid_max = 0
    recent: List[Tuple[int, int]] = []
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


def _process_holds_output(
    pid: int,
    kind: str,
    identities: Tuple[Any, ...],
    deadline: float,
    descriptor_budget: Optional[List[int]] = None,
) -> bool:
    if time.monotonic() >= deadline:
        return False
    if kind == "proc":
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
                if (metadata.st_dev, metadata.st_ino) in identities:
                    return True
        return False
    lsof = _fixed_command(("/usr/sbin/lsof", "/usr/bin/lsof"))
    if not lsof:
        return False
    try:
        result = subprocess.run(
            [lsof, "-n", "-P", "-F", "pfn", "-a", "-p", str(pid), "-d", "0-1023"],
            capture_output=True,
            text=True,
            timeout=max(0.01, deadline - time.monotonic()),
            env={"PATH": str(Path(lsof).parent), "LC_ALL": "C"},
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return any(line.startswith("n") and line[1:] in identities for line in result.stdout.splitlines())


def _process_identity(pid: int, deadline: float) -> Optional[Tuple[int, str]]:
    if time.monotonic() >= deadline:
        return None
    if sys.platform.startswith("linux"):
        try:
            stat_text = Path(f"/proc/{pid}/stat").read_text()
            status = Path(f"/proc/{pid}/status").read_text()
            start_time = stat_text[stat_text.rfind(")") + 2 :].split()[19]
            uid = int(next(line.split()[1] for line in status.splitlines() if line.startswith("Uid:")))
            return (uid, start_time)
        except (OSError, ValueError, IndexError, StopIteration):
            return None
    ps = _fixed_command(("/bin/ps", "/usr/bin/ps"))
    if not ps:
        return None
    try:
        result = subprocess.run(
            [ps, "-o", "pid=,uid=,lstart=,pgid=,comm=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=max(0.01, deadline - time.monotonic()),
            env={"PATH": str(Path(ps).parent), "LC_ALL": "C"},
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    fields = result.stdout.strip().split(None, 2)
    if len(fields) < 3 or fields[0] != str(pid):
        return None
    try:
        return (int(fields[1]), result.stdout.strip())
    except ValueError:
        return None


def _fixed_command(candidates: Sequence[str]) -> Optional[str]:
    return next((candidate for candidate in candidates if os.access(candidate, os.X_OK)), None)
