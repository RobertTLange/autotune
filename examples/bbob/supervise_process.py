#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import errno
import os
import select
import signal
import subprocess
import sys
import time


def parse_args() -> tuple[float, list[str]]:
    parser = argparse.ArgumentParser(description="Run a command in a supervised process group.")
    parser.add_argument("--grace-seconds", type=float, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not 0.1 <= args.grace_seconds <= 60:
        parser.error("--grace-seconds must be between 0.1 and 60")
    if not command:
        parser.error("a command is required after --")
    return args.grace_seconds, command


ProcessRow = tuple[int, int, int, str]


def enable_child_subreaper() -> None:
    if not sys.platform.startswith("linux"):
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))


def process_table() -> list[ProcessRow] | None:
    try:
        probe = subprocess.Popen(
            ["/bin/ps", "-axo", "pid=,ppid=,pgid=,stat="],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, _ = probe.communicate(timeout=1)
    except (OSError, subprocess.TimeoutExpired):
        if "probe" in locals():
            probe.kill()
            probe.wait()
        return None
    if probe.returncode != 0:
        return None

    rows: list[ProcessRow] = []
    for line in stdout.splitlines():
        fields = line.split()
        if len(fields) < 4:
            continue
        try:
            pid = int(fields[0])
            parent_pid = int(fields[1])
            process_group = int(fields[2])
        except ValueError:
            continue
        if pid != probe.pid:
            rows.append((pid, parent_pid, process_group, fields[3]))
    return rows


def active_group_members(group_id: int) -> list[int] | None:
    rows = process_table()
    if rows is None:
        return None
    return [
        pid
        for pid, _, process_group, state in rows
        if process_group == group_id and not state.startswith("Z")
    ]


def active_children(parent_pid: int, excluded_pid: int) -> list[int] | None:
    rows = process_table()
    if rows is None:
        return None
    return [
        pid
        for pid, process_parent, _, state in rows
        if process_parent == parent_pid and pid != excluded_pid and not state.startswith("Z")
    ]


def signal_group(group_id: int, signal_number: int) -> None:
    try:
        os.killpg(group_id, signal_number)
    except OSError as exc:
        if exc.errno != errno.ESRCH:
            raise


def clean_process_group(group_id: int, grace_seconds: float, deadline: float | None = None) -> None:
    members = active_group_members(group_id)
    if members == []:
        return

    signal_group(group_id, signal.SIGTERM)
    if deadline is None:
        deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        members = active_group_members(group_id)
        if members == []:
            return
        time.sleep(0.05)
    signal_group(group_id, signal.SIGKILL)


def clean_adopted_descendants(leader_pid: int, grace_seconds: float, deadline: float | None = None) -> None:
    if not sys.platform.startswith("linux"):
        return
    children = active_children(os.getpid(), leader_pid)
    if children == []:
        return
    minimum_deadline = time.monotonic() + min(grace_seconds, 1)
    deadline = max(deadline or minimum_deadline, minimum_deadline)

    while time.monotonic() < deadline:
        if children is not None:
            for pid in children:
                try:
                    os.kill(pid, signal.SIGTERM)
                except OSError as exc:
                    if exc.errno != errno.ESRCH:
                        raise
        time.sleep(0.05)
        children = active_children(os.getpid(), leader_pid)
        if children == []:
            return

    kill_deadline = time.monotonic() + 1
    while time.monotonic() < kill_deadline:
        children = active_children(os.getpid(), leader_pid)
        if children == []:
            return
        if children is not None:
            for pid in children:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError as exc:
                    if exc.errno != errno.ESRCH:
                        raise
        time.sleep(0.05)


def reap_adopted_children() -> None:
    if not sys.platform.startswith("linux"):
        return
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return


def exit_code(wait_status: int, forwarded_signal: int | None) -> int:
    if forwarded_signal is not None:
        return 128 + forwarded_signal
    if os.WIFEXITED(wait_status):
        return os.WEXITSTATUS(wait_status)
    if os.WIFSIGNALED(wait_status):
        return 128 + os.WTERMSIG(wait_status)
    return 1


def create_exit_queue(child_pid: int) -> tuple[object | None, bool]:
    if hasattr(os, "waitid"):
        return None, False
    if not hasattr(select, "kqueue"):
        raise RuntimeError("this platform provides neither waitid nor kqueue")

    queue = select.kqueue()
    event = select.kevent(
        child_pid,
        filter=select.KQ_FILTER_PROC,
        flags=select.KQ_EV_ADD | select.KQ_EV_ENABLE,
        fflags=select.KQ_NOTE_EXIT,
    )
    while True:
        try:
            queue.control([event], 0, 0)
            break
        except InterruptedError:
            continue
        except ProcessLookupError:
            queue.close()
            return None, True
        except BaseException:
            queue.close()
            raise
    return queue, False


def child_has_exited(child_pid: int, exit_queue: object | None, already_exited: bool) -> bool:
    if already_exited:
        return True
    if exit_queue is not None:
        return bool(exit_queue.control(None, 1, 0))
    return os.waitid(os.P_PID, child_pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None


def main() -> int:
    grace_seconds, command = parse_args()
    enable_child_subreaper()
    child_pid: int | None = None
    forwarded_signal: int | None = None
    termination_deadline: float | None = None

    def forward(signal_number: int, _frame: object) -> None:
        nonlocal forwarded_signal, termination_deadline
        if forwarded_signal is None:
            forwarded_signal = signal_number
            termination_deadline = time.monotonic() + grace_seconds
        if child_pid is not None:
            signal_group(child_pid, signal_number)

    for signal_number in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
        signal.signal(signal_number, forward)

    child = subprocess.Popen(command, start_new_session=True)
    child_pid = child.pid
    try:
        exit_queue, already_exited = create_exit_queue(child_pid)
    except BaseException:
        clean_process_group(child_pid, grace_seconds)
        clean_adopted_descendants(child_pid, grace_seconds)
        os.waitpid(child_pid, 0)
        reap_adopted_children()
        raise
    if forwarded_signal is not None:
        signal_group(child_pid, forwarded_signal)

    try:
        while True:
            try:
                if child_has_exited(child_pid, exit_queue, already_exited):
                    break
            except InterruptedError:
                continue
            if termination_deadline is not None and time.monotonic() >= termination_deadline:
                signal_group(child_pid, signal.SIGKILL)
            time.sleep(0.05)
    finally:
        if exit_queue is not None:
            exit_queue.close()

    clean_process_group(child_pid, grace_seconds, termination_deadline)
    clean_adopted_descendants(child_pid, grace_seconds, termination_deadline)
    _, wait_status = os.waitpid(child_pid, 0)
    reap_adopted_children()
    return exit_code(wait_status, forwarded_signal)


if __name__ == "__main__":
    raise SystemExit(main())
