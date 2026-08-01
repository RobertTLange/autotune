from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

import autotune_cli.processes as processes_module
from autotune_cli.processes import OutputPipes, ProcessIdentity

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX process helpers")


class _Entries:
    def __init__(self, *entries: object) -> None:
        self._entries = entries

    def __enter__(self) -> _Entries:  # noqa: PYI034
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def __iter__(self) -> Iterator[object]:
        return iter(self._entries)


def test_capture_output_pipes_records_only_valid_linux_descriptors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(processes_module.sys, "platform", "linux")

    def fstat(descriptor: int) -> object:
        if descriptor == 4:
            raise OSError("closed")
        return SimpleNamespace(st_dev=descriptor, st_ino=descriptor + 10)

    monkeypatch.setattr(processes_module.os, "fstat", fstat)

    assert processes_module.capture_output_pipes([3, 4]) == OutputPipes(
        proc_identities=frozenset({(3, 13)})
    )


def test_capture_output_pipes_uses_darwin_handle_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipe_handles = MagicMock(return_value=frozenset({21, 22}))
    darwin_process = ModuleType("autotune_cli.darwin_process")
    darwin_process.__dict__["pipe_handles"] = pipe_handles
    monkeypatch.setattr(processes_module.sys, "platform", "darwin")
    monkeypatch.setitem(sys.modules, "autotune_cli.darwin_process", darwin_process)
    monkeypatch.setattr(processes_module.os, "getpid", lambda: 7)

    assert processes_module.capture_output_pipes([3, 4]) == OutputPipes(
        darwin_handles=frozenset({21, 22})
    )
    pipe_handles.assert_called_once_with(7, (3, 4))


def test_capture_output_pipes_parses_lsof_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = subprocess.CompletedProcess([], 0, "p1\nn->pipe-a\nnignored\n")
    run = MagicMock(return_value=result)
    monkeypatch.setattr(processes_module.sys, "platform", "freebsd")
    monkeypatch.setattr(processes_module, "_trusted_command", lambda *_paths: "/bin/lsof")
    monkeypatch.setattr(processes_module.subprocess, "run", run)

    assert processes_module.capture_output_pipes([3]) == OutputPipes(
        lsof_devices=frozenset({"pipe-a"})
    )
    assert run.call_args.args[0][-2:] == ["-d", "3"]


def test_terminate_output_holders_revalidates_identity_and_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipes = OutputPipes(proc_identities=frozenset({(1, 2)}))
    killed = MagicMock()
    identity = MagicMock(
        side_effect=lambda pid, _deadline: {
            30: (999, "other"),
            31: (501, "stale"),
            32: (501, "valid"),
        }.get(pid)
    )
    monkeypatch.setattr(processes_module, "_output_holder_pids", lambda *_args: [10, 20, 30, 31, 32])
    monkeypatch.setattr(processes_module, "_process_identity", identity)
    monkeypatch.setattr(processes_module, "_process_holds_output", lambda pid, *_args: pid == 32)
    monkeypatch.setattr(processes_module, "_kill_matching_process", killed)
    monkeypatch.setattr(processes_module.os, "getpid", lambda: 10)
    monkeypatch.setattr(processes_module.os, "getuid", lambda: 501)

    processes_module.terminate_output_holders(pipes, excluded_pid=20)

    killed.assert_called_once()
    assert killed.call_args.args[:2] == (32, (501, "valid"))
    assert [call.args[0] for call in identity.call_args_list] == [30, 31, 32]


def test_capture_process_group_members_keeps_only_revalidated_members(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identities = {40: (501, "leader"), 42: (501, "member"), 43: (999, "foreign")}
    monkeypatch.setattr(processes_module.os, "getpid", lambda: 41)
    monkeypatch.setattr(processes_module.os, "getuid", lambda: 501)
    monkeypatch.setattr(processes_module.os, "getpgid", lambda pid: 40 if pid in {40, 42} else 99)
    monkeypatch.setattr(processes_module, "_process_group_pids", lambda *_args: [41, 42, 43, 44])
    monkeypatch.setattr(processes_module, "_process_identity", lambda pid, _deadline: identities.get(pid))

    assert processes_module.capture_process_group_members(40, 44) == (
        ProcessIdentity(42, (501, "member")),
    )


def test_capture_process_group_members_discards_snapshot_if_leader_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identities = iter([(501, "before"), (501, "member"), (501, "after")])
    monkeypatch.setattr(processes_module.os, "getuid", lambda: 501)
    monkeypatch.setattr(processes_module.os, "getpid", lambda: 99)
    monkeypatch.setattr(processes_module.os, "getpgid", lambda _pid: 40)
    monkeypatch.setattr(processes_module, "_process_group_pids", lambda *_args: [42])
    monkeypatch.setattr(processes_module, "_process_identity", lambda *_args: next(identities))

    assert processes_module.capture_process_group_members(40, 99) == ()


def test_signal_processes_uses_requested_signals(monkeypatch: pytest.MonkeyPatch) -> None:
    send = MagicMock()
    members = [ProcessIdentity(42, (501, "a")), ProcessIdentity(43, (501, "b"))]
    monkeypatch.setattr(processes_module, "_signal_matching_process", send)

    processes_module.signal_processes(members, signal.SIGTERM)
    processes_module.terminate_process_group_members(members[:1])

    assert [call.args[:3] for call in send.call_args_list] == [
        (42, (501, "a"), signal.SIGTERM),
        (43, (501, "b"), signal.SIGTERM),
        (42, (501, "a"), signal.SIGKILL),
    ]


def test_linux_process_group_discovery_filters_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = _Entries(SimpleNamespace(name="42"), SimpleNamespace(name="text"), SimpleNamespace(name="43"))
    monkeypatch.setattr(processes_module.sys, "platform", "linux")
    monkeypatch.setattr(processes_module.os, "scandir", lambda _path: entries)
    monkeypatch.setattr(processes_module, "_process_group_matches", lambda pid, _pgid: pid == 43)

    assert processes_module._process_group_pids(40, time.monotonic() + 1) == [43]


def test_lsof_output_holder_discovery_ignores_malformed_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = "pbad\ndpipe-a\np42\ndpipe-a\np43\ndother\n"
    monkeypatch.setattr(processes_module, "_trusted_command", lambda *_paths: "/bin/lsof")
    monkeypatch.setattr(
        processes_module.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, output),
    )

    assert processes_module._output_holder_pids(
        OutputPipes(lsof_devices=frozenset({"pipe-a"})), time.monotonic() + 1
    ) == [42]


def test_proc_output_holder_discovery_obeys_candidate_and_descriptor_budgets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = _Entries(SimpleNamespace(name="42"), SimpleNamespace(name="43"))

    def exhaust_budget(
        _pid: int,
        _pipes: OutputPipes,
        _deadline: float,
        descriptor_budget: list[int],
    ) -> bool:
        descriptor_budget[0] = 0
        return False

    holds = MagicMock(side_effect=exhaust_budget)
    monkeypatch.setattr(processes_module.os, "scandir", lambda _path: entries)
    monkeypatch.setattr(processes_module, "_recent_process_pids", lambda *_args: [42, 43])
    monkeypatch.setattr(processes_module, "_process_holds_output", holds)

    assert processes_module._proc_output_holder_pids(
        OutputPipes(proc_identities=frozenset({(1, 2)})), time.monotonic() + 1
    ) == []
    holds.assert_called_once()


def test_process_holds_output_matches_proc_descriptor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptors = _Entries(
        SimpleNamespace(path="/bad"),
        SimpleNamespace(path="/match"),
    )
    budget = [2]

    def stat(path: str) -> object:
        if path == "/bad":
            raise OSError("gone")
        return SimpleNamespace(st_dev=1, st_ino=2)

    monkeypatch.setattr(processes_module.os, "scandir", lambda _path: descriptors)
    monkeypatch.setattr(processes_module.os, "stat", stat)

    assert processes_module._process_holds_output(
        42,
        OutputPipes(proc_identities=frozenset({(1, 2)})),
        time.monotonic() + 1,
        budget,
    )
    assert budget == [0]


def test_recent_process_pids_prioritizes_recent_wrapped_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = iter(
        [
            SimpleNamespace(name="text"),
            SimpleNamespace(name="98"),
            SimpleNamespace(name="2"),
            SimpleNamespace(name="50"),
        ]
    )

    def read_text(path: Path) -> str:
        return "3" if path.name == "ns_last_pid" else "100"

    monkeypatch.setattr(processes_module.Path, "read_text", read_text)

    assert processes_module._recent_process_pids(
        entries, time.monotonic() + 1, limit=2
    ) == [2, 98]


def test_linux_process_identity_parses_uid_and_start_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokens = ["S", *(str(index) for index in range(1, 19)), "start"]

    def read_text(path: Path) -> str:
        if path.name == "stat":
            return f"42 (worker name) {' '.join(tokens)}"
        return "Name:\tworker\nUid:\t501\t501\t501\t501\n"

    monkeypatch.setattr(processes_module.sys, "platform", "linux")
    monkeypatch.setattr(processes_module.Path, "read_text", read_text)

    assert processes_module._process_identity(42, time.monotonic() + 1) == (
        501,
        "start",
    )


def test_signal_matching_process_prefers_revalidated_pidfd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    send = MagicMock()
    close = MagicMock()
    kill = MagicMock()
    monkeypatch.setattr(processes_module.sys, "platform", "linux")
    monkeypatch.setattr(processes_module.os, "pidfd_open", lambda _pid: 9, raising=False)
    monkeypatch.setattr(processes_module.signal, "pidfd_send_signal", send, raising=False)
    monkeypatch.setattr(processes_module.os, "close", close)
    monkeypatch.setattr(processes_module.os, "kill", kill)
    monkeypatch.setattr(processes_module, "_process_identity", lambda *_args: (501, "start"))

    processes_module._signal_matching_process(
        42, (501, "start"), signal.SIGTERM, time.monotonic() + 1
    )

    send.assert_called_once_with(9, signal.SIGTERM)
    close.assert_called_once_with(9)
    kill.assert_not_called()


def test_signal_matching_process_falls_back_after_pidfd_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kill = MagicMock(side_effect=OSError("gone"))
    monkeypatch.setattr(processes_module.sys, "platform", "linux")
    monkeypatch.setattr(
        processes_module.os,
        "pidfd_open",
        MagicMock(side_effect=OSError("unsupported")),
        raising=False,
    )
    monkeypatch.setattr(processes_module.os, "kill", kill)
    monkeypatch.setattr(processes_module, "_process_identity", lambda *_args: (501, "start"))

    processes_module._signal_matching_process(
        42, (501, "start"), signal.SIGKILL, time.monotonic() + 1
    )

    kill.assert_called_once_with(42, signal.SIGKILL)


def test_signal_matching_process_rejects_reused_pidfd_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    send = MagicMock()
    close = MagicMock()
    kill = MagicMock()
    monkeypatch.setattr(processes_module.sys, "platform", "linux")
    monkeypatch.setattr(processes_module.os, "pidfd_open", lambda _pid: 9, raising=False)
    monkeypatch.setattr(processes_module.signal, "pidfd_send_signal", send, raising=False)
    monkeypatch.setattr(processes_module.os, "close", close)
    monkeypatch.setattr(processes_module.os, "kill", kill)
    monkeypatch.setattr(processes_module, "_process_identity", lambda *_args: (501, "reused"))

    processes_module._signal_matching_process(
        42, (501, "expected"), signal.SIGTERM, time.monotonic() + 1
    )

    send.assert_not_called()
    kill.assert_not_called()
    close.assert_called_once_with(9)


def test_signal_matching_process_rejects_reused_fallback_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kill = MagicMock()
    monkeypatch.setattr(processes_module.sys, "platform", "linux")
    monkeypatch.setattr(
        processes_module.os,
        "pidfd_open",
        MagicMock(side_effect=OSError("unsupported")),
        raising=False,
    )
    monkeypatch.setattr(processes_module.os, "kill", kill)
    monkeypatch.setattr(processes_module, "_process_identity", lambda *_args: (501, "reused"))

    processes_module._signal_matching_process(
        42, (501, "expected"), signal.SIGKILL, time.monotonic() + 1
    )

    kill.assert_not_called()
