from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from autotune_cli import AsyncAutotune, Autotune
from autotune_cli.errors import AutotuneError
from autotune_cli.models import CommandResult
from autotune_cli.protocol import parse_sdk_envelope
from autotune_cli.transport import discover_binary


def test_analyze_parses_a_typed_search_space(fake_autotune: Path) -> None:
    result = Autotune(binary=fake_autotune).analyze("train.py", agent="codex")

    assert result.direction == "maximize"
    assert result.parameters[0].name == "learning_rate"
    assert result.parameters[0].choices is None


def test_run_requires_explicit_noninteractive_approval(fake_autotune: Path) -> None:
    client = Autotune(binary=fake_autotune)

    with pytest.raises(ValueError, match="yes=True or config"):
        client.run("train.py", trials=2)

    result = client.run("train.py", trials=2, config="space.yaml")
    assert result.study_name == "example"
    assert result.command.argv[-1] == "json"
    assert "--yes" in result.command.argv


def test_structured_errors_raise_or_are_returned_when_unchecked(fake_autotune: Path) -> None:
    client = Autotune(binary=fake_autotune)

    with pytest.raises(AutotuneError, match="requested failure"):
        client.results("fail")

    outcome = client.results("fail", check=False)
    assert outcome.message == "requested failure"


def test_raw_invoke_keeps_command_output(fake_autotune: Path) -> None:
    result = Autotune(binary=fake_autotune).invoke(["capabilities", "--sdk-format", "json"])

    assert json.loads(result.stdout)["command"] == "capabilities"


def test_core_commands_return_typed_values(fake_autotune: Path) -> None:
    client = Autotune(binary=fake_autotune)

    checks = client.doctor()
    resumed = client.resume(storage="sqlite:///study.db", trials=1)
    plot = client.plot_progress("ablation", output="progress.svg")

    assert checks[0].status == "ok"
    assert resumed.n_trials == 1
    assert plot.name == "progress.svg"


def test_async_client_matches_sync_interface(fake_autotune: Path) -> None:
    async def exercise() -> str:
        result = await AsyncAutotune(binary=fake_autotune).run("train.py", trials=2, yes=True)
        return result.study_name

    assert asyncio.run(exercise()) == "example"


def test_async_client_supports_all_core_commands(fake_autotune: Path) -> None:
    async def exercise() -> tuple[str, str, int, str, str]:
        client = AsyncAutotune(binary=fake_autotune)
        analysis = await client.analyze("train.py")
        checks = await client.doctor()
        results = await client.results()
        resumed = await client.resume(storage="sqlite:///study.db", trials=1)
        plot = await client.plot_progress("ablation", output="progress.svg")
        assert not hasattr(analysis, "message")
        assert not hasattr(checks, "message")
        assert not hasattr(results, "message")
        assert not hasattr(resumed, "message")
        return analysis.direction, checks[0].name, results.n_trials, resumed.study_name, plot.name

    assert asyncio.run(exercise()) == ("maximize", "python", 0, "example", "progress.svg")


def test_binary_discovery_honors_the_autotune_environment_variable(fake_autotune: Path) -> None:
    assert discover_binary(env={"AUTOTUNE_CLI_BIN": str(fake_autotune)}) == str(fake_autotune)


def test_invalid_protocol_output_is_rejected() -> None:
    command = CommandResult(0, "not json", "", ("autotune", "results"))

    with pytest.raises(AutotuneError, match="invalid SDK JSON"):
        parse_sdk_envelope(command)


def test_command_result_repr_redacts_sensitive_argument_values() -> None:
    result = CommandResult(1, "", "", ("autotune", "run", "--storage", "postgres://user:secret@db", "--agent-guidance", "api-key"))

    assert "secret" not in repr(result)
    assert "api-key" not in repr(result)


def test_protocol_rejects_exit_code_mismatches() -> None:
    command = CommandResult(
        1,
        json.dumps({"protocolVersion": 1, "type": "result", "command": "results", "exitCode": 1, "data": {}}),
        "",
        ("autotune", "results"),
    )

    with pytest.raises(AutotuneError, match="exit code"):
        parse_sdk_envelope(command)


@pytest.fixture
def fake_autotune(tmp_path: Path) -> Path:
    binary = tmp_path / "autotune"
    binary.write_text(
        """#!/usr/bin/env python3
import json
import sys

args = sys.argv[1:]
command = next((value for value in args if value in {"analyze", "doctor", "plot-progress", "results", "resume", "run", "capabilities"}), "cli")
if command == "capabilities":
    data = {"protocolVersion": 1, "commands": ["analyze", "doctor", "plot-progress", "results", "resume", "run"]}
elif command == "analyze":
    data = {"parameters": [{"name": "learning_rate", "cli_flag": "--lr", "type": "float", "low": 0.0001, "high": 0.1}], "has_arg_parsing": True, "needs_wrapper": False, "direction": "maximize"}
elif command == "run":
    data = {"study_name": "example", "direction": "maximize", "n_trials": 2, "best_trial": {"number": 0, "value": 1.0, "params": {}}, "all_trials": []}
elif command == "doctor":
    data = [{"name": "python", "status": "ok", "detail": "3.12"}]
elif command == "plot-progress":
    data = {"output": "progress.svg"}
elif command == "resume":
    data = {"study_name": "example", "direction": "maximize", "n_trials": 1, "best_trial": None, "all_trials": []}
elif args[1:2] == ["fail"]:
    print(json.dumps({"protocolVersion": 1, "type": "error", "command": command, "exitCode": 2, "error": {"message": "requested failure"}}))
    raise SystemExit(2)
else:
    data = {"study_name": "example", "direction": "maximize", "n_trials": 0, "best_trial": None, "all_trials": []}
print(json.dumps({"protocolVersion": 1, "type": "result", "command": command, "exitCode": 0, "data": data}))
""",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    return binary
