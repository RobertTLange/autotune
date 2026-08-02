from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

Direction = Literal["maximize", "minimize"]
ParameterType = Literal["float", "int", "categorical"]

_SENSITIVE_FLAGS = {
    "--agent-guidance",
    "--build-command",
    "--command",
    "--storage",
}
_SENSITIVE_FLAG_SUFFIXES = (
    "-access-key",
    "-api-key",
    "-auth-token",
    "-authorization",
    "-client-secret",
    "-credential",
    "-credentials",
    "-passphrase",
    "-password",
    "-private-key",
    "-secret",
    "-secret-key",
    "-token",
)


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
    argv: tuple[str, ...] = field(repr=False)

    def __repr__(self) -> str:
        return (
            f"CommandResult(returncode={self.returncode!r}, stdout={self.stdout!r}, "
            f"stderr={self.stderr!r}, argv={redacted_argv(self.argv)!r})"
        )


def redacted_argv(argv: Sequence[str]) -> tuple[str, ...]:
    values = list(argv)
    for index, value in enumerate(values):
        flag, separator, _ = value.partition("=")
        if not _is_sensitive_flag(flag):
            continue
        if separator:
            values[index] = f"{flag}=[REDACTED]"
        elif index + 1 < len(values):
            values[index + 1] = "[REDACTED]"
    return tuple(values)


def _is_sensitive_flag(flag: str) -> bool:
    normalized = flag.lower().replace("_", "-")
    if not normalized.startswith("--"):
        return False
    return normalized in _SENSITIVE_FLAGS or normalized.endswith(
        _SENSITIVE_FLAG_SUFFIXES
    )


@dataclass(frozen=True)
class SearchParameter:
    name: str
    cli_flag: str
    type: ParameterType
    low: float | int | None = None
    high: float | int | None = None
    log: bool | None = None
    choices: tuple[str | int | float | bool, ...] | None = None
    current_value: Any = None


@dataclass(frozen=True)
class FixedParameter:
    name: str
    cli_flag: str
    value: str | int | float | bool


@dataclass(frozen=True)
class SearchSpace:
    parameters: tuple[SearchParameter, ...]
    has_arg_parsing: bool
    needs_wrapper: bool
    direction: Direction
    fixed_parameters: tuple[FixedParameter, ...] = ()
    has_metric_output: bool | None = None
    failure_value: float | None = None
    optuna: dict[str, Any] | None = None
    reasoning: str | None = None


@dataclass(frozen=True)
class TrialResult:
    number: int
    value: float | None
    params: dict[str, Any]
    state: str | None = None
    user_attrs: dict[str, Any] | None = None


@dataclass(frozen=True)
class StudyResult:
    study_name: str
    direction: Direction
    n_trials: int
    best_trial: TrialResult | None
    all_trials: tuple[TrialResult, ...]
    command: CommandResult


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    status: Literal["ok", "fail", "skip"]
    detail: str


@dataclass(frozen=True)
class SdkResult:
    protocol_version: int
    command: str
    exit_code: int
    data: Any
    command_result: CommandResult


@dataclass(frozen=True)
class SdkError:
    protocol_version: int
    command: str
    exit_code: int
    message: str
    command_result: CommandResult
