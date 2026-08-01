from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Direction = Literal["maximize", "minimize"]
ParameterType = Literal["float", "int", "categorical"]


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
    argv: tuple[str, ...] = field(repr=False)

    def __repr__(self) -> str:
        return (
            f"CommandResult(returncode={self.returncode!r}, stdout={self.stdout!r}, "
            f"stderr={self.stderr!r}, argv={_redacted_argv(self.argv)!r})"
        )


def _redacted_argv(argv: tuple[str, ...]) -> tuple[str, ...]:
    sensitive_flags = {"--agent-guidance", "--storage"}
    values = list(argv)
    for index, value in enumerate(values[:-1]):
        if value in sensitive_flags:
            values[index + 1] = "[REDACTED]"
    return tuple(values)


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
    status: Literal["pass", "fail"]
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
