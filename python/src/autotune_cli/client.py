from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import Any, Literal

from .errors import AutotuneError, AutotuneProtocolError, AutotuneVersionError
from .models import (
    CommandResult,
    DoctorCheck,
    FixedParameter,
    SdkError,
    SdkResult,
    SearchParameter,
    SearchSpace,
    StudyResult,
    TrialResult,
)
from .protocol import SDK_PROTOCOL_VERSION, parse_sdk_envelope, protocol_data
from .transport import SubprocessTransport

Direction = Literal["maximize", "minimize"]


class Autotune:
    def __init__(
        self,
        binary: str | PathLike[str] | None = None,
        *,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._transport = SubprocessTransport(binary, env=env)
        self._protocol_compatible = False

    def invoke(
        self,
        args: Sequence[str],
        *,
        cwd: str | PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
        process_timeout_seconds: float | None = None,
        check: bool = True,
    ) -> CommandResult:
        return self._transport.invoke(
            args, cwd=cwd, env=env, timeout=process_timeout_seconds, check=check
        )

    def analyze(
        self,
        script: str | PathLike[str],
        *,
        agent: str = "claude",
        model: str | None = None,
        reasoning_effort: str | None = None,
        agent_guidance: str | None = None,
        agent_guidance_file: str | PathLike[str] | None = None,
        max_parameters: int | None = None,
        output: str | PathLike[str] | None = None,
        work_dir: str | PathLike[str] | None = None,
        command: str | None = None,
        **invoke_options: Any,
    ) -> SearchSpace | SdkError:
        args = ["analyze", os.fspath(script), "--agent", agent]
        _append_value(args, "--model", model)
        _append_value(args, "--reasoning-effort", reasoning_effort)
        _append_value(args, "--agent-guidance", agent_guidance)
        _append_value(args, "--agent-guidance-file", agent_guidance_file)
        _append_value(args, "--max-parameters", max_parameters)
        _append_value(args, "--output", output)
        _append_value(args, "--work-dir", work_dir)
        _append_value(args, "--command", command)
        envelope = self._structured(args, **invoke_options)
        return envelope if isinstance(envelope, SdkError) else _search_space(envelope)

    def run(
        self,
        script: str | PathLike[str],
        *,
        trials: int,
        yes: bool = False,
        config: str | PathLike[str] | None = None,
        direction: Direction | None = None,
        sampler: str | None = None,
        sampler_seed: int | None = None,
        pruner: str | None = None,
        n_jobs: int | None = None,
        agent: str = "claude",
        model: str | None = None,
        reasoning_effort: str | None = None,
        agent_guidance: str | None = None,
        agent_guidance_file: str | PathLike[str] | None = None,
        max_parameters: int | None = None,
        command: str | None = None,
        build_command: str | None = None,
        timeout_seconds: int | None = None,
        time_budget_seconds: int | None = None,
        refine_rounds: int | None = None,
        refine_trials: int | None = None,
        refine_mode: str | None = None,
        output: str | PathLike[str] | None = None,
        work_dir: str | PathLike[str] | None = None,
        storage: str | None = None,
        study_name: str | None = None,
        **invoke_options: Any,
    ) -> StudyResult | SdkError:
        if not yes and config is None:
            raise ValueError("Autotune.run requires yes=True or config=... for noninteractive approval")
        args = ["run", os.fspath(script), "--trials", str(trials), "--agent", agent, "--yes"]
        for flag, value in (
            ("--config", config), ("--direction", direction), ("--sampler", sampler),
            ("--sampler-seed", sampler_seed), ("--pruner", pruner), ("--n-jobs", n_jobs),
            ("--model", model), ("--reasoning-effort", reasoning_effort),
            ("--agent-guidance", agent_guidance), ("--agent-guidance-file", agent_guidance_file),
            ("--max-parameters", max_parameters), ("--command", command),
            ("--build-command", build_command), ("--timeout-seconds", timeout_seconds),
            ("--time-budget-seconds", time_budget_seconds), ("--refine-rounds", refine_rounds),
            ("--refine-trials", refine_trials), ("--refine-mode", refine_mode),
            ("--output", output), ("--work-dir", work_dir), ("--storage", storage),
            ("--study-name", study_name),
        ):
            _append_value(args, flag, value)
        envelope = self._structured(args, **invoke_options)
        return envelope if isinstance(envelope, SdkError) else _study_result(envelope)

    def doctor(self, script: str | PathLike[str] | None = None, *, agent: str = "claude", model: str | None = None, command: str | None = None, **invoke_options: Any) -> tuple[DoctorCheck, ...] | SdkError:
        args = ["doctor"]
        if script is not None:
            args.append(os.fspath(script))
        _append_value(args, "--agent", agent)
        _append_value(args, "--model", model)
        _append_value(args, "--command", command)
        envelope = self._structured(args, **invoke_options)
        if isinstance(envelope, SdkError):
            return envelope
        data = envelope.data
        if not isinstance(data, list):
            raise AutotuneProtocolError("doctor SDK data must be an array", envelope.command_result)
        return tuple(_doctor_check(value, envelope.command_result) for value in data)

    def results(self, location: str | PathLike[str] = "autotune", *, top: int | None = None, **invoke_options: Any) -> StudyResult | SdkError:
        args = ["results", os.fspath(location)]
        _append_value(args, "--top", top)
        envelope = self._structured(args, **invoke_options)
        if isinstance(envelope, SdkError):
            return envelope
        return _study_result(envelope)

    def resume(self, *, storage: str, trials: int, work_dir: str | PathLike[str] = "autotune", study_name: str | None = None, n_jobs: int | None = None, direction: Direction | None = None, **invoke_options: Any) -> StudyResult | SdkError:
        args = ["resume", "--storage", storage, "--trials", str(trials), "--work-dir", os.fspath(work_dir)]
        _append_value(args, "--study-name", study_name)
        _append_value(args, "--n-jobs", n_jobs)
        _append_value(args, "--direction", direction)
        envelope = self._structured(args, **invoke_options)
        return envelope if isinstance(envelope, SdkError) else _study_result(envelope)

    def plot_progress(self, run_dir: str | PathLike[str], *, output: str | PathLike[str], **invoke_options: Any) -> Path | SdkError:
        envelope = self._structured(["plot-progress", os.fspath(run_dir), "--output", os.fspath(output)], **invoke_options)
        if isinstance(envelope, SdkError):
            return envelope
        data = protocol_data(envelope)
        value = data.get("output")
        if not isinstance(value, str):
            raise AutotuneProtocolError("plot-progress SDK data must contain output", envelope.command_result)
        return Path(value)

    def _structured(self, args: list[str], **invoke_options: Any) -> SdkResult | SdkError:
        check = bool(invoke_options.pop("check", True))
        self._ensure_protocol_compatible(**invoke_options)
        result = self.invoke([*args, "--sdk-format", "json"], check=False, **invoke_options)
        envelope = parse_sdk_envelope(result)
        if isinstance(envelope, SdkError) and check:
            raise AutotuneError(envelope.message, result)
        return envelope

    def _ensure_protocol_compatible(self, **invoke_options: Any) -> None:
        if self._protocol_compatible:
            return
        result = self.invoke(["capabilities", "--sdk-format", "json"], check=False, **invoke_options)
        envelope = parse_sdk_envelope(result)
        data = protocol_data(envelope)
        if data.get("protocolVersion") != SDK_PROTOCOL_VERSION:
            raise AutotuneVersionError(
                f"incompatible autotune SDK protocol; expected {SDK_PROTOCOL_VERSION}", result
            )
        self._protocol_compatible = True


def _append_value(args: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        args.extend((flag, os.fspath(value) if isinstance(value, PathLike) else str(value)))


def _search_space(envelope: SdkResult | SdkError) -> SearchSpace:
    data = protocol_data(envelope)
    parameters = data.get("parameters")
    if not isinstance(parameters, list):
        raise AutotuneProtocolError("search-space SDK data must contain parameters", envelope.command_result)
    return SearchSpace(
        parameters=tuple(_search_parameter(value, envelope.command_result) for value in parameters),
        fixed_parameters=tuple(_fixed_parameter(value, envelope.command_result) for value in data.get("fixed_parameters", [])),
        has_arg_parsing=_bool(data, "has_arg_parsing", envelope.command_result),
        needs_wrapper=_bool(data, "needs_wrapper", envelope.command_result),
        direction=_direction(data, envelope.command_result),
        has_metric_output=data.get("has_metric_output") if isinstance(data.get("has_metric_output"), bool) else None,
        failure_value=_number_or_none(data.get("failure_value"), envelope.command_result),
        optuna=data.get("optuna") if isinstance(data.get("optuna"), dict) else None,
        reasoning=data.get("reasoning") if isinstance(data.get("reasoning"), str) else None,
    )


def _search_parameter(value: Any, result: CommandResult) -> SearchParameter:
    if not isinstance(value, dict):
        raise AutotuneProtocolError("search-space parameter must be an object", result)
    kind = value.get("type")
    if kind not in {"float", "int", "categorical"}:
        raise AutotuneProtocolError("search-space parameter type is invalid", result)
    choices = value.get("choices")
    return SearchParameter(
        name=_string(value, "name", result), cli_flag=_string(value, "cli_flag", result), type=kind,
        low=_number_or_none(value.get("low"), result), high=_number_or_none(value.get("high"), result),
        log=value.get("log") if isinstance(value.get("log"), bool) else None,
        choices=tuple(choices) if isinstance(choices, list) else None, current_value=value.get("current_value"),
    )


def _fixed_parameter(value: Any, result: CommandResult) -> FixedParameter:
    if not isinstance(value, dict):
        raise AutotuneProtocolError("fixed parameter must be an object", result)
    raw = value.get("value")
    if not isinstance(raw, (str, int, float, bool)):
        raise AutotuneProtocolError("fixed parameter value is invalid", result)
    return FixedParameter(_string(value, "name", result), _string(value, "cli_flag", result), raw)


def _study_result(envelope: SdkResult | SdkError) -> StudyResult:
    data = protocol_data(envelope)
    return StudyResult(
        study_name=_string(data, "study_name", envelope.command_result), direction=_direction(data, envelope.command_result),
        n_trials=_integer(data, "n_trials", envelope.command_result),
        best_trial=_trial_result(data.get("best_trial"), envelope.command_result),
        all_trials=tuple(
            trial
            for value in data.get("all_trials", [])
            if (trial := _trial_result(value, envelope.command_result)) is not None
        ),
        command=envelope.command_result,
    )


def _trial_result(value: Any, result: CommandResult) -> TrialResult | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise AutotuneProtocolError("trial result must be an object", result)
    raw_value = value.get("value")
    if raw_value is not None and (not isinstance(raw_value, (int, float)) or isinstance(raw_value, bool)):
        raise AutotuneProtocolError("trial value is invalid", result)
    params = value.get("params")
    if not isinstance(params, dict):
        raise AutotuneProtocolError("trial params must be an object", result)
    return TrialResult(_integer(value, "number", result), None if raw_value is None else float(raw_value), params, value.get("state") if isinstance(value.get("state"), str) else None, value.get("user_attrs") if isinstance(value.get("user_attrs"), dict) else None)


def _doctor_check(value: Any, result: CommandResult) -> DoctorCheck:
    if not isinstance(value, dict) or value.get("status") not in {"ok", "fail", "skip"}:
        raise AutotuneProtocolError("doctor check is invalid", result)
    return DoctorCheck(_string(value, "name", result), value["status"], _string(value, "detail", result))


def _string(data: dict[str, Any], key: str, result: CommandResult) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise AutotuneProtocolError(f"SDK data field {key} must be a string", result)
    return value


def _bool(data: dict[str, Any], key: str, result: CommandResult) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise AutotuneProtocolError(f"SDK data field {key} must be a boolean", result)
    return value


def _integer(data: dict[str, Any], key: str, result: CommandResult) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise AutotuneProtocolError(f"SDK data field {key} must be an integer", result)
    return value


def _number_or_none(value: Any, result: CommandResult) -> float | int | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise AutotuneProtocolError("SDK data field must be numeric", result)
    return value


def _direction(data: dict[str, Any], result: CommandResult) -> Direction:
    value = data.get("direction")
    if value == "maximize":
        return "maximize"
    if value == "minimize":
        return "minimize"
    raise AutotuneProtocolError("SDK data direction is invalid", result)
