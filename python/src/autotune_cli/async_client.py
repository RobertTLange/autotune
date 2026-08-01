from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import Any, Literal

from .async_transport import AsyncSubprocessTransport
from .client import _append_value, _doctor_check, _search_space, _study_result
from .errors import AutotuneError, AutotuneProtocolError, AutotuneVersionError
from .models import (
    CommandResult,
    DoctorCheck,
    SdkError,
    SdkResult,
    SearchSpace,
    StudyResult,
)
from .protocol import SDK_PROTOCOL_VERSION, parse_sdk_envelope, protocol_data

Direction = Literal["maximize", "minimize"]


class AsyncAutotune:
    def __init__(self, binary: str | PathLike[str] | None = None, *, env: Mapping[str, str] | None = None) -> None:
        self._transport = AsyncSubprocessTransport(binary, env=env)
        self._protocol_compatible = False

    async def invoke(self, args: Sequence[str], *, cwd: str | PathLike[str] | None = None, env: Mapping[str, str] | None = None, process_timeout_seconds: float | None = None, check: bool = True) -> CommandResult:
        return await self._transport.invoke(args, cwd=cwd, env=env, timeout=process_timeout_seconds, check=check)

    async def analyze(self, script: str | PathLike[str], *, agent: str = "claude", model: str | None = None, reasoning_effort: str | None = None, agent_guidance: str | None = None, agent_guidance_file: str | PathLike[str] | None = None, max_parameters: int | None = None, output: str | PathLike[str] | None = None, work_dir: str | PathLike[str] | None = None, command: str | None = None, **invoke_options: Any) -> SearchSpace | SdkError:
        args = ["analyze", os.fspath(script), "--agent", agent]
        for flag, value in (("--model", model), ("--reasoning-effort", reasoning_effort), ("--agent-guidance", agent_guidance), ("--agent-guidance-file", agent_guidance_file), ("--max-parameters", max_parameters), ("--output", output), ("--work-dir", work_dir), ("--command", command)):
            _append_value(args, flag, value)
        envelope = await self._structured(args, **invoke_options)
        return envelope if isinstance(envelope, SdkError) else _search_space(envelope)

    async def run(self, script: str | PathLike[str], *, trials: int, yes: bool = False, config: str | PathLike[str] | None = None, direction: Direction | None = None, sampler: str | None = None, sampler_seed: int | None = None, pruner: str | None = None, n_jobs: int | None = None, agent: str = "claude", model: str | None = None, reasoning_effort: str | None = None, agent_guidance: str | None = None, agent_guidance_file: str | PathLike[str] | None = None, max_parameters: int | None = None, command: str | None = None, build_command: str | None = None, timeout_seconds: int | None = None, time_budget_seconds: int | None = None, refine_rounds: int | None = None, refine_trials: int | None = None, refine_mode: str | None = None, output: str | PathLike[str] | None = None, work_dir: str | PathLike[str] | None = None, storage: str | None = None, study_name: str | None = None, **invoke_options: Any) -> StudyResult | SdkError:
        if not yes and config is None:
            raise ValueError("AsyncAutotune.run requires yes=True or config=... for noninteractive approval")
        args = ["run", os.fspath(script), "--trials", str(trials), "--agent", agent, "--yes"]
        for flag, value in (("--config", config), ("--direction", direction), ("--sampler", sampler), ("--sampler-seed", sampler_seed), ("--pruner", pruner), ("--n-jobs", n_jobs), ("--model", model), ("--reasoning-effort", reasoning_effort), ("--agent-guidance", agent_guidance), ("--agent-guidance-file", agent_guidance_file), ("--max-parameters", max_parameters), ("--command", command), ("--build-command", build_command), ("--timeout-seconds", timeout_seconds), ("--time-budget-seconds", time_budget_seconds), ("--refine-rounds", refine_rounds), ("--refine-trials", refine_trials), ("--refine-mode", refine_mode), ("--output", output), ("--work-dir", work_dir), ("--storage", storage), ("--study-name", study_name)):
            _append_value(args, flag, value)
        envelope = await self._structured(args, **invoke_options)
        return envelope if isinstance(envelope, SdkError) else _study_result(envelope)

    async def doctor(self, script: str | PathLike[str] | None = None, *, agent: str = "claude", model: str | None = None, command: str | None = None, **invoke_options: Any) -> tuple[DoctorCheck, ...] | SdkError:
        args = ["doctor"]
        if script is not None:
            args.append(os.fspath(script))
        for flag, value in (("--agent", agent), ("--model", model), ("--command", command)):
            _append_value(args, flag, value)
        envelope = await self._structured(args, **invoke_options)
        if isinstance(envelope, SdkError):
            return envelope
        if not isinstance(envelope.data, list):
            raise AutotuneProtocolError("doctor SDK data must be an array", envelope.command_result)
        return tuple(_doctor_check(value, envelope.command_result) for value in envelope.data)

    async def results(self, location: str | PathLike[str] = "autotune", *, top: int | None = None, **invoke_options: Any) -> StudyResult | SdkError:
        args = ["results", os.fspath(location)]
        _append_value(args, "--top", top)
        envelope = await self._structured(args, **invoke_options)
        return envelope if isinstance(envelope, SdkError) else _study_result(envelope)

    async def resume(self, *, storage: str, trials: int, work_dir: str | PathLike[str] = "autotune", study_name: str | None = None, n_jobs: int | None = None, direction: Direction | None = None, **invoke_options: Any) -> StudyResult | SdkError:
        args = ["resume", "--storage", storage, "--trials", str(trials), "--work-dir", os.fspath(work_dir)]
        for flag, value in (("--study-name", study_name), ("--n-jobs", n_jobs), ("--direction", direction)):
            _append_value(args, flag, value)
        envelope = await self._structured(args, **invoke_options)
        return envelope if isinstance(envelope, SdkError) else _study_result(envelope)

    async def plot_progress(self, run_dir: str | PathLike[str], *, output: str | PathLike[str], **invoke_options: Any) -> Path | SdkError:
        envelope = await self._structured(["plot-progress", os.fspath(run_dir), "--output", os.fspath(output)], **invoke_options)
        if isinstance(envelope, SdkError):
            return envelope
        data = protocol_data(envelope)
        value = data.get("output")
        if not isinstance(value, str):
            raise AutotuneProtocolError("plot-progress SDK data must contain output", envelope.command_result)
        return Path(value)

    async def _structured(self, args: list[str], **invoke_options: Any) -> SdkResult | SdkError:
        check = bool(invoke_options.pop("check", True))
        await self._ensure_protocol_compatible(**invoke_options)
        result = await self.invoke([*args, "--sdk-format", "json"], check=False, **invoke_options)
        envelope = parse_sdk_envelope(result)
        if isinstance(envelope, SdkError) and check:
            raise AutotuneError(envelope.message, result)
        return envelope

    async def _ensure_protocol_compatible(self, **invoke_options: Any) -> None:
        if self._protocol_compatible:
            return
        result = await self.invoke(["capabilities", "--sdk-format", "json"], check=False, **invoke_options)
        envelope = parse_sdk_envelope(result)
        data = protocol_data(envelope)
        if data.get("protocolVersion") != SDK_PROTOCOL_VERSION:
            raise AutotuneVersionError(f"incompatible autotune SDK protocol; expected {SDK_PROTOCOL_VERSION}", result)
        self._protocol_compatible = True
