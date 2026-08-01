from __future__ import annotations

import json
from typing import Any

from .errors import AutotuneProtocolError, AutotuneVersionError
from .models import CommandResult, SdkError, SdkResult

SDK_PROTOCOL_VERSION = 1


def parse_sdk_envelope(result: CommandResult) -> SdkResult | SdkError:
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise AutotuneProtocolError("autotune returned invalid SDK JSON", result) from error
    if not isinstance(payload, dict) or payload.get("protocolVersion") != SDK_PROTOCOL_VERSION:
        raise AutotuneVersionError(
            f"incompatible autotune SDK protocol; expected {SDK_PROTOCOL_VERSION}", result
        )
    command = payload.get("command")
    exit_code = payload.get("exitCode")
    if not isinstance(command, str) or not isinstance(exit_code, int) or isinstance(exit_code, bool):
        raise AutotuneProtocolError("autotune SDK envelope has invalid command metadata", result)
    if payload.get("type") == "result":
        if exit_code != 0 or result.returncode != 0 or exit_code != result.returncode:
            raise AutotuneProtocolError("autotune SDK result has inconsistent exit code", result)
        return SdkResult(SDK_PROTOCOL_VERSION, command, exit_code, payload.get("data"), result)
    if payload.get("type") == "error":
        error_data = payload.get("error")
        message = error_data.get("message") if isinstance(error_data, dict) else None
        if isinstance(message, str) and exit_code > 0 and exit_code == result.returncode:
            return SdkError(SDK_PROTOCOL_VERSION, command, exit_code, message, result)
    raise AutotuneProtocolError("autotune SDK envelope has an unknown shape", result)


def protocol_data(envelope: SdkResult | SdkError) -> dict[str, Any]:
    if isinstance(envelope, SdkError):
        raise AutotuneProtocolError(envelope.message, envelope.command_result)
    if not isinstance(envelope.data, dict):
        raise AutotuneProtocolError("autotune SDK result data must be an object", envelope.command_result)
    return envelope.data
