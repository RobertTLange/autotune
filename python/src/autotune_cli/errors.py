from __future__ import annotations

from .models import CommandResult


class AutotuneError(RuntimeError):
    def __init__(self, message: str, result: CommandResult) -> None:
        super().__init__(message)
        self.result = result


class AutotuneNotFoundError(AutotuneError):
    pass


class AutotuneProtocolError(AutotuneError):
    pass


class AutotuneVersionError(AutotuneError):
    pass
