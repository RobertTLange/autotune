from __future__ import annotations

import os
import threading
import time
from collections.abc import Sequence

from . import processes
from .processes import OutputPipes


class OutputPipeCapture:
    def __init__(self, pipe_fds: Sequence[int]) -> None:
        descriptors: list[int] = []
        try:
            for descriptor in pipe_fds:
                try:
                    duplicate = os.dup(descriptor)
                except OSError:
                    continue
                descriptors.append(duplicate)
                os.set_inheritable(duplicate, False)
        except BaseException:
            for descriptor in descriptors:
                os.close(descriptor)
            raise
        self._pipe_fds = tuple(descriptors)
        self._done = threading.Event()
        self._output_pipes = OutputPipes()
        self._error: BaseException | None = None
        self._started = False

    def start(self) -> None:
        thread = threading.Thread(target=self._capture, daemon=True)
        self._started = True
        try:
            thread.start()
        except BaseException:
            self._started = False
            raise

    @property
    def done(self) -> bool:
        return self._done.is_set()

    def result(
        self, timeout: float, cancel_event: threading.Event | None = None
    ) -> OutputPipes:
        if not self._started:
            self._started = True
            self._capture()
        deadline = time.monotonic() + timeout
        while not self._done.is_set():
            if cancel_event is not None and cancel_event.is_set():
                return OutputPipes()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return OutputPipes()
            self._done.wait(min(0.01, remaining))
        if self._error is not None:
            raise self._error
        return self._output_pipes

    def _capture(self) -> None:
        try:
            self._output_pipes = processes.capture_output_pipes(self._pipe_fds)
        except BaseException as error:
            self._error = error
        finally:
            for descriptor in self._pipe_fds:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            self._done.set()
