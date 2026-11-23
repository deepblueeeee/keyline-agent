"""Run identity shared by progress output and process-local input requests."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from handshake.runtime.state import Run, RunStatus

_CURRENT_RUN: ContextVar[Run | None] = ContextVar("current_run", default=None)


@contextmanager
def run_identity(run: Run) -> Iterator[None]:
    token = _CURRENT_RUN.set(run)
    try:
        yield
    finally:
        _CURRENT_RUN.reset(token)


def current_run() -> Run | None:
    return _CURRENT_RUN.get()


def format_run_message(run: Run, message: str, *, waiting: str | None = None) -> str:
    status = "waiting" if waiting else run.status.value
    prefix = f"[agent={run.agent_spec_id} run={run.run_id} status={status}"
    prefix += f" wait={waiting}]" if waiting else "]"
    return "\n".join(f"{prefix} {line}" for line in message.splitlines() or [""])


@dataclass(frozen=True, slots=True)
class RunReporter:
    run: Run
    sink: Callable[[str], None]

    def __call__(self, message: str) -> None:
        waiting = None
        if self.run.status is RunStatus.WAITING:
            waiting = ",".join(sorted({d.kind.value for d in self.run.open_dependencies()}))
            waiting = waiting or "children"
        self.sink(format_run_message(self.run, message, waiting=waiting))


def bind_reporter(emit: Callable[[str], None], run: Run) -> Callable[[str], None]:
    return RunReporter(run, emit.sink) if isinstance(emit, RunReporter) else emit


def emit_wait(emit: Callable[[str], None], message: str, reason: str) -> None:
    if isinstance(emit, RunReporter):
        emit.sink(format_run_message(emit.run, message, waiting=reason))
    else:
        emit(message)
