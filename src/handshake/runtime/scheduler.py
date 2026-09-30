"""Cooperative scheduling for a durable batch of child runs."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from handshake.runtime.errors import AgentLoopError
from handshake.runtime.runs import TERMINAL_RUN_STATUSES
from handshake.runtime.state import Run, RunStatus


class CooperativeScheduler:
    """Advance each runnable child once before considering a blocked sibling."""

    def runnable(self, children: Sequence[Run]) -> tuple[Run, ...]:
        return tuple(child for child in children if child.status is RunStatus.RUNNING)

    def advance(
        self,
        children: Sequence[Run],
        resume: Callable[[Run], Run],
    ) -> tuple[Run, ...]:
        outcomes = {child.run_id: child for child in children}
        for child in self.runnable(children):
            outcomes[child.run_id] = resume(child)
        return tuple(outcomes[child.run_id] for child in children)

    def ready_to_join(self, children: Sequence[Run]) -> bool:
        return bool(children) and all(
            child.status in TERMINAL_RUN_STATUSES for child in children
        )

    def advance_runnable(self, run: Run, resume: Callable[[Run], Run]) -> Run:
        if run.status is not RunStatus.RUNNING:
            raise AgentLoopError("Only a runnable run can advance")
        return resume(run)

    def waiting_child(self, children: Sequence[Run]) -> Run | None:
        return next(
            (
                child for child in children
                if child.status is RunStatus.WAITING and child.open_dependencies()
            ),
            None,
        )
