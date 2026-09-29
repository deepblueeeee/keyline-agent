"""Build the Anthropic Messages context exposed to the model."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import replace

from handshake.context.budget import (
    ContextBudgetPolicy,
    TokenEstimator,
    Utf8TokenEstimator,
    compact_json,
    measure_model_context,
    measure_model_turn,
)
from handshake.context.condenser import HistoryCondenser
from handshake.context.errors import ContextPressureError, ContextViewError
from handshake.context.managed_state import (
    ManagedContextState,
    compacted_end,
    validate_managed_state,
)
from handshake.context.projection import project_message
from handshake.context.reduction import (
    RawMessageReader,
    compact_semantic_context,
    mask_oldest_stale_result,
    same_tool_round,
    validate_episode,
)
from handshake.context.render import render_context_view
from handshake.context.state import (
    ContextCheckpoint,
    ContextDependency,
    ContextView,
    ContextWorkItem,
    ExactContext,
    HistorySegment,
    RawHistorySegment,
    RecentContext,
    SequencedMessage,
)
from handshake.models.base import ToolDefinition
from handshake.models.protocol import ModelContext, ModelTurn
from handshake.runtime.state import ContextScope, Run, Session

ContextMessageReader = Callable[[int, int | None], Iterable[SequencedMessage]]


class AgentContextManager:
    """Expose one ContextScope's model-visible view in its Session workspace."""

    def __init__(
        self,
        policy: ContextBudgetPolicy | None = None,
        estimator: TokenEstimator | None = None,
        condenser: HistoryCondenser | None = None,
        *,
        scope_id: str | None = None,
    ) -> None:
        self.policy = policy or ContextBudgetPolicy()
        self.estimator = estimator or Utf8TokenEstimator()
        self.condenser = condenser
        self._scope_id: str | None = None
        self._state: ManagedContextState | None = None
        if scope_id is not None:
            self.bind_scope(scope_id)

    def bind_scope(self, scope_id: str) -> None:
        """Bind this manager to exactly one durable context stream."""
        if not isinstance(scope_id, str) or not scope_id.strip():
            raise ContextViewError("Context scope ID must not be empty")
        if self._scope_id is None:
            self._scope_id = scope_id
        elif self._scope_id != scope_id:
            raise ContextViewError(
                f"Context manager belongs to scope {self._scope_id}, not {scope_id}"
            )

    def build(
        self,
        session: Session,
        run: Run,
        tools: tuple[ToolDefinition, ...] = (),
        *,
        instructions: str = "",
    ) -> ModelContext:
        """Render the already-maintained state without triggering compaction."""
        return self.render(self.capture(session, run), tools, instructions=instructions)

    def message_count(self, scope: ContextScope) -> int:
        """Return the durable history prefix already represented by this view."""
        return self._synchronized_state(scope.scope_id).message_count

    def is_initialized(self, scope: ContextScope) -> bool:
        """Return whether this process has reconstructed the context stream view."""
        self.bind_scope(scope.scope_id)
        return self._state is not None

    def prepare(
        self,
        session: Session,
        run: Run,
        tools: tuple[ToolDefinition, ...],
        *,
        instructions: str,
        raw_message_reader: RawMessageReader | None = None,
    ) -> ModelContext:
        """Apply context pressure policy and return the next model-visible context."""
        pressure = self.measure_pressure(session, run, tools, instructions=instructions)
        state = self._synchronized_state(run.context_scope_id)
        if (
            pressure >= self.policy.pressure_trigger_tokens
            and state.reduced_message_count < state.message_count
        ):
            while pressure > self.policy.pressure_target_tokens:
                if not mask_oldest_stale_result(state):
                    break
                pressure = self.measure_pressure(
                    session, run, tools, instructions=instructions
                )
            if pressure > self.policy.pressure_target_tokens and self.condenser is not None:
                compact_semantic_context(
                    state,
                    run_id=run.context_scope_id,
                    policy=self.policy,
                    estimator=self.estimator,
                    condenser=self.condenser,
                    tools=tools,
                    instructions=instructions,
                    measure_pressure=lambda: self.measure_pressure(
                        session, run, tools, instructions=instructions
                    ),
                    capture_view=lambda: self.capture(session, run),
                    raw_message_reader=raw_message_reader,
                )
                pressure = self.measure_pressure(
                    session, run, tools, instructions=instructions
                )
            state.reduced_message_count = state.message_count
            state.reduction_dirty = True
        self.ensure_below_hard_limit(pressure)
        return self.build(session, run, tools, instructions=instructions)

    def restore(
        self,
        scope: ContextScope,
        checkpoint: ContextCheckpoint | None,
        source_message_count: int,
        read_messages: ContextMessageReader,
    ) -> None:
        """Rebuild a bounded view from metadata plus only its uncovered raw tail."""
        self.bind_scope(scope.scope_id)
        if self._state is not None:
            raise ContextViewError("Context state is already initialized")
        if source_message_count < 0:
            raise ContextViewError("Durable message count cannot be negative")
        if checkpoint is None:
            checkpoint = ContextCheckpoint(
                max_tool_result_tokens=self.policy.max_tool_result_tokens,
            )
        _validate_checkpoint(checkpoint, source_message_count)
        restored_start = (
            checkpoint.episodes[-1].source_range.end_sequence + 1
            if checkpoint.episodes
            else 0
        )
        state = ManagedContextState(
            message_count=restored_start,
            recent_start=checkpoint.recent_start,
            episodes=checkpoint.episodes,
            stale_mask_through=checkpoint.stale_mask_through,
            max_tool_result_tokens=checkpoint.max_tool_result_tokens,
            reduced_message_count=checkpoint.reduced_message_count,
            checkpoint_revision=checkpoint.revision,
            token_measurement=checkpoint.token_measurement,
        )
        cursor_seen = (
            checkpoint.stale_mask_through is None
            or checkpoint.stale_mask_through[0] < restored_start
        )
        for item in read_messages(restored_start, source_message_count):
            if item.sequence != state.message_count:
                raise ContextViewError("Durable history contains missing sequences")
            if (
                checkpoint.stale_mask_through is not None
                and item.sequence == checkpoint.stale_mask_through[0]
            ):
                index = checkpoint.stale_mask_through[1]
                cursor_seen = (
                    item.message.role == "user"
                    and index < len(item.message.content)
                    and item.message.content[index].get("type") == "tool_result"
                )
            state.messages[item.sequence] = project_message(
                item.message,
                run_id=scope.scope_id,
                sequence=item.sequence,
                stale_mask_through=state.stale_mask_through,
                max_tool_result_tokens=state.max_tool_result_tokens,
                estimator=self.estimator,
            )
            state.message_count += 1
        if state.message_count != source_message_count:
            raise ContextViewError("Durable history ended before its reported tail")
        if not cursor_seen:
            raise ContextViewError("Stale mask cursor does not reference a tool result")
        if source_message_count > checkpoint.source_message_count:
            self._update_recent(state)
        self._state = state

    def pending_checkpoint(self, scope: ContextScope) -> ContextCheckpoint | None:
        """Return unsaved reduction metadata, if this view changed semantically."""
        state = self._synchronized_state(scope.scope_id)
        if not state.reduction_dirty:
            return None
        return ContextCheckpoint(
            source_message_count=state.message_count,
            recent_start=state.recent_start,
            episodes=state.episodes,
            stale_mask_through=state.stale_mask_through,
            max_tool_result_tokens=state.max_tool_result_tokens,
            reduced_message_count=state.reduced_message_count,
            revision=state.checkpoint_revision,
            token_measurement=state.token_measurement,
        )

    def checkpoint_saved(
        self, scope: ContextScope, checkpoint: ContextCheckpoint
    ) -> None:
        """Acknowledge a storage-owned successful checkpoint transaction."""
        state = self._synchronized_state(scope.scope_id)
        pending = self.pending_checkpoint(scope)
        if pending is None or replace(checkpoint, revision=pending.revision) != pending:
            raise ContextViewError("Saved context checkpoint does not match current view")
        state.checkpoint_revision = checkpoint.revision
        state.reduction_dirty = False

    def _update_recent(self, state: ManagedContextState) -> None:
        start = compacted_end(state)
        messages = tuple(
            SequencedMessage(sequence, state.messages[sequence])
            for sequence in range(start, state.message_count)
        )
        relative_cutoff = _recent_start(messages, self.policy, self.estimator)
        cutoff = (
            state.message_count
            if relative_cutoff == len(messages)
            else messages[relative_cutoff].sequence
        )
        state.recent_start = max(state.recent_start, start, cutoff)

    def ingest_messages(
        self,
        scope: ContextScope,
        messages: Iterable[SequencedMessage],
    ) -> None:
        """Add newly committed durable messages to the model-visible view."""
        self.bind_scope(scope.scope_id)
        current = self._state
        if current is None:
            current = ManagedContextState(
                max_tool_result_tokens=self.policy.max_tool_result_tokens
            )
        state = deepcopy(current)
        changed = False
        for item in messages:
            sequence = item.sequence
            if sequence != state.message_count:
                raise ContextViewError(
                    "Context messages must extend the durable transcript contiguously"
                )
            message = project_message(
                item.message,
                run_id=scope.scope_id,
                sequence=sequence,
                stale_mask_through=state.stale_mask_through,
                max_tool_result_tokens=state.max_tool_result_tokens,
                estimator=self.estimator,
            )
            state.messages[sequence] = message
            state.message_count = sequence + 1
            changed = True
        if not changed:
            self._state = state
            return
        self._update_recent(state)
        self._state = state

    def capture(
        self,
        session: Session,
        run: Run,
        *,
        constraints: tuple[str, ...] = (),
    ) -> ContextView:
        """Snapshot the incrementally maintained bounded model view."""
        state = self._synchronized_state(run.context_scope_id)
        exact = ExactContext(
            goal=run.goal,
            workspace=str(session.workspace),
            plan=tuple(run.plan),
            constraints=constraints,
            work_items=tuple(
                ContextWorkItem(item.work_id, item.description, item.status.value)
                for item in run.work_items
            ),
            dependencies=tuple(
                ContextDependency(
                    dependency.dependency_id,
                    dependency.work_id,
                    dependency.tool_call_id,
                    dependency.kind.value,
                    dependency.prompt,
                    dependency.status.value,
                )
                for dependency in run.dependencies
            ),
        )
        cutoff = state.recent_start
        history_end = compacted_end(state)
        messages = tuple(
            SequencedMessage(sequence, deepcopy(state.messages[sequence]))
            for sequence in range(history_end, state.message_count)
        )
        history: tuple[HistorySegment, ...] = state.episodes
        if history_end < cutoff:
            history += (
                RawHistorySegment(
                    tuple(item for item in messages if item.sequence < cutoff)
                ),
            )
        return ContextView(
            exact=exact,
            history=history,
            protected_recent=RecentContext(
                tuple(item for item in messages if item.sequence >= cutoff)
            ),
            source_message_count=state.message_count,
        )

    def measure_pressure(
        self,
        session: Session,
        run: Run,
        tools: tuple[ToolDefinition, ...],
        *,
        instructions: str = "",
    ) -> int:
        """Measure the current view without changing its compaction state."""
        return measure_model_context(
            self.build(session, run, tools, instructions=instructions),
            tools,
            self.estimator,
            self._synchronized_state(run.context_scope_id).token_measurement,
        )

    def record_usage(
        self, scope: ContextScope, context: ModelContext,
        tools: tuple[ToolDefinition, ...], turn: ModelTurn,
    ) -> None:
        """Calibrate the next request from this response's measured token counts."""
        measurement = measure_model_turn(context, tools, turn, self.estimator)
        if measurement is not None:
            state = self._synchronized_state(scope.scope_id)
            state.token_measurement = measurement
            state.reduction_dirty = True

    def mask_oldest_stale_result(self, scope: ContextScope) -> bool:
        """Incrementally mask one oldest result outside protected recent context."""
        return mask_oldest_stale_result(self._synchronized_state(scope.scope_id))

    def _synchronized_state(self, scope_id: str) -> ManagedContextState:
        self.bind_scope(scope_id)
        state = self._state
        if state is None:
            return ManagedContextState(
                max_tool_result_tokens=self.policy.max_tool_result_tokens
            )
        validate_managed_state(state)
        return state

    def ensure_below_hard_limit(self, visible_tokens: int) -> None:
        """Reject a provider request that remains at or above its hard input limit."""
        if visible_tokens >= self.policy.hard_input_tokens:
            raise ContextPressureError(
                "Model input remains above the hard context limit after context reduction"
            )

    def render(
        self,
        view: ContextView,
        tools: tuple[ToolDefinition, ...] = (),
        *,
        instructions: str = "",
    ) -> ModelContext:
        """Validate and render a ContextView without changing its source data."""
        return render_context_view(view, instructions=instructions, tools=tools)


class ActiveContextManagers:
    """Keep one manager for each context scope in the current process."""

    def __init__(self) -> None:
        self._by_scope: dict[str, AgentContextManager] = {}

    def get(self, scope_id: str) -> AgentContextManager | None:
        return self._by_scope.get(scope_id)

    def register(self, scope_id: str, manager: AgentContextManager) -> None:
        if scope_id in self._by_scope:
            raise ContextViewError(f"Context scope {scope_id} already has a manager")
        manager.bind_scope(scope_id)
        self._by_scope[scope_id] = manager


def _recent_start(
    messages: tuple[SequencedMessage, ...],
    policy: ContextBudgetPolicy,
    estimator: TokenEstimator,
) -> int:
    count = len(messages)
    by_count = max(0, count - policy.protected_recent_messages)
    by_tokens, tokens = count, 0
    if policy.protected_recent_tokens:
        for index in range(count - 1, -1, -1):
            tokens += estimator.estimate_text(compact_json(messages[index].message.as_dict()))
            by_tokens = index
            if tokens >= policy.protected_recent_tokens:
                break
    cutoff = min(by_count, by_tokens)
    if cutoff and same_tool_round(messages[cutoff - 1], messages[cutoff]):
        cutoff -= 1
    return cutoff


def _validate_checkpoint(
    checkpoint: ContextCheckpoint,
    durable_message_count: int,
) -> None:
    integer_values = (
        checkpoint.source_message_count,
        checkpoint.recent_start,
        checkpoint.max_tool_result_tokens,
        checkpoint.reduced_message_count,
        checkpoint.revision,
    )
    if any(type(value) is not int for value in integer_values):
        raise ContextViewError("Context checkpoint contains invalid metadata types")
    if (
        not 0 <= checkpoint.recent_start <= checkpoint.source_message_count
        or checkpoint.source_message_count > durable_message_count
        or not 0 <= checkpoint.reduced_message_count <= checkpoint.source_message_count
        or checkpoint.max_tool_result_tokens <= 0
        or checkpoint.revision < 0
    ):
        raise ContextViewError("Context checkpoint contains invalid boundaries")
    expected = 0
    for episode in checkpoint.episodes:
        validate_episode(episode, expected, checkpoint.recent_start - 1)
        expected = episode.source_range.end_sequence + 1
    if expected > checkpoint.recent_start:
        raise ContextViewError("Compacted history overlaps protected recent context")
    cursor = checkpoint.stale_mask_through
    if cursor is not None and (
        not isinstance(cursor, tuple)
        or len(cursor) != 2
        or any(type(value) is not int for value in cursor)
        or cursor[0] < 0
        or cursor[0] >= checkpoint.recent_start
        or cursor[1] < 0
    ):
        raise ContextViewError("Context checkpoint contains an invalid stale mask cursor")
