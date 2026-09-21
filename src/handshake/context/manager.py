"""Build the provider-neutral context exposed to the model."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace

from handshake.context.budget import (
    ContextBudgetPolicy,
    TokenEstimator,
    Utf8TokenEstimator,
    compact_json,
    estimate_model_context,
)
from handshake.context.condenser import (
    HistoryCondenser,
    condenser_context,
    episode_data,
    raw_data,
)
from handshake.context.errors import ContextPressureError, ContextViewError
from handshake.context.state import (
    ContextCheckpoint,
    ContextDependency,
    ContextView,
    ContextWorkItem,
    ExactContext,
    HistoryEpisode,
    HistorySegment,
    RawHistorySegment,
    RecentContext,
    SequencedMessage,
    SourceRange,
)
from handshake.models.base import ToolDefinition
from handshake.models.protocol import ModelContext, ModelMessage
from handshake.runtime.errors import ModelProtocolError, ModelProviderError
from handshake.runtime.state import Run, Session

RawMessageReader = Callable[[str, int], ModelMessage]
ContextMessageReader = Callable[[int, int | None], Iterable[SequencedMessage]]


@dataclass(slots=True)
class _ContextState:
    """Bounded view state: Episodes cover a prefix; messages hold its raw tail."""

    message_count: int = 0
    recent_start: int = 0
    messages: dict[int, ModelMessage] = field(default_factory=dict)
    episodes: tuple[HistoryEpisode, ...] = ()
    stale_mask_through: tuple[int, int] | None = None
    max_tool_result_tokens: int = 16_000
    reduced_message_count: int = 0
    checkpoint_revision: int = 0
    reduction_dirty: bool = False


def _system_prompt(exact: ExactContext) -> str:
    plan = "\n".join(f"- {step}" for step in exact.plan) or "(none)"
    return f"""You are Handshake, a local coding agent operating on one workspace.

Current goal:
{exact.goal}

Workspace:
{exact.workspace}

Optional current plan:
{plan}

Rules:
- Investigate enough repository context before choosing an approach or modifying files.
- Use update_plan for non-trivial, multi-step work when it improves progress tracking.
- Simple requests may be completed without creating a plan.
- Treat the plan as provisional and revise it when tool results materially change the approach.
- All file paths must be relative to the workspace.
- Keep context use focused: discover with list_files/search_files, read only relevant line ranges,
  and prefer edit_file for targeted changes to existing files.
- Make targeted changes and run appropriate validation afterwards.
- Shell tools require user approval. If denied, adapt instead of pretending they ran.
- Ask for a non-sensitive user fact with wait_for_text only when it cannot be inferred from the
  repository. Call wait_for_text by itself in a turn.
- Never request passwords, API keys, tokens, or other secrets with wait_for_text.
- A successful wait_for_text result means an input dependency was registered. The user's later
  answer arrives as a new user continuation message, not as another result for the same tool call.
- Request a test credential with wait_for_secret, never through normal messages or file tools.
  Call wait_for_secret by itself. Its successful result provides an opaque handle, not the Secret.
- Pass that opaque handle only to run_command_with_secret when the requested credential is
  available. Handles are one-time: request a new one before rerunning a command that started.
- Use tool results as evidence and correct failures when possible.
- Do not claim a test passed unless its tool result says it passed.
- When the current request is complete, respond without tool calls with a concise summary of
  changes, validation performed, and any unresolved issue.
"""


class AgentContextManager:
    """Expose the current model-visible view of a Session and Run."""

    def __init__(
        self,
        policy: ContextBudgetPolicy | None = None,
        estimator: TokenEstimator | None = None,
        condenser: HistoryCondenser | None = None,
    ) -> None:
        self.policy = policy or ContextBudgetPolicy()
        self.estimator = estimator or Utf8TokenEstimator()
        self.condenser = condenser
        self._states: dict[str, _ContextState] = {}

    def build(
        self,
        session: Session,
        run: Run,
    ) -> ModelContext:
        """Render the already-maintained state without triggering compaction."""
        return self.render(self.capture(session, run))

    def message_count(self, session: Session) -> int:
        """Return the durable history prefix already represented by this view."""
        return self._synchronized_state(session).message_count

    def is_initialized(self, session: Session) -> bool:
        """Return whether this process has reconstructed the Session view."""
        return session.session_id in self._states

    def prepare(
        self,
        session: Session,
        run: Run,
        tools: tuple[ToolDefinition, ...],
        *,
        raw_message_reader: RawMessageReader | None = None,
    ) -> ModelContext:
        """Apply context pressure policy and return the next model-visible context."""
        pressure = self.measure_pressure(session, run, tools)
        state = self._synchronized_state(session)
        if (
            pressure >= self.policy.pressure_trigger_tokens
            and state.reduced_message_count < state.message_count
        ):
            while pressure > self.policy.pressure_target_tokens:
                if not self.mask_oldest_stale_result(session):
                    break
                pressure = self.measure_pressure(session, run, tools)
            if pressure > self.policy.pressure_target_tokens and self.condenser is not None:
                self._compact(session, run, tools, raw_message_reader)
                pressure = self.measure_pressure(session, run, tools)
            state.reduced_message_count = state.message_count
            state.reduction_dirty = True
        self.ensure_below_hard_limit(pressure)
        return self.build(session, run)

    def restore(
        self,
        session: Session,
        checkpoint: ContextCheckpoint | None,
        source_message_count: int,
        read_messages: ContextMessageReader,
    ) -> None:
        """Rebuild a bounded view from metadata plus only its uncovered raw tail."""
        if session.session_id in self._states:
            raise ContextViewError("Context state is already initialized")
        if source_message_count < 0:
            raise ContextViewError("Durable message count cannot be negative")
        if checkpoint is None:
            checkpoint = ContextCheckpoint(
                max_tool_result_tokens=self.policy.max_tool_result_tokens,
            )
        _validate_checkpoint(checkpoint, source_message_count)
        compacted_end = (
            checkpoint.episodes[-1].source_range.end_sequence + 1
            if checkpoint.episodes
            else 0
        )
        state = _ContextState(
            message_count=compacted_end,
            recent_start=checkpoint.recent_start,
            episodes=checkpoint.episodes,
            stale_mask_through=checkpoint.stale_mask_through,
            max_tool_result_tokens=checkpoint.max_tool_result_tokens,
            reduced_message_count=checkpoint.reduced_message_count,
            checkpoint_revision=checkpoint.revision,
        )
        cursor_seen = (
            checkpoint.stale_mask_through is None
            or checkpoint.stale_mask_through[0] < compacted_end
        )
        for item in read_messages(compacted_end, source_message_count):
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
            state.messages[item.sequence] = self._project_message(
                state, item.sequence, item.message
            )
            state.message_count += 1
        if state.message_count != source_message_count:
            raise ContextViewError("Durable history ended before its reported tail")
        if not cursor_seen:
            raise ContextViewError("Stale mask cursor does not reference a tool result")
        if source_message_count > checkpoint.source_message_count:
            self._update_recent(state)
        self._states[session.session_id] = state

    def pending_checkpoint(self, session: Session) -> ContextCheckpoint | None:
        """Return unsaved reduction metadata, if this view changed semantically."""
        state = self._synchronized_state(session)
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
        )

    def checkpoint_saved(
        self, session: Session, checkpoint: ContextCheckpoint
    ) -> None:
        """Acknowledge a storage-owned successful checkpoint transaction."""
        state = self._synchronized_state(session)
        pending = self.pending_checkpoint(session)
        if pending is None or replace(checkpoint, revision=pending.revision) != pending:
            raise ContextViewError("Saved context checkpoint does not match current view")
        state.checkpoint_revision = checkpoint.revision
        state.reduction_dirty = False

    def _project_message(
        self, state: _ContextState, sequence: int, raw: ModelMessage
    ) -> ModelMessage:
        message = deepcopy(raw)
        if message.role != "user":
            return message
        policy = replace(
            self.policy, max_tool_result_tokens=state.max_tool_result_tokens
        )
        for index, block in enumerate(message.content):
            if block.get("type") != "tool_result":
                continue
            if (
                state.stale_mask_through is not None
                and (sequence, index) <= state.stale_mask_through
            ):
                block["content"] = _stale_marker(sequence, block.get("is_error"))
                continue
            call_id = block.get("tool_use_id")
            if not isinstance(call_id, str):
                continue
            content = block.get("content", "")
            text = content if isinstance(content, str) else compact_json(content)
            replacement = _ingress_content(
                sequence,
                call_id,
                block.get("is_error") is not False,
                text,
                policy,
                self.estimator,
            )
            if replacement is not None:
                block["content"] = replacement
        return message

    def _update_recent(self, state: _ContextState) -> None:
        compacted_end = state.episodes[-1].source_range.end_sequence + 1 if state.episodes else 0
        messages = tuple(
            SequencedMessage(sequence, state.messages[sequence])
            for sequence in range(compacted_end, state.message_count)
        )
        relative_cutoff = _recent_start(messages, self.policy, self.estimator)
        cutoff = (
            state.message_count
            if relative_cutoff == len(messages)
            else messages[relative_cutoff].sequence
        )
        state.recent_start = max(state.recent_start, compacted_end, cutoff)

    def ingest_messages(
        self,
        session: Session,
        messages: Iterable[SequencedMessage],
    ) -> None:
        """Add newly committed durable messages to the model-visible view."""
        current = self._states.get(
            session.session_id,
            _ContextState(max_tool_result_tokens=self.policy.max_tool_result_tokens),
        )
        state = deepcopy(current)
        changed = False
        for item in messages:
            sequence = item.sequence
            if sequence != state.message_count:
                raise ContextViewError(
                    "Context messages must extend the durable transcript contiguously"
                )
            message = self._project_message(state, sequence, item.message)
            state.messages[sequence] = message
            state.message_count = sequence + 1
            changed = True
        if not changed:
            self._states[session.session_id] = state
            return
        self._update_recent(state)
        self._states[session.session_id] = state

    def capture(
        self,
        session: Session,
        run: Run,
        *,
        constraints: tuple[str, ...] = (),
    ) -> ContextView:
        """Snapshot the incrementally maintained bounded model view."""
        state = self._synchronized_state(session)
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
        compacted_end = state.episodes[-1].source_range.end_sequence + 1 if state.episodes else 0
        messages = tuple(
            SequencedMessage(sequence, deepcopy(state.messages[sequence]))
            for sequence in range(compacted_end, state.message_count)
        )
        history: tuple[HistorySegment, ...] = state.episodes
        if compacted_end < cutoff:
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
    ) -> int:
        """Measure the current view without changing its compaction state."""
        return estimate_model_context(
            self.build(session, run),
            tools,
            self.estimator,
        )

    def mask_oldest_stale_result(self, session: Session) -> bool:
        """Incrementally mask one oldest result outside protected recent context."""
        state = self._synchronized_state(session)
        start = state.episodes[-1].source_range.end_sequence + 1 if state.episodes else 0
        for sequence in range(start, state.recent_start):
            message = state.messages[sequence]
            if message.role != "user":
                continue
            for index, block in enumerate(message.content):
                if block.get("type") != "tool_result":
                    continue
                if (
                    state.stale_mask_through is not None
                    and (sequence, index) <= state.stale_mask_through
                ):
                    continue
                block["content"] = _stale_marker(sequence, block.get("is_error"))
                state.stale_mask_through = (sequence, index)
                state.reduction_dirty = True
                return True
        return False

    def _fits_request(
        self, episodes: tuple[HistoryEpisode, ...], target: tuple[SequencedMessage, ...] | None
    ) -> bool:
        return estimate_model_context(
            condenser_context(episodes, target), (), self.estimator
        ) <= self.policy.condenser_input_tokens

    def _raw_candidate(
        self,
        session: Session,
        raw_message_reader: RawMessageReader | None,
    ) -> tuple[tuple[HistoryEpisode, ...], tuple[SequencedMessage, ...]]:
        state = self._synchronized_state(session)
        background: tuple[HistoryEpisode, ...] = ()
        for episode in reversed(state.episodes):
            proposed = (episode, *background)
            if self.estimator.estimate_text(compact_json(episode_data(proposed))) > (
                self.policy.background_episode_tokens
            ):
                break
            background = proposed
        start = state.episodes[-1].source_range.end_sequence + 1 if state.episodes else 0
        target: tuple[SequencedMessage, ...] = ()
        pending: list[SequencedMessage] = []
        for sequence in range(start, state.recent_start):
            message = deepcopy(state.messages[sequence])
            if message.role == "user":
                for index, block in enumerate(message.content):
                    if block.get("type") == "tool_result" and block.get("is_error") is True:
                        raw = (
                            raw_message_reader(session.session_id, sequence)
                            if raw_message_reader is not None
                            else state.messages[sequence]
                        )
                        block["content"] = deepcopy(raw.content[index].get("content", ""))
            pending.append(SequencedMessage(sequence, message))
            if sequence + 1 < state.message_count and _same_tool_round(
                pending[-1], SequencedMessage(sequence + 1, state.messages[sequence + 1])
            ):
                continue
            proposed_target = (*target, *pending)
            if self.estimator.estimate_text(compact_json(raw_data(proposed_target))) > (
                self.policy.raw_target_tokens
            ):
                break
            # Preserve target capacity when the total cap is tighter than the two sub-budgets.
            while background and not self._fits_request(background, proposed_target):
                background = background[1:]
            if not self._fits_request(background, proposed_target):
                break
            target = proposed_target
            pending = []
        return background, target

    def _accept_episodes(
        self, session: Session, run: Run, tools: tuple[ToolDefinition, ...],
        episodes: tuple[HistoryEpisode, ...],
    ) -> bool:
        view = self.capture(session, run)
        end = episodes[-1].source_range.end_sequence + 1
        state = self._synchronized_state(session)
        history: tuple[HistorySegment, ...] = episodes
        if end < state.recent_start:
            history += (RawHistorySegment(tuple(
                SequencedMessage(i, state.messages[i]) for i in range(end, state.recent_start)
            )),)
        proposed = self.render(replace(view, history=history))
        if estimate_model_context(proposed, tools, self.estimator) >= (
            estimate_model_context(self.render(view), tools, self.estimator)
        ):
            return False
        state.episodes = episodes
        for sequence in tuple(state.messages):
            if sequence < end:
                del state.messages[sequence]
        state.reduction_dirty = True
        return True

    def _compact(
        self,
        session: Session,
        run: Run,
        tools: tuple[ToolDefinition, ...],
        raw_message_reader: RawMessageReader | None,
    ) -> None:
        assert self.condenser is not None
        state = self._synchronized_state(session)
        try:
            while self.measure_pressure(session, run, tools) > self.policy.pressure_target_tokens:
                background, target = self._raw_candidate(session, raw_message_reader)
                if not target:
                    break
                episodes = self.condenser.compact_raw(background, target)
                if not episodes:
                    break
                expected = target[0].sequence
                for episode in episodes:
                    _validate_episode(episode, expected, target[-1].sequence)
                    end = episode.source_range.end_sequence
                    if end + 1 < state.message_count and _same_tool_round(
                        SequencedMessage(end, state.messages[end]),
                        SequencedMessage(end + 1, state.messages[end + 1]),
                    ):
                        raise ContextViewError("Episode splits a tool round")
                    expected = end + 1
                if not self._accept_episodes(session, run, tools, state.episodes + episodes):
                    break

            while self.measure_pressure(session, run, tools) > self.policy.pressure_target_tokens:
                selected: tuple[HistoryEpisode, ...] = ()
                for episode in state.episodes:
                    proposed = (*selected, episode)
                    if (self.estimator.estimate_text(compact_json(episode_data(proposed)))
                            > self.policy.rollup_episode_tokens
                            or not self._fits_request(proposed, None)):
                        break
                    selected = proposed
                if len(selected) < 2:
                    break
                merged = self.condenser.roll_up(selected)
                if merged is None:
                    break
                start = selected[0].source_range.start_sequence
                end = selected[-1].source_range.end_sequence
                _validate_episode(merged, start, end)
                if merged.source_range.end_sequence != end:
                    raise ContextViewError("Roll-up must cover its complete input")
                if not self._accept_episodes(
                    session, run, tools, (merged, *state.episodes[len(selected):])
                ):
                    break
        except (ContextViewError, ModelProtocolError, ModelProviderError):
            # A failed condenser request never replaces the last validated view.
            return

    def _synchronized_state(self, session: Session) -> _ContextState:
        state = self._states.get(session.session_id)
        if state is None:
            return _ContextState(
                max_tool_result_tokens=self.policy.max_tool_result_tokens
            )
        if not 0 <= state.recent_start <= state.message_count:
            raise ContextViewError("Context state has an invalid protected recent boundary")
        compacted_end = (
            state.episodes[-1].source_range.end_sequence + 1
            if state.episodes
            else 0
        )
        if compacted_end > state.recent_start:
            raise ContextViewError("Compacted history overlaps protected recent context")
        if len(state.messages) != state.message_count - compacted_end:
            raise ContextViewError(
                "Context raw tail must cover every uncompacted source sequence"
            )
        for sequence in range(compacted_end, state.message_count):
            if sequence not in state.messages:
                raise ContextViewError(
                    "Context raw tail must cover every uncompacted source sequence"
                )
        return state

    def ensure_below_hard_limit(self, visible_tokens: int) -> None:
        """Reject a provider request that remains at or above its hard input limit."""
        if visible_tokens >= self.policy.hard_input_tokens:
            raise ContextPressureError(
                "Model input remains above the hard context limit after context reduction"
            )

    def render(self, view: ContextView) -> ModelContext:
        """Validate and render a ContextView without changing its source data."""
        _validate_view(view)
        exact = view.exact
        prompt = _system_prompt(exact)
        state = {
            "constraints": exact.constraints,
            "work_items": [asdict(item) for item in exact.work_items],
            "dependencies": [asdict(item) for item in exact.dependencies],
        }
        prompt += "\nExact current state:\n" + json.dumps(state, ensure_ascii=False)

        messages: list[ModelMessage] = []
        for segment in view.history:
            if isinstance(segment, RawHistorySegment):
                messages.extend(item.message for item in segment.messages)
            else:
                messages.append(_render_episode(segment))
        messages.extend(item.message for item in view.protected_recent.messages)

        return ModelContext(
            system_prompt=prompt,
            goal=exact.goal,
            workspace=exact.workspace,
            plan=exact.plan,
            messages=deepcopy(tuple(messages)),
        )


def _ingress_content(
    sequence: int,
    call_id: str,
    is_error: bool,
    content: str,
    policy: ContextBudgetPolicy,
    estimator: TokenEstimator,
) -> str | None:
    original_tokens = _result_tokens(call_id, is_error, content, estimator)
    if original_tokens <= policy.max_tool_result_tokens:
        return None
    marker = (
        f"[tool result ingress-limited; "
        f"status={'failure' if is_error else 'success'}; "
        f"source_sequence={sequence}; original_estimated_tokens={original_tokens}]"
    )
    kept = len(content)
    replacement = marker
    while kept:
        head, tail = (kept + 1) // 2, kept // 2
        replacement = (
            f"{marker}\n--- head ---\n{content[:head]}\n"
            f"--- tail ---\n{content[-tail:] if tail else ''}"
        )
        if (
            _result_tokens(call_id, is_error, replacement, estimator)
            <= policy.max_tool_result_tokens
        ):
            break
        kept //= 2
    while replacement and _result_tokens(
        call_id, is_error, replacement, estimator
    ) > policy.max_tool_result_tokens:
        replacement = replacement[: len(replacement) // 2]
    return replacement


def _stale_marker(sequence: int, is_error: object) -> str:
    status = "failure" if is_error is not False else "success"
    return (
        f"[stale tool result masked; status={status}; "
        f"source_sequence={sequence}; details omitted from active context]"
    )


def _result_tokens(
    call_id: str,
    is_error: bool,
    content: str,
    estimator: TokenEstimator,
) -> int:
    block = {
        "type": "tool_result",
        "tool_use_id": call_id,
        "content": content,
        "is_error": is_error,
    }
    return estimator.estimate_text(compact_json(block))


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
    if cutoff and _same_tool_round(messages[cutoff - 1], messages[cutoff]):
        cutoff -= 1
    return cutoff


def _same_tool_round(previous: SequencedMessage, current: SequencedMessage) -> bool:
    if previous.message.role != "assistant" or current.message.role != "user":
        return False
    if previous.message.run_id != current.message.run_id:
        return False
    calls = {
        block.get("id")
        for block in previous.message.content
        if block.get("type") == "tool_use"
    }
    results = {
        block.get("tool_use_id")
        for block in current.message.content
        if block.get("type") == "tool_result"
    }
    return bool(calls & results)


def _segment_range(segment: HistorySegment) -> SourceRange:
    if isinstance(segment, HistoryEpisode):
        return segment.source_range
    if not segment.messages:
        raise ContextViewError("A raw history segment cannot be empty")
    return SourceRange(segment.messages[0].sequence, segment.messages[-1].sequence)


def _validate_sequenced_messages(messages: tuple[SequencedMessage, ...], description: str) -> None:
    if not messages:
        return
    expected = messages[0].sequence
    for item in messages:
        if item.sequence != expected:
            raise ContextViewError(f"{description} messages must have continuous sequences")
        expected += 1


def _validate_view(view: ContextView) -> None:
    if view.source_message_count < 0:
        raise ContextViewError("source_message_count cannot be negative")

    expected_sequence = 0
    for segment in view.history:
        if isinstance(segment, RawHistorySegment):
            _validate_sequenced_messages(segment.messages, "Raw history")
        source_range = _segment_range(segment)
        if (
            source_range.start_sequence < 0
            or source_range.end_sequence < source_range.start_sequence
        ):
            raise ContextViewError("History segments must use valid source ranges")
        if source_range.start_sequence != expected_sequence:
            raise ContextViewError(
                "Context history must cover source sequences in order without gaps or overlaps"
            )
        expected_sequence = source_range.end_sequence + 1

    recent = view.protected_recent.messages
    _validate_sequenced_messages(recent, "Protected recent")
    if recent:
        if recent[0].sequence != expected_sequence:
            raise ContextViewError(
                "Protected recent history must follow older history without gaps or overlaps"
            )
        expected_sequence = recent[-1].sequence + 1

    if expected_sequence != view.source_message_count:
        raise ContextViewError("ContextView must cover every source message exactly once")


def _render_episode(episode: HistoryEpisode) -> ModelMessage:
    source_range = episode.source_range
    text = (
        "[Compacted history episode; source messages "
        f"{source_range.start_sequence}-{source_range.end_sequence}]\n"
        f"Title: {episode.title}\n"
        f"Summary: {episode.summary}"
    )
    return ModelMessage(role="user", content=({"type": "text", "text": text},))


def _validate_episode(episode: HistoryEpisode, start: int, maximum: int) -> None:
    source = episode.source_range
    if (type(source.start_sequence) is not int or type(source.end_sequence) is not int
            or source.start_sequence != start or not start <= source.end_sequence <= maximum
            or not isinstance(episode.title, str) or not episode.title.strip()
            or not isinstance(episode.summary, str) or not episode.summary.strip()):
        raise ContextViewError("Invalid condenser episode range or content")


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
        _validate_episode(episode, expected, checkpoint.recent_start - 1)
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
