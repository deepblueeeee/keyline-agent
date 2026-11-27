from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from keyline_agent.context import (
    AgentContextManager,
    ContextBudgetPolicy,
    ContextPressureError,
    HistoryEpisode,
    LLMHistoryCondenser,
    RawHistorySegment,
    SequencedMessage,
    SourceRange,
)
from keyline_agent.context.budget import compact_json, estimate_model_context
from keyline_agent.context.condenser import condenser_context, episode_data, raw_data
from keyline_agent.context.reduction import select_raw_candidate
from keyline_agent.models.errors import ModelProtocolError, ModelProviderError
from keyline_agent.models.protocol import ModelMessage, ModelTurn
from keyline_agent.models.scripted import ScriptedModel
from keyline_agent.runtime.loop import start_run
from keyline_agent.runtime.state import Run, RunStatus, Session
from keyline_agent.tools.registry import ToolRegistry
from tests.history_support import TEST_AGENT_SPEC, fresh_store, history, ingest, run_async

_HISTORY: dict[str, list[ModelMessage]] = {}


class Characters:
    def estimate_text(self, text: str) -> int:
        return len(text)


class Condenser:
    def __init__(self, *responses: tuple[HistoryEpisode, ...] | Exception) -> None:
        self.responses = deque(responses)
        self.calls: list[tuple[str, object, object]] = []
        self.rollups: deque[HistoryEpisode | None] = deque()

    async def compact_raw(
        self, background: tuple[HistoryEpisode, ...], target: tuple[SequencedMessage, ...],
    ) -> tuple[HistoryEpisode, ...]:
        self.calls.append(("raw", background, target))
        result = self.responses.popleft() if self.responses else ()
        if isinstance(result, Exception):
            raise result
        return result

    async def roll_up(self, episodes: tuple[HistoryEpisode, ...]) -> HistoryEpisode | None:
        self.calls.append(("rollup", episodes, None))
        return self.rollups.popleft() if self.rollups else None


def episode(start: int, end: int, text: str = "Investigation complete.") -> HistoryEpisode:
    return HistoryEpisode(SourceRange(start, end), "Stage", text)


def setup(
    tmp_path: Path, condenser: Condenser, **policy: int | float,
) -> tuple[AgentContextManager, Session, Run]:
    session = Session(workspace=tmp_path)
    run = Run(goal="Continue", session_id=session.session_id, agent_spec_id="test")
    messages = list(
        ModelMessage("user", ({"type": "text", "text": f"stage {i}: " + "x" * 2000},),
                     run.run_id)
        for i in range(7)
    )
    _HISTORY[session.session_id] = messages
    budget = replace(ContextBudgetPolicy(
        context_window_tokens=100_000, reserved_output_tokens=0,
        pressure_trigger_ratio=0.01, pressure_target_ratio=0.001,
        protected_recent_messages=1, protected_recent_tokens=0,
    ), **policy)
    manager = AgentContextManager(budget, Characters(), condenser)
    ingest(manager, session, messages)
    return manager, session, run


def _messages(session: Session) -> list[ModelMessage]:
    return _HISTORY[session.session_id]


async def _prepare(
    manager: AgentContextManager,
    session: Session,
    run: Run,
) -> None:
    await manager.prepare(
        session,
        run,
        (),
        instructions="",
        raw_message_reader=lambda _session_id, sequence: _messages(session)[sequence],
    )


@run_async
async def test_frontier_background_suffix_and_incremental_growth(tmp_path: Path) -> None:
    condenser = Condenser((episode(0, 1), episode(2, 3)), (), ())
    manager, session, run = setup(tmp_path, condenser)
    original = deepcopy(_messages(session))
    await _prepare(manager, session, run)
    view = manager.capture(session, run)
    assert view.history[:2] == (episode(0, 1), episode(2, 3))
    assert isinstance(view.history[2], RawHistorySegment)
    assert [m.sequence for m in view.history[2].messages] == [4, 5]
    assert condenser.calls[1][1] == view.history[:2]
    assert [c[0] for c in condenser.calls] == ["raw", "raw", "rollup"]
    assert tuple(manager._synchronized_state(run.context_scope_id).messages) == (4, 5, 6)
    assert manager._synchronized_state(run.context_scope_id).message_count == 7
    assert _messages(session) == original
    assert view.protected_recent.messages[0].message == original[6]
    _messages(session).append(ModelMessage("user", ({"type": "text", "text": "new"},)))
    ingest(manager, session, _messages(session)[-1:], start=7)
    await _prepare(manager, session, run)
    target = condenser.calls[3][2]
    assert isinstance(target, tuple)
    assert [item.sequence for item in target] == [4, 5, 6]
    assert tuple(manager._synchronized_state(run.context_scope_id).messages) == (4, 5, 6, 7)
    assert manager._synchronized_state(run.context_scope_id).message_count == 8
    calls = len(condenser.calls)
    manager.build(session, run)
    manager.capture(session, run)
    assert len(condenser.calls) == calls


@run_async
async def test_rollup_after_raw_exhaustion_and_repeated_rollup(tmp_path: Path) -> None:
    condenser = Condenser(tuple(episode(i, i + 1) for i in (0, 2, 4)))
    condenser.rollups.extend((episode(0, 3, "Merged."), episode(0, 5, "Done.")))
    two_budget = len(compact_json(episode_data((episode(0, 1), episode(2, 3)))))
    manager, session, run = setup(tmp_path, condenser, rollup_episode_tokens=two_budget)
    await _prepare(manager, session, run)
    assert [c[0] for c in condenser.calls] == ["raw", "rollup", "rollup"]
    assert manager.capture(session, run).history == (episode(0, 5, "Done."),)
    assert tuple(manager._synchronized_state(run.context_scope_id).messages) == (6,)


@run_async
async def test_raw_no_progress_allows_rollup_without_crossing_suffix(tmp_path: Path) -> None:
    condenser = Condenser((episode(0, 1), episode(2, 3)), ())
    condenser.rollups.append(episode(0, 3, "Done."))
    manager, session, run = setup(tmp_path, condenser)
    await _prepare(manager, session, run)
    assert manager.capture(session, run).history[0] == episode(0, 3, "Done.")
    assert [c[0] for c in condenser.calls] == ["raw", "raw", "rollup"]
    assert condenser.calls[-1][1] == (episode(0, 1), episode(2, 3))


@pytest.mark.parametrize("result", [
    (episode(1, 3),), (episode(0, 1), episode(1, 3)),
    (episode(0, 1), episode(3, 4)), (episode(0, 6),),
    (episode(0, 2, ""),), ModelProtocolError("bad JSON"), ModelProviderError("unavailable"),
])
@run_async
async def test_invalid_response_preserves_view_and_stops(
    tmp_path: Path, result: tuple[HistoryEpisode, ...] | Exception,
) -> None:
    condenser = Condenser(result)
    manager, session, run = setup(tmp_path, condenser)
    before = manager.capture(session, run)
    await _prepare(manager, session, run)
    assert manager.capture(session, run) == before
    assert len(condenser.calls) == 1


def tool_round(failed: bool, text: str) -> tuple[ModelMessage, ModelMessage]:
    return (
        ModelMessage("assistant", ({"type": "tool_use", "id": "same", "name": "test",
                                    "input": {}},), "run"),
        ModelMessage("user", ({"type": "tool_result", "tool_use_id": "same",
                               "is_error": failed, "content": text},), "run"),
    )


@run_async
async def test_masked_success_and_full_failure_and_durable_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KEYLINE_HOME", str(tmp_path / "keyline-agent-home"))
    condenser = Condenser()
    manager, session, run = setup(tmp_path, condenser, max_tool_result_tokens=600)
    _HISTORY[session.session_id] = [
        *tool_round(False, "SUCCESS" * 1000),
        *tool_round(True, "ERROR DETAILS" * 1000),
        ModelMessage("user", ({"type": "text", "text": "recent"},)),
    ]
    _HISTORY[session.session_id] = [
        replace(message, run_id=run.run_id) for message in _messages(session)
    ]
    manager = AgentContextManager(manager.policy, Characters(), condenser)
    ingest(manager, session, _messages(session))
    original = deepcopy(_messages(session))
    store = fresh_store(tmp_path, session)
    store.create_run(session, run, tuple(original))
    await _prepare(manager, session, run)
    target = condenser.calls[0][2]
    assert isinstance(target, tuple)
    assert target[1].message.content[0]["content"].startswith("[stale tool result masked;")
    assert target[3].message.content[0]["content"] == "ERROR DETAILS" * 1000
    assert _messages(session) == original
    assert history(store, session) == original


@run_async
async def test_candidate_budget_and_tool_round_boundaries(tmp_path: Path) -> None:
    condenser = Condenser((episode(0, 0),))  # Cannot split the tool round.
    manager, session, run = setup(tmp_path, condenser)
    _HISTORY[session.session_id] = [
        *tool_round(True, "details" * 100),
        ModelMessage("user", ({"type": "text", "text": "recent"},)),
    ]
    manager = AgentContextManager(manager.policy, Characters(), condenser)
    ingest(manager, session, _messages(session))
    await _prepare(manager, session, run)
    assert all(isinstance(s, RawHistorySegment) for s in manager.capture(session, run).history)
    assert len(condenser.calls) == 1
    target = condenser.calls[0][2]
    assert isinstance(target, tuple)
    manager.policy = replace(manager.policy,
                             raw_target_tokens=len(compact_json(raw_data(target))) - 1)
    await _prepare(manager, session, run)
    assert len(condenser.calls) == 1  # Full failure cannot fit; do not truncate or skip it.


@run_async
async def test_independent_background_raw_and_total_budgets(tmp_path: Path) -> None:
    condenser = Condenser((episode(0, 0), episode(1, 1)), ())
    manager, session, run = setup(tmp_path, condenser, background_episode_tokens=1)
    await _prepare(manager, session, run)
    assert condenser.calls[1][1] == ()
    state = manager._synchronized_state(run.context_scope_id)
    background, target = select_raw_candidate(
        state,
        run_id=run.run_id,
        policy=manager.policy,
        estimator=manager.estimator,
        raw_message_reader=None,
    )
    assert len(target) == 4
    first_size = len(compact_json(raw_data(target[:1])))
    manager.policy = replace(manager.policy, raw_target_tokens=first_size)
    assert len(
        select_raw_candidate(
            state,
            run_id=run.run_id,
            policy=manager.policy,
            estimator=manager.estimator,
            raw_message_reader=None,
        )[1]
    ) == 1
    total = estimate_model_context(condenser_context((), target[:1]), (), Characters())
    manager.policy = replace(manager.policy, condenser_input_tokens=total - 1)
    assert select_raw_candidate(
        state,
        run_id=run.run_id,
        policy=manager.policy,
        estimator=manager.estimator,
        raw_message_reader=None,
    )[1] == ()


@run_async
async def test_masking_reaches_target_without_condenser(tmp_path: Path) -> None:
    condenser = Condenser()
    manager, session, run = setup(tmp_path, condenser, context_window_tokens=15_000,
                                 pressure_trigger_ratio=0.8, pressure_target_ratio=0.5)
    _HISTORY[session.session_id] = [
        *tool_round(False, "x" * 12_000),
        ModelMessage("user", ({"type": "text", "text": "recent"},)),
    ]
    manager = AgentContextManager(manager.policy, Characters(), condenser)
    ingest(manager, session, _messages(session))
    await _prepare(manager, session, run)
    assert not condenser.calls
    assert manager.measure_pressure(session, run, ()) <= 7500


@run_async
async def test_no_gain_and_hard_limit(tmp_path: Path) -> None:
    condenser = Condenser((episode(0, 5, "x" * 30_000),))
    manager, session, run = setup(tmp_path, condenser, context_window_tokens=10_000)
    before = manager.capture(session, run)
    with pytest.raises(ContextPressureError):
        await _prepare(manager, session, run)
    assert manager.capture(session, run) == before


@run_async
async def test_stop_after_raw_reaches_target(tmp_path: Path) -> None:
    condenser = Condenser((episode(0, 5),))
    manager, session, run = setup(tmp_path, condenser, context_window_tokens=17_000,
                                 pressure_trigger_ratio=0.8, pressure_target_ratio=0.5)
    await _prepare(manager, session, run)
    assert [c[0] for c in condenser.calls] == ["raw"]


@run_async
async def test_llm_request_and_json_protocol() -> None:
    model = ScriptedModel([ModelTurn(({"type": "text", "text":
        '{"episodes":[{"start_sequence":0,"end_sequence":1,"title":"Stage",'
        '"summary":"Investigation complete."}]}'},))])
    condenser = LLMHistoryCondenser(model)
    assert await condenser.compact_raw((), ()) == (episode(0, 1),)
    assert model.tool_sets == [()]
    assert model.contexts[0].goal == ""
    assert model.contexts[0].plan == ()


@pytest.mark.parametrize("text,stop", [
    ("not json", None), ('{"episodes":[{}]}', None),
    ('{"episodes":[]}', "max_tokens"),
    ('{"episodes":[{"start_sequence":true,"end_sequence":1,"title":"x",'
     '"summary":"y"}]}', None),
])
@run_async
async def test_llm_rejects_malformed_or_truncated_output(text: str, stop: str | None) -> None:
    model = ScriptedModel([ModelTurn(({"type": "text", "text": text},), stop_reason=stop)])
    with pytest.raises(ModelProtocolError):
        await LLMHistoryCondenser(model).compact_raw((), ())


@run_async
async def test_loop_persists_compaction_and_restart_restores_without_llm(tmp_path: Path) -> None:
    condenser = Condenser((episode(0, 5),))
    manager, session, old_run = setup(tmp_path, condenser)
    store = fresh_store(tmp_path, session)
    old_run.transition(RunStatus.RUNNING)
    old_run.transition(RunStatus.COMPLETED)
    store.create_run(session, old_run, tuple(_messages(session)))
    run = Run(
        goal="Next request",
        session_id=session.session_id,
        sequence=2,
        previous_run_id=old_run.run_id,
        agent_spec_id="test",
    )
    model = ScriptedModel([ModelTurn(({"type": "text", "text": "Done"},),
                                    stop_reason="end_turn")])
    await start_run(
        session,
        run,
        model,
        ToolRegistry(),
        context_manager=manager,
        store=store,
        agent_spec=TEST_AGENT_SPEC,
    )
    assert run.model_turns == 1
    assert condenser.calls
    checkpoint = store.load_context_checkpoint(run.context_scope_id)
    assert checkpoint is not None
    restored_condenser = Condenser()
    restored = AgentContextManager(
        manager.policy, Characters(), restored_condenser,
    )
    restored.restore(
        session.context_scope,
        checkpoint,
        store.message_count(run.context_scope_id),
        lambda start, end: store.read_messages(run.context_scope_id, start, end),
    )
    assert restored.build(
        session, run, instructions=TEST_AGENT_SPEC.instructions
    ) == manager.build(session, run, instructions=TEST_AGENT_SPEC.instructions)
    assert restored.capture(session, run).history[0] == episode(0, 5)
    assert restored_condenser.calls == []
    assert old_run.model_turns == 0


@run_async
async def test_below_trigger_preserves_history_without_requests(tmp_path: Path) -> None:
    condenser = Condenser()
    manager, session, run = setup(tmp_path, condenser, pressure_trigger_ratio=0.8,
                                 pressure_target_ratio=0.5)
    before = manager.capture(session, run)
    await _prepare(manager, session, run)
    assert manager.capture(session, run) == before
    assert not condenser.calls


@pytest.mark.parametrize("merged", [episode(0, 2), episode(1, 3),
                                     episode(0, 3, "x" * 5000)])
@run_async
async def test_invalid_or_larger_rollup_keeps_accepted_raw_compaction(
    tmp_path: Path, merged: HistoryEpisode,
) -> None:
    condenser = Condenser((episode(0, 1), episode(2, 3)), ())
    condenser.rollups.append(merged)
    manager, session, run = setup(tmp_path, condenser)
    await _prepare(manager, session, run)
    assert manager.capture(session, run).history[:2] == (episode(0, 1), episode(2, 3))
    assert len(condenser.calls) == 3


@run_async
async def test_multiple_results_repeated_ids_restore_by_sequence_and_block(tmp_path: Path) -> None:
    condenser = Condenser()
    manager, session, run = setup(tmp_path, condenser)
    first = tool_round(False, "success")
    failure = tool_round(True, "first failure")[1]
    second = tuple(replace(m, run_id="other-run") for m in tool_round(True, "second failure"))
    _HISTORY[session.session_id] = [
        first[0],
        replace(first[1], content=first[1].content + failure.content),
        *second,
        ModelMessage("user", ({"type": "text", "text": "recent"},)),
    ]
    manager = AgentContextManager(manager.policy, Characters(), condenser)
    ingest(manager, session, _messages(session))
    await _prepare(manager, session, run)
    target = condenser.calls[0][2]
    assert isinstance(target, tuple)
    assert "masked" in target[1].message.content[0]["content"]
    assert target[1].message.content[1]["content"] == "first failure"
    assert target[3].message.content[0]["content"] == "second failure"


@run_async
async def test_shared_model_summary_never_enters_transcript_or_turn_count(tmp_path: Path) -> None:
    manager, session, _ = setup(tmp_path, Condenser(), context_window_tokens=20_000,
                                pressure_trigger_ratio=0.8, pressure_target_ratio=0.5)
    model = ScriptedModel([
        ModelTurn(({"type": "text", "text": compact_json({"episodes": [{
            "start_sequence": 0, "end_sequence": 6, "title": "Summary",
            "summary": "Earlier work completed.",
        }]})},), stop_reason="end_turn"),
        ModelTurn(({"type": "text", "text": "Done"},), stop_reason="end_turn"),
    ])
    manager.condenser = LLMHistoryCondenser(model)
    run = Run(goal="Next request", session_id=session.session_id, agent_spec_id="test")
    original = deepcopy(_messages(session))
    store = fresh_store(tmp_path, session)
    old_run = Run(
        goal="Earlier", session_id=session.session_id,
        run_id=original[0].run_id, agent_spec_id="test",
    )
    old_run.transition(RunStatus.RUNNING)
    old_run.transition(RunStatus.COMPLETED)
    store.create_run(session, old_run, tuple(original))
    run.sequence = 2
    run.previous_run_id = old_run.run_id
    await start_run(
        session,
        run,
        model,
        ToolRegistry(),
        context_manager=manager,
        store=store,
        agent_spec=TEST_AGENT_SPEC,
    )
    assert run.model_turns == 1
    assert len(model.contexts) == 2
    assert model.tool_sets[0] == ()
    assert model.tool_sets[1]
    assert "Compacted history episode" in repr(model.contexts[1].messages)
    persisted = history(store, session)
    assert persisted[:7] == original
    assert len(persisted) == 9
