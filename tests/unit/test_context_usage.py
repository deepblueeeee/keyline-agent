from dataclasses import replace
from pathlib import Path

import pytest

from keyline_agent.context import (
    AgentContextManager,
    ContextBudgetPolicy,
    ContextPressureError,
    HistoryEpisode,
    SequencedMessage,
    SourceRange,
)
from keyline_agent.context.budget import (
    Utf8TokenEstimator,
    compact_json,
    estimate_model_context,
    measure_model_context,
    measure_model_turn,
)
from keyline_agent.models.protocol import (
    ModelContext,
    ModelMessage,
    ModelTurn,
    ModelUsage,
    ToolCall,
)
from keyline_agent.models.scripted import ScriptedModel
from keyline_agent.runtime.dependencies import resolve_text_dependency
from keyline_agent.runtime.loop import continue_run, start_run
from keyline_agent.runtime.state import Run, RunStatus, Session
from keyline_agent.tools.registry import ToolRegistry
from tests.history_support import TEST_AGENT_SPEC, fresh_store, ingest, run_async


def text_message(text: str, role: str = "user") -> ModelMessage:
    return ModelMessage(
        "assistant" if role == "assistant" else "user", ({"type": "text", "text": text},)
    )


def test_measured_history_new_inputs_and_next_response_correction() -> None:
    estimator = Utf8TokenEstimator()
    context = ModelContext("system", "goal", "/tmp", (), (text_message("old"),))
    turn = ModelTurn(({"type": "text", "text": "reply"},), usage=ModelUsage(100, 20, 30, 40))
    measurement = measure_model_turn(context, (), turn, estimator)
    after = replace(context, messages=(*context.messages, text_message("reply", "assistant")))
    assert measure_model_context(after, (), estimator, measurement) == 190

    tool_result = ModelMessage(
        "user",
        (
            {
                "type": "tool_result",
                "tool_use_id": "call",
                "content": "result",
            },
        ),
    )
    added = (tool_result, text_message("next input"))
    next_context = replace(after, messages=(*after.messages, *added))
    expected_delta = sum(estimator.estimate_text(compact_json(item.as_dict())) for item in added)
    assert measure_model_context(next_context, (), estimator, measurement) == 190 + expected_delta

    # The new input count includes the whole request: replace rather than sum baselines.
    next_turn = ModelTurn(({"type": "text", "text": "done"},), usage=ModelUsage(240, 10))
    corrected = measure_model_turn(next_context, (), next_turn, estimator)
    corrected_context = replace(
        next_context,
        messages=(
            *next_context.messages,
            text_message("done", "assistant"),
        ),
    )
    assert measure_model_context(corrected_context, (), estimator, corrected) == 250


def test_changed_header_is_estimated_without_recounting_measured_history() -> None:
    estimator = Utf8TokenEstimator()
    context = ModelContext("system", "goal", "/tmp", (), (text_message("old"),))
    turn = ModelTurn(({"type": "text", "text": "reply"},), usage=ModelUsage(100, 20))
    measurement = measure_model_turn(context, (), turn, estimator)
    after = replace(context, messages=(*context.messages, text_message("reply", "assistant")))
    changed = replace(after, system_prompt="system with updated state")
    delta = estimate_model_context(changed, (), estimator) - estimate_model_context(
        after, (), estimator
    )
    assert abs(measure_model_context(changed, (), estimator, measurement) - (120 + delta)) <= 1


def test_replaced_or_compressed_history_uses_estimate_until_recalibrated() -> None:
    estimator = Utf8TokenEstimator()
    context = ModelContext("system", "goal", "/tmp", (), (text_message("large history"),))
    turn = ModelTurn(({"type": "text", "text": "reply"},), usage=ModelUsage(9000, 200))
    measurement = measure_model_turn(context, (), turn, estimator)
    for messages in (
        (text_message("summary"),),
        (text_message("masked history"), text_message("reply", "assistant")),
    ):
        changed = replace(context, messages=messages)
        assert measure_model_context(changed, (), estimator, measurement) == (
            estimate_model_context(changed, (), estimator)
        )
        next_turn = ModelTurn(({"type": "text", "text": "continued"},), usage=ModelUsage(50, 5))
        recalibrated = measure_model_turn(changed, (), next_turn, estimator)
        after = replace(changed, messages=(*messages, text_message("continued", "assistant")))
        assert measure_model_context(after, (), estimator, recalibrated) == 55


def test_missing_usage_uses_local_estimator() -> None:
    estimator = Utf8TokenEstimator()
    context = ModelContext("system", "goal", "/tmp", (), (text_message("input"),))
    turn = ModelTurn(({"type": "text", "text": "reply"},))
    assert measure_model_turn(context, (), turn, estimator) is None
    assert measure_model_context(context, (), estimator, None) == (
        estimate_model_context(context, (), estimator)
    )


@run_async
async def test_usage_survives_wait_checkpoint_and_process_recovery(tmp_path: Path) -> None:
    session = Session(workspace=tmp_path)
    run = Run(goal="ask", session_id=session.session_id, agent_spec_id="test")
    store = fresh_store(tmp_path, session)
    wait_turn = ModelTurn(
        (
            {
                "type": "tool_use",
                "id": "ask",
                "name": "wait_for_text",
                "input": {"prompt": "Which color?"},
            },
        ),
        (ToolCall("ask", "wait_for_text", {"prompt": "Which color?"}),),
        "tool_use",
        ModelUsage(10000, 100),
    )
    model = ScriptedModel((wait_turn,))
    manager = AgentContextManager()
    await start_run(
        session, run, model, ToolRegistry(), store=store,
        context_manager=manager, agent_spec=TEST_AGENT_SPEC,
    )
    assert run.status is RunStatus.WAITING
    checkpoint = store.load_context_checkpoint(run.context_scope_id)
    assert checkpoint is not None and checkpoint.token_measurement is not None
    assert checkpoint.token_measurement.tokens == 10100

    restored = AgentContextManager()
    restored.restore(
        session.context_scope,
        checkpoint,
        store.message_count(run.context_scope_id),
        lambda start, end: store.read_messages(run.context_scope_id, start, end),
    )
    assert restored.measure_pressure(
        session, run, model.tool_sets[0], instructions=TEST_AGENT_SPEC.instructions
    ) == (
        manager.measure_pressure(
            session, run, model.tool_sets[0], instructions=TEST_AGENT_SPEC.instructions
        )
    )
    resumed_model = ScriptedModel(
        (
            ModelTurn(
                ({"type": "text", "text": "done"},),
                stop_reason="end_turn",
                usage=ModelUsage(11000, 5),
            ),
        )
    )
    continuation = resolve_text_dependency(
        session,
        run,
        run.open_dependencies()[0].dependency_id,
        "blue",
    )
    completed = await continue_run(
        session,
        run,
        resumed_model,
        ToolRegistry(),
        store=store,
        context_manager=restored,
        pending_messages=(continuation,),
        agent_spec=TEST_AGENT_SPEC,
    )
    assert completed.status is RunStatus.COMPLETED
    saved = store.load_context_checkpoint(run.context_scope_id)
    assert saved is not None and saved.token_measurement is not None
    assert saved.token_measurement.tokens == 11005


@run_async
async def test_low_actual_usage_avoids_unnecessary_compaction(tmp_path: Path) -> None:
    session = Session(workspace=tmp_path)
    run = Run(goal="continue", session_id=session.session_id, agent_spec_id="test")
    policy = ContextBudgetPolicy(context_window_tokens=1000, reserved_output_tokens=0)
    manager = AgentContextManager(policy)
    ingest(manager, session, (text_message("x" * 10000),))
    context = manager.build(session, run)
    turn = ModelTurn(({"type": "text", "text": "reply"},), usage=ModelUsage(100, 10))
    manager.record_usage(session.context_scope, context, (), turn)
    ingest(manager, session, (text_message("reply", "assistant"),), start=1)
    assert estimate_model_context(manager.build(session, run), (), manager.estimator) > 1000
    assert manager.measure_pressure(session, run, ()) == 110
    assert (await manager.prepare(
        session, run, (), instructions=""
    )).messages == manager.build(session, run).messages


@run_async
async def test_high_actual_usage_triggers_semantic_compaction_even_with_low_estimate(
    tmp_path: Path,
) -> None:
    class Condenser:
        calls = 0

        async def compact_raw(
            self,
            background: tuple[HistoryEpisode, ...],
            target: tuple[SequencedMessage, ...],
        ) -> tuple[HistoryEpisode, ...]:
            self.calls += 1
            return (
                HistoryEpisode(
                    SourceRange(target[0].sequence, target[-1].sequence),
                    "Old work",
                    "Done.",
                ),
            )

        async def roll_up(self, episodes: tuple[HistoryEpisode, ...]) -> HistoryEpisode | None:
            return None

    session = Session(workspace=tmp_path)
    run = Run(goal="continue", session_id=session.session_id, agent_spec_id="test")
    policy = ContextBudgetPolicy(
        context_window_tokens=10000,
        reserved_output_tokens=0,
        protected_recent_messages=1,
        protected_recent_tokens=0,
    )
    condenser = Condenser()
    manager = AgentContextManager(policy, condenser=condenser)
    ingest(manager, session, (text_message("x" * 2000),))
    context = manager.build(session, run)
    turn = ModelTurn(({"type": "text", "text": "reply"},), usage=ModelUsage(8900, 100))
    manager.record_usage(session.context_scope, context, (), turn)
    ingest(manager, session, (text_message("reply", "assistant"),), start=1)
    assert estimate_model_context(manager.build(session, run), (), manager.estimator) < 5000
    assert manager.measure_pressure(session, run, ()) == 9000
    prepared = await manager.prepare(session, run, (), instructions="")
    assert condenser.calls == 1
    assert "Compacted history episode" in prepared.messages[0].content[0]["text"]
    assert manager.measure_pressure(session, run, ()) == (
        estimate_model_context(prepared, (), manager.estimator)
    )


@run_async
async def test_actual_usage_enforces_hard_limit_for_protected_context(tmp_path: Path) -> None:
    session = Session(workspace=tmp_path)
    run = Run(goal="continue", session_id=session.session_id, agent_spec_id="test")
    manager = AgentContextManager(
        ContextBudgetPolicy(
            context_window_tokens=10000,
            reserved_output_tokens=0,
        )
    )
    ingest(manager, session, (text_message("history"),))
    context = manager.build(session, run)
    turn = ModelTurn(({"type": "text", "text": "reply"},), usage=ModelUsage(9990, 10))
    manager.record_usage(session.context_scope, context, (), turn)
    ingest(manager, session, (text_message("reply", "assistant"),), start=1)
    with pytest.raises(ContextPressureError):
        await manager.prepare(session, run, (), instructions="")


@run_async
async def test_prepare_measures_the_instructions_passed_for_this_request(tmp_path: Path) -> None:
    session = Session(workspace=tmp_path)
    run = Run(goal="review", session_id=session.session_id, agent_spec_id="reviewer")
    manager = AgentContextManager()
    short_instructions = "Review the change."
    long_instructions = "Check every detail. " * 300
    short_pressure = manager.measure_pressure(
        session, run, (), instructions=short_instructions
    )
    long_pressure = manager.measure_pressure(
        session, run, (), instructions=long_instructions
    )
    assert short_pressure < long_pressure

    manager.policy = ContextBudgetPolicy(
        context_window_tokens=long_pressure, reserved_output_tokens=0
    )
    assert short_instructions in (await manager.prepare(
        session, run, (), instructions=short_instructions
    )).system_prompt
    with pytest.raises(ContextPressureError):
        await manager.prepare(session, run, (), instructions=long_instructions)
