from copy import deepcopy
from pathlib import Path

import pytest

from handshake.context import (
    AgentContextManager,
    ContextView,
    ContextViewError,
    HistoryEpisode,
    RawHistorySegment,
    RecentContext,
    SequencedMessage,
    SourceRange,
)
from handshake.models.protocol import ModelMessage
from handshake.runtime.state import Dependency, DependencyKind, Run, Session
from tests.history_support import ingest


def _session_run(tmp_path: Path) -> tuple[Session, Run]:
    session = Session(workspace=tmp_path)
    run = Run(goal="Make a focused change", session_id=session.session_id)
    return session, run


def test_build_preserves_state_and_includes_complete_context(tmp_path: Path) -> None:
    session, run = _session_run(tmp_path)
    messages = [
            ModelMessage(
                role="user",
                content=({"type": "text", "text": "first request"},),
                run_id="earlier-run",
            ),
            ModelMessage(
                role="assistant",
                content=({"type": "text", "text": "first result"},),
                run_id="earlier-run",
            ),
        ]
    original_messages = list(messages)
    original_plan = list(run.plan)
    manager = AgentContextManager()
    ingest(manager, run, messages)

    context = manager.build(session, run)

    assert context.goal == run.goal
    assert context.workspace == str(tmp_path)
    assert context.plan == ()
    assert context.messages == tuple(original_messages)
    assert "Optional current plan:\n(none)" in context.system_prompt
    assert "Investigate enough repository context" in context.system_prompt
    assert "Simple requests may be completed without creating a plan." in context.system_prompt
    assert "Treat the plan as provisional" in context.system_prompt
    assert "list_files/search_files" in context.system_prompt
    assert "prefer edit_file" in context.system_prompt
    assert "Keep a short current plan" not in context.system_prompt
    assert messages == original_messages
    assert run.plan == original_plan


def test_build_renders_current_plan(tmp_path: Path) -> None:
    session, run = _session_run(tmp_path)
    run.plan.extend(["Inspect the implementation", "Run focused tests"])

    context = AgentContextManager().build(session, run)

    assert context.plan == ("Inspect the implementation", "Run focused tests")
    assert (
        "Optional current plan:\n- Inspect the implementation\n- Run focused tests"
    ) in context.system_prompt


def test_build_is_read_only_and_requires_explicit_ingress(tmp_path: Path) -> None:
    session, run = _session_run(tmp_path)
    manager = AgentContextManager()

    first = manager.build(session, run)

    messages = [
        ModelMessage("user", ({"type": "text", "text": "new input"},), run.run_id)
    ]
    assert manager.build(session, run) == first

    ingest(manager, run, messages)
    original_messages = deepcopy(messages)
    second = manager.build(session, run)
    third = manager.build(session, run)

    assert first.messages == ()
    assert second == third
    assert messages == original_messages


def test_capture_preserves_exact_state_and_sequences_complete_history(
    tmp_path: Path,
) -> None:
    session, run = _session_run(tmp_path)
    dependency = Dependency(
        work_id=run.work_items[0].work_id,
        tool_call_id="ask",
        kind=DependencyKind.TEXT,
        prompt="Which version must remain supported?",
    )
    run.add_dependency(run.work_items[0], dependency)
    run.plan.append("Check compatibility")
    messages = [
            ModelMessage("user", ({"type": "text", "text": "request"},), "prior"),
            ModelMessage("assistant", ({"type": "text", "text": "response"},), "prior"),
        ]
    manager = AgentContextManager()
    ingest(manager, run, messages)
    view = manager.capture(
        session,
        run,
        constraints=("Preserve the public API",),
    )
    run.plan.clear()
    dependency.prompt = "changed after capture"

    context = manager.render(view)

    assert view.exact.plan == ("Check compatibility",)
    assert view.exact.dependencies[0].prompt == "Which version must remain supported?"
    assert view.exact.dependencies[0].dependency_id == dependency.dependency_id
    assert view.exact.dependencies[0].status == "open"
    assert view.exact.work_items[0].work_id == run.work_items[0].work_id
    assert view.history == ()
    assert view.source_message_count == 2
    assert tuple(item.sequence for item in view.protected_recent.messages) == (0, 1)
    assert tuple(item.message for item in view.protected_recent.messages) == tuple(messages)
    assert "Preserve the public API" in context.system_prompt
    assert dependency.dependency_id in context.system_prompt
    assert context.messages == tuple(messages)


def test_capture_and_render_do_not_alias_session_history(tmp_path: Path) -> None:
    session, run = _session_run(tmp_path)
    messages = [
        ModelMessage(
            "user",
            ({"type": "text", "text": "important original evidence"},),
            run.run_id,
        )
    ]
    manager = AgentContextManager()
    ingest(manager, run, messages)
    view = manager.capture(session, run)
    context = manager.render(view)

    assert context.messages == tuple(messages)
    context.messages[0].content[0]["text"] = "provider mutation"
    captured = view.protected_recent.messages[0].message
    assert captured.content[0]["text"] == "important original evidence"
    messages[0].content[0]["text"] = "later transcript mutation"
    assert captured.content[0]["text"] == "important original evidence"


def test_render_interleaves_raw_history_episode_and_protected_recent(
    tmp_path: Path,
) -> None:
    session, run = _session_run(tmp_path)
    exact = AgentContextManager().capture(session, run).exact
    source = tuple(
        SequencedMessage(
            sequence,
            ModelMessage(
                "user" if sequence % 2 == 0 else "assistant",
                ({"type": "text", "text": f"message {sequence}"},),
                "source-run",
            ),
        )
        for sequence in range(5)
    )
    view = ContextView(
        exact=exact,
        history=(
            RawHistorySegment(source[:2]),
            HistoryEpisode(
                SourceRange(2, 3),
                "Completed adapter investigation",
                "Confirmed the fallback is required and its focused tests passed.",
            ),
        ),
        protected_recent=RecentContext(source[4:]),
        source_message_count=5,
    )

    context = AgentContextManager().render(view)

    assert [message.content[0]["text"] for message in context.messages] == [
        "message 0",
        "message 1",
        (
            "[Compacted history episode; source messages 2-3]\n"
            "Title: Completed adapter investigation\n"
            "Summary: Confirmed the fallback is required and its focused tests passed."
        ),
        "message 4",
    ]
    assert context.messages[2].role == "user"
    assert context.messages[2].run_id == ""
    context.messages[0].content[0]["text"] = "mutated"
    assert source[0].message.content[0]["text"] == "message 0"


def _invalid_view(tmp_path: Path, case: str) -> ContextView:
    session, run = _session_run(tmp_path)
    exact = AgentContextManager().capture(session, run).exact

    def item(sequence: int) -> SequencedMessage:
        return SequencedMessage(
            sequence,
            ModelMessage("user", ({"type": "text", "text": str(sequence)},)),
        )

    if case == "missing":
        return ContextView(
            exact,
            protected_recent=RecentContext((item(1),)),
            source_message_count=2,
        )
    if case == "overlap":
        return ContextView(
            exact,
            history=(HistoryEpisode(SourceRange(0, 1), "done", "done"),),
            protected_recent=RecentContext((item(1),)),
            source_message_count=2,
        )
    if case == "out_of_order":
        return ContextView(
            exact,
            history=(
                HistoryEpisode(SourceRange(1, 1), "later", "later"),
                HistoryEpisode(SourceRange(0, 0), "earlier", "earlier"),
            ),
            source_message_count=2,
        )
    if case == "noncontinuous_raw":
        return ContextView(
            exact,
            history=(RawHistorySegment((item(0), item(2))),),
            source_message_count=3,
        )
    if case == "out_of_bounds":
        return ContextView(
            exact,
            history=(HistoryEpisode(SourceRange(0, 2), "too far", "too far"),),
            source_message_count=2,
        )
    if case == "invalid_range":
        return ContextView(
            exact,
            history=(HistoryEpisode(SourceRange(1, 0), "invalid", "invalid"),),
        )
    if case == "empty_raw":
        return ContextView(exact, history=(RawHistorySegment(()),))
    return ContextView(exact, source_message_count=-1)


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "overlap",
        "out_of_order",
        "noncontinuous_raw",
        "out_of_bounds",
        "invalid_range",
        "empty_raw",
        "negative_count",
    ],
)
def test_render_rejects_invalid_context_views(tmp_path: Path, case: str) -> None:
    with pytest.raises(ContextViewError):
        AgentContextManager().render(_invalid_view(tmp_path, case))
