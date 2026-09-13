from pathlib import Path

from handshake.models.protocol import ModelMessage
from handshake.runtime.context import AgentContextManager
from handshake.runtime.state import Run, Session


def _session_run(tmp_path: Path) -> tuple[Session, Run]:
    session = Session(workspace=tmp_path)
    run = Run(goal="Make a focused change", session_id=session.session_id)
    return session, run


def test_build_preserves_state_and_includes_complete_context(tmp_path: Path) -> None:
    session, run = _session_run(tmp_path)
    session.messages.extend(
        [
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
    )
    original_messages = list(session.messages)
    original_plan = list(run.plan)
    manager = AgentContextManager()

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
    assert session.messages == original_messages
    assert run.plan == original_plan


def test_build_renders_current_plan(tmp_path: Path) -> None:
    session, run = _session_run(tmp_path)
    run.plan.extend(["Inspect the implementation", "Run focused tests"])

    context = AgentContextManager().build(session, run)

    assert context.plan == ("Inspect the implementation", "Run focused tests")
    assert (
        "Optional current plan:\n"
        "- Inspect the implementation\n"
        "- Run focused tests"
    ) in context.system_prompt
