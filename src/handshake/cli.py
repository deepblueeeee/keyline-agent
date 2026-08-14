"""CLI for real-model tasks, durable text waits, and the scripted demo."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from handshake.config import DeepSeekConfig, handshake_home, read_deepseek_api_key
from handshake.models.base import ModelAdapter
from handshake.models.deepseek_anthropic import DeepSeekAnthropicModel
from handshake.models.scripted import ScriptedModel
from handshake.runtime.errors import HandshakeError, UserCancelledError
from handshake.runtime.loop import continue_agent, run_agent
from handshake.runtime.sessions import (
    build_session_history,
    create_followup_task,
    create_initial_task,
    latest_task,
)
from handshake.runtime.state import (
    ModelTurn,
    Session,
    SessionHistoryEntry,
    Task,
    TaskStatus,
    ToolCall,
    WaitCondition,
)
from handshake.runtime.waits import resolve_text_wait
from handshake.storage import SQLiteStateStore
from handshake.tools import create_default_registry
from handshake.tools.registry import ToolRegistry

_WAITING_EXIT_CODE = 2


def _interactive_approval(cwd: Path, argv: tuple[str, ...]) -> bool:
    print("\nApproval required")
    print(f"  cwd:  {cwd}")
    print(f"  argv: {shlex.join(argv)}")
    if not sys.stdin.isatty():
        print("Denied: stdin is not interactive")
        return False
    try:
        answer = input("Execute this command once? [y/N] ")
    except (EOFError, KeyboardInterrupt) as exc:
        print()
        raise UserCancelledError("User cancelled shell approval") from exc
    return answer.strip().lower() in {"y", "yes"}


def _state_store() -> SQLiteStateStore:
    return SQLiteStateStore(handshake_home() / "handshake.db")


def _create_real_model(model_name: str) -> ModelAdapter:
    config = DeepSeekConfig.from_environment(model_override=model_name)
    api_key = read_deepseek_api_key()
    model = DeepSeekAnthropicModel(
        api_key=api_key,
        base_url=config.base_url,
        model=config.model,
        max_tokens=config.max_tokens,
        timeout_seconds=config.timeout_seconds,
        max_retries=config.max_retries,
    )
    del api_key
    return model


def _single_open_wait(task: Task) -> WaitCondition:
    waits = task.open_waits()
    if len(waits) != 1:
        raise HandshakeError(
            f"Task {task.task_id} must have exactly one open text wait, found {len(waits)}"
        )
    return waits[0]


def _resume_command(session: Session) -> str:
    return f"handshake resume {session.session_id}"


def _read_wait_response(
    session: Session, task: Task, wait: WaitCondition
) -> str | None:
    print("\nInput required")
    print(f"  session:  {session.session_id}")
    print(f"  task:     {task.task_id}")
    print(f"  wait:     {wait.wait_id}")
    print(f"  question: {wait.prompt}")
    if not sys.stdin.isatty():
        print("Task remains waiting because stdin is not interactive.")
        print(f"Resume with: {_resume_command(session)}")
        return None
    try:
        return input("Response: ")
    except EOFError:
        print("\nTask remains waiting; no response was recorded.")
        print(f"Resume with: {_resume_command(session)}")
        return None
    except KeyboardInterrupt as exc:
        print("\nTask remains waiting; no response was recorded.")
        print(f"Resume with: {_resume_command(session)}")
        raise UserCancelledError("User interrupted text input") from exc


def _print_final_summary(task: Task) -> None:
    print("\nFinal summary")
    print(task.final_summary or "(none)")


def _drive_task(
    session: Session,
    task: Task,
    model: ModelAdapter,
    registry: ToolRegistry,
    store: SQLiteStateStore,
    session_history: tuple[SessionHistoryEntry, ...],
) -> tuple[int, Task]:
    while task.status is TaskStatus.WAITING:
        wait = _single_open_wait(task)
        response = _read_wait_response(session, task, wait)
        if response is None:
            return _WAITING_EXIT_CODE, task
        if not response.strip():
            print("Response must not be empty; the task is still waiting.")
            continue
        resolve_text_wait(task, wait.wait_id, response)
        store.checkpoint_task(task)
        print("Input received; resuming task")
        task = continue_agent(
            task,
            model,
            registry,
            emit=print,
            store=store,
            session_history=session_history,
        )
    if task.status is not TaskStatus.COMPLETED:
        raise HandshakeError(
            f"Task {task.task_id} stopped in unsupported state {task.status.value}"
        )
    _print_final_summary(task)
    return 0, task


def _print_terminal_task(task: Task) -> int:
    if task.status is TaskStatus.COMPLETED:
        _print_final_summary(task)
        return 0
    if task.status in {TaskStatus.FAILED, TaskStatus.CANCELLED}:
        print(f"\nTask {task.status.value}")
        print(task.error or "(no error recorded)")
        return 1
    raise HandshakeError(
        f"Task {task.task_id} cannot be restored from {task.status.value}; "
        "only stable waiting or terminal tasks are supported"
    )


def _read_next_goal(session: Session) -> str | None:
    if not sys.stdin.isatty():
        return None
    print(f"\nSession {session.session_id} is ready for the next task.")
    while True:
        try:
            goal = input("You: ").strip()
        except EOFError:
            print()
            return None
        except KeyboardInterrupt as exc:
            print()
            raise UserCancelledError("User exited the session prompt") from exc
        if goal == "/exit":
            return None
        if goal:
            return goal
        print("Task goal must not be empty. Enter /exit to leave the session.")


def _task_history(
    session: Session, task: Task, store: SQLiteStateStore
) -> tuple[SessionHistoryEntry, ...]:
    tasks = store.list_tasks(session.session_id)
    return build_session_history(
        session,
        tasks,
        before_sequence=task.sequence,
    )


def _process_current_task(
    session: Session,
    task: Task,
    registry: ToolRegistry,
    store: SQLiteStateStore,
    model: ModelAdapter | None = None,
) -> tuple[int, Task]:
    if task.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}:
        return _print_terminal_task(task), task
    if task.status not in {TaskStatus.CREATED, TaskStatus.WAITING}:
        return _print_terminal_task(task), task

    history = _task_history(session, task, store)
    current_model = model or _create_real_model(task.model_name)
    try:
        if task.status is TaskStatus.CREATED:
            task = run_agent(
                task,
                current_model,
                registry,
                max_steps=task.max_steps,
                emit=print,
                store=store,
                session_history=history,
            )
        return _drive_task(session, task, current_model, registry, store, history)
    except UserCancelledError:
        raise
    except HandshakeError as exc:
        try:
            persisted = store.load_task(task.task_id)
        except HandshakeError:
            raise exc from None
        if persisted.status is not TaskStatus.FAILED:
            raise
        return _print_terminal_task(persisted), persisted


def _drive_session(
    session: Session,
    task: Task,
    registry: ToolRegistry,
    store: SQLiteStateStore,
    initial_model: ModelAdapter | None = None,
) -> int:
    result_code, task = _process_current_task(
        session, task, registry, store, initial_model
    )
    if result_code == _WAITING_EXIT_CODE or not sys.stdin.isatty():
        return result_code

    while True:
        goal = _read_next_goal(session)
        if goal is None:
            return 0
        tasks = store.list_tasks(session.session_id)
        task = create_followup_task(session, tasks, goal)
        result_code, task = _process_current_task(session, task, registry, store)
        if result_code == _WAITING_EXIT_CODE:
            return result_code


def _run_real(args: argparse.Namespace) -> int:
    workspace = Path(args.repo).expanduser().resolve()
    if not workspace.is_dir():
        raise HandshakeError(f"Repository directory does not exist: {workspace}")
    config = DeepSeekConfig.from_environment(model_override=args.model)
    model = _create_real_model(config.model)
    store = _state_store()
    session = Session(workspace=workspace)
    task = create_initial_task(
        session,
        args.task,
        model_name=config.model,
        max_steps=args.max_steps,
    )
    store.create_session(session)
    print(f"Session created: {session.session_id}")
    registry = create_default_registry(_interactive_approval)
    return _drive_session(session, task, registry, store, model)


def _resume_session(args: argparse.Namespace) -> int:
    store = _state_store()
    session = store.load_session(args.session_id)
    if not session.workspace.is_dir():
        raise HandshakeError(f"Repository directory does not exist: {session.workspace}")
    task = latest_task(session, store.list_tasks(session.session_id))
    registry = create_default_registry(_interactive_approval)
    return _drive_session(session, task, registry, store)


def _list_sessions(_args: argparse.Namespace) -> int:
    store = _state_store()
    sessions = store.list_sessions()
    if not sessions:
        print("No durable sessions.")
        return 0
    for session in sessions:
        tasks = store.list_tasks(session.session_id)
        if not tasks:
            print(f"{session.session_id}  empty      {session.created_at}  [{session.workspace}]")
            continue
        task = latest_task(session, tasks)
        goal = task.goal.replace("\n", " ")
        if len(goal) > 60:
            goal = goal[:57] + "..."
        print(
            f"{session.session_id}  {task.status.value:<9}  {task.updated_at}  "
            f"{goal}  [{session.workspace}]"
        )
    return 0


def _inspect_session(args: argparse.Namespace) -> int:
    store = _state_store()
    session = store.load_session(args.session_id)
    tasks = store.list_tasks(session.session_id)
    print(f"Session:   {session.session_id}")
    print(f"Workspace: {session.workspace}")
    print(f"Created:   {session.created_at}")
    print("Tasks:")
    for task in tasks:
        goal = task.goal.replace("\n", " ")
        print(
            f"  - #{task.sequence} {task.task_id} [{task.status.value}] "
            f"{task.model_turns}/{task.max_steps} {goal}"
        )
        for wait in task.open_waits():
            print(f"      wait {wait.wait_id} [{wait.kind.value}] {wait.prompt}")
    return 0


def _inspect_task(args: argparse.Namespace) -> int:
    task = _state_store().load_task(args.task_id)
    print(f"Task:        {task.task_id}")
    print(f"Session:     {task.session_id}")
    print(f"Sequence:    {task.sequence}")
    print(f"Status:      {task.status.value}")
    print(f"Goal:        {task.goal}")
    print(f"Workspace:   {task.workspace}")
    print(f"Model:       {task.model_name or '(not recorded)'}")
    print(f"Model turns: {task.model_turns}/{task.max_steps}")
    print(f"Messages:    {len(task.messages)}")
    print("Plan:")
    if task.plan:
        for step in task.plan:
            print(f"  - {step}")
    else:
        print("  (not set)")
    print("Waits:")
    if task.waits:
        for wait in task.waits:
            print(f"  - {wait.wait_id} [{wait.kind.value}/{wait.status.value}] {wait.prompt}")
    else:
        print("  (none)")
    if task.error:
        print(f"Error:       {task.error}")
    return 0


def _tool_turn(*calls: ToolCall, text: str = "") -> ModelTurn:
    content: list[dict[str, object]] = []
    if text:
        content.append({"type": "text", "text": text})
    content.extend(
        {
            "type": "tool_use",
            "id": call.id,
            "name": call.name,
            "input": call.arguments,
        }
        for call in calls
    )
    return ModelTurn(content=tuple(content), tool_calls=calls, stop_reason="tool_use")


def _demo_model() -> ScriptedModel:
    implementation = '''"""Tiny demo module for Handshake."""


def add(left: int, right: int) -> int:
    """Return the sum of two integers."""
    return left + right
'''
    return ScriptedModel(
        [
            _tool_turn(
                ToolCall(
                    "plan-1",
                    "update_plan",
                    {"steps": ["Inspect the stub", "Implement add", "Run tests"]},
                )
            ),
            _tool_turn(
                ToolCall("read-1", "read_file", {"path": "calculator.py"}),
                ToolCall(
                    "search-1",
                    "search_files",
                    {"query": "NotImplementedError", "path": "."},
                ),
                text="I will inspect the bundled fixture.",
            ),
            _tool_turn(
                ToolCall(
                    "write-1",
                    "write_file",
                    {"path": "calculator.py", "content": implementation},
                )
            ),
            _tool_turn(
                ToolCall(
                    "test-1",
                    "run_tests",
                    {"argv": [sys.executable, "-m", "pytest", "-q"]},
                )
            ),
            ModelTurn(
                content=(
                    {
                        "type": "text",
                        "text": "Implemented calculator.add and verified the fixture tests pass.",
                    },
                ),
                stop_reason="end_turn",
            ),
        ]
    )


def _run_demo(_args: argparse.Namespace) -> int:
    fixture = Path(__file__).resolve().parents[2] / "examples" / "minimal_task"
    if not fixture.is_dir():
        raise HandshakeError(f"Bundled demo fixture is missing: {fixture}")
    with tempfile.TemporaryDirectory(prefix="handshake-demo-") as temporary:
        workspace = Path(temporary) / "workspace"
        shutil.copytree(fixture, workspace)
        print(f"Running isolated scripted demo in {workspace}")
        session = Session(workspace=workspace)
        task = Task(
            goal="Implement the missing calculator.add function",
            workspace=workspace,
            session_id=session.session_id,
            model_name="scripted",
        )
        registry = create_default_registry(lambda _cwd, _argv: True)
        run_agent(task, _demo_model(), registry, emit=print)
        _print_final_summary(task)
        return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="handshake")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="start a durable DeepSeek session")
    run_parser.add_argument("repo", help="repository workspace")
    run_parser.add_argument("task", help="natural-language coding task")
    run_parser.add_argument(
        "--model",
        default=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        help="DeepSeek model name",
    )
    run_parser.add_argument("--max-steps", type=int, default=50)
    run_parser.set_defaults(handler=_run_real)

    resume_parser = subparsers.add_parser("resume", help="resume a durable session")
    resume_parser.add_argument("session_id", help="durable session ID")
    resume_parser.set_defaults(handler=_resume_session)

    list_parser = subparsers.add_parser("list", help="list durable sessions")
    list_parser.set_defaults(handler=_list_sessions)

    inspect_parser = subparsers.add_parser("inspect", help="inspect durable session state")
    inspect_parser.add_argument("session_id", help="durable session ID")
    inspect_parser.set_defaults(handler=_inspect_session)

    inspect_task_parser = subparsers.add_parser(
        "inspect-task", help="inspect one durable task"
    )
    inspect_task_parser.add_argument("task_id", help="durable task ID")
    inspect_task_parser.set_defaults(handler=_inspect_task)

    demo_parser = subparsers.add_parser("demo", help="run the offline scripted demo")
    demo_parser.set_defaults(handler=_run_demo)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        handler = args.handler
        return int(handler(args))
    except UserCancelledError as exc:
        print(f"Cancelled: {exc}", file=sys.stderr)
        return 130
    except HandshakeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
