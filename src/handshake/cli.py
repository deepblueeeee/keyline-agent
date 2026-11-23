"""CLI for durable model runs, routed dependencies, and scripted demos."""

from __future__ import annotations

import argparse
import asyncio
import getpass
import inspect
import os
import readline  # noqa: F401  # Enable line editing for interactive input().
import shlex
import shutil
import sys
import tempfile
import warnings
from collections.abc import Awaitable, Mapping, Sequence
from functools import partial
from pathlib import Path

from handshake.cli_input import TerminalInput, read_terminal_line
from handshake.config import (
    ApprovalMode,
    handshake_home,
    load_runtime_settings,
    read_deepseek_api_key,
    requires_approval,
    save_deepseek_api_key,
    settings_path,
)
from handshake.config.settings import RuntimeSettings
from handshake.context import ActiveContextManagers, AgentContextManager
from handshake.errors import HandshakeError
from handshake.inputs import EphemeralSecretStore, SecretWriter
from handshake.models.base import ModelAdapter, close_model
from handshake.models.deepseek_anthropic import DeepSeekAnthropicModel
from handshake.models.protocol import ModelMessage, ModelTurn, ToolCall
from handshake.models.scripted import ScriptedModel
from handshake.runtime.agents import (
    AgentSpec,
    child_agent_specs,
    demo_agent_spec,
    main_agent_spec,
    model_name_for_run,
)
from handshake.runtime.delegation import DelegationRuntime, continue_delegation
from handshake.runtime.dependencies import resolve_text_dependency
from handshake.runtime.errors import AgentLoopError, UserCancelledError
from handshake.runtime.loop import (
    context_manager_for_run,
    continue_run,
    prepare_run_for_resume,
    start_run,
)
from handshake.runtime.persistence import RunStore
from handshake.runtime.progress import RunReporter, current_run, format_run_message
from handshake.runtime.recovery import resume_run
from handshake.runtime.runs import (
    create_followup_run,
    create_initial_run,
    latest_run,
)
from handshake.runtime.scheduler import CooperativeScheduler
from handshake.runtime.secret_dependencies import resolve_secret_dependency
from handshake.runtime.state import (
    Dependency,
    DependencyKind,
    Run,
    RunStatus,
    Session,
)
from handshake.storage import SQLiteStateStore, StorageError
from handshake.tools import create_default_registry, runtime_tool_definitions
from handshake.tools.registry import ToolRegistry
from handshake.tools.secret_command import SecretCommandRunner
from handshake.tools.shell import ApprovalCallback, AsyncApprovalCallback

_WAITING_EXIT_CODE = 2


def _agent_specs(settings: RuntimeSettings, registry: ToolRegistry) -> dict[str, AgentSpec]:
    main = main_agent_spec(
        settings.provider.model,
        (definition.name for definition in (*runtime_tool_definitions(), *registry.definitions())),
        settings.context,
    )
    specs = (main, *child_agent_specs(settings.provider.model, settings.context))
    return {spec.spec_id: spec for spec in specs}


def _spec_for_run(run: Run, specs: Mapping[str, AgentSpec]) -> AgentSpec:
    try:
        spec = specs[run.agent_spec_id]
    except KeyError:
        raise AgentLoopError(f"Agent spec {run.agent_spec_id} is unavailable") from None
    model_name_for_run(run, spec)
    return spec


def _interactive_approval(cwd: Path, argv: tuple[str, ...], *, with_secret: bool = False) -> bool:
    details = f"Approval required\n  cwd: {cwd}\n  argv: {shlex.join(argv)}"
    if with_secret:
        details += "\n  A one-time Secret will be supplied to this command."
    run = current_run()
    print(format_run_message(run, details, waiting="approval") if run else details, flush=True)
    if not sys.stdin.isatty():
        print("Denied: stdin is not interactive")
        return False
    try:
        answer = _read_approval_answer("Execute this command once? [y/N] ")
    except (EOFError, KeyboardInterrupt) as exc:
        print()
        raise UserCancelledError("User cancelled shell approval") from exc
    return answer.strip().lower() in {"y", "yes"}


def _read_approval_answer(prompt: str) -> str:
    return read_terminal_line(prompt)


def _approval_callback(
    mode: ApprovalMode, *, with_secret: bool = False, terminal: TerminalInput | None = None
) -> ApprovalCallback:
    def approve(cwd: Path, argv: tuple[str, ...]) -> bool:
        if requires_approval(mode, cwd, argv, with_secret=with_secret):
            run = current_run()
            if run is not None:
                print(format_run_message(
                    run, "Waiting for command approval", waiting="approval"
                ), flush=True)
            if terminal is not None:
                with terminal.slot():
                    return _interactive_approval(cwd, argv, with_secret=with_secret)
            return _interactive_approval(cwd, argv, with_secret=with_secret)
        return True

    return approve


def _async_approval_callback(
    mode: ApprovalMode, terminal: TerminalInput, *, with_secret: bool = False
) -> AsyncApprovalCallback:
    async def approve(cwd: Path, argv: tuple[str, ...]) -> bool:
        if not requires_approval(mode, cwd, argv, with_secret=with_secret):
            return True
        run = current_run()
        if run is not None:
            print(format_run_message(
                run, "Waiting for command approval", waiting="approval"
            ), flush=True)
        return await terminal.read(
            lambda: _interactive_approval(cwd, argv, with_secret=with_secret)
        )

    return approve


def _announce_approval_mode(mode: ApprovalMode) -> None:
    if mode is ApprovalMode.DANGEROUS_ONLY:
        print(
            "Approval mode: dangerous_only (conservative heuristic; tests, scripts, "
            "and Secret commands still require confirmation)."
        )
    elif mode is ApprovalMode.ALLOW_ALL:
        print(
            "Warning: approval.mode=allow_all runs commands, including Secret commands, "
            "without confirmation."
        )


def _state_store() -> RunStore:
    return SQLiteStateStore(handshake_home() / "handshake.db")


def _create_real_model(model_name: str) -> ModelAdapter:
    config = load_runtime_settings(model_override=model_name).provider
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


def _create_secret_capabilities(
    approval: ApprovalCallback, async_approval: AsyncApprovalCallback | None = None,
) -> tuple[SecretWriter, SecretCommandRunner]:
    store = EphemeralSecretStore()
    return store.writer(), SecretCommandRunner(store.consumer(), approval, async_approval)


def _next_open_dependency(run: Run) -> Dependency:
    dependencies = run.open_dependencies()
    if not dependencies:
        raise HandshakeError(f"Waiting run {run.run_id} has no open dependency")
    return dependencies[0]


def _resume_command(session: Session) -> str:
    return f"handshake resume {session.session_id}"


def _read_dependency_response(session: Session, run: Run, dependency: Dependency) -> str | None:
    details = (
        f"Input required\n  session: {session.session_id}\n"
        f"  dependency: {dependency.dependency_id}\n  question: {dependency.prompt}"
    )
    print(format_run_message(run, details, waiting="text"), flush=True)
    if not sys.stdin.isatty():
        print("Run remains waiting because stdin is not interactive.")
        print(f"Resume with: {_resume_command(session)}")
        return None
    try:
        return read_terminal_line("Response: ")
    except EOFError:
        print("\nRun remains waiting; no response was recorded.")
        print(f"Resume with: {_resume_command(session)}")
        return None
    except KeyboardInterrupt as exc:
        print("\nRun remains waiting; no response was recorded.")
        print(f"Resume with: {_resume_command(session)}")
        raise UserCancelledError("User interrupted text input") from exc


def _read_secret_value(session: Session, run: Run, dependency: Dependency) -> str | None:
    handle = run.secret_handle_for(dependency)
    if handle is None:
        raise HandshakeError(f"Secret dependency {dependency.dependency_id} has no handle metadata")
    details = (
        f"Secret input required\n  session: {session.session_id}\n"
        f"  dependency: {dependency.dependency_id}\n  request: {dependency.prompt}\n"
        f"  purpose: {handle.purpose}\n  target: {handle.environment_variable}"
    )
    print(format_run_message(run, details, waiting="secret"), flush=True)
    if not sys.stdin.isatty():
        print("Run remains waiting because stdin is not interactive.")
        print(f"Resume with: {_resume_command(session)}")
        return None
    try:
        return read_terminal_line("Secret (input hidden): ", hidden=True)
    except EOFError:
        print("\nRun remains waiting; no Secret was recorded.")
        print(f"Resume with: {_resume_command(session)}")
        return None
    except KeyboardInterrupt as exc:
        print("\nRun remains waiting; no Secret was recorded.")
        print(f"Resume with: {_resume_command(session)}")
        raise UserCancelledError("User interrupted Secret input") from exc


def _print_final_summary(run: Run) -> None:
    print(format_run_message(run, "Final summary\n" + (run.final_summary or "(none)")), flush=True)


async def _resolve_run_input(
    session: Session, run: Run, secret_writer: SecretWriter, terminal: TerminalInput
) -> ModelMessage | None:
    while True:
        dependency = _next_open_dependency(run)
        if dependency.kind is DependencyKind.TEXT:
            response = await terminal.read(
                partial(_read_dependency_response, session, run, dependency)
            )
            if response is None:
                return None
            if not response.strip():
                print("Response must not be empty; the run is still waiting.")
                continue
            continuation = resolve_text_dependency(session, run, dependency.dependency_id, response)
        elif dependency.kind is DependencyKind.SECRET:
            secret_value = await terminal.read(
                partial(_read_secret_value, session, run, dependency)
            )
            if secret_value is None:
                return None
            if secret_value == "":
                print("Secret must not be empty; the run is still waiting.")
                continue
            continuation = resolve_secret_dependency(
                session,
                run,
                dependency.dependency_id,
                secret_value,
                secret_writer,
            )
            del secret_value
        else:
            raise HandshakeError(f"Unsupported dependency kind: {dependency.kind.value}")
        return continuation


async def _drive_run(
    session: Session,
    run: Run,
    model: ModelAdapter,
    registry: ToolRegistry,
    store: RunStore,
    secret_writer: SecretWriter,
    secret_command_runner: SecretCommandRunner,
    agent_spec: AgentSpec,
    context_manager: AgentContextManager | None = None,
    delegation: DelegationRuntime | None = None,
    terminal: TerminalInput | None = None,
) -> tuple[int, Run]:
    terminal = terminal or TerminalInput()
    while run.status is RunStatus.WAITING:
        if not run.open_dependencies():
            return _WAITING_EXIT_CODE, run
        continuation = await _resolve_run_input(session, run, secret_writer, terminal)
        if continuation is None:
            return _WAITING_EXIT_CODE, run
        run = await continue_run(
            session,
            run,
            model,
            registry,
            emit=RunReporter(run, print),
            store=store,
            secret_command_runner=secret_command_runner,
            context_manager=context_manager,
            agent_spec=agent_spec,
            delegation=delegation,
            pending_messages=(continuation,),
        )
    if run.status is not RunStatus.COMPLETED:
        raise HandshakeError(f"Run {run.run_id} stopped in unsupported state {run.status.value}")
    _print_final_summary(run)
    return 0, run


def _print_terminal_run(run: Run) -> int:
    if run.status is RunStatus.COMPLETED:
        _print_final_summary(run)
        return 0
    if run.status in {RunStatus.FAILED, RunStatus.CANCELLED}:
        print(format_run_message(
            run, f"Run {run.status.value}\n" + (run.error or "(no error recorded)")
        ), flush=True)
        return 1
    raise HandshakeError(
        f"Run {run.run_id} cannot be restored from {run.status.value}; "
        "only stable waiting or terminal runs are supported"
    )


def _read_next_goal(session: Session) -> str | None:
    if not sys.stdin.isatty():
        return None
    print(f"\nSession {session.session_id} is ready for the next request.")
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
        print("Request must not be empty. Enter /exit to leave the session.")


async def _process_current_run(
    session: Session,
    run: Run,
    registry: ToolRegistry,
    store: RunStore,
    secret_writer: SecretWriter,
    secret_command_runner: SecretCommandRunner,
    agent_specs: Mapping[str, AgentSpec],
    context_managers: ActiveContextManagers,
    model: ModelAdapter | None = None,
    scheduler: CooperativeScheduler | None = None,
    terminal: TerminalInput | None = None,
) -> tuple[int, Run]:
    if run.status in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}:
        return _print_terminal_run(run), run
    if run.status not in {RunStatus.CREATED, RunStatus.WAITING, RunStatus.RUNNING}:
        return _print_terminal_run(run), run

    terminal = terminal or TerminalInput()
    reporter = RunReporter(run, print)
    agent_spec = _spec_for_run(run, agent_specs)
    current_model = model or _create_real_model(model_name_for_run(run, agent_spec))
    delegation = DelegationRuntime(
        agent_specs, lambda spec: _create_real_model(spec.model_name), context_managers,
        scheduler=scheduler or CooperativeScheduler(),
        resolve_input=lambda child: _resolve_run_input(session, child, secret_writer, terminal),
    )
    current_model = delegation.scheduler.model(current_model)
    current_context = context_manager_for_run(context_managers, run, agent_spec, current_model)
    try:
        if run.status is RunStatus.CREATED:
            run = await start_run(
                session,
                run,
                current_model,
                registry,
                emit=reporter,
                store=store,
                secret_command_runner=secret_command_runner,
                context_manager=current_context,
                agent_spec=agent_spec,
                delegation=delegation,
            )
            if run.status is RunStatus.WAITING and not run.open_dependencies():
                return _WAITING_EXIT_CODE, run
        elif run.status is RunStatus.RUNNING:
            run = await delegation.scheduler.advance_runnable(
                run,
                lambda runnable: resume_run(
                    session, runnable, current_model, registry, store=store,
                    agent_spec=agent_spec, delegation=delegation, emit=reporter,
                    secret_command_runner=secret_command_runner,
                    context_manager=current_context,
                ),
            )
        elif run.status is RunStatus.WAITING and not run.open_dependencies():
            run = await continue_delegation(
                session,
                run,
                current_model,
                registry,
                store=store,
                agent_spec=agent_spec,
                delegation=delegation,
                emit=reporter,
                secret_command_runner=secret_command_runner,
            )
            if run.status is RunStatus.WAITING and not run.open_dependencies():
                return _WAITING_EXIT_CODE, run
        return await _drive_run(
            session,
            run,
            current_model,
            registry,
            store,
            secret_writer,
            secret_command_runner,
            agent_spec,
            current_context,
            delegation,
            terminal,
        )
    except UserCancelledError:
        raise
    except HandshakeError as exc:
        try:
            persisted = store.load_run(run.run_id)
        except HandshakeError:
            raise exc from None
        if persisted.status is not RunStatus.FAILED:
            raise
        return _print_terminal_run(persisted), persisted
    finally:
        await close_model(current_model)


async def _drive_session(
    session: Session,
    run: Run,
    registry: ToolRegistry,
    store: RunStore,
    secret_writer: SecretWriter,
    secret_command_runner: SecretCommandRunner,
    agent_specs: Mapping[str, AgentSpec],
    initial_model: ModelAdapter | None = None,
    max_model_requests: int = 4,
    terminal: TerminalInput | None = None,
) -> int:
    terminal = terminal or TerminalInput()
    context_managers = ActiveContextManagers()
    scheduler = CooperativeScheduler(max_model_requests)
    result_code, run = await _settle_run(
        session,
        run,
        registry,
        store,
        secret_writer,
        secret_command_runner,
        agent_specs,
        context_managers,
        initial_model,
        scheduler,
        terminal,
    )
    if result_code == _WAITING_EXIT_CODE or not sys.stdin.isatty():
        return result_code

    while True:
        goal = _read_next_goal(session)
        if goal is None:
            return 0
        runs = store.list_runs(session.session_id)
        run = create_followup_run(session, runs, goal)
        result_code, run = await _settle_run(
            session,
            run,
            registry,
            store,
            secret_writer,
            secret_command_runner,
            agent_specs,
            context_managers,
            scheduler=scheduler,
            terminal=terminal,
        )
        if result_code == _WAITING_EXIT_CODE:
            return result_code


async def _settle_run(
    session: Session,
    run: Run,
    registry: ToolRegistry,
    store: RunStore,
    secret_writer: SecretWriter,
    secret_command_runner: SecretCommandRunner,
    agent_specs: Mapping[str, AgentSpec],
    context_managers: ActiveContextManagers,
    initial_model: ModelAdapter | None = None,
    scheduler: CooperativeScheduler | None = None,
    terminal: TerminalInput | None = None,
) -> tuple[int, Run]:
    """Advance the parent while its child tasks route and resume their own waits."""
    if run.parent_run_id is not None:
        run = store.load_run(run.parent_run_id)
    return await _process_current_run(
        session, run, registry, store, secret_writer, secret_command_runner, agent_specs,
        context_managers, initial_model, scheduler, terminal,
    )


async def _run_real(args: argparse.Namespace) -> int:
    workspace = Path(args.repo).expanduser().resolve()
    if not workspace.is_dir():
        raise HandshakeError(f"Repository directory does not exist: {workspace}")
    settings = load_runtime_settings(model_override=args.model)
    _announce_approval_mode(settings.approval_mode)
    store = _state_store()
    session = Session(workspace=workspace)
    terminal = TerminalInput()
    approval = _approval_callback(settings.approval_mode, terminal=terminal)
    registry = create_default_registry(
        approval, async_approval=_async_approval_callback(settings.approval_mode, terminal)
    )
    agent_specs = _agent_specs(settings, registry)
    main_spec = agent_specs["main"]
    model = _create_real_model(main_spec.model_name)
    run = create_initial_run(
        session,
        args.request,
        model_name=main_spec.model_name,
        max_steps=(main_spec.default_max_steps if args.max_steps is None else args.max_steps),
        agent_spec_id=main_spec.spec_id,
    )
    store.create_session(session)
    print(f"Session created: {session.session_id}")
    secret_writer, secret_command_runner = _create_secret_capabilities(
        _approval_callback(settings.approval_mode, with_secret=True, terminal=terminal),
        _async_approval_callback(settings.approval_mode, terminal, with_secret=True),
    )
    return await _drive_session(
        session,
        run,
        registry,
        store,
        secret_writer,
        secret_command_runner,
        agent_specs,
        model,
        max_model_requests=settings.max_model_requests,
        terminal=terminal,
    )


async def _resume_session(args: argparse.Namespace) -> int:
    store = _state_store()
    session = store.load_session(args.session_id)
    if not session.workspace.is_dir():
        raise HandshakeError(f"Repository directory does not exist: {session.workspace}")
    settings = load_runtime_settings()
    _announce_approval_mode(settings.approval_mode)
    terminal = TerminalInput()
    approval = _approval_callback(settings.approval_mode, terminal=terminal)
    registry = create_default_registry(
        approval, async_approval=_async_approval_callback(settings.approval_mode, terminal)
    )
    secret_writer, secret_command_runner = _create_secret_capabilities(
        _approval_callback(settings.approval_mode, with_secret=True, terminal=terminal),
        _async_approval_callback(settings.approval_mode, terminal, with_secret=True),
    )
    secrets_reissued = False
    for active in store.list_runs(session.session_id):
        if active.status in {RunStatus.RUNNING, RunStatus.WAITING}:
            secrets_reissued = (
                prepare_run_for_resume(session, active, secret_command_runner, store=store)
                or secrets_reissued
            )
    if secrets_reissued:
        print("A process-local Secret was lost and must be entered again.")
    run = latest_run(session, store.list_runs(session.session_id))
    return await _drive_session(
        session,
        run,
        registry,
        store,
        secret_writer,
        secret_command_runner,
        _agent_specs(settings, registry),
        max_model_requests=settings.max_model_requests,
        terminal=terminal,
    )


def _list_sessions(_args: argparse.Namespace) -> int:
    store = _state_store()
    sessions = store.list_sessions()
    if not sessions:
        print("No durable sessions.")
        return 0
    for session in sessions:
        runs = store.list_runs(session.session_id)
        if not runs:
            print(f"{session.session_id}  empty      {session.created_at}  [{session.workspace}]")
            continue
        run = latest_run(session, runs)
        goal = run.goal.replace("\n", " ")
        if len(goal) > 60:
            goal = goal[:57] + "..."
        print(
            f"{session.session_id}  {run.status.value:<9}  {run.updated_at}  "
            f"{goal}  [{session.workspace}]"
        )
    return 0


def _inspect_session(args: argparse.Namespace) -> int:
    store = _state_store()
    session = store.load_session(args.session_id)
    runs = store.list_runs(session.session_id)
    print(f"Session:   {session.session_id}")
    print(f"Workspace: {session.workspace}")
    print(f"Created:   {session.created_at}")
    print(f"Latest run messages: {store.message_count(runs[-1].context_scope_id) if runs else 0}")
    print("Runs:")
    for run in runs:
        goal = run.goal.replace("\n", " ")
        print(
            f"  - #{run.sequence} {run.run_id} agent={run.agent_spec_id} [{run.status.value}] "
            f"{run.model_turns}/{run.max_steps} {goal}"
        )
        for dependency in run.open_dependencies():
            print(
                f"      dependency {dependency.dependency_id} "
                f"[{dependency.kind.value}] {dependency.prompt}"
            )
    return 0


def _inspect_run(args: argparse.Namespace) -> int:
    store = _state_store()
    run = store.load_run(args.run_id)
    session = store.load_session(run.session_id)
    message_count = store.message_count(run.context_scope_id)
    owner_id = run.workspace_binding.worktree_owner_id
    worktree = store.load_worktree(owner_id) if owner_id is not None else None
    print(f"Run:         {run.run_id}")
    print(f"Session:     {run.session_id}")
    print(f"Sequence:    {run.sequence}")
    print(f"Status:      {run.status.value}")
    print(f"Goal:        {run.goal}")
    workspace = worktree.path if worktree else (session.workspace if owner_id is None else None)
    print(f"Workspace:   {workspace or '(unavailable)'}")
    if worktree is not None:
        status = worktree.status.value
        if worktree.cleaned:
            status += ", removed"
        elif worktree.cleanup_pending:
            status += ", cleanup pending"
        print(f"Worktree:    {worktree.branch} [{status}]")
        print(f"Owner run:   {worktree.owner_run_id}")
        if worktree.saved_commit is not None:
            print(f"Snapshot:    {worktree.saved_commit}")
    print(f"Agent:       {run.agent_spec_id}")
    print(f"Parent run:  {run.parent_run_id or '(root)'}")
    print(f"Model:       {run.model_name or '(not recorded)'}")
    print(f"Model turns: {run.model_turns}/{run.max_steps}")
    print(f"Messages:    {message_count}")
    print("Plan:")
    if run.plan:
        for step in run.plan:
            print(f"  - {step}")
    else:
        print("  (not set)")
    print("Dependencies:")
    if run.dependencies:
        for dependency in run.dependencies:
            print(
                f"  - {dependency.dependency_id} "
                f"[{dependency.kind.value}/{dependency.status.value}] "
                f"{dependency.prompt}"
            )
    else:
        print("  (none)")
    print("Secret handles:")
    if run.secret_handles:
        for handle in run.secret_handles:
            print(
                f"  - {handle.secret_handle} [{handle.status.value}] "
                f"{handle.consumer} → {handle.environment_variable}"
            )
    else:
        print("  (none)")
    if run.error:
        print(f"Error:       {run.error}")
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
                    "plan-1",
                    "update_plan",
                    {"steps": ["Implement add", "Run tests"]},
                )
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


async def _run_demo(_args: argparse.Namespace) -> int:
    fixture = Path(__file__).resolve().parents[2] / "examples" / "minimal_task"
    if not fixture.is_dir():
        raise HandshakeError(f"Bundled demo fixture is missing: {fixture}")
    with tempfile.TemporaryDirectory(prefix="handshake-demo-") as temporary:
        workspace = Path(temporary) / "workspace"
        shutil.copytree(fixture, workspace)
        print(f"Running isolated scripted demo in {workspace}")
        session = Session(workspace=workspace)
        registry = create_default_registry(lambda _cwd, _argv: True)
        demo_spec = demo_agent_spec(
            definition.name for definition in (*runtime_tool_definitions(), *registry.definitions())
        )
        run = Run(
            goal="Implement the missing calculator.add function",
            session_id=session.session_id,
            agent_spec_id=demo_spec.spec_id,
            model_name=demo_spec.model_name,
        )
        store = SQLiteStateStore(Path(temporary) / "demo.db")
        store.create_session(session)
        await start_run(
            session,
            run,
            _demo_model(),
            registry,
            emit=print,
            store=store,
            agent_spec=demo_spec,
        )
        _print_final_summary(run)
        return 0


def _set_api_key(_args: argparse.Namespace) -> int:
    if not sys.stdin.isatty():
        raise HandshakeError("Saving an API key requires an interactive terminal")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            value = getpass.getpass("DeepSeek API key (input hidden): ")
    except getpass.GetPassWarning:
        raise HandshakeError("Hidden API key input is unavailable in this terminal") from None
    except (EOFError, KeyboardInterrupt) as exc:
        raise UserCancelledError("User cancelled API key input") from exc
    try:
        save_deepseek_api_key(value)
    finally:
        del value
    print("DeepSeek API key saved in the user directory.")
    return 0


def _show_config(_args: argparse.Namespace) -> int:
    settings = load_runtime_settings()
    print(f"Configuration: {settings_path()}")
    print("[model]")
    print(f"max_tokens = {settings.provider.max_tokens}")
    print("\n[context]")
    print(f"context_window_tokens = {settings.context.context_window_tokens}")
    print("\n[scheduler]")
    print(f"max_model_requests = {settings.max_model_requests}")
    print("\n[approval]")
    print(f'mode = "{settings.approval_mode.value}"')
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="handshake")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="start a durable DeepSeek session")
    run_parser.add_argument("repo", help="repository workspace")
    run_parser.add_argument("request", help="natural-language coding request")
    run_parser.add_argument(
        "--model",
        default=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        help="DeepSeek model name",
    )
    run_parser.add_argument("--max-steps", type=int)
    run_parser.set_defaults(handler=_run_real)

    resume_parser = subparsers.add_parser("resume", help="resume a durable session")
    resume_parser.add_argument("session_id", help="durable session ID")
    resume_parser.set_defaults(handler=_resume_session)

    list_parser = subparsers.add_parser("list", help="list durable sessions")
    list_parser.set_defaults(handler=_list_sessions)

    inspect_parser = subparsers.add_parser("inspect", help="inspect durable session state")
    inspect_parser.add_argument("session_id", help="durable session ID")
    inspect_parser.set_defaults(handler=_inspect_session)

    inspect_run_parser = subparsers.add_parser("inspect-run", help="inspect one durable run")
    inspect_run_parser.add_argument("run_id", help="durable run ID")
    inspect_run_parser.set_defaults(handler=_inspect_run)

    demo_parser = subparsers.add_parser("demo", help="run the offline scripted demo")
    demo_parser.set_defaults(handler=_run_demo)

    config_parser = subparsers.add_parser(
        "config",
        help="manage local configuration",
        description=(
            "Edit ~/.handshake/config.toml to set [approval] mode to "
            "always_ask, dangerous_only, or allow_all."
        ),
    )
    config_subparsers = config_parser.add_subparsers(dest="config_command", required=True)
    key_parser = config_subparsers.add_parser(
        "set-api-key", help="save the DeepSeek API key in the user directory"
    )
    key_parser.set_defaults(handler=_set_api_key)
    show_parser = config_subparsers.add_parser(
        "show", help="show effective non-secret model, context, and approval settings"
    )
    show_parser.set_defaults(handler=_show_config)
    return parser


async def _await_command(result: Awaitable[int]) -> int:
    return await result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        handler = args.handler
        result = handler(args)
        if inspect.isawaitable(result):
            return int(asyncio.run(_await_command(result)))
        return int(result)
    except KeyboardInterrupt:
        print("Cancelled: user interrupted execution", file=sys.stderr)
        return 130
    except UserCancelledError as exc:
        print(f"Cancelled: {exc}", file=sys.stderr)
        return 130
    except (HandshakeError, StorageError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
