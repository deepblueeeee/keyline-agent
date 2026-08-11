"""Command-line interface for the first Handshake vertical slice."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from handshake.config import DeepSeekConfig, read_deepseek_api_key
from handshake.models.deepseek_anthropic import DeepSeekAnthropicModel
from handshake.models.scripted import ScriptedModel
from handshake.runtime.errors import HandshakeError, UserCancelledError
from handshake.runtime.loop import run_agent
from handshake.runtime.state import ModelTurn, Task, ToolCall
from handshake.tools import create_default_registry


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


def _run_real(args: argparse.Namespace) -> int:
    workspace = Path(args.repo).expanduser().resolve()
    if not workspace.is_dir():
        raise HandshakeError(f"Repository directory does not exist: {workspace}")
    config = DeepSeekConfig.from_environment(model_override=args.model)
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
    task = Task(goal=args.task, workspace=workspace)
    registry = create_default_registry(_interactive_approval)
    run_agent(task, model, registry, max_steps=args.max_steps, emit=print)
    print("\nFinal summary")
    print(task.final_summary or "(none)")
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
        task = Task(goal="Implement the missing calculator.add function", workspace=workspace)
        registry = create_default_registry(lambda _cwd, _argv: True)
        run_agent(task, _demo_model(), registry, emit=print)
        print("\nFinal summary")
        print(task.final_summary or "(none)")
        return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="handshake")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run a task with DeepSeek")
    run_parser.add_argument("repo", help="repository workspace")
    run_parser.add_argument("task", help="natural-language coding task")
    run_parser.add_argument(
        "--model",
        default=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        help="DeepSeek model name",
    )
    run_parser.add_argument("--max-steps", type=int, default=50)
    run_parser.set_defaults(handler=_run_real)

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
