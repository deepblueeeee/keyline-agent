import os
import shutil
from pathlib import Path

import pytest

from keyline_agent.config import DeepSeekConfig
from keyline_agent.models.deepseek_anthropic import DeepSeekAnthropicModel
from keyline_agent.runtime.agents import AgentSpec
from keyline_agent.runtime.loop import start_run
from keyline_agent.runtime.state import Run, RunStatus, Session
from keyline_agent.tools import create_default_registry, runtime_tool_definitions
from tests.history_support import fresh_store, run_async


@pytest.mark.live
@pytest.mark.skipif(
    os.environ.get("KEYLINE_LIVE_TEST") != "1" or not os.environ.get("DEEPSEEK_API_KEY"),
    reason="set KEYLINE_LIVE_TEST=1 and DEEPSEEK_API_KEY to run",
)
@run_async
async def test_deepseek_completes_isolated_fixture(tmp_path: Path) -> None:
    fixture = Path(__file__).resolve().parents[2] / "examples" / "minimal_task"
    workspace = tmp_path / "workspace"
    shutil.copytree(fixture, workspace)
    config = DeepSeekConfig.from_environment()
    api_key = os.environ["DEEPSEEK_API_KEY"]
    model = DeepSeekAnthropicModel(
        api_key=api_key,
        base_url=config.base_url,
        model=config.model,
    )
    del api_key
    session = Session(workspace=workspace)
    run = Run(
        goal="Implement calculator.add and run the existing tests.",
        session_id=session.session_id,
        agent_spec_id="main",
        model_name=config.model,
    )
    registry = create_default_registry(lambda _cwd, _argv: True)
    agent_spec = AgentSpec(
        spec_id="main",
        model_name=config.model,
        instructions="Complete the coding request.",
        allowed_tools=frozenset(
            definition.name
            for definition in (*runtime_tool_definitions(), *registry.definitions())
        ),
    )

    await start_run(
        session,
        run,
        model,
        registry,
        store=fresh_store(tmp_path, session),
        agent_spec=agent_spec,
    )

    assert run.status is RunStatus.COMPLETED
    assert "return left + right" in (workspace / "calculator.py").read_text()
