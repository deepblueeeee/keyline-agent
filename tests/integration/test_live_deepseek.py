import os
import shutil
from pathlib import Path

import pytest

from handshake.config import DeepSeekConfig
from handshake.models.deepseek_anthropic import DeepSeekAnthropicModel
from handshake.runtime.loop import run_agent
from handshake.runtime.state import Task, TaskStatus
from handshake.tools import create_default_registry


@pytest.mark.live
@pytest.mark.skipif(
    os.environ.get("HANDSHAKE_LIVE_TEST") != "1" or not os.environ.get("DEEPSEEK_API_KEY"),
    reason="set HANDSHAKE_LIVE_TEST=1 and DEEPSEEK_API_KEY to run",
)
def test_deepseek_completes_isolated_fixture(tmp_path: Path) -> None:
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
    task = Task(
        goal="Implement calculator.add and run the existing tests.",
        workspace=workspace,
    )

    run_agent(task, model, create_default_registry(lambda _cwd, _argv: True))

    assert task.status is TaskStatus.COMPLETED
    assert "return left + right" in (workspace / "calculator.py").read_text()
