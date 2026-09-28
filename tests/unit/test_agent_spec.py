import pytest

from handshake.runtime.agents import AgentSpec


def test_agent_spec_holds_stable_configuration() -> None:
    spec = AgentSpec(
        spec_id="reader",
        model_name="deepseek-v4-flash",
        instructions="Read the code.",
        allowed_tools=frozenset({"read_file", "search_files"}),
    )
    assert spec.spec_id == "reader"
    assert spec.instructions == "Read the code."
    assert spec.default_max_steps == 50
    assert spec.allowed_tools == {"read_file", "search_files"}


def test_agent_spec_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        AgentSpec("", "model", "prompt")
    with pytest.raises(ValueError):
        AgentSpec("reader", "model", "prompt", default_max_steps=0)
