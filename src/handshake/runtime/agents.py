"""Stable agent configuration; execution state belongs to Run."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from handshake.context.budget import ContextBudgetPolicy
from handshake.runtime.errors import AgentLoopError
from handshake.runtime.state import Run


@dataclass(frozen=True, slots=True)
class AgentSpec:
    """An agent's profile, model, tool allowance and default budget."""

    spec_id: str
    model_name: str
    instructions: str
    profile: str = ""
    allowed_tools: frozenset[str] = frozenset()
    default_budget: ContextBudgetPolicy = field(default_factory=ContextBudgetPolicy)
    default_max_steps: int = 50

    def __post_init__(self) -> None:
        if not self.spec_id or not self.model_name or not self.instructions:
            raise ValueError("Agent spec needs an identity, model, and instructions")
        if isinstance(self.allowed_tools, str):
            raise ValueError("Allowed tools must be a collection of names")
        tools = frozenset(self.allowed_tools)
        if any(not isinstance(tool, str) or not tool for tool in tools):
            raise ValueError("Allowed tool names must be non-empty strings")
        if type(self.default_max_steps) is not int or self.default_max_steps <= 0:
            raise ValueError("Agent step budget must be positive")
        if self.default_budget.hard_input_tokens <= 0:
            raise ValueError("Agent context budget must allow model input")
        object.__setattr__(self, "allowed_tools", tools)


_MAIN_INSTRUCTIONS = """You are Handshake's main coding agent.
Complete the user's request in the workspace.
- Investigate enough repository context before choosing an approach or changing files.
- Keep a provisional plan for non-trivial work and revise it when evidence changes.
- Keep context focused: read relevant files and ranges before making targeted edits.
- Validate changes with appropriate checks and report what they actually showed.
- Finish with a concise summary of changes and any unresolved issue."""

_DEMO_INSTRUCTIONS = "Complete the isolated coding demo and verify the result."


def main_agent_spec(
    model_name: str,
    allowed_tools: Iterable[str],
    budget: ContextBudgetPolicy,
) -> AgentSpec:
    """Build the configured main coding agent."""
    return AgentSpec(
        spec_id="main",
        model_name=model_name,
        instructions=_MAIN_INSTRUCTIONS,
        allowed_tools=frozenset(allowed_tools),
        default_budget=budget,
    )


def demo_agent_spec(allowed_tools: Iterable[str]) -> AgentSpec:
    """Build the scripted demo agent."""
    return AgentSpec(
        spec_id="demo",
        model_name="scripted",
        instructions=_DEMO_INSTRUCTIONS,
        allowed_tools=frozenset(allowed_tools),
    )


def model_name_for_run(run: Run, agent_spec: AgentSpec) -> str:
    """Select the model from current config, never the stored run record."""
    if agent_spec.spec_id != run.agent_spec_id:
        raise AgentLoopError(f"Agent spec {run.agent_spec_id} is unavailable")
    return agent_spec.model_name
