"""Stable agent configuration; execution state belongs to Run."""

from __future__ import annotations

from dataclasses import dataclass, field

from handshake.context.budget import ContextBudgetPolicy


@dataclass(frozen=True, slots=True)
class AgentSpec:
    """An agent's profile, model, tool allowance and default budget."""

    spec_id: str
    model_name: str
    system_prompt: str
    profile: str = ""
    allowed_tools: frozenset[str] = frozenset()
    default_budget: ContextBudgetPolicy = field(default_factory=ContextBudgetPolicy)
    default_max_steps: int = 50

    def __post_init__(self) -> None:
        if not self.spec_id or not self.model_name or not self.system_prompt:
            raise ValueError("Agent spec needs an identity, model, and system prompt")
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
