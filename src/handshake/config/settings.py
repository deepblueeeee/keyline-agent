"""User-local model token limits and the derived internal context policy."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

from handshake.config.paths import handshake_home
from handshake.config.provider import DeepSeekConfig
from handshake.context.budget import ContextBudgetPolicy
from handshake.runtime.errors import HandshakeError


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    provider: DeepSeekConfig
    context: ContextBudgetPolicy


def settings_path() -> Path:
    return handshake_home() / "config.toml"


def load_runtime_settings(*, model_override: str | None = None) -> RuntimeSettings:
    """Load optional TOML overrides, leaving existing defaults unchanged."""
    path = settings_path()
    try:
        with path.open("rb") as source:
            data = tomllib.load(source)
    except FileNotFoundError:
        data = {}
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise HandshakeError(f"Cannot read configuration {path}: {exc}") from exc

    _check_keys(data, {"model", "context"}, "configuration")
    model = _section(data, "model")
    context = _section(data, "context")
    _check_keys(model, {"max_tokens"}, "model")
    _check_keys(context, {"context_window_tokens"}, "context")

    provider = DeepSeekConfig.from_environment(model_override=model_override)
    if "max_tokens" in model:
        provider = replace(
            provider,
            max_tokens=_integer(model["max_tokens"], "model.max_tokens", 1),
        )

    defaults = ContextBudgetPolicy()
    context_window_tokens = _integer(
        context.get("context_window_tokens", defaults.context_window_tokens),
        "context.context_window_tokens",
        1,
    )
    input_tokens = context_window_tokens - provider.max_tokens
    if input_tokens <= 0:
        raise HandshakeError(
            "context.context_window_tokens must exceed model.max_tokens"
        )
    policy = replace(
        defaults,
        context_window_tokens=context_window_tokens,
        reserved_output_tokens=provider.max_tokens,
        condenser_input_tokens=min(defaults.condenser_input_tokens, input_tokens),
    )
    if policy.pressure_target_tokens <= 0 or (
        policy.pressure_target_tokens >= policy.pressure_trigger_tokens
    ):
        raise HandshakeError("Model token limits leave too little context input budget")
    return RuntimeSettings(provider=provider, context=policy)


def _section(data: dict[str, object], name: str) -> dict[str, object]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise HandshakeError(f"Configuration section [{name}] must be a table")
    return value


def _check_keys(data: dict[str, object], allowed: set[str], section: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise HandshakeError(
            f"Unknown {section} configuration key: {sorted(unknown)[0]}"
        )


def _integer(value: object, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise HandshakeError(f"{name} must be an integer >= {minimum}")
    return value

