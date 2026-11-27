"""User-local model token limits and the derived internal context policy."""

from __future__ import annotations

import os
import stat
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

from keyline_agent.config.approval import ApprovalMode
from keyline_agent.config.provider import DeepSeekConfig
from keyline_agent.context.budget import ContextBudgetPolicy
from keyline_agent.errors import KeylineError


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    provider: DeepSeekConfig
    context: ContextBudgetPolicy
    approval_mode: ApprovalMode
    max_model_requests: int = 4


def settings_path() -> Path:
    # Approval policy must not move into a workspace via KEYLINE_HOME.
    return Path.home() / ".keyline-agent" / "config.toml"


def load_runtime_settings(*, model_override: str | None = None) -> RuntimeSettings:
    """Load optional TOML overrides, leaving existing defaults unchanged."""
    path = settings_path()
    data = _read_config(path)

    _check_keys(data, {"model", "context", "approval", "scheduler"}, "configuration")
    model = _section(data, "model")
    context = _section(data, "context")
    approval = _section(data, "approval")
    scheduler = _section(data, "scheduler")
    _check_keys(scheduler, {"max_model_requests"}, "scheduler")
    max_model_requests = _integer(
        scheduler.get("max_model_requests", 4), "scheduler.max_model_requests", 1
    )
    _check_keys(model, {"max_tokens"}, "model")
    _check_keys(context, {"context_window_tokens"}, "context")
    _check_keys(approval, {"mode"}, "approval")
    raw_mode = approval.get("mode", ApprovalMode.ALWAYS_ASK.value)
    if not isinstance(raw_mode, str):
        raise KeylineError(
            "approval.mode must be always_ask, dangerous_only, or allow_all"
        )
    try:
        mode = ApprovalMode(raw_mode)
    except ValueError:
        raise KeylineError(
            "approval.mode must be always_ask, dangerous_only, or allow_all"
        ) from None

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
        raise KeylineError(
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
        raise KeylineError("Model token limits leave too little context input budget")
    return RuntimeSettings(
        provider=provider, context=policy, approval_mode=mode,
        max_model_requests=max_model_requests,
    )


def _read_config(path: Path) -> dict[str, object]:
    try:
        parent = path.parent.lstat()
    except FileNotFoundError:
        return {}
    except OSError:
        raise KeylineError("Cannot inspect the user configuration directory") from None
    if (
        not stat.S_ISDIR(parent.st_mode)
        or parent.st_mode & 0o022
        or (hasattr(os, "getuid") and parent.st_uid != os.getuid())
    ):
        raise KeylineError("User configuration directory has unsafe permissions")
    try:
        info = path.lstat()
    except FileNotFoundError:
        return {}
    except OSError:
        raise KeylineError("Cannot inspect the user configuration file") from None
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_mode & 0o022
        or (hasattr(os, "getuid") and info.st_uid != os.getuid())
    ):
        raise KeylineError("User configuration file has unsafe permissions")
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as source:
            opened = os.fstat(source.fileno())
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_mode & 0o022
                or (hasattr(os, "getuid") and opened.st_uid != os.getuid())
            ):
                raise KeylineError("User configuration file has unsafe permissions")
            return tomllib.load(source)
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise KeylineError(f"Cannot read configuration {path}: {exc}") from exc


def _section(data: dict[str, object], name: str) -> dict[str, object]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise KeylineError(f"Configuration section [{name}] must be a table")
    return value


def _check_keys(data: dict[str, object], allowed: set[str], section: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise KeylineError(
            f"Unknown {section} configuration key: {sorted(unknown)[0]}"
        )


def _integer(value: object, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise KeylineError(f"{name} must be an integer >= {minimum}")
    return value
