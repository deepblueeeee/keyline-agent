"""Local configuration and provider credential access."""

from handshake.config.approval import ApprovalMode, requires_approval
from handshake.config.credentials import save_deepseek_api_key
from handshake.config.paths import artifact_directory, handshake_home
from handshake.config.provider import DeepSeekConfig, read_deepseek_api_key
from handshake.config.settings import RuntimeSettings, load_runtime_settings, settings_path

__all__ = [
    "DeepSeekConfig",
    "ApprovalMode",
    "RuntimeSettings",
    "artifact_directory",
    "handshake_home",
    "load_runtime_settings",
    "read_deepseek_api_key",
    "requires_approval",
    "save_deepseek_api_key",
    "settings_path",
]
