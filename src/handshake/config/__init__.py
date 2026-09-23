"""Local configuration and provider credential access."""

from handshake.config.credentials import save_deepseek_api_key
from handshake.config.paths import artifact_directory, handshake_home
from handshake.config.provider import DeepSeekConfig, read_deepseek_api_key

__all__ = [
    "DeepSeekConfig",
    "artifact_directory",
    "handshake_home",
    "read_deepseek_api_key",
    "save_deepseek_api_key",
]
