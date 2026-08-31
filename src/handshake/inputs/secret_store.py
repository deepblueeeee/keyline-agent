"""Process-local Secret storage exposed only through narrow capabilities."""

from __future__ import annotations

from typing import Protocol

from handshake.runtime.errors import SecretStoreError


class SecretWriter(Protocol):
    """Capability that can write, but cannot consume, a Secret value."""

    def put(self, secret_handle: str, value: str) -> None:
        """Associate a non-empty raw value with an opaque handle."""
        ...


class SecretConsumer(Protocol):
    """Capability that can check and consume a Secret value once."""

    def contains(self, secret_handle: str) -> bool:
        """Return whether the process still holds the handle's raw value."""
        ...

    def consume(self, secret_handle: str) -> str:
        """Remove and return the raw value for one-time secure execution."""
        ...


class _SecretWriter:
    __slots__ = ("_values",)

    def __init__(self, values: dict[str, str]) -> None:
        self._values = values

    def put(self, secret_handle: str, value: str) -> None:
        if value == "":
            raise SecretStoreError("Secret value must not be empty")
        if secret_handle in self._values:
            raise SecretStoreError(f"Secret handle already has a value: {secret_handle}")
        self._values[secret_handle] = value

    def __repr__(self) -> str:
        return "SecretWriter(<write-only capability>)"


class _SecretConsumer:
    __slots__ = ("_values",)

    def __init__(self, values: dict[str, str]) -> None:
        self._values = values

    def contains(self, secret_handle: str) -> bool:
        return secret_handle in self._values

    def consume(self, secret_handle: str) -> str:
        try:
            return self._values.pop(secret_handle)
        except KeyError:
            raise SecretStoreError(
                f"Secret value is unavailable for handle: {secret_handle}"
            ) from None

    def __repr__(self) -> str:
        return "SecretConsumer(<consume-only capability>)"


class EphemeralSecretStore:
    """Own raw Secret values for the lifetime of one local process."""

    __slots__ = ("_values",)

    def __init__(self) -> None:
        self._values: dict[str, str] = {}

    def writer(self) -> SecretWriter:
        return _SecretWriter(self._values)

    def consumer(self) -> SecretConsumer:
        return _SecretConsumer(self._values)

    def __repr__(self) -> str:
        return f"EphemeralSecretStore(entries={len(self._values)})"
