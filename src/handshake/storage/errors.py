"""Errors raised by durable state storage."""


class StorageError(Exception):
    """Raised when durable session state cannot be read or written safely."""


class RunNotFoundError(StorageError):
    """Raised when a requested durable run does not exist."""


class SessionNotFoundError(StorageError):
    """Raised when a requested durable session does not exist."""
