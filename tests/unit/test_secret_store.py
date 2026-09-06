import pytest

from handshake.inputs import EphemeralSecretStore
from handshake.runtime.errors import SecretStoreError
from handshake.tools import create_default_registry


def test_secret_store_exposes_separate_write_and_consume_capabilities() -> None:
    canary = "canary-secret-value"
    store = EphemeralSecretStore()
    writer = store.writer()
    consumer = store.consumer()

    assert not hasattr(writer, "consume")
    assert not hasattr(consumer, "put")
    writer.put("secret_handle", canary)
    assert consumer.contains("secret_handle")
    assert consumer.consume("secret_handle") == canary
    assert not consumer.contains("secret_handle")

    with pytest.raises(SecretStoreError) as captured:
        consumer.consume("secret_handle")
    assert canary not in str(captured.value)


def test_secret_store_representations_do_not_expose_values() -> None:
    canary = "repr-canary-secret"
    store = EphemeralSecretStore()
    writer = store.writer()
    consumer = store.consumer()
    writer.put("secret_handle", canary)

    assert canary not in repr(store)
    assert canary not in repr(writer)
    assert canary not in repr(consumer)


def test_secret_store_preserves_whitespace_and_rejects_only_empty_string() -> None:
    store = EphemeralSecretStore()
    writer = store.writer()
    consumer = store.consumer()

    writer.put("secret_whitespace", "  value with spaces  ")
    assert consumer.consume("secret_whitespace") == "  value with spaces  "
    with pytest.raises(SecretStoreError, match="must not be empty"):
        writer.put("secret_empty", "")


def test_normal_tool_registry_has_no_secret_resolution_capability() -> None:
    registry = create_default_registry(lambda _cwd, _argv: True)

    assert not hasattr(registry, "resolve")
    assert not hasattr(registry, "secret_store")
    assert {tool.name for tool in registry.definitions()} == {
        "list_files",
        "read_file",
        "search_files",
        "edit_file",
        "write_file",
        "run_shell",
        "run_tests",
    }
