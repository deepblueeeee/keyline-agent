from unittest.mock import Mock

import pytest

from evals._swebench_docker import _block_host_signals, _isolated_create


@pytest.mark.parametrize(
    "mode, mounts", [("none", []), ("bridge", []), ("none", [{"Type": "bind"}])]
)
def test_scoring_container_must_be_isolated_before_start(
    mode: str, mounts: list[dict[str, str]]
) -> None:
    create, collection = Mock(), Mock()
    container = create.return_value
    container.attrs = {
        "HostConfig": {
            "NetworkMode": mode,
            "Privileged": False,
            "CapDrop": ["ALL"],
            "SecurityOpt": ["no-new-privileges"],
        },
        "Mounts": mounts,
    }
    if mode == "none" and not mounts:
        assert _isolated_create(create, collection, image="pinned-image") is container
        container.remove.assert_not_called()
    else:
        with pytest.raises(RuntimeError, match="isolated scoring container"):
            _isolated_create(create, collection, image="pinned-image")
        container.remove.assert_called_once_with(force=True)

    create.assert_called_once_with(
        collection,
        image="pinned-image",
        network_mode="none",
        cap_add=[],
        cap_drop=["ALL"],
        security_opt=["no-new-privileges"],
        use_config_proxy=False,
    )
    container.reload.assert_called_once()
    container.start.assert_not_called()


def test_official_cleanup_cannot_signal_a_host_process() -> None:
    with pytest.raises(PermissionError, match="Docker API"):
        _block_host_signals("os.kill", (123, 9))
