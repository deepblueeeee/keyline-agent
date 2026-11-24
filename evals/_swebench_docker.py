"""Start the official CLI with an isolated Docker container policy."""

import runpy
import sys
from collections.abc import Callable
from typing import Any
from unittest.mock import patch


def _isolated_create(create: Callable[..., Any], collection: Any, **kwargs: Any) -> Any:
    kwargs.update(
        network_mode="none",
        cap_add=[],
        cap_drop=["ALL"],
        security_opt=["no-new-privileges"],
        use_config_proxy=False,
    )
    container = create(collection, **kwargs)
    try:
        container.reload()
        config = container.attrs["HostConfig"]
        if (
            config["NetworkMode"] != "none"
            or config["Privileged"]
            or config.get("Binds")
            or container.attrs["Mounts"]
            or config.get("CapAdd")
            or "ALL" not in (config.get("CapDrop") or [])
            or "no-new-privileges" not in (config.get("SecurityOpt") or [])
        ):
            raise RuntimeError("Docker did not create an isolated scoring container")
        print(f"Verified scoring container {container.id}: network=none, mounts=0", flush=True)
    except Exception:
        container.remove(force=True)
        raise
    return container


def _block_host_signals(event: str, args: tuple[Any, ...]) -> None:
    # The CLI's cleanup fallback can mistake a Docker VM PID for a local PID.
    if event == "os.kill":
        raise PermissionError("Use the Docker API to stop scoring containers")


if __name__ == "__main__":
    from docker.models.containers import ContainerCollection

    create = ContainerCollection.create
    sys.addaudithook(_block_host_signals)
    with patch.object(
        ContainerCollection, "create", lambda self, **kw: _isolated_create(create, self, **kw)
    ):
        runpy.run_module("swebench.harness.run_evaluation", run_name="__main__", alter_sys=True)
