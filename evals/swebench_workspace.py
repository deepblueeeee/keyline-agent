"""Run the tools for one SWE-bench trial in the same isolated container."""

from __future__ import annotations

import json
import tarfile
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path
from time import sleep
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from docker import DockerClient
from docker.errors import APIError, DockerException, ImageNotFound, NotFound
from docker.models.containers import Container
from docker.types import LogConfig
from requests.exceptions import ConnectionError as DockerConnectionError
from requests.exceptions import ReadTimeout

import keyline_agent
from keyline_agent.runtime.state import ToolResult
from keyline_agent.tools.files import read_file
from keyline_agent.tools.shell import parse_process_arguments

COMMAND_TIMEOUT_SECONDS = 60
_TESTBED = "/testbed"
_PATH = "/opt/miniconda3/envs/testbed/bin:/usr/local/bin:/usr/bin:/bin"
_PRIVATE = "/.keyline-agent-eval"
_BASELINE = f"{_PRIVATE}/baseline.git"
_TOOL_APP = f"{_PRIVATE}/tools.pyz"
_PYTHON = "/opt/miniconda3/bin/python"
_AGENT = "1000:1000"
_RETRYABLE_IMAGE_ERRORS = {408, 429, 500, 502, 503, 504}


def _tool_archive() -> bytes:
    """Package the current file handlers and their standard-library dependencies."""
    package = Path(keyline_agent.__file__).parent
    app = BytesIO()
    with ZipFile(app, "w", ZIP_DEFLATED) as zip_app:
        zip_app.write(Path(__file__).with_name("_swebench_tools.py"), "__main__.py")
        for source in package.rglob("*.py"):
            zip_app.write(source, source.relative_to(package.parent))
    archive = BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as tar:
        entry = tarfile.TarInfo("tools.pyz")
        entry.size = app.tell()
        entry.mode = 0o644
        app.seek(0)
        tar.addfile(entry, app)
    return archive.getvalue()


class DockerWorkspace:
    path = Path(_TESTBED)

    def __init__(self, client: DockerClient, base_commit: str, *, image: str) -> None:
        self.client = client
        self.base_commit = base_commit
        self.image = image
        self._container: Container | None = None

    @property
    def container(self) -> Container:
        if self._container is None:
            raise RuntimeError("Docker workspace is not prepared")
        return self._container

    def close(self) -> None:
        if self._container is None:
            return
        try:
            self._container.remove(force=True, v=True)
        except NotFound:
            pass
        except Exception as exc:
            raise RuntimeError(f"Could not remove trial container {self._container.id}") from exc
        self._container = None

    def _ensure_image(self) -> None:
        for attempt in range(3):
            try:
                try:
                    self.client.images.get(self.image)
                except ImageNotFound:
                    pinned = self.image.replace(":latest@", "@")
                    self.client.images.pull(pinned, platform="linux/amd64")
                    self.client.images.get(self.image)
            except (ImageNotFound, DockerConnectionError, ReadTimeout):
                if attempt == 2:
                    raise
            except APIError as exc:
                if attempt == 2 or exc.status_code not in _RETRYABLE_IMAGE_ERRORS:
                    raise
            else:
                return
            sleep(2**attempt)

    def prepare(self, *, checkpoint: str | None = None) -> None:
        if self._container is not None:
            raise RuntimeError("Docker workspace already has a container")
        if checkpoint is None:
            self._ensure_image()
        self._container = self.client.containers.create(
            checkpoint or self.image,
            entrypoint=["/bin/sleep"],
            command=["infinity"],
            platform="linux/amd64",
            working_dir="/",
            user="0:0",
            init=True,
            network_mode="none",
            cap_drop=["ALL"],
            security_opt=["no-new-privileges"],
            use_config_proxy=False,
            environment={
                "PATH": _PATH,
                "HOME": "/tmp",
                "PYTHONPATH": _TESTBED,
                "GIT_CONFIG_COUNT": "1",
                "GIT_CONFIG_KEY_0": "safe.directory",
                "GIT_CONFIG_VALUE_0": _TESTBED,
            },
            log_config=LogConfig(
                type="local", config={"max-size": "4m", "max-file": "1", "compress": "false"}
            ),
            mem_limit="2g",
            pids_limit=128,
        )
        try:
            self.container.reload()
            config = self.container.attrs["HostConfig"]
            if (
                config["NetworkMode"] != "none"
                or config["Privileged"]
                or config.get("Binds")
                or config.get("CapAdd")
                or "ALL" not in (config.get("CapDrop") or [])
                or "no-new-privileges" not in (config.get("SecurityOpt") or [])
                or self.container.attrs["Mounts"]
            ):
                raise RuntimeError("Docker workspace isolation verification failed")
            self.container.start()
            if checkpoint is None:
                self._prepare_checkout()
            base = self._checked(("git", f"--git-dir={_BASELINE}", "rev-parse", "HEAD"), root=True)
            if base.strip().decode("ascii") != self.base_commit:
                raise RuntimeError("The task checkout differs from the pinned base commit")
        except BaseException:
            self.close()
            raise

    def _prepare_checkout(self) -> None:
        # Only the pinned commit goes into the bare template or the agent's checkout.
        self._checked(
            (
                "/bin/sh", "-ec",
                f"mkdir -m 755 {_PRIVATE}; "
                f"git -C {_TESTBED} update-ref refs/heads/eval-base \"$1\"; "
                f"git clone -q --bare --depth 1 --single-branch -b eval-base "
                f"file://{_TESTBED} {_BASELINE}; "
                f"git --git-dir={_BASELINE} remote remove origin; "
                f"rm -rf {_TESTBED}; mkdir -m 777 {_TESTBED}",
                "sh", self.base_commit,
            ),
            root=True,
        )
        self._checked((
            "/bin/sh", "-ec",
            f"cp -R {_BASELINE} .git; git config core.bare false; git reset -q --hard",
        ))
        if self._checked(("git", "status", "--porcelain", "--untracked-files=all")):
            raise RuntimeError("The task checkout differs from the pinned base commit")
        if not self.container.put_archive(_PRIVATE, _tool_archive()):
            raise RuntimeError("Could not install the trial's file tools")

    def _checked(self, argv: tuple[str, ...], *, root: bool = False) -> bytes:
        code, (stdout, stderr) = self.container.exec_run(
            argv, demux=True, workdir="/" if root else _TESTBED, user="0:0" if root else _AGENT,
            environment={
                "PATH": "/usr/bin:/bin",
                "GIT_CONFIG_GLOBAL": "/dev/null",
                "GIT_CONFIG_NOSYSTEM": "1",
            } if root else None,
        )
        if code:
            raise RuntimeError((stderr or stdout or b"").decode(errors="replace"))
        return stdout or b""

    def _check_workspace(self, workspace: Path) -> None:
        if workspace != self.path:
            raise RuntimeError("Docker command does not have a safe workspace")

    def execute_file(
        self, name: str, workspace: Path, arguments: Mapping[str, Any],
    ) -> ToolResult:
        self._check_workspace(workspace)
        path = arguments.get("path")
        if name == "read_file" and isinstance(path, str) and (
            path == "@keyline/artifacts" or path.startswith("@keyline/artifacts/")
        ):
            return read_file(workspace, arguments)
        # Split JSON into argv chunks to stay below Linux's per-argument byte limit.
        payload = json.dumps(dict(arguments))
        chunks = tuple(payload[i : i + 60_000] for i in range(0, len(payload), 60_000))
        output = self._checked((
            "/usr/bin/timeout", "-k", "1s", "60s",
            _PYTHON, "-I", "-B", _TOOL_APP, "file", name, *chunks,
        ))
        return ToolResult(**json.loads(output))

    def execute(
        self, workspace: Path, arguments: Mapping[str, Any], *, test_command: bool = False,
    ) -> ToolResult:
        self._check_workspace(workspace)
        argv, timeout = parse_process_arguments(arguments, timeout_limit=COMMAND_TIMEOUT_SECONDS)
        if test_command and Path(argv[0]).name == "sh":
            argv = ("/bin/bash", "--posix", "-o", "pipefail", *argv[1:])
        elif test_command and Path(argv[0]).name == "bash":
            argv = (argv[0], "-o", "pipefail", *argv[1:])
        try:
            output = self._checked((
                _PYTHON, "-I", "-B", _TOOL_APP, "command", str(timeout), *argv,
            ))
        except DockerException as exc:
            return ToolResult(ok=False, stderr=str(getattr(exc, "explanation", exc)))
        self._check_checkout_files()
        return ToolResult(**json.loads(output))

    def _check_checkout_files(self) -> None:
        # A FIFO or socket can block a later file tool.
        special = self._checked((
            "/usr/bin/find", ".", "!", "-type", "f", "!", "-type", "d",
            "!", "-type", "l", "-print", "-quit",
        ))
        if special:
            path = special.decode(errors="replace").strip()
            raise RuntimeError(f"Unsupported workspace file: {path}")

    def export_patch(self) -> str:
        self._check_checkout_files()
        return self._checked(
            (_PYTHON, "-I", "-B", _TOOL_APP, "patch", _BASELINE, self.base_commit)
        ).decode("utf-8")

    def checkpoint(self) -> str:
        """Save the filesystem of a paused trial so its container can be removed."""
        return self.container.commit().id
