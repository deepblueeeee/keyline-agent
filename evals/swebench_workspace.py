"""Expose one clean SWE-bench checkout to file tools and isolated commands."""

from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryFile
from time import monotonic
from typing import Any

from docker import DockerClient
from docker.errors import DockerException
from docker.models.containers import Container
from docker.types import LogConfig, Mount
from requests.exceptions import ConnectionError as DockerConnectionError
from requests.exceptions import ReadTimeout

from handshake.runtime.state import ToolResult

from .swebench import IMAGE_TAG

_IMAGE = f"swebench/sweb.eval.x86_64.sympy_1776_sympy-20590:{IMAGE_TAG}"
_TESTBED = "/testbed"
_PATH = "/opt/miniconda3/envs/testbed/bin:/usr/local/bin:/usr/bin:/bin"


class DockerWorkspace:
    def __init__(self, client: DockerClient, path: Path, base_commit: str) -> None:
        self.client = client
        self.path = path.resolve()
        self.base_commit = base_commit
        self.git_dir = self.path.parent / "baseline.git"
        self._cleanup_failed = False

    @contextmanager
    def _container(self, argv: tuple[str, ...], *, bind: bool) -> Iterator[Container]:
        # A fresh command container prevents background processes from racing host file tools.
        mounts = [Mount(_TESTBED, str(self.path), type="bind")] if bind else []
        container = self.client.containers.create(
            _IMAGE,
            command=list(argv),
            platform="linux/amd64",
            working_dir=_TESTBED,
            user=f"{os.getuid()}:{os.getgid()}" if bind else "0:0",
            mounts=mounts,
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
            container.reload()
            config = container.attrs["HostConfig"]
            actual = container.attrs["Mounts"]
            expected = {str(self.path), f"/host_mnt{self.path}"}
            if (
                config["NetworkMode"] != "none"
                or config["Privileged"]
                or config.get("Binds")
                or config.get("CapAdd")
                or "ALL" not in (config.get("CapDrop") or [])
                or "no-new-privileges" not in (config.get("SecurityOpt") or [])
                or len(actual) != int(bind)
                or (
                    bind
                    and (
                        actual[0]["Type"] != "bind"
                        or actual[0]["Source"] not in expected
                        or actual[0]["Destination"] != _TESTBED
                        or not actual[0]["RW"]
                    )
                )
            ):
                raise RuntimeError("Docker workspace isolation verification failed")
            container.start()
            yield container
        finally:
            try:
                container.remove(force=True)
            except Exception:
                self._cleanup_failed = True
                raise
            if bind:
                self._check_checkout_files()

    def prepare(self) -> None:
        # Upload-pack sends only the base snapshot. Copying /testbed/.git directly can
        # expose unrelated refs and objects from the official scoring image.
        command = (
            "/bin/sh",
            "-ec",
            'git update-ref refs/heads/eval-base "$1"; '
            "git clone -q --depth 1 -b eval-base file:///testbed /tmp/source; "
            "git -C /tmp/source remote remove origin",
            "sh",
            self.base_commit,
        )
        self.path.mkdir()
        with self._container(command, bind=False) as container:
            if container.wait(timeout=90)["StatusCode"] != 0:
                raise RuntimeError(container.logs().decode(errors="replace"))
            with TemporaryFile() as archive:
                for chunk in container.get_archive("/tmp/source/.")[0]:
                    archive.write(chunk)
                archive.seek(0)
                with tarfile.open(fileobj=archive) as checkout:
                    checkout.extractall(self.path, filter="data")
        shutil.copytree(self.path / ".git", self.git_dir)
        if self._git("rev-parse", "HEAD").strip() != self.base_commit.encode("ascii") or self._git(
            "status", "--porcelain", "--untracked-files=all"
        ):
            raise RuntimeError("The task checkout differs from the pinned base commit")

    def execute(self, workspace: Path, arguments: Mapping[str, Any]) -> ToolResult:
        from handshake.tools.shell import parse_process_arguments

        if workspace.resolve() != self.path or self._cleanup_failed:
            raise RuntimeError("Docker command does not have a safe workspace")
        argv, timeout = parse_process_arguments(arguments)
        timeout = min(timeout, 60)
        try:
            with self._container(argv, bind=True) as container:
                deadline = monotonic() + timeout
                try:
                    code = container.wait(timeout=timeout)["StatusCode"]
                except (ReadTimeout, DockerConnectionError) as exc:
                    # Requests may report a timeout on a streamed response as a connection error.
                    if isinstance(exc, DockerConnectionError) and monotonic() < deadline:
                        raise
                    return ToolResult(ok=False, stderr=f"Command timed out after {timeout:g}s")
                return ToolResult(
                    ok=code == 0,
                    stdout=container.logs(stdout=True, stderr=False).decode(errors="replace"),
                    stderr=container.logs(stdout=False, stderr=True).decode(errors="replace"),
                    exit_code=code,
                )
        except DockerException as exc:
            if self._cleanup_failed:
                raise RuntimeError("Could not remove the command container") from exc
            return ToolResult(ok=False, stderr=str(getattr(exc, "explanation", exc)))

    def _git(self, *args: str) -> bytes:
        return subprocess.run(
            [
                "git",
                f"--git-dir={self.git_dir}",
                f"--work-tree={self.path}",
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "core.fsmonitor=false",
                *args,
            ],
            cwd=self.path,
            env={"PATH": os.defpath, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1"},
            capture_output=True,
            check=True,
            timeout=60,
        ).stdout

    def _check_checkout_files(self) -> None:
        # A command can create FIFOs that would block a subsequent host search or read.
        for directory, _, names in os.walk(self.path):
            for name in names:
                path = Path(directory) / name
                if not path.is_symlink() and not path.is_file():
                    raise RuntimeError(f"Unsupported workspace file: {path.relative_to(self.path)}")

    def export_patch(self) -> str:
        if self._cleanup_failed:
            raise RuntimeError("Cannot export a checkout still in use by a command container")
        self._check_checkout_files()
        self._git("add", "-A")
        return self._git(
            "diff", "--cached", "--binary", "--no-ext-diff", "--no-textconv", self.base_commit
        ).decode("utf-8")
