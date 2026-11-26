"""Entrypoint for the eval-only tool zipapp inside a SWE-bench container."""

import json
import os
import shutil
import signal
import subprocess
import sys
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile

import handshake.config  # noqa: F401 - Initialize config before its artifact readers.
from handshake.runtime.state import ToolResult
from handshake.tools import create_default_registry

_MAX_OUTPUT = 4 * 1024 * 1024


def run_command(argv: list[str], timeout: float) -> ToolResult:
    # File-backed output keeps the exec stream separate from command descendants.
    with TemporaryFile() as stdout, TemporaryFile() as stderr:
        try:
            with subprocess.Popen(
                argv, stdout=stdout, stderr=stderr, start_new_session=True,
            ) as command:
                try:
                    code = command.wait(timeout=timeout)
                finally:
                    with suppress(ProcessLookupError):
                        os.killpg(command.pid, signal.SIGKILL)
        except subprocess.TimeoutExpired:
            return ToolResult(ok=False, stderr=f"Command timed out after {timeout:g}s")
        except OSError as exc:
            return ToolResult(ok=False, stderr=f"Could not start process: {exc}")
        if code < 0:
            code = 128 - code
        stdout.seek(0)
        stderr.seek(0)
        output = stdout.read(_MAX_OUTPUT)
        return ToolResult(
            ok=code == 0,
            stdout=output.decode(errors="replace"),
            stderr=stderr.read(_MAX_OUTPUT - len(output)).decode(errors="replace"),
            exit_code=code,
        )


def export_patch(workspace: Path, baseline: Path, base_commit: str) -> bytes:
    # Use the prepared Git metadata, independent of changes to /testbed/.git.
    with TemporaryDirectory() as directory:
        git_dir = Path(directory) / "baseline.git"
        shutil.copytree(baseline, git_dir)

        def git(*args: str) -> bytes:
            return subprocess.check_output(
                [
                    "/usr/bin/git", f"--git-dir={git_dir}", f"--work-tree={workspace}",
                    "-c", "core.bare=false", "-c", "core.hooksPath=/dev/null",
                    "-c", "core.fsmonitor=false", *args,
                ],
                cwd=workspace,
                env={
                    "PATH": os.defpath,
                    "GIT_CONFIG_GLOBAL": "/dev/null",
                    "GIT_CONFIG_NOSYSTEM": "1",
                },
            )

        git("read-tree", base_commit)
        git("add", "-A")
        diff = ("diff", "--cached", "--binary", "--no-ext-diff", "--no-textconv", base_commit)
        patch = git(*diff)
        try:
            patch.decode("utf-8")
        except UnicodeDecodeError:
            # SWE-bench predictions are UTF-8 JSON. Binary patches preserve arbitrary bytes.
            with (git_dir / "info" / "attributes").open("ab") as attributes:
                attributes.write(b"\n* -diff\n")
            patch = git(*diff)
        return patch


if __name__ == "__main__":
    # The ordinary workspace tools use the same standard-library code in both environments.
    if sys.argv[1] == "patch":
        sys.stdout.buffer.write(export_patch(Path.cwd(), Path(sys.argv[2]), sys.argv[3]))
    elif sys.argv[1] == "command":
        print(json.dumps(asdict(run_command(sys.argv[3:], float(sys.argv[2])))))
    else:
        registry = create_default_registry(lambda _path, _argv: False)
        result = registry.execute(sys.argv[2], Path.cwd(), json.loads("".join(sys.argv[3:])))
        print(json.dumps(asdict(result)))
