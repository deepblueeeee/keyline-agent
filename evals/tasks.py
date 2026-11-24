"""Pinned references to a small set of official SWE-bench tasks."""

import re
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

Suite = Literal["regression", "capability"]
DEFAULT_DATASET = Path(__file__).with_name("tasks.toml")


@dataclass(frozen=True, slots=True)
class Task:
    instance_id: str
    source: str
    revision: str
    suite: Suite
    image_digest: str
    tags: tuple[str, ...] = ()

    @property
    def image_tag(self) -> str:
        return f"latest@{self.image_digest}"

    def as_dict(self) -> dict[str, object]:
        return {**asdict(self), "tags": list(self.tags)}


def load_tasks(path: Path = DEFAULT_DATASET) -> tuple[Task, ...]:
    with path.open("rb") as stream:
        dataset = tomllib.load(stream)
    tasks = tuple(
        Task(**{**dataset.get("benchmark", {}), **entry, "tags": tuple(entry.get("tags", ()))})
        for entry in dataset.get("tasks", [])
    )
    for task in tasks:
        if (
            not re.fullmatch(r"[\w.-]+__[\w.-]+-\d+", task.instance_id)
            or task.suite not in {"regression", "capability"}
            or not re.fullmatch(r"[0-9a-f]{40}", task.revision)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", task.image_digest)
        ):
            raise ValueError(f"Invalid task configuration for {task.instance_id}")
    if not tasks or len({task.instance_id for task in tasks}) != len(tasks):
        raise ValueError("Dataset must contain unique task IDs")
    return tasks


def select_tasks(
    tasks: tuple[Task, ...], *, instance_ids: list[str] | None, suite: Suite | None
) -> tuple[Task, ...]:
    if instance_ids:
        selected = set(instance_ids)
        if missing := selected - {task.instance_id for task in tasks}:
            raise ValueError(f"Unknown instance IDs: {', '.join(sorted(missing))}")
        return tuple(task for task in tasks if task.instance_id in selected)
    matches = tuple(task for task in tasks if task.suite == suite)
    if not matches:
        raise ValueError(f"Dataset has no {suite} tasks")
    return matches
