# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Labels that tie scheduler jobs to the workflow, stage, and experiment they run.

Slurm jobs carry the label as their job comment and Ray Jobs as metadata, so
``nvflow status`` can show and select jobs the same way on either engine.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path, PurePosixPath

from nvflow.execution.contracts import TaskPlan
from nvflow.project.context import ProjectContext
from nvflow.project.workflow_selector import WorkflowSelector

_COMMENT_PREFIX = "nvflow"
# Characters shlex.quote leaves alone, without the "," and "=" that separate fields.
_UNSAFE_CHARACTER = re.compile(r"[^A-Za-z0-9_@%+:./-]")


@dataclass(frozen=True, slots=True)
class JobLabel:
    """The workflow, stage, and experiment one scheduler job belongs to."""

    workflow: str = ""
    stage: str = ""
    experiment: str = ""

    def __post_init__(self) -> None:
        for name in _FIELD_NAMES:
            object.__setattr__(self, name, label_value(getattr(self, name)))

    def to_comment(self) -> str:
        """Render the label as a Slurm job comment that needs no shell quoting."""
        return ",".join([_COMMENT_PREFIX, *(f"{name}={value}" for name, value in self._items())])

    @classmethod
    def from_comment(cls, comment: str) -> JobLabel | None:
        """Parse a comment written by :meth:`to_comment`; any other comment gives None."""
        prefix, _, body = comment.strip().partition(",")
        if prefix != _COMMENT_PREFIX:
            return None
        values = {}
        for item in body.split(","):
            name, separator, value = item.partition("=")
            if separator and name in _FIELD_NAMES:
                values[name] = value
        return cls(**values)

    def to_metadata(self) -> dict[str, str]:
        """Render the label as Ray Jobs metadata entries."""
        return {f"nvflow_{name}": value for name, value in self._items()}

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, str]) -> JobLabel | None:
        """Read a label from Ray Jobs metadata; metadata without one gives None."""
        values = {
            name: str(metadata[f"nvflow_{name}"])
            for name in _FIELD_NAMES
            if metadata.get(f"nvflow_{name}")
        }
        return cls(**values) if values else None

    def _items(self) -> tuple[tuple[str, str], ...]:
        return tuple((name, value) for name in _FIELD_NAMES if (value := getattr(self, name)))


_FIELD_NAMES = tuple(field.name for field in fields(JobLabel))


def label_value(value: str) -> str:
    """Return *value* as job labels store it, for comparing with a stored label."""
    return _UNSAFE_CHARACTER.sub("_", value)


def task_label(workflow: str | None, task: TaskPlan) -> JobLabel:
    """Return the label every scheduler job submitted for *task* carries."""
    return JobLabel(
        workflow=workflow or "",
        stage=task.stage,
        experiment=str(
            task.parameters.get("experiment_name") or task.parameters.get("expname") or task.name
        ),
    )


def workflow_label(context: ProjectContext, config_path: str | Path) -> str:
    """Return the selector naming a workflow file, or its path outside the project's workflows."""
    path = Path(config_path).resolve()
    try:
        relative = path.relative_to(context.workflows_root.resolve()).with_suffix("")
    except ValueError:
        return path.as_posix()
    return str(
        WorkflowSelector(project=context.identity, relative_path=PurePosixPath(relative.as_posix()))
    )


def planned_resources(task: TaskPlan) -> tuple[int | None, int | None]:
    """Return the nodes and GPUs per node *task* requests.

    GPUs per node is None when the task's resource groups request different counts.
    """
    if not task.resources:
        return None, None
    gpus = {resource.gpus_per_node for resource in task.resources}
    return (
        sum(resource.nodes for resource in task.resources),
        next(iter(gpus)) if len(gpus) == 1 else None,
    )


__all__ = ["JobLabel", "label_value", "planned_resources", "task_label", "workflow_label"]
