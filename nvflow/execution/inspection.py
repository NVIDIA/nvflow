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
"""Collect explicit filesystem observations during live stage planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from nvflow.execution.contracts import completion_task_ids

if TYPE_CHECKING:
    from collections.abc import Callable

    from nvflow.execution.contracts import LaunchPlan
    from nvflow.execution.filesystem import TargetFilesystem

OutputPresence = Literal["exists", "missing", "not_declared", "unavailable"]


@dataclass(frozen=True)
class StageOutputInspection:
    """Presence of the outputs declared by the tasks that finish one stage."""

    stage: str
    presence: OutputPresence
    declared_paths: tuple[str, ...] = ()
    existing_paths: tuple[str, ...] = ()
    hidden_paths: tuple[str, ...] = ()


def inspect_declared_outputs(
    plan: LaunchPlan,
    filesystem: TargetFilesystem | None,
    *,
    visible: Callable[[str], bool] | None = None,
) -> tuple[StageOutputInspection, ...]:
    """Inspect the stages' final output paths in one cached filesystem batch.

    Outputs of earlier tasks, such as data shared with other workflows, do not
    show that the stage itself produced anything. Paths that ``visible``
    rejects are not queried: their presence is unknown, not missing.
    """
    declared_by_stage: dict[str, tuple[str, ...]] = {}
    for stage in plan.stages:
        final = set(completion_task_ids(stage.tasks))
        declared_by_stage[stage.stage] = tuple(
            dict.fromkeys(
                output.path
                for task in stage.tasks
                if task.task_id in final
                for output in task.outputs
                if output.path
            )
        )

    def shown(path: str) -> bool:
        return visible is None or visible(path)

    query_paths = sorted(
        {path for paths in declared_by_stage.values() for path in paths if shown(path)}
    )
    existence = (
        filesystem.batch_exists(query_paths) if filesystem is not None and query_paths else {}
    )

    inspections = []
    for stage in plan.stages:
        declared = declared_by_stage[stage.stage]
        hidden = tuple(path for path in declared if not shown(path))
        existing = tuple(path for path in declared if existence.get(path, False))
        if existing:
            presence: OutputPresence = "exists"
        elif not stage.complete:
            presence = "unavailable"
        elif not declared:
            presence = "not_declared"
        elif filesystem is None or len(hidden) == len(declared):
            presence = "unavailable"
        else:
            presence = "missing"
        inspections.append(
            StageOutputInspection(
                stage=stage.stage,
                presence=presence,
                declared_paths=declared,
                existing_paths=existing,
                hidden_paths=hidden,
            )
        )
    return tuple(inspections)


__all__ = [
    "OutputPresence",
    "StageOutputInspection",
    "inspect_declared_outputs",
]
