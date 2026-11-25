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
"""Run one pipeline stage as an ordered group of substages."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from nvflow.execution import PlannedExecutionStage

if TYPE_CHECKING:
    from nvflow.core import BaseStage
    from nvflow.execution import PlanningContext, StagePlan, TaskPlan

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


class SubstagesStage(PlannedExecutionStage):
    """Run the stage's ``substages`` in order, in the stage's ``container``.

    Each substage is an ordinary stage type configured as it would be on its
    own. It inherits the stage's ``container``, which it cannot change, and
    ``installation_command``, and it keeps its own outputs and completion
    records. On Slurm, consecutive substage tasks on the same hardware run as one
    job.
    """

    def plan(self, context: PlanningContext) -> StagePlan:
        from nvflow.execution import StagePlan, internalize_dependencies, rekey_tasks
        from nvflow.execution.contracts import SLURM_GROUP_JOBS, completion_task_ids

        tasks: list[TaskPlan] = []
        aliases: dict[str, tuple[str, ...]] = {}
        upstream = tuple(context.run_after)
        for name in context.config["substages"]:
            expname = f"{context.expname}-{name}"
            child = self._substage(context.launcher_config, name).execution(
                replace(
                    context,
                    stage=name,
                    config=_substage_config(context.config, name),
                    launcher_config=_substage_config(context.launcher_config, name),
                    expname=expname,
                    run_after=upstream,
                )
            )
            if child.disposition == "noop":
                continue
            planned = rekey_tasks(
                child.tasks, stage=context.stage, prefix=f"{context.stage}.{name}"
            )
            tasks.extend(planned)
            aliases[expname] = completion_task_ids(planned)
            upstream = (expname,)
        if not tasks:
            return StagePlan.noop(context.stage, "every substage is complete")
        linked = internalize_dependencies(tuple(tasks), fallback_aliases=aliases)
        return StagePlan(
            stage=context.stage,
            tasks=tuple(
                replace(task, parameters={**task.parameters, SLURM_GROUP_JOBS: True})
                for task in linked
            ),
        )

    def validate_config(self, config: dict[str, Any]) -> None:
        container = config.get("container")
        if not isinstance(container, str) or not container:
            raise ValueError("a stage with 'substages' must set its 'container'")
        substages = config.get("substages")
        if not isinstance(substages, Mapping) or not substages:
            raise ValueError("'substages' must map each substage name to its configuration")
        for name, substage in substages.items():
            if not isinstance(name, str) or not _NAME.fullmatch(name):
                raise ValueError(f"invalid substage name {name!r}")
            if not isinstance(substage, Mapping) or not isinstance(substage.get("uses"), str):
                raise ValueError(f"substage {name!r} must declare 'uses'")
            if "dependencies" in substage:
                raise ValueError(
                    f"substage {name!r} runs after the one above it; drop 'dependencies'"
                )
            self._substage(config, name).validate_config(_substage_config(config, name))

    def _substage(self, config: Mapping[str, Any], name: str) -> BaseStage:
        from nvflow.core.workflow_definition import resolve_stage_uses

        project = self.bound_project_context
        stage = resolve_stage_uses(config["substages"][name]["uses"], project)()
        if project is not None:
            stage.bind_project_context(project)
        return stage


def _substage_config(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    substage = {
        key: value
        for key, value in config["substages"][name].items()
        if key not in {"uses", "description"}
    }
    if substage.setdefault("container", config["container"]) != config["container"]:
        raise ValueError(f"substage {name!r} must run in the stage's container")
    if config.get("installation_command") is not None:
        substage.setdefault("installation_command", config["installation_command"])
    if "_environment" in config:
        substage.setdefault("_environment", config["_environment"])
    return substage
