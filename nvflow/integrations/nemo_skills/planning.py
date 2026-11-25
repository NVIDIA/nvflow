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
"""Translate NeMo-Skills declarative jobs into backend-neutral launch plans."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from nvflow.execution.contracts import (
    ArtifactDeclaration,
    CommandPlan,
    DependencyPlan,
    ResourceRequest,
    StagePlan,
    TaskPlan,
)

PlannedJobs = list[dict[str, Any]]
"""Compatibility alias for NeMo-Skills job dictionaries."""


def stage_plan_from_jobs(
    stage: str,
    jobs: Sequence[dict[str, Any]],
    *,
    output_dir: str | None = None,
    task_prefix: str | None = None,
    experiment_name: str | None = None,
) -> StagePlan:
    """Convert already-built NeMo-Skills jobs without submitting them."""
    from nemo_skills.pipeline.utils.scripts import ServerScript

    if not jobs:
        return StagePlan.noop(stage, "no jobs were produced")

    prefix = task_prefix or stage
    task_ids = {id(job): f"{prefix}.{index}" for index, job in enumerate(jobs)}
    tasks = []
    for index, job in enumerate(jobs):
        groups = job.get("groups") or [job.get("group")]
        groups = [group for group in groups if group is not None]
        # hostname_ref() needs every heterogeneous group index before any script renders.
        heterogeneous = len(groups) > 1
        for group_index, group in enumerate(groups):
            for command in group.commands:
                command.script.het_group_index = group_index if heterogeneous else None
        commands = []
        resources = []
        group_log_dirs = {}
        for group_index, group in enumerate(groups):
            hardware = group.hardware
            group_name = str(group.name or f"group-{group_index}")
            group_log_dirs[group_name] = getattr(group, "log_dir", None)
            resources.append(
                ResourceRequest(
                    role=group_name,
                    nodes=hardware.num_nodes or 1,
                    gpus_per_node=hardware.num_gpus,
                    tasks_per_node=hardware.num_tasks or 1,
                    partition=hardware.partition,
                    account=hardware.account,
                    exclusive=(hardware.sbatch_kwargs or {}).get("exclusive"),
                )
            )
            for command in group.commands:
                inline = getattr(command.script, "inline", "")
                rendered = str(inline() if callable(inline) else inline)
                readiness = None
                if isinstance(command.script, ServerScript):
                    readiness = getattr(command.script, "nvflow_readiness", None)
                    if not isinstance(readiness, str) or not readiness:
                        raise ValueError(
                            "ServerScript must declare an executable nvflow_readiness contract"
                        )
                commands.append(
                    CommandPlan(
                        command=rendered,
                        container=str(command.container),
                        runtime_role=str(command.container),
                        role=str(command.name or group.name or "main"),
                        runtime_template=False,
                        group=group_name,
                        kind=("service" if isinstance(command.script, ServerScript) else "command"),
                        readiness=readiness,
                        ray_application=(
                            job.get("role") in {"collect-rollouts", "rejudge-rollouts"}
                            and not isinstance(command.script, ServerScript)
                        ),
                    )
                )

        dependencies = []
        dependency_condition = _group_dependency_condition(groups)
        for dependency in job.get("dependencies") or []:
            internal_id = task_ids.get(id(dependency))
            dependencies.append(
                DependencyPlan(
                    upstream=internal_id or str(dependency),
                    condition=dependency_condition,
                    external=internal_id is None,
                )
            )
        outputs = (
            (
                ArtifactDeclaration(
                    name="output",
                    path=output_dir,
                    kind="directory",
                ),
            )
            if output_dir and index == len(jobs) - 1
            else ()
        )
        tasks.append(
            TaskPlan(
                task_id=task_ids[id(job)],
                stage=stage,
                name=str(job["name"]),
                operation="nemo_skills.pipeline",
                commands=tuple(commands),
                resources=tuple(resources),
                dependencies=tuple(dependencies),
                outputs=outputs,
                parameters={
                    "experiment_name": job.get(
                        "_nvflow_experiment",
                        experiment_name,
                    ),
                    "job_index": index,
                    "logical_role": job.get("role"),
                    "managed_model_paths": tuple(job.get("_nvflow_managed_model_paths") or ()),
                    "group_log_dirs": group_log_dirs,
                },
            )
        )
    return StagePlan(stage=stage, tasks=tuple(tasks))


def _group_dependency_condition(
    groups: Sequence[Any],
) -> Literal["after_success", "after_start", "after_any"]:
    dependency_types = {
        (group.hardware.sbatch_kwargs or {}).get("dependency_type")
        for group in groups
        if (group.hardware.sbatch_kwargs or {}).get("dependency_type")
    }
    if len(dependency_types) > 1:
        raise ValueError(
            "NeMo-Skills job groups disagree on dependency_type: "
            + ", ".join(sorted(dependency_types))
        )
    dependency_type = next(iter(dependency_types), None)
    if dependency_type == "afterany":
        return "after_any"
    if dependency_type == "after":
        return "after_start"
    return "after_success"


def portable_cluster_config(*, gpus_per_node: int = 8) -> dict[str, Any]:
    """Return scheduler-free aliases used only while building exact commands."""
    if gpus_per_node < 1:
        raise ValueError("gpus_per_node must be positive")
    roles = (
        "nemo-gym",
        "nemo-rl",
        "nemo-skills",
        "sglang",
        "vllm",
    )
    return {
        "executor": "none",
        "gpus_per_node": gpus_per_node,
        "mounts": ["/:/"],
        "containers": {role: role for role in roles},
    }
