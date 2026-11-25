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
"""Plan a chunked data stage's steps as cluster tasks.

Each chunk runs its steps in order, then a final step runs after every chunk.
Consecutive steps share a task when they need the same container and GPUs, or
when a step is portable (needs only NVFlow and the standard library). Plans
include finished steps: only the worker can tell whether a completion record
still matches the inputs and settings, and it skips the work when it does.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from nvflow.execution import ArtifactDeclaration, DependencyPlan, StagePlan, declare_task_outputs
from nvflow.execution.operations import task_from_invocation

if TYPE_CHECKING:
    from nvflow.execution import PlanningContext, TaskPlan


def input_files(config: Mapping[str, Any]) -> list[str]:
    """Return ``input_file`` or ``input_files`` as a list of paths."""
    single, many = config.get("input_file"), config.get("input_files")
    if (single is None) == (many is None):
        raise ValueError("set exactly one of 'input_file' or 'input_files'")
    files = [single] if single is not None else many
    if not isinstance(files, list | tuple) or not all(
        isinstance(path, str) and path for path in files
    ):
        raise ValueError("'input_files' must be a non-empty list of paths")
    if not files:
        raise ValueError("'input_files' must be a non-empty list of paths")
    return list(files)


def validate_chunking(config: Mapping[str, Any]) -> None:
    for key, minimum in (("num_chunks", 1), ("dependent_jobs", 0)):
        value = config.get(key, minimum)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"'{key}' must be an integer of at least {minimum}")


def require_live_inputs(context: PlanningContext, paths: Sequence[str]) -> None:
    """Fail a live plan early when no upstream stage will create missing inputs."""
    if not context.live or context.run_after or context.filesystem is None:
        return
    present = context.filesystem.batch_exists(list(paths))
    missing = [path for path in paths if not present.get(path)]
    if missing:
        raise ValueError(
            f"live {context.stage} plan: required input artifact(s) are missing: "
            + ", ".join(missing)
        )


@dataclass(frozen=True)
class Step:
    """One idempotent command and the completion record it publishes."""

    name: str
    command: str
    completion: str
    container: str = "nemo-skills"
    num_gpus: int = 0
    portable: bool = False
    outputs: tuple[ArtifactDeclaration, ...] = ()


def _runtime(group: Sequence[Step]) -> tuple[str, int] | None:
    return next(((step.container, step.num_gpus) for step in group if not step.portable), None)


def _joins(group: Sequence[Step], step: Step) -> bool:
    runtime = _runtime(group)
    return step.portable or runtime is None or runtime == (step.container, step.num_gpus)


def _groups(steps: Sequence[Step]) -> list[list[Step]]:
    groups: list[list[Step]] = []
    for step in steps:
        if groups and _joins(groups[-1], step):
            groups[-1].append(step)
        else:
            groups.append([step])
    return groups


def _add_step_tasks(
    context: PlanningContext,
    tasks: list[TaskPlan],
    group: Sequence[Step],
    *,
    name: str,
    upstream: list[str] | None,
    log_dir: str,
    dependent_jobs: int,
) -> str:
    """Append one task running *group* (plus retries) and return the last task's id.

    Without *upstream* task ids, the task follows the stage's own dependencies.
    """
    container, num_gpus = _runtime(group) or (group[0].container, group[0].num_gpus)
    command = "set -euo pipefail\n" + "\n".join(step.command for step in group)
    outputs = [output for step in group for output in step.outputs]
    outputs += [ArtifactDeclaration(f"{step.name}_completion", step.completion) for step in group]
    previous = upstream
    for attempt in range(dependent_jobs + 1):
        keyword: dict[str, object] = {
            "cluster": context.target.cluster,
            "expname": name + (f"-dep{attempt}" if attempt else ""),
            "log_dir": log_dir,
            "container": container,
        }
        if num_gpus:
            keyword["num_gpus"] = num_gpus
        if context.config.get("installation_command") and container == str(
            context.config.get("container") or "nemo-skills"
        ):
            keyword["installation_command"] = context.config["installation_command"]
        with declare_task_outputs(*outputs):
            task = task_from_invocation(
                context,
                index=len(tasks),
                operation="nemo_skills.run_cmd",
                positional=(),
                keyword=keyword,
                arguments=command,
            )
        if previous is not None:
            # A retry follows any end of the attempt before it, to resume after a
            # timeout. A first attempt needs the steps before it to succeed: their
            # old outputs and completion records from an earlier run would
            # otherwise pass its checks.
            condition = "after_any" if attempt else "after_success"
            task = replace(
                task,
                dependencies=tuple(
                    DependencyPlan(task_id, condition=condition) for task_id in previous
                ),
            )
        tasks.append(task)
        previous = [task.task_id]
    return tasks[-1].task_id


def plan_steps(
    context: PlanningContext,
    *,
    chunks: Sequence[Sequence[Step]],
    final: Step,
    log_dir: str,
    dependent_jobs: int = 0,
) -> StagePlan:
    """Plan each chunk's steps, then *final* after all of them."""
    tasks: list[TaskPlan] = []
    terminals: list[str] = []
    final_planned = False
    single = len(chunks) == 1
    for index, steps in enumerate(chunks):
        groups = _groups(steps)
        if single and groups and _joins(groups[-1], final):
            groups[-1].append(final)
            final_planned = True
        previous: list[str] | None = None
        for group in groups:
            label = "-".join(step.name for step in group)
            name = (
                f"{context.expname}-{label}"
                if single
                else f"{context.expname}-chunk{index}-{label}"
            )
            previous = [
                _add_step_tasks(
                    context,
                    tasks,
                    group,
                    name=name,
                    upstream=previous,
                    log_dir=log_dir,
                    dependent_jobs=dependent_jobs,
                )
            ]
        if previous is not None:
            terminals.extend(previous)
    if not final_planned:
        _add_step_tasks(
            context,
            tasks,
            [final],
            name=f"{context.expname}-{final.name}",
            upstream=terminals or None,
            log_dir=log_dir,
            dependent_jobs=dependent_jobs,
        )
    return StagePlan(stage=context.stage, tasks=tuple(tasks))


__all__ = [
    "Step",
    "input_files",
    "plan_steps",
    "require_live_inputs",
    "validate_chunking",
]
