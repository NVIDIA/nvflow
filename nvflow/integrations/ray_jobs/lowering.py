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
"""Ray-specific lowering from portable grouped tasks to Jobs API tasks."""

from __future__ import annotations

from dataclasses import replace

from nvflow.execution.contracts import (
    DependencyPlan,
    ExecutionGraph,
    ResourceRequest,
    ServicePlan,
    TaskPlan,
)


def lower_execution_graph(graph: ExecutionGraph) -> ExecutionGraph:
    """Split heterogeneous command groups while retaining source identity."""
    split_by_source = {
        task.task_id: (
            ()
            if task.operation == "cleanup"
            and all(command.container == "host" for command in task.commands)
            else _split_task(task)
        )
        for task in graph.tasks
    }
    terminal_ids = {
        task_id: tuple(
            child.task_id
            for child in children
            if not any(command.kind == "service" for command in child.commands)
        )
        or tuple(child.task_id for child in children)
        for task_id, children in split_by_source.items()
    }
    lowered_stages = []
    for stage in graph.stages:
        tasks = []
        for source in stage.tasks:
            children = split_by_source[source.task_id]
            services = [
                child
                for child in children
                if any(command.kind == "service" for command in child.commands)
            ]
            consumers = [
                child
                for child in children
                if all(command.kind != "service" for command in child.commands)
            ]
            for child in children:
                dependencies = list(_lower_dependencies(source.dependencies, terminal_ids))
                if child in consumers:
                    dependencies.extend(
                        DependencyPlan(service.task_id, condition="after_ready")
                        for service in services
                    )
                child_services = ()
                if child in services:
                    declared = [
                        service for service in source.services if service.task_id == source.task_id
                    ]
                    if declared:
                        child_services = tuple(
                            replace(
                                service,
                                task_id=child.task_id,
                                consumer_task_id=_lower_consumer(
                                    service.consumer_task_id,
                                    terminal_ids,
                                ),
                            )
                            for service in declared
                        )
                    else:
                        consumer_id = consumers[0].task_id if consumers else None
                        command = child.commands[0]
                        child_services = (
                            ServicePlan(
                                name=command.role,
                                task_id=child.task_id,
                                readiness=command.readiness or "after_start",
                                cleanup="always",
                                consumer_task_id=consumer_id,
                            ),
                        )
                outputs = (
                    source.outputs
                    if child == (consumers[-1] if consumers else children[-1])
                    else ()
                )
                tasks.append(
                    replace(
                        child,
                        dependencies=tuple(dependencies),
                        services=child_services,
                        outputs=outputs,
                    )
                )
        lowered_stages.append(replace(stage, tasks=tuple(tasks)))
    return replace(
        graph,
        stages=tuple(lowered_stages),
    )


def _split_task(task: TaskPlan) -> tuple[TaskPlan, ...]:
    if len(task.commands) == 1:
        command = task.commands[0]
        if command.ray_application and not task.parameters.get("ray_application"):
            return (
                replace(
                    task,
                    parameters={**task.parameters, "ray_application": True},
                ),
            )
        return (task,)
    consumers = [command for command in task.commands if command.kind != "service"]
    if len(consumers) > 1:
        raise ValueError(
            f"Ray task {task.task_id!r} has multiple ordered non-service commands; "
            "declare separate dependency-linked tasks"
        )
    resources = {resource.role: resource for resource in task.resources}
    service_groups = {command.group for command in task.commands if command.kind == "service"}
    children = []
    for index, command in enumerate(task.commands):
        resource = resources.get(command.group)
        if resource is None or (command.kind != "service" and command.group in service_groups):
            resource = ResourceRequest(
                role="main",
                nodes=1,
                gpus_per_node=0,
                tasks_per_node=1,
            )
        parameters = dict(task.parameters)
        parameters.update(
            {
                "source_task_id": task.task_id,
                "source_task_name": task.name,
                "source_command_index": index,
            }
        )
        if command.ray_application:
            parameters["ray_application"] = True
        if command.kind == "service" and resource.nodes > 1:
            parameters["ray_application"] = True
        children.append(
            replace(
                task,
                task_id=f"{task.task_id}.command-{index}",
                name=f"{task.name}-{command.role}",
                operation=("service" if command.kind == "service" else task.operation),
                commands=(command,),
                resources=(resource,),
                dependencies=(),
                outputs=(),
                services=(),
                parameters=parameters,
            )
        )
    return tuple(children)


def _lower_dependencies(
    dependencies: tuple[DependencyPlan, ...],
    terminal_ids: dict[str, tuple[str, ...]],
) -> tuple[DependencyPlan, ...]:
    lowered = []
    for dependency in dependencies:
        if dependency.external:
            lowered.append(dependency)
            continue
        for upstream in terminal_ids[dependency.upstream]:
            lowered.append(replace(dependency, upstream=upstream))
    return tuple(lowered)


def _lower_consumer(
    consumer_task_id: str | None,
    terminal_ids: dict[str, tuple[str, ...]],
) -> str | None:
    if consumer_task_id is None:
        return None
    lowered = terminal_ids[consumer_task_id]
    if len(lowered) != 1:
        raise ValueError(
            f"Ray service consumer {consumer_task_id!r} lowers to {len(lowered)} terminal tasks"
        )
    return lowered[0]


__all__ = ["lower_execution_graph"]
