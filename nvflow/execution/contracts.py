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
"""Serializable contracts between workflow stages and execution backends."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from nvflow.project.provenance import ProjectProvenance

_RUNTIME_INSTALL = re.compile(
    r"(?:^|[\s;&|'\"(])(?:"
    r"python(?:\d+(?:\.\d+)*)?\s+-m\s+pip\s+install"
    r"|pip(?:\d+)?\s+install"
    r"|uv\s+(?:pip\s+install|sync|venv)(?=\s|$)"
    r"|conda\s+install"
    r"|gym\s+env\s+prefetch"
    r")\b",
    re.IGNORECASE | re.MULTILINE,
)
SLURM_GROUP_JOBS = "slurm_group_jobs"


@dataclass(frozen=True)
class ResourceRequest:
    """Resources requested by one allocation group."""

    role: str = "main"
    nodes: int = 1
    gpus_per_node: int | None = None
    tasks_per_node: int = 1
    partition: str | None = None
    account: str | None = None
    exclusive: bool | None = None


@dataclass(frozen=True)
class CommandPlan:
    """One containerized command within a planned task."""

    command: str
    container: str
    role: str = "main"
    runtime_template: bool = False
    group: str = "main"
    kind: Literal["command", "service"] = "command"
    readiness: str | None = None
    display_command: str | None = None
    runtime_role: str | None = None
    ray_application: bool = False

    def __post_init__(self) -> None:
        if not self.container:
            raise ValueError("Command container must not be empty")
        if self.runtime_role is not None and not self.runtime_role:
            raise ValueError("Command runtime_role must not be empty")

    def require_runtime_role(self) -> str:
        if self.runtime_role is None:
            raise ValueError("Exact execution command requires explicit runtime_role")
        return self.runtime_role


@dataclass(frozen=True)
class DependencyPlan:
    """Typed dependency on another plan-local task or external experiment."""

    upstream: str
    condition: Literal[
        "after_success",
        "after_start",
        "after_ready",
        "after_any",
    ] = "after_success"
    external: bool = False


@dataclass(frozen=True)
class ArtifactDeclaration:
    """Output expected from a task before it materializes."""

    name: str
    path: str
    kind: str = "file"
    optional: bool = False


@dataclass(frozen=True)
class ServicePlan:
    """Long-lived service lifecycle consumed by other planned tasks."""

    name: str
    task_id: str
    readiness: str
    cleanup: Literal["always", "on_success", "on_failure"] = "always"
    consumer_task_id: str | None = None


@dataclass(frozen=True)
class TaskPlan:
    """One plan-local scheduler task."""

    task_id: str
    stage: str
    name: str
    operation: str
    commands: tuple[CommandPlan, ...] = ()
    resources: tuple[ResourceRequest, ...] = ()
    dependencies: tuple[DependencyPlan, ...] = ()
    outputs: tuple[ArtifactDeclaration, ...] = ()
    services: tuple[ServicePlan, ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=dict)


def rekey_tasks(
    tasks: tuple[TaskPlan, ...],
    *,
    stage: str,
    prefix: str,
) -> tuple[TaskPlan, ...]:
    """Rekey a composed child plan while preserving internal references."""
    task_ids = {task.task_id: f"{prefix}.{index}" for index, task in enumerate(tasks)}
    rekeyed = []
    for task in tasks:
        dependencies = tuple(
            replace(
                dependency,
                upstream=task_ids.get(dependency.upstream, dependency.upstream),
            )
            if not dependency.external
            else dependency
            for dependency in task.dependencies
        )
        services = tuple(
            replace(
                service,
                task_id=task_ids.get(service.task_id, service.task_id),
                consumer_task_id=(
                    task_ids.get(service.consumer_task_id, service.consumer_task_id)
                    if service.consumer_task_id is not None
                    else None
                ),
            )
            for service in task.services
        )
        rekeyed.append(
            replace(
                task,
                task_id=task_ids[task.task_id],
                stage=stage,
                dependencies=dependencies,
                services=services,
            )
        )
    return tuple(rekeyed)


def completion_task_ids(tasks: Iterable[TaskPlan]) -> tuple[str, ...]:
    """Return, in task order, the tasks whose success completes a group.

    A service is stopped by its cleanup task rather than finishing, so neither
    one signals that the group's work succeeded.
    """
    ordered = tuple(tasks)
    work = tuple(task for task in ordered if task.operation not in {"service", "cleanup"})
    work = work or ordered
    members = {task.task_id for task in work}
    upstream = {
        dependency.upstream
        for task in work
        for dependency in task.dependencies
        if not dependency.external and dependency.upstream in members
    }
    return tuple(task.task_id for task in work if task.task_id not in upstream)


def internalize_dependencies(
    tasks: tuple[TaskPlan, ...],
    *,
    fallback_aliases: Mapping[str, tuple[str, ...]] | None = None,
) -> tuple[TaskPlan, ...]:
    """Resolve selected experiment dependencies to terminal plan-local tasks."""
    aliases: dict[str, set[str]] = {}
    for task in tasks:
        candidates = {
            task.name,
            str(task.parameters.get("experiment_name") or ""),
            str(task.parameters.get("expname") or ""),
        }
        for candidate in candidates - {""}:
            aliases.setdefault(candidate, set()).add(task.task_id)

    terminal_ids: dict[str, tuple[str, ...]] = {
        alias: completion_task_ids(task for task in tasks if task.task_id in members)
        for alias, members in aliases.items()
    }
    for alias, task_ids in (fallback_aliases or {}).items():
        terminal_ids.setdefault(alias, task_ids)

    internalized = []
    for task in tasks:
        dependencies: list[DependencyPlan] = []
        for dependency in task.dependencies:
            resolved = terminal_ids.get(dependency.upstream)
            if dependency.external and dependency.upstream in terminal_ids:
                dependencies.extend(
                    replace(
                        dependency,
                        upstream=task_id,
                        external=False,
                    )
                    for task_id in resolved or ()
                    if task_id != task.task_id
                )
            else:
                dependencies.append(dependency)
        internalized.append(
            replace(
                task,
                dependencies=tuple(dict.fromkeys(dependencies)),
            )
        )
    return tuple(internalized)


@dataclass(frozen=True)
class StagePlan:
    """Plan contribution from one stage.

    ``tasks`` contains work selected by the stage's planning logic. ``complete``
    describes plan coverage only; it is not a runtime stage status.
    """

    stage: str
    tasks: tuple[TaskPlan, ...] = ()
    complete: bool = True
    reason: str | None = None
    disposition: Literal["planned", "noop", "opaque"] = "planned"
    implementation: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.implementation is not None:
            required = {"kind", "name", "contract_version", "provider", "platform"}
            if set(self.implementation) != required:
                raise ValueError(
                    f"Stage {self.stage!r} implementation must contain: "
                    + ", ".join(sorted(required))
                )
            if self.implementation["kind"] not in {"shared", "project"}:
                raise ValueError(f"Stage {self.stage!r} has invalid implementation kind")
            platform = self.implementation["platform"]
            if (
                not isinstance(self.implementation["name"], str)
                or not self.implementation["name"]
                or not isinstance(self.implementation["provider"], str)
                or not self.implementation["provider"]
                or not isinstance(platform, dict)
                or not isinstance(platform.get("distribution"), str)
                or not platform["distribution"]
                or not isinstance(platform.get("version"), str)
                or not platform["version"]
                or not isinstance(platform.get("source"), str)
                or not platform["source"]
            ):
                raise ValueError(f"Stage {self.stage!r} has invalid implementation identity")
            if self.implementation["kind"] == "shared" and (
                isinstance(self.implementation["contract_version"], bool)
                or not isinstance(self.implementation["contract_version"], int)
            ):
                raise ValueError(
                    f"Shared stage {self.stage!r} requires an integer contract_version"
                )
        if self.disposition == "planned" and not self.tasks:
            raise ValueError(f"Planned stage {self.stage!r} must contain at least one task")
        if self.disposition == "noop" and self.tasks:
            raise ValueError(f"No-op stage {self.stage!r} cannot contain tasks")
        if self.disposition == "opaque" and self.complete:
            raise ValueError(f"Opaque stage {self.stage!r} cannot be complete")
        if self.disposition != "opaque" and not self.complete:
            raise ValueError(f"Inspectable stage {self.stage!r} must be complete")

    @property
    def outputs(self) -> tuple[ArtifactDeclaration, ...]:
        """Return task outputs in stable task/declaration order."""
        return tuple(output for task in self.tasks for output in task.outputs)

    @classmethod
    def opaque(cls, stage: str, reason: str) -> StagePlan:
        """Represent a legacy stage that cannot yet be inspected."""
        return cls(
            stage=stage,
            complete=False,
            reason=reason,
            disposition="opaque",
        )

    @classmethod
    def noop(
        cls,
        stage: str,
        reason: str,
    ) -> StagePlan:
        """Represent an inspectable stage that selected no tasks."""
        return cls(
            stage=stage,
            reason=reason,
            disposition="noop",
        )


@dataclass(frozen=True)
class StageExecution:
    """Exact executable contribution from one stage.

    Unlike :class:`StagePlan`, this contract cannot contain opaque stages or
    runtime command templates. It is the single source consumed by every
    execution engine; public plans are projections of it.
    """

    stage: str
    tasks: tuple[TaskPlan, ...] = ()
    reason: str | None = None
    disposition: Literal["executable", "noop"] = "executable"
    implementation: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.disposition == "executable" and not self.tasks:
            raise ValueError(f"Executable stage {self.stage!r} must contain at least one task")
        if self.disposition == "noop" and self.tasks:
            raise ValueError(f"No-op stage {self.stage!r} cannot contain tasks")
        for task in self.tasks:
            if task.stage != self.stage:
                raise ValueError(
                    f"Execution task {task.task_id!r} belongs to stage {task.stage!r}, "
                    f"expected {self.stage!r}"
                )
            if not task.commands:
                raise ValueError(f"Execution task {task.task_id!r} has no exact commands")
            for command in task.commands:
                command.require_runtime_role()
            templated = [command.role for command in task.commands if command.runtime_template]
            if templated:
                raise ValueError(
                    f"Execution task {task.task_id!r} contains runtime command templates: "
                    + ", ".join(templated)
                )

    @classmethod
    def noop(cls, stage: str, reason: str) -> StageExecution:
        """Represent an intentional no-op selected by exact stage logic."""
        return cls(stage=stage, reason=reason, disposition="noop")

    @classmethod
    def from_plan(cls, plan: StagePlan) -> StageExecution:
        """Promote a complete exact plan contribution."""
        if not plan.complete or plan.disposition == "opaque":
            raise ValueError(f"Stage {plan.stage!r} has no exact execution contract")
        return cls(
            stage=plan.stage,
            tasks=plan.tasks,
            reason=plan.reason,
            disposition="noop" if plan.disposition == "noop" else "executable",
            implementation=plan.implementation,
        )

    def to_plan(self) -> StagePlan:
        """Project this executable contract into the public plan schema."""
        return StagePlan(
            stage=self.stage,
            tasks=tuple(_public_task(task) for task in self.tasks),
            complete=True,
            reason=self.reason,
            disposition="noop" if self.disposition == "noop" else "planned",
            implementation=self.implementation,
        )


@dataclass(frozen=True)
class ExecutionGraph:
    """Immutable target-bound graph consumed by execution engines."""

    execution_version: int
    project: str
    workflow: str
    target: str
    engine: Literal["slurm", "ray"]
    target_fingerprint: str
    stages: tuple[StageExecution, ...]
    provenance: ProjectProvenance | None = None
    environments: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.execution_version != 1:
            raise ValueError(f"Unsupported execution_version: {self.execution_version}")
        plan = self.to_plan()
        if not plan.complete:
            raise ValueError("Execution graph cannot contain incomplete stages")
        unresolved = [
            f"{task.task_id}->{dependency.upstream}"
            for task in self.tasks
            for dependency in task.dependencies
            if dependency.external
        ]
        if unresolved:
            raise ValueError(
                "Execution graph contains unresolved selected-stage dependencies: "
                + ", ".join(unresolved)
            )

    @property
    def tasks(self) -> tuple[TaskPlan, ...]:
        """Return every executable task in stable stage order."""
        return tuple(task for stage in self.stages for task in stage.tasks)

    @property
    def digest(self) -> str:
        """Return a deterministic digest including target and engine identity."""
        payload = json.dumps(
            self.to_dict(include_digest=False),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
        return f"sha256:{hashlib.sha256(payload).hexdigest()}"

    def to_plan(self, *, planning_mode: Literal["static", "live"] = "static") -> LaunchPlan:
        """Derive the public display/inspection plan from exact execution."""
        return LaunchPlan(
            plan_version=2,
            project=self.project,
            workflow=self.workflow,
            target=self.target,
            stages=tuple(stage.to_plan() for stage in self.stages),
            provenance=self.provenance,
            planning_mode=planning_mode,
            environments=self.environments,
        )

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        """Convert the execution graph to a JSON-compatible mapping."""
        value = _json_compatible(asdict(self))
        assert isinstance(value, dict)
        if include_digest:
            value["digest"] = self.digest
        return value


@dataclass(frozen=True)
class LaunchPlan:
    """Immutable target-resolved workflow launch plan."""

    plan_version: int
    project: str
    workflow: str
    target: str
    stages: tuple[StagePlan, ...]
    provenance: ProjectProvenance | None = None
    planning_mode: Literal["static", "live"] = "static"
    environments: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.plan_version not in {1, 2}:
            raise ValueError(f"Unsupported workflow plan_version: {self.plan_version}")
        if self.plan_version == 2 and any(stage.implementation is None for stage in self.stages):
            raise ValueError("Workflow plan_version 2 requires stage implementation identity")
        task_ids = [task.task_id for task in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Launch plan contains duplicate task IDs")
        known = set(task_ids)
        graph: dict[str, set[str]] = {task_id: set() for task_id in task_ids}
        for task in self.tasks:
            for dependency in task.dependencies:
                if dependency.external:
                    continue
                if dependency.upstream not in known:
                    raise ValueError(
                        f"Task {task.task_id!r} depends on unknown task {dependency.upstream!r}"
                    )
                graph[task.task_id].add(dependency.upstream)
            for service in task.services:
                if service.task_id not in known:
                    raise ValueError(
                        f"Task {task.task_id!r} references unknown service task {service.task_id!r}"
                    )
                if service.consumer_task_id is not None and service.consumer_task_id not in known:
                    raise ValueError(
                        f"Service {service.name!r} references unknown consumer task "
                        f"{service.consumer_task_id!r}"
                    )
        _require_acyclic(graph)

    @property
    def complete(self) -> bool:
        """Return whether every stage supplied an inspectable plan."""
        return all(stage.complete for stage in self.stages)

    @property
    def tasks(self) -> tuple[TaskPlan, ...]:
        """Return every task in stable stage order."""
        return tuple(task for stage in self.stages for task in stage.tasks)

    @property
    def digest(self) -> str:
        """Return a deterministic digest of all execution-relevant fields."""
        digest_value = self.to_dict(include_digest=False)
        # Static/live describes how the task graph was obtained, not what will
        # execute. Identical task graphs should therefore have identical IDs.
        digest_value.pop("planning_mode", None)
        payload = json.dumps(
            digest_value,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
        return f"sha256:{hashlib.sha256(payload).hexdigest()}"

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        """Convert the plan to a JSON-compatible mapping."""
        value = _json_compatible(asdict(self))
        assert isinstance(value, dict)
        for stage in value["stages"]:
            for task in stage["tasks"]:
                for command in task["commands"]:
                    display = command.pop("display_command", None)
                    if display is not None:
                        command["command"] = display
        if self.plan_version == 1:
            for stage in value["stages"]:
                stage.pop("implementation", None)
        else:
            value["kind"] = "workflow"
        value["complete"] = self.complete
        if include_digest:
            value["digest"] = self.digest
        return value


def _public_task(task: TaskPlan) -> TaskPlan:
    return replace(
        task,
        commands=tuple(
            replace(
                command,
                command=command.display_command or command.command,
                display_command=None,
            )
            for command in task.commands
        ),
    )


def installs_packages(task: TaskPlan) -> bool:
    """Whether *task* installs packages when it starts, in a command or a parameter."""
    return any(_RUNTIME_INSTALL.search(command.command) for command in task.commands) or (
        _contains_runtime_install(task.parameters)
    )


def _contains_runtime_install(value: object) -> bool:
    if isinstance(value, str):
        return bool(_RUNTIME_INSTALL.search(value))
    if isinstance(value, Mapping):
        return any(_contains_runtime_install(nested) for nested in value.values())
    if isinstance(value, list | tuple | set):
        return any(_contains_runtime_install(nested) for nested in value)
    return False


def _require_acyclic(graph: Mapping[str, set[str]]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str) -> None:
        if task_id in visiting:
            raise ValueError(f"Launch plan contains a dependency cycle at {task_id!r}")
        if task_id in visited:
            return
        visiting.add(task_id)
        for upstream in graph[task_id]:
            visit(upstream)
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in graph:
        visit(task_id)


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(nested) for key, nested in value.items()}
    if isinstance(value, list | tuple | set):
        return [_json_compatible(nested) for nested in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)
