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
"""Compose immutable workflow launch plans without scheduler submission."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from nvflow.core.base_stage import BaseStage
from nvflow.core.code_root import CODE_ROOT_TOKEN
from nvflow.core.dependencies import resolve_dependency_names
from nvflow.core.workflow_definition import WorkflowDefinition
from nvflow.execution.contracts import (
    ExecutionGraph,
    LaunchPlan,
    StageExecution,
    StagePlan,
    completion_task_ids,
    installs_packages,
    internalize_dependencies,
)
from nvflow.execution.filesystem import TargetFilesystem
from nvflow.execution.inspection import (
    StageOutputInspection,
    inspect_declared_outputs,
)
from nvflow.execution.target import ResolvedTarget, TargetBinding
from nvflow.project.context import ProjectContext
from nvflow.project.provenance import ProjectProvenance


@dataclass(frozen=True)
class PlanningContext:
    """Bound inputs supplied to one stage's plan builder."""

    stage: str
    config: Mapping[str, Any]
    launcher_config: Mapping[str, Any]
    target: TargetBinding
    expname: str
    run_after: tuple[str, ...]
    project_context: ProjectContext | None = None
    live: bool = False
    filesystem: TargetFilesystem | None = None
    engine: Literal["slurm", "ray"] | None = None
    gpu_topology: Mapping[str, int] = field(default_factory=dict)

    def target_gpus_per_node(self, runtime_role: str) -> int | None:
        """Return the declared node shape for one runtime role, if known."""
        return self.gpu_topology.get(runtime_role, self.gpu_topology.get("*"))


def plan_stage(stage: BaseStage, context: PlanningContext) -> StagePlan | None:
    """Return the public projection of one exact stage execution."""
    return stage.execution(context).to_plan()


class WorkflowPlanner:
    """Build one target-resolved plan from a workflow definition."""

    def __init__(
        self,
        definition: WorkflowDefinition,
        target: TargetBinding,
        filesystem: TargetFilesystem | None = None,
    ) -> None:
        self.definition = definition
        self.target = target
        self.filesystem = filesystem
        self.output_inspections: tuple[StageOutputInspection, ...] = ()

    def plan(
        self,
        stages: Sequence[str] | None = None,
        environment: Sequence[str] | None = None,
        *,
        live: bool = False,
        inspect_outputs: bool = True,
    ) -> LaunchPlan:
        """Build a plan without submitting or mutating scheduler state.

        ``inspect_outputs`` controls the informational declaration batch only;
        it does not disable live stage selection or resume checks.
        """
        self.output_inspections = ()
        provenance, executions = self._stage_executions(
            stages,
            environment,
            live=live,
        )
        launch_plan = LaunchPlan(
            plan_version=2,
            project=self.definition.project,
            workflow=self.definition.workflow_name,
            target=self.target.cluster,
            stages=tuple(execution.to_plan() for execution in executions),
            provenance=provenance,
            planning_mode="live" if live else "static",
            environments=tuple(environment or ()),
        )
        if live and inspect_outputs:
            self.output_inspections = inspect_declared_outputs(
                launch_plan,
                self.filesystem,
            )
        return launch_plan

    def execution_graph(
        self,
        resolved_target: ResolvedTarget,
        stages: Sequence[str] | None = None,
        environment: Sequence[str] | None = None,
        *,
        live: bool = True,
    ) -> ExecutionGraph:
        """Build the exact graph consumed by either execution engine."""
        if resolved_target.binding != self.target:
            raise ValueError("Resolved target does not match planner target binding")
        provenance, executions = self._stage_executions(
            stages,
            environment,
            live=live,
            resolved_target=resolved_target,
        )
        if resolved_target.engine == "ray":
            _reject_ray_incompatible_controls(executions)
        return ExecutionGraph(
            execution_version=1,
            project=self.definition.project,
            workflow=self.definition.workflow_name,
            target=self.target.cluster,
            engine=resolved_target.engine,
            target_fingerprint=resolved_target.fingerprint,
            stages=executions,
            provenance=provenance,
            environments=tuple(environment or ()),
        )

    def _stage_executions(
        self,
        stages: Sequence[str] | None,
        environment: Sequence[str] | None,
        *,
        live: bool,
        resolved_target: ResolvedTarget | None = None,
    ) -> tuple[ProjectProvenance, tuple[StageExecution, ...]]:
        provenance = self.definition.project_context.source_provenance()
        stages_to_plan = self.definition.select_active_stages(stages)
        self.definition.validate_stages(stages_to_plan)
        executions: list[StageExecution] = []
        selected_stages = set(stages_to_plan)
        for stage_name in stages_to_plan:
            raw_config = self.definition.stage_runtime_config(stage_name)
            stage_config = self.definition.project_context.render_execution_paths(raw_config)
            launcher_config = self.definition.project_context.render_launcher_paths(raw_config)
            if environment is not None:
                stage_config["_environment"] = list(environment)
                launcher_config["_environment"] = list(environment)
            stage_class = self.definition.resolve_stage_class(stage_name)
            stage = stage_class()
            stage.bind_project_context(self.definition.project_context)
            stage.validate_config(launcher_config)
            context = PlanningContext(
                stage=stage_name,
                config=stage_config,
                launcher_config=launcher_config,
                target=self.target,
                expname=self.definition.get_expname(stage_name, stage_config),
                run_after=tuple(
                    resolve_dependency_names(
                        stage_config.get("dependencies", []),
                        self.definition.config["stages"],
                        self.definition.get_expname,
                        environment=environment,
                        selected_stages=selected_stages,
                    )
                ),
                project_context=self.definition.project_context,
                live=live,
                filesystem=self.filesystem if live else None,
                engine=(None if resolved_target is None else resolved_target.engine),
                gpu_topology=_gpu_topology(resolved_target),
            )
            execution = stage.execution(context)
            if execution.stage != stage_name:
                raise ValueError(
                    f"Stage execution identity {execution.stage!r} does not match "
                    f"configured stage {stage_name!r}"
                )
            executions.append(
                replace(
                    _portable_code_root(execution),
                    implementation=self.definition.stage_implementation(stage_name),
                )
            )

        fallback_aliases: dict[str, tuple[str, ...]] = {}
        for stage_name, execution in zip(stages_to_plan, executions, strict=True):
            aliases = resolve_dependency_names(
                [stage_name],
                self.definition.config["stages"],
                self.definition.get_expname,
                environment=environment,
                selected_stages={stage_name},
            )
            terminals = completion_task_ids(execution.tasks)
            for alias in aliases:
                fallback_aliases[alias] = terminals
        internalized = internalize_dependencies(
            tuple(task for execution in executions for task in execution.tasks),
            fallback_aliases=fallback_aliases,
        )
        tasks_by_id = {task.task_id: task for task in internalized}
        exact = tuple(
            replace(
                execution,
                tasks=tuple(tasks_by_id[task.task_id] for task in execution.tasks),
            )
            for execution in executions
        )
        return provenance, exact


def _gpu_topology(target: ResolvedTarget | None) -> Mapping[str, int]:
    if target is None:
        return {}
    if target.engine == "slurm":
        value = target.raw.get("gpus_per_node")
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return {"*": value}
        return {}
    assert target.ray is not None
    return {
        role: int(cluster.gpus_per_node)
        for role, cluster_name in target.ray.routes.items()
        if (cluster := target.ray.clusters[cluster_name]).gpus_per_node is not None
    }


def _reject_ray_incompatible_controls(
    executions: tuple[StageExecution, ...],
) -> None:
    """Fail before engine dispatch when exact tasks retain scheduler controls."""
    unsupported: list[str] = []
    for execution in executions:
        for task in execution.tasks:
            for resource in task.resources:
                fields = [
                    name
                    for name, value in (
                        ("partition", resource.partition),
                        ("account", resource.account),
                        ("exclusive", resource.exclusive),
                    )
                    if value is not None
                ]
                if fields:
                    unsupported.append(f"{task.task_id}: {', '.join(fields)}")
            scheduler = task.parameters.get("slurm") or {}
            if isinstance(scheduler, Mapping) and scheduler.get("sbatch_kwargs"):
                unsupported.append(f"{task.task_id}: sbatch_kwargs")
            if task.parameters.get("installation_command"):
                unsupported.append(f"{task.task_id}: installation_command")
            elif installs_packages(task):
                unsupported.append(f"{task.task_id}: runtime package installation")
    if unsupported:
        raise ValueError(
            "Ray execution cannot apply direct-scheduler task controls: " + "; ".join(unsupported)
        )


def _portable_code_root(execution: StageExecution) -> StageExecution:
    return replace(
        execution,
        tasks=tuple(
            replace(
                task,
                commands=tuple(
                    replace(
                        command,
                        command=command.command.replace(
                            "/nemo_run/code",
                            CODE_ROOT_TOKEN,
                        ),
                        display_command=(
                            command.display_command.replace(
                                "/nemo_run/code",
                                CODE_ROOT_TOKEN,
                            )
                            if command.display_command is not None
                            else None
                        ),
                    )
                    for command in task.commands
                ),
            )
            for task in execution.tasks
        ),
    )
