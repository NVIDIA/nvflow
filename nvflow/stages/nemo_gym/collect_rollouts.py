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
"""Concrete shared stage for canonical native NeMo Gym rollout collection."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from nvflow.execution import ArtifactDeclaration, PlannedExecutionStage
from nvflow.integrations.nemo_gym.environments import (
    collect_target_config,
    selected_environments,
)
from nvflow.integrations.nemo_gym.operations import (
    GymOperationContext,
    normalize_rollout_operation,
    parse_worker_extensions,
    run_rollout_operation,
)

from ._common import dataset_selection_paths

if TYPE_CHECKING:
    from nvflow.execution import PlanningContext, StagePlan, TaskPlan


class CollectRolloutsStage(PlannedExecutionStage):
    """Collect canonical Gym bundles for explicitly configured environments."""

    @staticmethod
    def target_configs(config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        if not isinstance(config.get("output_dir"), str) or not config["output_dir"]:
            raise ValueError("'output_dir' must be a non-empty string in collect config")
        if not isinstance(config.get("rollout"), dict):
            raise ValueError("'rollout' must be a mapping in collect config")
        return [
            (environment.output_name, collect_target_config(config, environment))
            for environment in selected_environments(config)
        ]

    def validate_config(self, config: dict[str, Any]) -> None:
        extensions = parse_worker_extensions(config.get("extensions"))
        for name, target in self.target_configs(config):
            try:
                normalize_rollout_operation(target, extensions)
            except ValueError as exc:
                raise ValueError(f"environment {name!r}: {exc}") from exc

    def plan(self, context: PlanningContext) -> StagePlan:
        from nvflow.execution import StagePlan
        from nvflow.integrations.nemo_skills import stage_plan_from_jobs

        extensions = parse_worker_extensions(context.config.get("extensions"))
        tasks: list[TaskPlan] = []
        for name, target in self.target_configs(dict(context.config)):
            operation = normalize_rollout_operation(target, extensions)
            planned = run_rollout_operation(
                operation,
                GymOperationContext(
                    cluster=context.target.cluster,
                    expname=f"{context.expname}-{name}",
                    run_after=context.run_after,
                    inspect_live_state=context.live,
                    filesystem=context.filesystem,
                ),
            )
            contribution = stage_plan_from_jobs(
                context.stage,
                planned,
                output_dir=operation.output_dir,
                task_prefix=f"{context.stage}.{name}",
                experiment_name=f"{context.expname}-{name}",
            )
            contribution_tasks = list(contribution.tasks)
            artifacts = operation.layout.analyzed_artifacts
            canonical_outputs = (
                ArtifactDeclaration("rollouts", artifacts.rollouts.as_posix()),
                ArtifactDeclaration(
                    "materialized_inputs",
                    artifacts.materialized_inputs.as_posix(),
                ),
                ArtifactDeclaration(
                    "failures",
                    artifacts.failures.as_posix(),
                    optional=True,
                ),
                ArtifactDeclaration(
                    "aggregate_metrics",
                    artifacts.aggregate_metrics.as_posix(),
                ),
                ArtifactDeclaration(
                    "reward_profile",
                    artifacts.reward_profiling.as_posix(),
                    optional=operation.num_repeats < 2,
                ),
                ArtifactDeclaration(
                    "agent_metrics",
                    artifacts.agent_metrics.as_posix(),
                    optional=operation.num_repeats < 2,
                ),
                ArtifactDeclaration(
                    "repeat_level_metrics",
                    artifacts.repeat_level_metrics.as_posix(),
                    optional=operation.num_repeats < 2,
                ),
                ArtifactDeclaration(
                    "quality_summary",
                    artifacts.quality_summary.as_posix(),
                ),
                ArtifactDeclaration(
                    "rollout_verdicts",
                    artifacts.rollout_verdicts.as_posix(),
                ),
                ArtifactDeclaration("completion", artifacts.completion.as_posix()),
            )
            analyze_index = next(
                index
                for index, task in enumerate(contribution_tasks)
                if task.parameters.get("logical_role") == "analyze-rollouts"
            )
            contribution_tasks[analyze_index] = replace(
                contribution_tasks[analyze_index],
                outputs=canonical_outputs,
            )
            if operation.dataset_selection:
                profiled, train, completion, report, diagnostics = dataset_selection_paths(
                    operation.layout,
                    operation.dataset_selection,
                )
                selection_outputs = (
                    ArtifactDeclaration(
                        "profiled_dataset",
                        profiled,
                    ),
                    ArtifactDeclaration(
                        "training_dataset",
                        train,
                    ),
                    ArtifactDeclaration(
                        "dataset_completion",
                        completion,
                    ),
                    ArtifactDeclaration(
                        "selection_report",
                        report,
                    ),
                    ArtifactDeclaration(
                        "unprofiled_samples",
                        diagnostics,
                        optional=True,
                    ),
                )
                dataset_index = next(
                    index
                    for index, task in enumerate(contribution_tasks)
                    if task.parameters.get("logical_role") == "build-training-dataset"
                )
                contribution_tasks[dataset_index] = replace(
                    contribution_tasks[dataset_index],
                    outputs=selection_outputs,
                )
            tasks.extend(contribution_tasks)
        if not tasks:
            return StagePlan.noop(
                context.stage,
                (
                    "filesystem inspection selected no rollout work"
                    if context.live
                    else "environment filter selected no rollout work"
                ),
            )
        return StagePlan(stage=context.stage, tasks=tuple(tasks))


__all__ = ["CollectRolloutsStage"]
