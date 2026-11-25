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
"""Concrete shared stage for native Gym reward recomputation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from nvflow.execution import ArtifactDeclaration, PlannedExecutionStage
from nvflow.integrations.nemo_gym.environments import (
    deep_merge,
    environment_payload,
    selected_environments,
)
from nvflow.integrations.nemo_gym.native_contract import NativeGymOutputLayout
from nvflow.integrations.nemo_gym.operations import (
    GymOperationContext,
    normalize_reward_operation,
    parse_worker_extensions,
    run_reward_operation,
)

from ._common import dataset_selection_paths

if TYPE_CHECKING:
    from nvflow.execution import PlanningContext, StagePlan, TaskPlan


def _target_configs(config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    required = ("output_dir", "rollouts_dir")
    missing = [
        field for field in required if not isinstance(config.get(field), str) or not config[field]
    ]
    if missing:
        raise ValueError("reward verification requires non-empty path(s): " + ", ".join(missing))
    result = []
    for environment in selected_environments(config, require_input=False):
        output_dir = f"{str(config['output_dir']).rstrip('/')}/{environment.output_name}"
        raw_environment = environment_payload(config, environment)
        gym = dict(raw_environment.get("gym") or {})
        if gym.get("observability_enabled") or "model_call_capture_dir" in gym:
            gym["model_call_capture_dir"] = (
                NativeGymOutputLayout.rejudge(output_dir).initial_outputs_dir
                / "captures"
                / "model-calls"
            ).as_posix()
            raw_environment["gym"] = gym

        configured_verification = config.get("verification")
        raw_verification = {} if configured_verification is None else configured_verification
        if not isinstance(raw_verification, Mapping):
            raise ValueError("verification must be a mapping")
        verification = dict(raw_verification)
        configured_raw_models = verification.get("models")
        raw_models = {} if configured_raw_models is None else configured_raw_models
        if not isinstance(raw_models, Mapping):
            raise ValueError("verification.models must be a mapping")
        configured_models = dict(raw_models)
        unknown_roles = sorted(set(configured_models) - {"judge"})
        if unknown_roles:
            raise ValueError(f"unknown verification model role(s): {', '.join(unknown_roles)}")
        configured_judge = configured_models.get("judge")
        if configured_judge is not None and not isinstance(configured_judge, Mapping):
            raise ValueError("verification.models.judge must be a mapping")
        configured_environment_judge = environment.models.get("judge")
        if configured_environment_judge is not None and not isinstance(
            configured_environment_judge,
            Mapping,
        ):
            raise ValueError(f"environment {environment.name!r}.models.judge must be a mapping")
        environment_judge = dict(configured_environment_judge or {})
        judge = dict(configured_judge) if configured_judge is not None else environment_judge
        verification["models"] = {"judge": judge}
        verification["mode"] = verification.get(
            "mode",
            raw_environment.get("reward_recomputation", "fallback"),
        )

        target = {
            key: value
            for key, value in config.items()
            if key not in {"contract_version", "extensions"}
        }
        target["output_dir"] = output_dir
        target["rollouts_dir"] = (
            f"{str(config['rollouts_dir']).rstrip('/')}/{environment.output_name}"
        )
        target["environments"] = {environment.name: raw_environment}
        target["rejudge"] = verification
        configured_source_responses = config.get("source_responses_create_params", {})
        if not isinstance(configured_source_responses, Mapping):
            raise ValueError("source_responses_create_params must be a mapping")
        target["source_responses_create_params"] = {
            **dict(configured_source_responses),
            **dict(environment.responses_create_params),
        }
        configured_blade = config.get("blade")
        raw_blade = {} if configured_blade is None else configured_blade
        if not isinstance(raw_blade, Mapping):
            raise ValueError("blade must be a mapping")
        blade = dict(raw_blade)
        if environment.blade:
            blade = deep_merge(blade, environment.blade)
        target["blade"] = blade
        raw_selection = config.get("dataset_selection")
        if raw_selection is not None:
            if not isinstance(raw_selection, Mapping):
                raise ValueError("dataset_selection must be a mapping")
            if raw_selection:
                target["dataset_selection"] = {
                    **dict(raw_selection),
                    "source": environment.input_data,
                }
        result.append((environment.output_name, target))
    return result


class RejudgeRolloutsStage(PlannedExecutionStage):
    """Recompute rewards for completed Gym rollout bundles."""

    target_configs = staticmethod(_target_configs)

    def validate_config(self, config: dict[str, Any]) -> None:
        extensions = parse_worker_extensions(config.get("extensions"))
        for name, target in self.target_configs(config):
            try:
                normalize_reward_operation(target, extensions)
            except ValueError as exc:
                raise ValueError(f"environment {name!r}: {exc}") from exc

    def plan(self, context: PlanningContext) -> StagePlan:
        from nvflow.execution import StagePlan
        from nvflow.integrations.nemo_skills import stage_plan_from_jobs

        extensions = parse_worker_extensions(context.config.get("extensions"))
        tasks: list[TaskPlan] = []
        for name, target in self.target_configs(dict(context.config)):
            operation = normalize_reward_operation(target, extensions)
            planned = run_reward_operation(
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
            artifacts = operation.analyzed_artifacts
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
                    optional=not bool(operation.dataset_selection),
                ),
                ArtifactDeclaration(
                    "agent_metrics",
                    artifacts.agent_metrics.as_posix(),
                    optional=True,
                ),
                ArtifactDeclaration(
                    "repeat_level_metrics",
                    artifacts.repeat_level_metrics.as_posix(),
                    optional=True,
                ),
                ArtifactDeclaration(
                    "quality_summary",
                    artifacts.quality_summary.as_posix(),
                ),
                ArtifactDeclaration(
                    "rollout_verdicts",
                    artifacts.rollout_verdicts.as_posix(),
                ),
                ArtifactDeclaration(
                    "compare_json",
                    (operation.layout.compare_analysis_dir / "compare_report.json").as_posix(),
                ),
                ArtifactDeclaration(
                    "compare_markdown",
                    (operation.layout.compare_analysis_dir / "compare_report.md").as_posix(),
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
                    "filesystem inspection selected no reward work"
                    if context.live
                    else "environment filter selected no reward work"
                ),
            )
        return StagePlan(stage=context.stage, tasks=tuple(tasks))


__all__ = ["RejudgeRolloutsStage"]
