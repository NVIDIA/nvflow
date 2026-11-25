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
"""Concrete shared stage for profiled and training dataset publication."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from nvflow.execution import PlannedExecutionStage
from nvflow.integrations.nemo_gym.contracts import require_contract_version
from nvflow.integrations.nemo_gym.native_contract import NativeGymOutputLayout
from nvflow.integrations.nemo_gym.postprocess import build_dataset_selection_cmd
from nvflow.integrations.nemo_gym.reward_profiles import parse_dataset_selection_spec

from ._common import dataset_selection_paths

if TYPE_CHECKING:
    from nvflow.execution import PlanningContext, StagePlan


def _selection_config(config: Mapping[str, Any]) -> dict[str, Any]:
    configured = config.get("dataset_selection")
    selection = {} if configured is None else configured
    if not isinstance(selection, Mapping):
        raise ValueError("'dataset_selection' must be a mapping")
    return dict(selection)


def _response_overrides(config: Mapping[str, Any]) -> dict[str, object]:
    configured = config.get("responses_create_params_overrides", {})
    if not isinstance(configured, Mapping):
        raise ValueError("'responses_create_params_overrides' must be a mapping")
    return dict(configured)


def build_training_dataset_command(config: dict[str, Any]) -> str:
    """Render the shared worker command from a validated stage config."""
    raw_spec = _selection_config(config)
    spec = parse_dataset_selection_spec(raw_spec)
    serialized = asdict(spec)
    layout = NativeGymOutputLayout.collection(str(config["output_dir"]))
    return build_dataset_selection_cmd(
        source=str(config["source"]),
        reward_profile=str(config["profile"]),
        materialized_inputs=str(config["materialized_inputs"]),
        rollouts=str(config["rollouts"]),
        rollout_completion=str(config["rollout_completion"]),
        profiled_output=layout.profiled_dataset.as_posix(),
        training_output=layout.training_dataset.as_posix(),
        report_output=layout.dataset_selection_report.as_posix(),
        completion_output=layout.dataset_completion.as_posix(),
        diagnostics_output=layout.unprofiled_samples.as_posix(),
        selection=serialized,
        responses_create_params_overrides=_response_overrides(config),
    )


class BuildTrainingDatasetStage(PlannedExecutionStage):
    """Publish standardized profiled data with an explicit selection policy."""

    def validate_config(self, config: dict[str, Any]) -> None:
        require_contract_version(config.get("contract_version"))
        for field in (
            "source",
            "profile",
            "materialized_inputs",
            "rollouts",
            "rollout_completion",
            "output_dir",
            "container",
        ):
            if not isinstance(config.get(field), str) or not config[field]:
                raise ValueError(
                    f"'{field}' must be a non-empty string in dataset publication config"
                )
        build_training_dataset_command(config)

    def plan(self, context: PlanningContext) -> StagePlan:
        from nvflow.execution import ArtifactDeclaration, declare_task_outputs, plan_run_cmd

        config = dict(context.config)
        output_dir = str(config["output_dir"])
        layout = NativeGymOutputLayout.collection(output_dir)
        profiled, train, completion, report, diagnostics = dataset_selection_paths(
            layout,
            _selection_config(config),
        )
        outputs = (
            ArtifactDeclaration("profiled_dataset", profiled),
            ArtifactDeclaration("training_dataset", train),
            ArtifactDeclaration("dataset_completion", completion),
            ArtifactDeclaration("selection_report", report),
            ArtifactDeclaration("unprofiled_samples", diagnostics, optional=True),
        )
        with declare_task_outputs(*outputs):
            return plan_run_cmd(
                context,
                build_training_dataset_command(config),
                cluster=context.target.cluster,
                container=config["container"],
                num_gpus=0,
                log_dir=layout.dataset_logs_dir.as_posix(),
                expname=context.expname,
                run_after=list(context.run_after) or None,
                installation_command=config.get("installation_command"),
            )


__all__ = ["BuildTrainingDatasetStage", "build_training_dataset_command"]
