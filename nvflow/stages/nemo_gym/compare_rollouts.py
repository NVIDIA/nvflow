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
"""Concrete shared stage for native comparison of completed Gym bundles."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nvflow.execution import ArtifactDeclaration, PlannedExecutionStage
from nvflow.integrations.nemo_gym.compare import build_compare_command
from nvflow.integrations.nemo_gym.contracts import require_contract_version

if TYPE_CHECKING:
    from nvflow.execution import PlanningContext, StagePlan


def _outputs(config: dict[str, Any]) -> tuple[ArtifactDeclaration, ...]:
    output_dir = str(config["output_dir"]).rstrip("/")
    report_format = config.get("report_format", "both")
    outputs = []
    if report_format in {"json", "both"}:
        outputs.append(ArtifactDeclaration("compare_json", f"{output_dir}/compare_report.json"))
    if report_format in {"md", "both"}:
        outputs.append(ArtifactDeclaration("compare_markdown", f"{output_dir}/compare_report.md"))
    return tuple(outputs)


class CompareRolloutsStage(PlannedExecutionStage):
    """Compare two compatible completed Gym runs without model inference."""

    def validate_config(self, config: dict[str, Any]) -> None:
        require_contract_version(config.get("contract_version"))
        for field in (
            "baseline_rollouts",
            "baseline_completion",
            "candidate_rollouts",
            "candidate_completion",
            "output_dir",
            "container",
        ):
            if not isinstance(config.get(field), str) or not config[field]:
                raise ValueError(f"'{field}' is required in compare config")
        if config["baseline_rollouts"] == config["candidate_rollouts"]:
            raise ValueError("compare requires different baseline and candidate rollouts")
        if config.get("report_format", "both") not in {"md", "json", "both"}:
            raise ValueError("compare.report_format must be md, json, or both")

    def plan(self, context: PlanningContext) -> StagePlan:
        from nvflow.execution import declare_task_outputs, plan_run_cmd

        config = dict(context.config)
        with declare_task_outputs(*_outputs(config)):
            return plan_run_cmd(
                context,
                build_compare_command(config),
                cluster=context.target.cluster,
                container=config["container"],
                num_gpus=0,
                log_dir=f"{config['output_dir']}/logs",
                expname=context.expname,
                run_after=list(context.run_after) or None,
                installation_command=config.get("installation_command"),
            )


__all__ = ["CompareRolloutsStage"]
