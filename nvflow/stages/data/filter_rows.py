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
"""Drop the records whose id appears in another file (``data.filter``)."""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING, Any

from nvflow.execution import ArtifactDeclaration, PlannedExecutionStage, declare_task_outputs
from nvflow.stages.data.step_plan import require_live_inputs

if TYPE_CHECKING:
    from nvflow.execution import PlanningContext, StagePlan

OUTPUTS = {"kept": "kept.jsonl", "dropped": "dropped.jsonl", "report": "report.json"}


class FilterRowsStage(PlannedExecutionStage):
    """Write ``kept.jsonl``, ``dropped.jsonl``, and ``report.json``.

    A row of ``input_file`` is excluded when its ``id_field`` value appears as an
    ``exclude_id_field`` value (default: ``id_field``) in ``exclude_file``.
    """

    def plan(self, context: PlanningContext) -> StagePlan:
        from nvflow.execution import plan_run_cmd

        config = context.config
        output_dir = str(config["output_dir"]).rstrip("/")
        input_file, exclude_file = str(config["input_file"]), str(config["exclude_file"])
        require_live_inputs(context, [input_file, exclude_file])
        id_field = str(config["id_field"])
        command = (
            f"python3 -m nvflow.stages.data.filter_rows_worker {shlex.quote(input_file)} "
            f"--exclude_file {shlex.quote(exclude_file)} --output_dir {shlex.quote(output_dir)} "
            f"--id_field {shlex.quote(id_field)} "
            f"--exclude_id_field {shlex.quote(str(config.get('exclude_id_field') or id_field))}"
        )
        with declare_task_outputs(
            *(ArtifactDeclaration(name, f"{output_dir}/{file}") for name, file in OUTPUTS.items())
        ):
            return plan_run_cmd(
                context,
                command,
                cluster=context.target.cluster,
                container=str(config.get("container") or "nemo-skills"),
                installation_command=config.get("installation_command"),
                log_dir=f"{output_dir}/logs",
                output_dir=output_dir,
                expname=context.expname,
                run_after=list(context.run_after) or None,
                sbatch_kwargs=None,
            )

    def validate_config(self, config: dict[str, Any]) -> None:
        from nvflow.core.workflow_structure import reject_unknown_options

        reject_unknown_options(
            config,
            {"input_file", "exclude_file", "output_dir", "id_field", "exclude_id_field"},
            "data.filter",
        )
        for key in ("input_file", "exclude_file", "output_dir", "id_field"):
            if not isinstance(config.get(key), str) or not config[key]:
                raise ValueError(f"'{key}' is required in data.filter config")
