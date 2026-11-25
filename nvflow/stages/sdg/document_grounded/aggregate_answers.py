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
"""Aggregate repeated native Gym evaluation results."""

from pathlib import Path
from typing import Any

from nvflow.core import console
from nvflow.execution import CapturedExecutionStage, declare_task_outputs

from ._helpers import build_trim_cmd


class AggregateAnswersStage(CapturedExecutionStage):
    """Aggregate repeated native Gym evaluation results."""

    capture_strategy = "run_cmd"

    def build_tasks(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Execute parsing and aggregation of multi-seed evaluation results."""
        from nvflow.integrations.nemo_skills.commands import run_cmd, wrap_arguments

        input_dir = config["input_dir"]
        output_file = config["output_file"]
        num_repeats = config.get("num_repeats", 5)

        console.status("Parsing and aggregating repeated evaluation results")
        console.detail("Input dir", input_dir)
        console.detail("Output file", output_file)
        console.detail("Repeats", str(num_repeats))
        console.blank()

        generation_folder = Path(input_dir) / "selected_answers"
        aggregate_cmd = (
            "python -m nvflow.stages.sdg.document_grounded.aggregate "
            f"--input_dir {generation_folder} "
            f"--output_file {output_file} "
            f"--num_repeats {num_repeats}"
        )
        trim_cmd = build_trim_cmd(
            stage_name="aggregate_answers",
            paths=[output_file],
            domain_keep_fields=config.get("domain_keep_fields"),
        )
        full_cmd = f"{aggregate_cmd} && {trim_cmd}"

        console.status("Running aggregation (streaming, no intermediate files)")
        with declare_task_outputs(output_file=output_file):
            run_cmd(
                ctx=wrap_arguments(full_cmd),
                cluster=cluster,
                expname=expname,
                run_after=run_after,
            )

        console.success("Completed aggregation")
        console.detail("Output", output_file)

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate stage configuration."""
        required = ["input_dir", "output_file"]
        for field in required:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
