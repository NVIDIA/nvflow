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
"""Aggregate multi-seed evaluation results."""

from pathlib import Path
from typing import Any

from nvflow.core import BaseStage, StageRegistry, console


@StageRegistry.register(
    recipe="finance",
    workflow="document_grounded_sdg",
    stage="aggregate_answers",
)
class AggregateAnswersStage(BaseStage):
    """Aggregate multi-seed evaluation results.

    This stage processes output-rs*.jsonl files in streaming mode:
    - Reads all seed files line-by-line in parallel (no intermediate files)
    - Parses evaluate_generation inline
    - Only keeps records where ALL seeds have correct=YES
    - Only keeps records where ALL seeds have consistent answerable (all YES or all NO)
    - Adds a final 'answerable' field based on the consistent value

    This ensures high-quality data where the evaluation is confident and consistent
    across multiple random samples. Uses O(1) memory regardless of file size.
    """

    workflow = "document_grounded_sdg"

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Execute parsing and aggregation of multi-seed evaluation results."""
        from nemo_skills.pipeline.cli import run_cmd, wrap_arguments

        input_dir = config["input_dir"]
        output_file = config["output_file"]
        num_seeds = config.get("num_seeds", 5)

        console.status("Parsing and aggregating multi-seed evaluation results")
        console.detail("Input dir", input_dir)
        console.detail("Output file", output_file)
        console.detail("Num seeds", str(num_seeds))
        console.blank()

        # The evaluate_answers stage creates: {input_dir}/{input_file_stem}/output-rsN.jsonl
        # Input file stem is "selected_answers" based on workflow config
        generation_folder = Path(input_dir) / "selected_answers"

        aggregate_module = "nvflow.recipes.finance.utils.sdg.aggregate_evaluate"

        # Aggregate results (parse + aggregate combined, no intermediate files)
        full_cmd = (
            f"python3 -m {aggregate_module} "
            f"--input_dir {generation_folder} "
            f"--output_file {output_file} "
            f"--num_seeds {num_seeds}"
        )

        console.status("Running aggregation (streaming, no intermediate files)")

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
