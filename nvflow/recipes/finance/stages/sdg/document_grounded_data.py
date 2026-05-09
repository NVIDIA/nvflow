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
"""Document grounded sdg data post processing stage."""

from typing import Any

from nvflow.core import BaseStage, StageRegistry, console


@StageRegistry.register(
    recipe="finance",
    workflow="document_grounded_sdg",
    stage="dgsdg_post_process",
)
class DGSDGPostProcessStage(BaseStage):
    """Post process document grounded sdg data by cleaning fields and creating subsets.

    This stage:
    1. Removes unwanted fields (solutions, generations_list, etc.)
    2. Renames reference_reasoning -> reasoning_content, reference_answer -> answer
    3. Creates full_data.jsonl with all cleaned records
    4. Creates medium_sft_data.jsonl:
       - Only records with difficulty_score in [1, 2, 3, 4]
       - For 10-K filings: excludes Risk_Factors questions
       - For 10-Q filings: only includes Risk_Factors questions
    5. Creates hard_rl_data.jsonl:
       - Only records with difficulty_score = 0
    """

    workflow = "document_grounded_sdg"

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Execute document grounded sdg data post processing."""
        from nemo_skills.pipeline.cli import run_cmd, wrap_arguments

        input_file = config["input_file"]
        output_dir = config["output_dir"]
        seed = config.get("seed", 42)

        console.status("Post processing document grounded sdg data")
        console.detail("Input file", input_file)
        console.detail("Output dir", output_dir)
        console.detail("Random seed", str(seed))
        console.blank()

        module = "nvflow.recipes.finance.utils.sdg.dgsdg_post_process"

        cmd = (
            f"python3 -m {module} --input_file {input_file} --output_dir {output_dir} --seed {seed}"
        )

        run_cmd(
            ctx=wrap_arguments(cmd),
            cluster=cluster,
            expname=expname,
            run_after=run_after,
        )

        console.success("Document grounded sdg data post processing job submitted")
        console.detail("Output files will be in", output_dir)

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate stage configuration."""
        required = ["input_file", "output_dir"]
        for field in required:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
