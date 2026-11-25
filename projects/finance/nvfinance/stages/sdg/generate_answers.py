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
"""Generate questions for financial reasoning."""

from typing import Any

from nvflow.core import console
from nvflow.execution import CapturedExecutionStage

# Run with uv run nvflow run generate_answers --config=projects/finance/workflows/sdg/template-based-sdg.yaml --cluster my_cluster


class GenerateAnswersStage(CapturedExecutionStage):
    """Generate answers for financial reasoning."""

    capture_strategy = "run_cmd"

    def build_tasks(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Execute answer generation."""
        from nvflow.integrations.nemo_skills.commands import generate, wrap_arguments

        input_file = config["input_file"]
        output_dir = config["output_dir"]
        prompt_config = config.get("prompt_config")
        inline_args = config.get("inline_args", "")

        console.status("Generating SDG answers")
        console.detail("Input file", input_file)
        console.detail("Output dir", output_dir)

        console.detail("Prompt config", str(prompt_config))
        console.detail("Inline args", str(inline_args))
        console.blank()

        stage_kwargs = config.get("stage_kwargs", {})
        ctx = wrap_arguments(f"++prompt_config={prompt_config} {inline_args}")
        generate(
            ctx=ctx,
            cluster=cluster,
            input_file=input_file,
            output_dir=output_dir,
            expname=expname,
            run_after=run_after,
            **stage_kwargs,
            rerun_done=True,
        )

        console.success(f"Completed Answer Generation for: {input_file}")
