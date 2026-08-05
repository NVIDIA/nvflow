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
"""Filter answers based on answerability assessment."""

from pathlib import Path
from typing import Any

from nvflow.core import BaseStage, StageRegistry, console

# Run with uv run nflow run filter_answers --config=nvflow/recipes/finance/workflows/sdg/template-based-sdg.yaml


@StageRegistry.register(recipe="finance", workflow="template_based_sdg", stage="filter_answers")
class FilterAnswersStage(BaseStage):
    """Filter answers based on answerability assessment."""

    workflow = "template_based_sdg"

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Execute answer filtering.

        This stage:
        1. Uses an LLM to tag answers as ANSWERABLE or UNANSWERABLE
        2. Parses the LLM responses to extract tags
        3. Filters out UNANSWERABLE entries
        """
        from nemo_skills.pipeline.cli import generate, wrap_arguments

        input_file = config["input_file"]
        output_file = config["output_file"]
        prompt_config = config.get("prompt_config")
        inline_args = config.get("inline_args", "")

        console.status("Filtering answers based on answerability")
        console.detail("Input file", input_file)
        console.detail("Output file", output_file)
        console.detail("Prompt config", str(prompt_config))
        console.detail("Inline args", str(inline_args))
        console.blank()

        generation_folder = Path(output_file.replace(".jsonl", ""))

        console.detail("Generation folder", str(generation_folder))

        # Intermediate file paths
        generated_file = str(generation_folder / "output.jsonl")
        parsed_file = str(generation_folder / "parsed.jsonl")

        # Step 1: Parse LLM responses to extract filter tags
        parse_cmd = (
            f"python3 -m nvflow.recipes.finance.utils.sdg.parse_filter_responses "
            f"--input_file {generated_file} --output_file {parsed_file}"
        )

        # Step 2: Apply filter to keep only ANSWERABLE entries
        filter_cmd = (
            f"python3 -m nvflow.recipes.finance.utils.sdg.apply_answer_filter "
            f"--input_file {parsed_file} --output_file {output_file} --keep_tag ANSWERABLE"
        )

        # Combine both commands
        postprocess_cmd = f"{parse_cmd} && {filter_cmd}"

        console.status("Running LLM generation to tag answers")

        stage_kwargs = config.get("stage_kwargs", {})
        ctx = wrap_arguments(f"++prompt_config={prompt_config} {inline_args}")
        generate(
            ctx=ctx,
            cluster=cluster,
            input_file=input_file,
            output_dir=str(generation_folder),
            expname=expname,
            run_after=run_after,
            **stage_kwargs,
            rerun_done=True,
            postprocess_cmd=postprocess_cmd,
        )

        console.success(f"Completed Answer Filtering for: {input_file}")
        console.detail("Filtered output", output_file)
