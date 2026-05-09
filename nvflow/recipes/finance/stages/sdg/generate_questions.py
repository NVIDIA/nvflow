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

from pathlib import Path
from typing import Any

from nvflow.core import BaseStage, StageRegistry, console
from nvflow.lib.vllm_compat import inject_server_entrypoint

# Run with uv run nflow run generate_questions --config=nvflow/recipes/finance/workflows/sdg/template-based-sdg.yaml


@StageRegistry.register(recipe="finance", workflow="template_based_sdg", stage="generate_questions")
class GenerateQuestionsStage(BaseStage):
    """Generate questions for financial reasoning."""

    workflow = "template_based_sdg"

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Execute question generation."""
        from nemo_skills.pipeline.cli import generate, run_cmd, wrap_arguments

        input_file = config["input_file"]
        output_file = config["output_file"]
        prompt_config = config.get("prompt_config")
        inline_args = config.get("inline_args", "")
        company_info_file = config["company_info_file"]
        start_year = config["start_year"]
        end_year = config["end_year"]

        console.status("Generating SDG questions")
        console.detail("Input file", input_file)
        console.detail("Company info file", company_info_file)
        console.detail("Start year", start_year)
        console.detail("End year", end_year)
        console.detail("Output file", output_file)
        console.detail("Prompt config", str(prompt_config))
        console.detail("Inline args", str(inline_args))
        console.blank()

        prepped_file = input_file.replace(".jsonl", "_prepped.jsonl")
        console.status(f"Prepped file: {prepped_file}")

        output_dir = Path(output_file).parent
        generation_folder = output_dir / Path(input_file).stem
        prep_log_dir = str(generation_folder / "prep-logs")

        # Run prep command to generate company/year combinations
        run_cmd(
            ctx=wrap_arguments(
                f"pip install -q --root-user-action=ignore jsonlines && "
                f"python3 -m nvflow.recipes.finance.utils.sdg.prepare_question_gen_data "
                f"--input_file {input_file} --company_info_file {company_info_file} "
                f"--start_year {start_year} --end_year {end_year} --output_file {prepped_file}"
            ),
            cluster=cluster,
            log_dir=prep_log_dir,
            expname=f"{expname}-prep",
            run_after=run_after,
        )

        console.detail("Output dir", str(generation_folder))

        generated_file = str(generation_folder / "output.jsonl")

        postprocess_cmd = f"python3 -m nvflow.recipes.finance.utils.sdg.parse_generated_questions --input_file {generated_file} --output_file {output_file}"

        stage_kwargs = inject_server_entrypoint(  # WORKAROUND(vllm-0.17-hermes, harmony-aarch64)
            config.get("stage_kwargs", {}),
            config.get("stage_kwargs", {}).get("model", ""),
        )
        ctx = wrap_arguments(f"++prompt_config={prompt_config} {inline_args}")
        generate(
            ctx=ctx,
            cluster=cluster,
            input_file=prepped_file,
            output_dir=str(generation_folder),
            expname=expname,
            run_after=[f"{expname}-prep"],
            **stage_kwargs,
            rerun_done=True,
            postprocess_cmd=postprocess_cmd,
        )

        console.success(f"Completed Question Generation for: {input_file}")
