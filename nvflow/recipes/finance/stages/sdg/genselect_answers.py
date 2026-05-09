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
"""Generate and select best answers using parallel thinking genselect mode."""

from typing import Any

from nvflow.core import BaseStage, StageRegistry, console
from nvflow.lib.vllm_compat import inject_server_entrypoint

# Run with uv run nflow run genselect_answers --config=nvflow/recipes/finance/workflows/sdg/template-based-sdg.yaml


@StageRegistry.register(recipe="finance", workflow="template_based_sdg", stage="genselect_answers")
@StageRegistry.register(
    recipe="finance",
    workflow="document_grounded_sdg",
    stage="genselect_answers",
)
class GenselectAnswersStage(BaseStage):
    """Generate and select best answers using parallel thinking."""

    workflow = "template_based_sdg"

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Execute genselect answer generation."""
        from nemo_skills.pipeline.cli import generate, run_cmd, wrap_arguments

        input_dir = config["input_dir"]
        output_file = config["output_file"]
        prompt_config = config.get("prompt_config")
        inline_args = config.get("inline_args", "")

        # Calculate intermediate file paths
        output_dir = output_file.replace(".jsonl", "")
        prepped_file = output_dir + "_prepped.jsonl"

        console.status("Generating and selecting best answers")
        console.detail("Input dir", input_dir)
        console.detail("Output file", output_file)
        console.detail("Prepped file", prepped_file)
        console.detail("Output dir", output_dir)
        console.detail("Prompt config", str(prompt_config))
        console.detail("Inline args", str(inline_args))
        console.blank()

        console.status("Step 1: Preparing genselect data")
        run_cmd(
            ctx=wrap_arguments(
                f"python3 -m nvflow.recipes.finance.utils.sdg.prepare_genselect_data --input_dir={input_dir} --output_file={prepped_file}"
            ),
            cluster=cluster,
            expname=f"{expname}-prep",
            log_dir=f"{output_dir}/prep-data-logs",
            run_after=run_after,
        )

        postprocess_cmd = f"python3 -m nvflow.recipes.finance.utils.sdg.postprocess_genselect --input_dir={output_dir} --output_file={output_file}"

        console.status("Generating answers with genselect")

        stage_kwargs = inject_server_entrypoint(  # WORKAROUND(vllm-0.17-hermes, harmony-aarch64)
            config.get("stage_kwargs", {}),
            config.get("stage_kwargs", {}).get("model", ""),
        )
        ctx = wrap_arguments(f"++prompt_config={prompt_config} {inline_args}")
        generate(
            ctx=ctx,
            cluster=cluster,
            input_file=prepped_file,
            output_dir=output_dir,
            expname=expname,
            run_after=[f"{expname}-prep"],
            **stage_kwargs,
            rerun_done=True,
            postprocess_cmd=postprocess_cmd,
        )

        console.success(f"Genselect answer generation submitted → {output_file}")
