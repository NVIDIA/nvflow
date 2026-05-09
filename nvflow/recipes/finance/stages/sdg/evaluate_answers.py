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
"""Evaluate answers for correctness and answerability."""

from pathlib import Path
from typing import Any

from nvflow.core import BaseStage, StageRegistry, console
from nvflow.lib.vllm_compat import inject_server_entrypoint


@StageRegistry.register(
    recipe="finance",
    workflow="document_grounded_sdg",
    stage="evaluate_answers",
)
class EvaluateAnswersStage(BaseStage):
    """Evaluate answers for correctness and answerability.

    This stage evaluates each answer on two dimensions:
    1. ANSWERABLE: Can the question be answered from the context?
    2. CORRECT: Is the response appropriate?
       - If answerable: Is the answer accurate?
       - If not answerable: Does the model correctly identify this?

    When num_random_seeds > 1, this stage only parses the results.
    The filtering and aggregation is done by aggregate_answers stage.

    This is valuable for training because:
    - Answerable + Correct: Model learns to answer correctly
    - Unanswerable + Correct: Model learns to recognize and decline unanswerable questions
    """

    workflow = "document_grounded_sdg"

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Execute answer evaluation and filtering."""
        from nemo_skills.pipeline.cli import generate, wrap_arguments

        input_file = config["input_file"]
        output_dir = config.get("output_dir")
        output_file = config.get("output_file")
        prompt_config = config.get("prompt_config")
        inline_args = config.get("inline_args", "")
        stage_kwargs = inject_server_entrypoint(  # WORKAROUND(vllm-0.17-hermes, harmony-aarch64)
            config.get("stage_kwargs", {}),
            config.get("stage_kwargs", {}).get("model", ""),
        )
        num_random_seeds = stage_kwargs.get("num_random_seeds", 1)

        console.status("Evaluating answers for correctness and answerability")
        console.detail("Input file", input_file)
        console.detail("Output dir", str(output_dir))
        console.detail("Prompt config", str(prompt_config))
        console.detail("Inline args", str(inline_args))
        console.detail("Num random seeds", str(num_random_seeds))
        console.blank()

        # Determine output directory
        if output_dir:
            generation_folder = Path(output_dir) / Path(input_file).stem
        else:
            generation_folder = Path(output_file).parent / Path(input_file).stem

        console.detail("Generation folder", str(generation_folder))

        module = "nvflow.recipes.finance.utils.sdg.parse_evaluate_responses"

        console.status("Running LLM generation to evaluate answers")
        ctx = wrap_arguments(f"++prompt_config={prompt_config} {inline_args}")

        if num_random_seeds > 1:
            # Multi-seed mode: no postprocess here
            # The aggregate_answers stage will handle parse and aggregation
            generate(
                ctx=ctx,
                cluster=cluster,
                input_file=input_file,
                output_dir=str(generation_folder),
                expname=expname,
                run_after=run_after,
                **stage_kwargs,
                rerun_done=True,
            )
        else:
            # Single seed mode: parse and filter as before
            generated_file = str(generation_folder / "output.jsonl")
            parsed_file = str(generation_folder / "parsed.jsonl")
            final_output = (
                output_file if output_file else str(generation_folder / "evaluated.jsonl")
            )

            parse_cmd = f"python3 -m {module} parse --input_file {generated_file} --output_file {parsed_file}"
            filter_cmd = f"python3 -m {module} filter --input_file {parsed_file} --output_file {final_output}"
            postprocess_cmd = f"{parse_cmd} && {filter_cmd}"

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

        console.success(f"Completed Answer Evaluation for: {input_file}")
        if num_random_seeds > 1:
            console.detail("Parsed outputs in", str(generation_folder))
        else:
            console.detail(
                "Output (correct answers only, with 'answerable' field)", str(output_file)
            )
