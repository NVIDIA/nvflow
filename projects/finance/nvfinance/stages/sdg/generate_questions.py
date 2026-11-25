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

from nvflow.core import console
from nvflow.execution import CapturedExecutionStage, declare_task_outputs
from nvflow.execution.commands import build_python_cmd

# Run with uv run nvflow run generate_questions --config=projects/finance/workflows/sdg/template-based-sdg.yaml --cluster my_cluster


class GenerateQuestionsStage(CapturedExecutionStage):
    """Generate questions for financial reasoning."""

    capture_strategy = "run_cmd"

    def build_tasks(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Execute question generation."""
        from nvflow.integrations.nemo_skills.commands import generate, run_cmd, wrap_arguments

        input_file = config["input_file"]
        output_file = config["output_file"]
        prompt_config = config.get("prompt_config")
        inline_args = config.get("inline_args", "")
        company_info_file = config["company_info_file"]
        start_year = config["start_year"]
        end_year = config["end_year"]
        sampling_mode = config.get("sampling_mode", "exhaustive")
        sampling_seed = config.get("sampling_seed", 42)
        form_weights = config.get("form_weights")

        console.status("Generating SDG questions")
        console.detail("Input file", input_file)
        console.detail("Company info file", company_info_file)
        console.detail("Start year", start_year)
        console.detail("End year", end_year)
        console.detail("Sampling mode", sampling_mode)
        if sampling_mode == "sample":
            console.detail("Total samples", config["total_samples"])
            console.detail("Sampling seed", sampling_seed)
            console.detail("Form weights", str(form_weights))
        console.detail("Output file", output_file)
        console.detail("Prompt config", str(prompt_config))
        console.detail("Inline args", str(inline_args))
        console.blank()

        output_dir = Path(output_file).parent
        prepped_file = str(output_dir / (Path(input_file).stem + "_prepped.jsonl"))
        console.status(f"Prepped file: {prepped_file}")

        generation_folder = output_dir / Path(input_file).stem
        prep_log_dir = str(generation_folder / "prep-logs")

        prep_kwargs: dict[str, Any] = {
            "input_file": input_file,
            "company_info_file": company_info_file,
            "start_year": start_year,
            "end_year": end_year,
            "mode": sampling_mode,
            "output_file": prepped_file,
        }
        if sampling_mode == "sample":
            prep_kwargs.update(total_samples=config["total_samples"], seed=sampling_seed)
            if form_weights:
                import json

                prep_kwargs["form_weights"] = json.dumps(form_weights)

        # Run prep command to generate or sample company/year combinations.
        with declare_task_outputs(output_file=prepped_file):
            run_cmd(
                ctx=wrap_arguments(
                    build_python_cmd("nvfinance.utils.sdg.prepare_question_gen_data", **prep_kwargs)
                ),
                cluster=cluster,
                log_dir=prep_log_dir,
                expname=f"{expname}-prep",
                run_after=run_after,
            )

        console.detail("Output dir", str(generation_folder))

        generated_file = str(generation_folder / "output.jsonl")

        postprocess_cmd = f"python3 -m nvfinance.utils.sdg.parse_generated_questions --input_file {generated_file} --output_file {output_file}"

        stage_kwargs = config.get("stage_kwargs", {})
        ctx = wrap_arguments(f"++prompt_config={prompt_config} {inline_args}")
        with declare_task_outputs(
            output_dir=generation_folder,
            output_file=output_file,
        ):
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

    def validate_config(self, config: dict[str, Any]) -> None:
        mode = config.get("sampling_mode", "exhaustive")
        if mode not in {"exhaustive", "sample"}:
            raise ValueError("'sampling_mode' must be either 'exhaustive' or 'sample'")
        if mode == "sample" and int(config.get("total_samples", 0)) <= 0:
            raise ValueError("'total_samples' must be positive in sample mode")
