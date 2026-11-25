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
"""GenSelect: run multiple rollout seeds through an LLM judge to pick the best answer.

Pipeline position: after rollout generation, before filter_answers/SFT data prep.

The stage has two sub-steps:
  1. prepare_genselect_data  — merges per-seed rollout files into one JSONL of
                               (problem, solutions, reasonings) records.
  2. LLM judge (vLLM/SGLang) — reads the prepared JSONL, scores each solution,
                               and writes a selected_answers JSONL.
  3. postprocess_genselect   — extracts the numeric judgment index and maps it
                               back to the winning generation/reasoning.

Rollout inputs (one of, mutually exclusive):
  input_dir   — single directory globbed for output-rs*.jsonl. Baseline
                single-run behaviour (one model's N seeds in one dir).
  input_files — explicit ordered list of rollout JSONLs. Use for multi-model
                rollouts (e.g. 4 DeepSeek seeds + 4 Qwen seeds) or files not
                co-located in one dir. List order = seed index. No manual
                symlinking; prepare_genselect_data merges them per problem.
                Example YAML usage:
                  input_files:
                    - /workspace/.../deepseekv4-.../rollout/output-rs0.jsonl  # seed 0
                    - ...                                                     # seeds 1-3 (DSV4)
                    - /workspace/.../qwen3plus-.../rollout/output-rs0.jsonl   # seed 4
                    - ...                                                     # seeds 5-7 (Qwen)

Key config options:
  prompt_config   — path to the GenSelect judge prompt YAML.
                    Use genselect_answers_no_context.yaml for Q+A-only runs
                    (tool-calling pipeline where context is not available).
  preprocess_args — extra CLI flags forwarded to prepare_genselect_data.
                    Controls what the judge sees without touching stage code:
                      --strip_context       remove context from all records
                                            (required for tool-calling pipeline)
                      --shuffle_solutions   randomise solution order per problem
                                            to eliminate LLM position bias
                    Example YAML usage:
                      preprocess_args: "--strip_context --shuffle_solutions"
"""

from typing import Any

from nvflow.core import console
from nvflow.execution import CapturedExecutionStage, declare_task_outputs

# Run with uv run nvflow run genselect_answers --config=projects/finance/workflows/sdg/template-based-sdg.yaml --cluster my_cluster


class GenselectAnswersStage(CapturedExecutionStage):
    """Generate and select best answers using parallel thinking."""

    capture_strategy = "run_cmd"

    def build_tasks(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Execute genselect answer generation."""
        from nvflow.integrations.nemo_skills.commands import generate, run_cmd, wrap_arguments

        # Rollout inputs — two mutually-exclusive ways:
        #   input_files: explicit ordered list of rollout JSONLs (multi-model:
        #                4 DeepSeek + 4 Qwen, etc.; order = seed index).
        #   input_dir:   single directory globbed for output-rs*.jsonl (baseline
        #                single-run behaviour).
        input_files = config.get("input_files")
        input_dir = config.get("input_dir")
        output_file = config["output_file"]
        prompt_config = config.get("prompt_config")
        inline_args = config.get("inline_args", "")
        preprocess_args = config.get("preprocess_args", "")

        if input_files:
            input_arg = "--input_files " + " ".join(input_files)
            input_desc = f"{len(input_files)} file(s)"
        elif input_dir:
            input_arg = f"--input_dir={input_dir}"
            input_desc = input_dir
        else:
            raise ValueError("genselect_answers requires either 'input_files' or 'input_dir'")

        # Calculate intermediate file paths
        output_dir = output_file.replace(".jsonl", "")
        prepped_file = output_dir + "_prepped.jsonl"

        console.status("Generating and selecting best answers")
        console.detail("Input", input_desc)
        console.detail("Output file", output_file)
        console.detail("Prepped file", prepped_file)
        console.detail("Output dir", output_dir)
        console.detail("Prompt config", str(prompt_config))
        console.detail("Inline args", str(inline_args))
        console.blank()

        console.status("Step 1: Preparing genselect data")
        with declare_task_outputs(output_file=prepped_file):
            run_cmd(
                ctx=wrap_arguments(
                    f"python3 -m nvfinance.utils.sdg.prepare_genselect_data {input_arg} --output_file={prepped_file} {preprocess_args}"
                ),
                cluster=cluster,
                expname=f"{expname}-prep",
                log_dir=f"{output_dir}/prep-data-logs",
                run_after=run_after,
            )

        postprocess_cmd = f"python3 -m nvfinance.utils.sdg.postprocess_genselect --input_dir={output_dir} --output_file={output_file}"

        console.status("Generating answers with genselect")

        stage_kwargs = config.get("stage_kwargs", {})
        ctx = wrap_arguments(f"++prompt_config={prompt_config} {inline_args}")
        with declare_task_outputs(output_dir=output_dir, output_file=output_file):
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
