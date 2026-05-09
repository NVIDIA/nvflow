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
"""Estimate question difficulty by testing with a smaller model."""

from typing import Any

from nvflow.core import BaseStage, StageRegistry, console
from nvflow.lib.vllm_compat import inject_server_entrypoint


@StageRegistry.register(
    recipe="finance",
    workflow="document_grounded_sdg",
    stage="difficulty_estimation",
)
class DifficultyEstimationStage(BaseStage):
    """Estimate question difficulty using a smaller model.

    This stage:
    1. Prepares input for small model
    2. Uses a smaller model (e.g., Qwen3-4B) to answer each question N times
    3. Prepares judge input (one file per seed for streaming)
    4. Uses a larger model (e.g., GPT-OSS120) to judge each answer (N parallel jobs)
    5. Aggregates results: difficulty_score = number of correct answers
       - Score 0: Very hard (small model always wrong)
       - Score N: Easy (small model always correct)

    Uses streaming processing to handle large datasets without loading all data into memory.
    """

    workflow = "document_grounded_sdg"

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Execute difficulty estimation pipeline."""
        from nemo_skills.pipeline.cli import generate, run_cmd, wrap_arguments

        input_file = config["input_file"]
        output_file = config["output_file"]

        # Model configs
        answer_model_kwargs = config.get("answer_model_kwargs", {})
        judge_model_kwargs = config.get("judge_model_kwargs", {})

        # Prompt configs
        answer_prompt = config.get("answer_prompt_config")
        judge_prompt = config.get("judge_prompt_config")

        num_seeds = config.get("num_random_seeds", 5)
        work_dir = config.get("work_dir", output_file.rsplit("/", 1)[0] + "/difficulty_work")

        console.status("Estimating question difficulty")
        console.detail("Input file", input_file)
        console.detail("Output file", output_file)
        console.detail("Work dir", work_dir)
        console.detail("Number of seeds", str(num_seeds))
        console.blank()

        # Define intermediate paths
        prep_file = f"{work_dir}/prep_input.jsonl"
        answer_dir = f"{work_dir}/answers"
        judge_input_dir = f"{work_dir}/judge_inputs"
        judge_output_dir = f"{work_dir}/judged"

        module = "nvflow.recipes.finance.utils.sdg.difficulty_estimation"

        # =====================================================================
        # Step 1: Prepare input (keep reference_answer for later comparison)
        # =====================================================================
        console.status("Step 1/4: Preparing input for small model")

        prep_cmd = (
            f"python3 -m {module} prepare_input --input_file {input_file} --output_file {prep_file}"
        )

        run_cmd(
            ctx=wrap_arguments(prep_cmd),
            cluster=cluster,
            expname=f"{expname}-prep",
            run_after=run_after,
        )
        console.success("Step 1 job submitted")

        # =====================================================================
        # Step 2: Generate answers with small model (N random seeds)
        # =====================================================================
        console.status(f"Step 2/4: Generating answers with small model ({num_seeds} seeds)")

        answer_args = answer_model_kwargs.get("args", {}).copy()
        answer_args = inject_server_entrypoint(
            answer_args, answer_args.get("model", "")
        )  # WORKAROUND(vllm-0.17-hermes, harmony-aarch64)
        answer_ctx_args = answer_model_kwargs.get("ctx_args", "")

        if answer_prompt:
            answer_ctx_args += f" ++prompt_config={answer_prompt}"

        ctx = wrap_arguments(answer_ctx_args)

        generate(
            ctx=ctx,
            cluster=cluster,
            input_file=prep_file,
            output_dir=answer_dir,
            expname=f"{expname}-answer",
            run_after=[f"{expname}-prep"],
            num_random_seeds=num_seeds,
            **answer_args,
        )
        console.success("Step 2 job submitted")

        # =====================================================================
        # Step 3: Prepare judge input (pair reference with candidate answers)
        # =====================================================================
        console.status("Step 3/4: Preparing input for judge model")

        max_answer_chars = config.get("max_answer_chars", 20000)
        judge_prep_cmd = f"python3 -m {module} prepare_judge --input_dir {answer_dir} --output_dir {judge_input_dir} --max_answer_chars {max_answer_chars}"

        run_cmd(
            ctx=wrap_arguments(judge_prep_cmd),
            cluster=cluster,
            expname=f"{expname}-judge-prep",
            run_after=[f"{expname}-answer"],
        )
        console.success("Step 3 job submitted")

        # =====================================================================
        # Step 4: Judge answers with large model (one job per seed)
        # =====================================================================
        console.status(f"Step 4/5: Judging answers ({num_seeds} separate jobs)")

        judge_args = judge_model_kwargs.get("args", {}).copy()
        judge_args = inject_server_entrypoint(
            judge_args, judge_args.get("model", "")
        )  # WORKAROUND(vllm-0.17-hermes, harmony-aarch64)
        judge_ctx_args = judge_model_kwargs.get("ctx_args", "")

        if judge_prompt:
            judge_ctx_args += f" ++prompt_config={judge_prompt}"
        judge_ctx_args += " ++generation_key=judge_generation"

        ctx = wrap_arguments(judge_ctx_args)

        # Submit a judge job for each seed
        for seed_idx in range(num_seeds):
            judge_input_file = f"{judge_input_dir}/judge_input_rs{seed_idx}.jsonl"
            seed_output_dir = f"{judge_output_dir}/rs{seed_idx}"

            generate(
                ctx=ctx,
                cluster=cluster,
                input_file=judge_input_file,
                output_dir=seed_output_dir,
                expname=f"{expname}-judge-rs{seed_idx}",
                run_after=[f"{expname}-judge-prep"],
                **judge_args,
            )
            console.detail(f"Seed {seed_idx}", f"{judge_input_file} -> {seed_output_dir}")

        console.success("Step 4 jobs submitted")

        # =====================================================================
        # Step 5: Aggregate difficulty scores
        # =====================================================================
        console.status("Step 5/5: Aggregating difficulty scores")

        # Wait for all judge jobs to complete
        judge_job_names = [f"{expname}-judge-rs{i}" for i in range(num_seeds)]

        aggregate_cmd = f"python3 -m {module} aggregate --input_dir {judge_output_dir} --output_file {output_file} --num_seeds {num_seeds}"

        run_cmd(
            ctx=wrap_arguments(aggregate_cmd),
            cluster=cluster,
            expname=expname,  # Use base expname for downstream dependencies
            run_after=judge_job_names,
        )
        console.success("Step 5 job submitted")

        console.blank()
        console.success("Difficulty estimation pipeline submitted")
        console.detail("Final output will be in", output_file)

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate stage configuration."""
        required = ["input_file", "output_file"]
        for field in required:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")

        if "answer_model_kwargs" not in config:
            raise ValueError("Missing required field: answer_model_kwargs")
        if "judge_model_kwargs" not in config:
            raise ValueError("Missing required field: judge_model_kwargs")
