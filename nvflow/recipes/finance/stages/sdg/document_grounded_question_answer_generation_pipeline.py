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
"""Question and answer generation pipeline for document-grounded synthetic data generation."""

from typing import Any

from nvflow.core import BaseStage, StageRegistry, console
from nvflow.lib.vllm_compat import inject_server_entrypoint


@StageRegistry.register(
    recipe="finance",
    workflow="document_grounded_sdg",
    stage="generate_verified_qa",
)
class DocumentGroundedQuestionAnswerGenerationPipelineStage(BaseStage):
    """
    Combined stage for question generation, verification, and answer generation.

    Executes 6 sub-steps internally:
    1. Preprocess input data for question generation
    2. Generate questions using LLM
    3. Preprocess generated questions for verification
    4. Verify questions using LLM
    5. Preprocess verified questions for answer generation
    6. Generate answers using LLM
    """

    workflow = "document_grounded_sdg"

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        from nemo_skills.pipeline.cli import generate, run_cmd, wrap_arguments

        input_folder = config["input_folder"]
        output_dir = config["output_dir"]

        # Question pipeline config
        question_generation_kwargs = config.get("question_generation_kwargs", {})
        question_verify_kwargs = config.get("question_verify_kwargs", {})

        # Answer pipeline config
        answer_preprocess_kwargs = config.get("answer_preprocess_kwargs", {})
        answer_generation_kwargs = config.get("answer_generation_kwargs", {})

        # Define paths
        # Question pipeline paths
        question_output_dir = f"{output_dir}/question_pipeline"
        q_generate_input_file = f"{question_output_dir}/generate_input.jsonl"
        q_generate_output_dir = f"{question_output_dir}/generated"
        q_verify_input_file = f"{question_output_dir}/verify_input.jsonl"
        q_verify_output_dir = f"{question_output_dir}/verified"

        # Answer pipeline paths
        answer_output_dir = f"{output_dir}/answer_pipeline"
        a_generate_input_file = f"{answer_output_dir}/answer_input.jsonl"
        a_generate_output_dir = f"{answer_output_dir}/generated"

        module = "nvflow.recipes.finance.utils.sdg.document_grounded_preprocess"

        # =====================================================================
        # Step 1: Construct question generate input
        # =====================================================================
        console.status("Step 1/6: Preparing data for question generation")
        console.detail("Input folder", input_folder)
        console.detail("Output file", q_generate_input_file)

        cmd = f"python3 -m {module} construct_question_generate_input --input_folder {input_folder} --output_file {q_generate_input_file}"

        run_cmd(
            ctx=wrap_arguments(cmd),
            cluster=cluster,
            expname=f"{expname}-step1-q-prep",
            run_after=run_after,
        )
        console.success("Step 1 job submitted")

        # =====================================================================
        # Step 2: Generate questions
        # =====================================================================
        console.status("Step 2/6: Generating questions")

        q_gen_args = question_generation_kwargs.get("args", {}).copy()
        q_gen_args = inject_server_entrypoint(
            q_gen_args, q_gen_args.get("model", "")
        )  # WORKAROUND(vllm-0.17-hermes, harmony-aarch64)
        q_gen_ctx_args = question_generation_kwargs.get("ctx_args", "")

        # Handle skip_filled
        if q_gen_args.pop("skip_filled", True):
            q_gen_ctx_args += " ++skip_filled=True"

        ctx = wrap_arguments(q_gen_ctx_args)

        generate(
            ctx=ctx,
            cluster=cluster,
            input_file=q_generate_input_file,
            output_dir=q_generate_output_dir,
            expname=f"{expname}-step2-q-gen",
            run_after=[f"{expname}-step1-q-prep"],
            **q_gen_args,
        )
        console.success("Step 2 job submitted")

        # =====================================================================
        # Step 3: Construct question verify input
        # =====================================================================
        console.status("Step 3/6: Preparing data for question verification")

        cmd = f"python3 -m {module} construct_question_verify_input --input_dir {q_generate_output_dir} --output_file {q_verify_input_file}"

        run_cmd(
            ctx=wrap_arguments(cmd),
            cluster=cluster,
            expname=f"{expname}-step3-q-verify-prep",
            run_after=[f"{expname}-step2-q-gen"],
        )
        console.success("Step 3 job submitted")

        # =====================================================================
        # Step 4: Verify questions
        # =====================================================================
        console.status("Step 4/6: Verifying questions")

        verify_args = question_verify_kwargs.get("args", {}).copy()
        verify_args = inject_server_entrypoint(
            verify_args, verify_args.get("model", "")
        )  # WORKAROUND(vllm-0.17-hermes, harmony-aarch64)
        verify_ctx_args = question_verify_kwargs.get("ctx_args", "")

        # Handle skip_filled
        if verify_args.pop("skip_filled", True):
            verify_ctx_args += " ++skip_filled=True"

        ctx = wrap_arguments(verify_ctx_args)

        generate(
            ctx=ctx,
            cluster=cluster,
            input_file=q_verify_input_file,
            output_dir=q_verify_output_dir,
            expname=f"{expname}-step4-q-verify",
            run_after=[f"{expname}-step3-q-verify-prep"],
            **verify_args,
        )
        console.success("Step 4 job submitted")

        # =====================================================================
        # Step 5: Construct answer generate input
        # =====================================================================
        # Input for answer generation is the verified questions output
        answer_input_dir = q_verify_output_dir

        console.status("Step 5/6: Preparing data for answer generation")
        console.detail("Input dir", answer_input_dir)
        console.detail("Output file", a_generate_input_file)

        threshold = answer_preprocess_kwargs.get("threshold", 0.5)
        sbatch_kwargs = answer_preprocess_kwargs.get("sbatch_kwargs", "")

        cmd = f"python3 -m {module} construct_answer_generate_input --input_dir {answer_input_dir} --output_file {a_generate_input_file} --threshold {threshold}"

        run_cmd(
            ctx=wrap_arguments(cmd),
            cluster=cluster,
            expname=f"{expname}-step5-a-prep",
            run_after=[f"{expname}-step4-q-verify"],
            sbatch_kwargs=sbatch_kwargs,
        )
        console.success("Step 5 job submitted")
        prep_job_name = [f"{expname}-step5-a-prep"]

        # =====================================================================
        # Step 6: Generate answers
        # =====================================================================
        console.status("Step 6/6: Generating answers")

        a_gen_args = answer_generation_kwargs.get("args", {}).copy()
        a_gen_args = inject_server_entrypoint(
            a_gen_args, a_gen_args.get("model", "")
        )  # WORKAROUND(vllm-0.17-hermes, harmony-aarch64)
        a_gen_ctx_args = answer_generation_kwargs.get("ctx_args", "")

        # Handle skip_filled
        if a_gen_args.pop("skip_filled", False):
            a_gen_ctx_args += " ++skip_filled=True"

        ctx = wrap_arguments(a_gen_ctx_args)

        # IMPORTANT: Use base expname (no suffix) for the LAST job
        # This allows downstream stages to correctly depend on this stage
        generate(
            ctx=ctx,
            cluster=cluster,
            input_file=a_generate_input_file,
            output_dir=a_generate_output_dir,
            expname=expname,  # Use base expname for downstream dependencies
            run_after=prep_job_name,
            **a_gen_args,
        )
        console.success("Step 6 job submitted")

        console.blank()
        console.success("Question and Answer pipeline jobs submitted successfully")
        console.detail("Final output will be in", a_generate_output_dir)

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate that required configuration fields are present."""
        required = ["input_folder", "output_dir"]
        for field in required:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")

        if "question_generation_kwargs" not in config:
            raise ValueError("Missing required field: question_generation_kwargs")
        if "question_verify_kwargs" not in config:
            raise ValueError("Missing required field: question_verify_kwargs")
        if "answer_generation_kwargs" not in config:
            raise ValueError("Missing required field: answer_generation_kwargs")
