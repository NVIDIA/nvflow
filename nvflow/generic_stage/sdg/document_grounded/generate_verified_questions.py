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
"""Question generation + verification pipeline for document-grounded SDG."""

from typing import Any

from nvflow.core import BaseStage, console
from nvflow.lib.rl.helpers import resolve_host_path

from ._helpers import (
    build_trim_cmd,
    clean_stale_experiments,
    parse_stage_kwargs,
    submit_gym_generation,
)


class GenerateVerifiedQuestionsStage(BaseStage):
    """Q-side of DG-SDG: prep -> generate -> verify-prep -> verify.

    Output layout under ``output_dir``::

        generate_input.jsonl   # step 1 output
        generated/             # step 2 output (Q-gen rollouts)
        verify_input.jsonl     # step 3 output
        verified/              # step 4 output (Q-verify rollouts; consumed by
                               # the generate_answers stage)
    """

    workflow = "document_grounded_sdg"

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        from nemo_skills.pipeline.cli import run_cmd, wrap_arguments

        clean_stale_experiments(
            cluster,
            [
                f"{expname}-step1-q-prep",
                f"{expname}-step2-q-gen",
                f"{expname}-step2-q-gen-render",
                f"{expname}-step3-q-verify-prep",
                f"{expname}-step4-q-verify",
                f"{expname}-step4-q-verify-render",
                expname,
            ],
        )

        input_folder = config["input_folder"]
        output_dir = config["output_dir"]
        question_prep_script = config["question_prep_script"]

        gym_path = config["gym_path"]
        gym_uv_venv_dir = config.get("gym_uv_venv_dir", "")
        gym_config_paths_default = config.get("gym_config_paths", [])
        gym_agent_name_default = config.get("gym_agent_name")
        gym_container = config.get("container", "nemo-rl")
        installation_command = config.get("installation_command")
        extra_record_fields_default = config.get("extra_record_fields")
        extra_record_field_mappers_default = config.get("extra_record_field_mappers")

        def _substep(prefix: str) -> dict[str, Any]:
            agent = config.get(f"{prefix}_gym_agent_name", gym_agent_name_default)
            if not agent:
                raise ValueError(
                    f"generate_verified_questions: '{prefix}_gym_agent_name' "
                    "(or stage-level 'gym_agent_name') is required."
                )
            return {
                "gym_config_paths": config.get(
                    f"{prefix}_gym_config_paths", gym_config_paths_default
                ),
                "gym_agent_name": agent,
                "extra_record_fields": config.get(
                    f"{prefix}_extra_record_fields", extra_record_fields_default
                ),
                "extra_record_field_mappers": config.get(
                    f"{prefix}_extra_record_field_mappers",
                    extra_record_field_mappers_default,
                ),
            }

        q_gen_overrides = _substep("question_generation")
        q_verify_overrides = _substep("question_verify")

        question_generation_kwargs = config.get("question_generation_kwargs", {})
        question_verify_kwargs = config.get("question_verify_kwargs", {})
        rerun_q_prep = config.get("question_prep_rerun_done", False)
        rerun_q_verify_prep = config.get("question_verify_prep_rerun_done", False)

        q_generate_input_file = f"{output_dir}/generate_input.jsonl"
        q_generate_output_dir = f"{output_dir}/generated"
        q_verify_input_file = f"{output_dir}/verify_input.jsonl"
        q_verify_output_dir = f"{output_dir}/verified"

        lib_preprocess = "python -m nvflow.lib.sdg.document_grounded.preprocess"

        step1_expname = f"{expname}-step1-q-prep"
        # execute() runs on the orchestrator node: resolve the container path to
        # its host path before checking existence (see _helpers.host_path).
        step1_host = resolve_host_path(q_generate_input_file)
        step1_exists = step1_host.exists() and step1_host.stat().st_size > 0
        step1_submitted = False
        if step1_exists and not rerun_q_prep:
            console.status("Step 1/4: Preparing data for question generation")
            console.detail("Output file", q_generate_input_file)
            console.success("Step 1 skipped (reusing existing generate_input.jsonl)")
        else:
            console.status("Step 1/4: Preparing data for question generation")
            console.detail("Input folder", input_folder)
            console.detail("Output file", q_generate_input_file)
            cmd = (
                f"python {question_prep_script} "
                f"--input_folder {input_folder} "
                f"--output_file {q_generate_input_file}"
            )
            run_cmd(
                ctx=wrap_arguments(cmd),
                cluster=cluster,
                expname=step1_expname,
                run_after=run_after,
            )
            step1_submitted = True
            console.success("Step 1 job submitted")

        console.status("Step 2/4: Generating questions")
        q_gen_params = parse_stage_kwargs(question_generation_kwargs)
        submit_gym_generation(
            cluster=cluster,
            rollout_expname=f"{expname}-step2-q-gen",
            run_after=[step1_expname] if step1_submitted else run_after,
            input_file=q_generate_input_file,
            output_dir=q_generate_output_dir,
            prompt_template=q_gen_params["prompt_template"],
            gym_path=gym_path,
            gym_config_paths=q_gen_overrides["gym_config_paths"],
            gym_agent_name=q_gen_overrides["gym_agent_name"],
            container=gym_container,
            installation_command=installation_command,
            gym_uv_venv_dir=gym_uv_venv_dir,
            model_path=q_gen_params["model_path"],
            num_gpus=q_gen_params["num_gpus"],
            server_nodes=q_gen_params["server_nodes"],
            num_chunks=q_gen_params["num_chunks"],
            num_random_seeds=q_gen_params["num_random_seeds"],
            inference_params=q_gen_params["inference_params"],
            vllm_extra=q_gen_params["vllm_extra"],
            extra_record_fields=q_gen_overrides["extra_record_fields"],
            extra_record_field_mappers=q_gen_overrides["extra_record_field_mappers"],
            rerun_done=question_generation_kwargs.get("rerun_done", False),
        )
        console.success("Step 2 job submitted")

        step3_expname = f"{expname}-step3-q-verify-prep"
        step3_host = resolve_host_path(q_verify_input_file)
        step3_exists = step3_host.exists() and step3_host.stat().st_size > 0
        step3_submitted = False
        if step3_exists and not rerun_q_verify_prep:
            console.status("Step 3/4: Preparing data for question verification")
            console.detail("Output file", q_verify_input_file)
            console.success("Step 3 skipped (reusing existing verify_input.jsonl)")
        else:
            console.status("Step 3/4: Preparing data for question verification")
            cmd = (
                f"{lib_preprocess} construct_question_verify_input "
                f"--input_dir {q_generate_output_dir} "
                f"--output_file {q_verify_input_file}"
            )
            run_cmd(
                ctx=wrap_arguments(cmd),
                cluster=cluster,
                expname=step3_expname,
                run_after=[f"{expname}-step2-q-gen"],
            )
            step3_submitted = True
            console.success("Step 3 job submitted")

        console.status("Step 4/4: Verifying questions")
        q_verify_params = parse_stage_kwargs(question_verify_kwargs)
        q_verify_expname = f"{expname}-step4-q-verify"
        submit_gym_generation(
            cluster=cluster,
            rollout_expname=q_verify_expname,
            run_after=[step3_expname] if step3_submitted else [f"{expname}-step2-q-gen"],
            input_file=q_verify_input_file,
            output_dir=q_verify_output_dir,
            prompt_template=q_verify_params["prompt_template"],
            gym_path=gym_path,
            gym_config_paths=q_verify_overrides["gym_config_paths"],
            gym_agent_name=q_verify_overrides["gym_agent_name"],
            container=gym_container,
            installation_command=installation_command,
            gym_uv_venv_dir=gym_uv_venv_dir,
            model_path=q_verify_params["model_path"],
            num_gpus=q_verify_params["num_gpus"],
            server_nodes=q_verify_params["server_nodes"],
            num_chunks=q_verify_params["num_chunks"],
            num_random_seeds=q_verify_params["num_random_seeds"],
            inference_params=q_verify_params["inference_params"],
            vllm_extra=q_verify_params["vllm_extra"],
            extra_record_fields=q_verify_overrides["extra_record_fields"],
            extra_record_field_mappers=q_verify_overrides["extra_record_field_mappers"],
            rerun_done=question_verify_kwargs.get("rerun_done", False),
        )

        # Stage trim runs under the stage expname (depends on q-verify) so the
        # downstream stage's `run_after=[stage_expname]` waits for trimmed output.
        trim_cmd = build_trim_cmd(
            stage_name="generate_verified_questions",
            paths=[q_verify_output_dir],
            domain_keep_fields=config.get("domain_keep_fields"),
        )
        run_cmd(
            ctx=wrap_arguments(trim_cmd),
            cluster=cluster,
            expname=expname,
            log_dir=f"{q_verify_output_dir}/trim-logs",
            run_after=[q_verify_expname],
        )
        console.success("Step 4 job submitted")

        console.blank()
        console.success("Question generation + verification pipeline jobs submitted")
        console.detail("Verified questions will be in", q_verify_output_dir)

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate required configuration fields."""
        required = ["input_folder", "output_dir", "gym_path", "question_prep_script"]
        for field in required:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
        if "question_generation_kwargs" not in config:
            raise ValueError("Missing required field: question_generation_kwargs")
        if "question_verify_kwargs" not in config:
            raise ValueError("Missing required field: question_verify_kwargs")
