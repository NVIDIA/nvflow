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
"""Answer generation pipeline for document-grounded SDG.

Consumes verified-question records produced by GenerateVerifiedQuestionsStage
and emits N candidate answers per question for downstream genselect.
"""

from typing import Any

from nvflow.core import BaseStage, console
from nvflow.lib.rl.helpers import resolve_host_path

from ._helpers import (
    build_trim_cmd,
    clean_stale_experiments,
    parse_stage_kwargs,
    submit_gym_generation,
)


class GenerateAnswersStage(BaseStage):
    """A-side of DG-SDG: a-prep (threshold filter) -> A-gen.

    Output layout under ``output_dir``::

        answer_input.jsonl     # step 1 output (questions surviving the
                               # verification threshold)
        generated/             # step 2 output (A-gen rollouts; consumed by
                               # gym_genselect_answers)
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
                f"{expname}-step1-a-prep",
                f"{expname}-step2-a-gen",
                f"{expname}-step2-a-gen-render",
                expname,
            ],
        )

        input_dir = config["input_dir"]
        output_dir = config["output_dir"]

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
                    f"generate_answers: '{prefix}_gym_agent_name' "
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

        a_gen_overrides = _substep("answer_generation")

        answer_preprocess_kwargs = config.get("answer_preprocess_kwargs", {})
        answer_generation_kwargs = config.get("answer_generation_kwargs", {})

        a_generate_input_file = f"{output_dir}/answer_input.jsonl"
        a_generate_output_dir = f"{output_dir}/generated"

        lib_preprocess = "python -m nvflow.lib.sdg.document_grounded.preprocess"

        # execute() runs on the orchestrator node: resolve the container path to
        # its host path before checking existence (see _helpers.host_path).
        step1_expname = f"{expname}-step1-a-prep"
        rerun_a_prep = config.get("answer_prep_rerun_done", False)
        a_prep_host = resolve_host_path(a_generate_input_file)
        a_prep_exists = a_prep_host.exists() and a_prep_host.stat().st_size > 0
        a_prep_submitted = False
        console.status("Step 1/2: Preparing data for answer generation")
        console.detail("Output file", a_generate_input_file)
        if a_prep_exists and not rerun_a_prep:
            console.success("Step 1 skipped (reusing existing answer_input.jsonl)")
        else:
            console.detail("Input dir", input_dir)
            threshold = answer_preprocess_kwargs.get("threshold", 0.5)
            sbatch_kwargs = answer_preprocess_kwargs.get("sbatch_kwargs", "")
            cmd = (
                f"{lib_preprocess} construct_answer_generate_input "
                f"--input_dir {input_dir} "
                f"--output_file {a_generate_input_file} "
                f"--threshold {threshold}"
            )
            run_cmd(
                ctx=wrap_arguments(cmd),
                cluster=cluster,
                expname=step1_expname,
                run_after=run_after,
                sbatch_kwargs=sbatch_kwargs,
            )
            a_prep_submitted = True
            console.success("Step 1 job submitted")

        console.status("Step 2/2: Generating answers")
        params = parse_stage_kwargs(answer_generation_kwargs)
        a_gen_expname = f"{expname}-step2-a-gen"
        submit_gym_generation(
            cluster=cluster,
            rollout_expname=a_gen_expname,
            run_after=[step1_expname] if a_prep_submitted else run_after,
            input_file=a_generate_input_file,
            output_dir=a_generate_output_dir,
            prompt_template=params["prompt_template"],
            gym_path=gym_path,
            gym_config_paths=a_gen_overrides["gym_config_paths"],
            gym_agent_name=a_gen_overrides["gym_agent_name"],
            container=gym_container,
            installation_command=installation_command,
            gym_uv_venv_dir=gym_uv_venv_dir,
            model_path=params["model_path"],
            num_gpus=params["num_gpus"],
            server_nodes=params["server_nodes"],
            num_chunks=params["num_chunks"],
            num_random_seeds=params["num_random_seeds"],
            inference_params=params["inference_params"],
            vllm_extra=params["vllm_extra"],
            extra_record_fields=a_gen_overrides["extra_record_fields"],
            extra_record_field_mappers=a_gen_overrides["extra_record_field_mappers"],
            rerun_done=answer_generation_kwargs.get("rerun_done", False),
        )

        # Per-stage trim runs under the *stage* expname (depends on a-gen) so
        # downstream `run_after=[stage_expname]` waits for the trimmed output.
        trim_cmd = build_trim_cmd(
            stage_name="generate_answers",
            paths=[a_generate_output_dir],
            domain_keep_fields=config.get("domain_keep_fields"),
        )
        run_cmd(
            ctx=wrap_arguments(trim_cmd),
            cluster=cluster,
            expname=expname,
            log_dir=f"{a_generate_output_dir}/trim-logs",
            run_after=[a_gen_expname],
        )
        console.success("Step 2 job submitted")

        console.blank()
        console.success("Answer generation pipeline jobs submitted")
        console.detail("Generated answers will be in", a_generate_output_dir)

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate required configuration fields."""
        required = ["input_dir", "output_dir", "gym_path"]
        for field in required:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
        if "answer_generation_kwargs" not in config:
            raise ValueError("Missing required field: answer_generation_kwargs")
