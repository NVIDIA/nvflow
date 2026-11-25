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
"""Generate and select best answers using NeMo-Gym inference."""

from typing import Any

from nvflow.core import console
from nvflow.execution import CapturedExecutionStage, declare_task_outputs
from nvflow.execution.capture import declare_slurm_experiment_cleanup, reusable_output
from nvflow.integrations.nemo_gym.generation import submit_gym_generation
from nvflow.integrations.nemo_gym.native_contract import NativeGymOutputLayout

from ._helpers import build_trim_cmd


class GymGenselectAnswersStage(CapturedExecutionStage):
    """Generate and select best answers via NeMo-Gym collect_rollouts."""

    capture_strategy = "gym_generation"

    def build_tasks(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Execute genselect answer generation via rollout()."""
        from nvflow.integrations.nemo_skills.commands import run_cmd, wrap_arguments

        declare_slurm_experiment_cleanup(
            [f"{expname}-prep", f"{expname}-gen", f"{expname}-gen-render", expname]
        )

        input_dir = config["input_dir"]
        output_file = config["output_file"]
        prompt_template = config["prompt_template"]

        output_dir = output_file.replace(".jsonl", "")
        prepped_file = output_dir + "_prepped.jsonl"

        console.status("Generating and selecting best answers (NeMo-Gym)")
        console.detail("Input dir", input_dir)
        console.detail("Output file", output_file)
        console.detail("Prepped file", prepped_file)
        console.detail("Output dir", output_dir)
        console.detail("Prompt template", prompt_template)
        console.blank()

        prep_expname = f"{expname}-prep"
        rerun_prep = config.get("genselect_prep_rerun_done", False)
        prep_exists = reusable_output(prepped_file)
        prep_submitted = False
        console.status("Step 1: Preparing genselect data")
        shuffle_solutions = config.get("shuffle_solutions", False)
        shuffle_seed = config.get("shuffle_seed", 42)
        shuffle_args = (
            f" --shuffle_solutions --shuffle_seed={shuffle_seed}" if shuffle_solutions else ""
        )
        include_context_args = " --include_context" if config.get("include_context", False) else ""
        if prep_exists and not rerun_prep:
            console.success("Step 1 skipped (reusing existing prepped genselect input)")
        else:
            with declare_task_outputs(output_file=prepped_file):
                run_cmd(
                    ctx=wrap_arguments(
                        f"python -m nvflow.stages.sdg.document_grounded.genselect merge "
                        f"--input_dir={input_dir} --output_file={prepped_file}"
                        f"{shuffle_args}{include_context_args}"
                    ),
                    cluster=cluster,
                    expname=prep_expname,
                    log_dir=f"{output_dir}/prep-data-logs",
                    run_after=run_after,
                )
            prep_submitted = True

        policy = dict((config.get("models") or {}).get("policy") or {})
        model_path = policy.pop("model", "")
        num_gpus = policy.pop("num_gpus", 8)
        server_nodes = policy.pop("server_nodes", 1)
        server = dict(policy.pop("server", {}))

        console.status("Step 2: Generating answers via NeMo-Gym")
        gen_expname = f"{expname}-gen"
        submit_gym_generation(
            cluster=cluster,
            rollout_expname=gen_expname,
            run_after=[prep_expname] if prep_submitted else run_after,
            input_file=prepped_file,
            output_dir=output_dir,
            prompt_template=prompt_template,
            gym_config_paths=config.get("gym_config_paths", []),
            gym_agent_name=config["gym_agent_name"],
            container=config.get("container", "nemo-rl"),
            vllm_container=config.get("vllm_container"),
            installation_command=config.get("installation_command"),
            model_path=model_path,
            num_gpus=num_gpus,
            server_nodes=server_nodes,
            num_chunks=config.get("num_chunks", 1),
            num_repeats=config.get("num_repeats", 1),
            inference_params=config.get("inference_params", {}),
            vllm_extra=server,
            extra_record_fields=config.get("extra_record_fields"),
            extra_record_field_mappers=config.get("extra_record_field_mappers"),
            force_render=config.get("force_render", False),
        )

        # Genselect postprocess (select best answer -> output_file) + trim, run
        # under the stage expname so downstream `run_after=[stage_expname]` waits.
        trim_cmd = build_trim_cmd(
            stage_name="gym_genselect_answers",
            paths=[output_file],
            domain_keep_fields=config.get("domain_keep_fields"),
        )
        analyzed_rollouts = NativeGymOutputLayout.collection(output_dir).analyzed_artifacts.rollouts
        postprocess_cmd = (
            f"cp {analyzed_rollouts} {output_dir}/output.jsonl && "
            "python -m nvflow.stages.sdg.document_grounded.genselect postprocess "
            f"--input_dir={output_dir} "
            f"--output_file={output_file} && "
            f"{trim_cmd}"
        )
        with declare_task_outputs(output_file=output_file):
            run_cmd(
                ctx=wrap_arguments(postprocess_cmd),
                cluster=cluster,
                expname=expname,
                log_dir=f"{output_dir}/postprocess-logs",
                run_after=[gen_expname],
            )

        console.success(f"Genselect answer generation submitted -> {output_file}")

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate required configuration fields."""
        for field in (
            "input_dir",
            "output_file",
            "prompt_template",
            "gym_agent_name",
        ):
            if not config.get(field):
                raise ValueError(f"'{field}' is required in genselect_answers config")
        if not (config.get("models") or {}).get("policy", {}).get("model"):
            raise ValueError("'models.policy.model' is required")
