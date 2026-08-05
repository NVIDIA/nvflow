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

from nvflow.core import BaseStage, console
from nvflow.lib.rl.helpers import resolve_host_path

from ._helpers import (
    build_trim_cmd,
    clean_stale_experiments,
    submit_gym_generation,
)


class GymGenselectAnswersStage(BaseStage):
    """Generate and select best answers via NeMo-Gym collect_rollouts."""

    workflow = "document_grounded_sdg"

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Execute genselect answer generation via rollout()."""
        from nemo_skills.pipeline.cli import run_cmd, wrap_arguments

        clean_stale_experiments(
            cluster,
            [f"{expname}-prep", f"{expname}-gen", f"{expname}-gen-render", expname],
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

        # execute() runs on the orchestrator node: resolve the container path to
        # its host path before checking existence (see _helpers.host_path).
        prep_expname = f"{expname}-prep"
        rerun_prep = config.get("genselect_prep_rerun_done", False)
        prep_host = resolve_host_path(prepped_file)
        prep_exists = prep_host.exists() and prep_host.stat().st_size > 0
        prep_submitted = False
        console.status("Step 1: Preparing genselect data")
        if prep_exists and not rerun_prep:
            console.success("Step 1 skipped (reusing existing prepped genselect input)")
        else:
            run_cmd(
                ctx=wrap_arguments(
                    f"python -m nvflow.lib.sdg.document_grounded.genselect merge "
                    f"--input_dir={input_dir} --output_file={prepped_file}"
                ),
                cluster=cluster,
                expname=prep_expname,
                log_dir=f"{output_dir}/prep-data-logs",
                run_after=run_after,
            )
            prep_submitted = True

        pv = dict(config.get("policy_vllm", {}))
        model_path = pv.pop("model_path", "") or config.get("model", "")
        num_gpus = pv.pop("num_gpus", 0) or config.get("server_gpus", 8)
        server_nodes = pv.pop("server_nodes", 1)

        console.status("Step 2: Generating answers via NeMo-Gym")
        gen_expname = f"{expname}-gen"
        submit_gym_generation(
            cluster=cluster,
            rollout_expname=gen_expname,
            run_after=[prep_expname] if prep_submitted else run_after,
            input_file=prepped_file,
            output_dir=output_dir,
            prompt_template=prompt_template,
            gym_path=config["gym_path"],
            gym_config_paths=config.get("gym_config_paths", []),
            gym_agent_name=config["gym_agent_name"],
            container=config.get("container", "nemo-rl"),
            installation_command=config.get("installation_command"),
            gym_uv_venv_dir=config.get("gym_uv_venv_dir", ""),
            model_path=model_path,
            num_gpus=num_gpus,
            server_nodes=server_nodes,
            num_chunks=config.get("num_chunks", 1),
            num_random_seeds=config.get("num_random_seeds", 1),
            inference_params=config.get("inference_params", {}),
            vllm_extra=pv,
            extra_record_fields=config.get("extra_record_fields"),
            extra_record_field_mappers=config.get("extra_record_field_mappers"),
            rerun_done=config.get("rerun_done", False),
        )

        # Genselect postprocess (select best answer -> output_file) + trim, run
        # under the stage expname so downstream `run_after=[stage_expname]` waits.
        trim_cmd = build_trim_cmd(
            stage_name="gym_genselect_answers",
            paths=[output_file],
            domain_keep_fields=config.get("domain_keep_fields"),
        )
        postprocess_cmd = (
            f"cp {output_dir}/output-rs0.jsonl {output_dir}/output.jsonl && "
            "python -m nvflow.lib.sdg.document_grounded.genselect postprocess "
            f"--input_dir={output_dir} "
            f"--output_file={output_file} && "
            f"{trim_cmd}"
        )
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
            "gym_path",
            "gym_agent_name",
        ):
            if not config.get(field):
                raise ValueError(f"'{field}' is required in genselect_answers config")
        if not config.get("policy_vllm") and not config.get("model"):
            raise ValueError("Either 'policy_vllm.model_path' or 'model' is required")
