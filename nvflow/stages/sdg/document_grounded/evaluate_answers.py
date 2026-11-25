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

from nvflow.core import console
from nvflow.execution import CapturedExecutionStage, declare_task_outputs
from nvflow.execution.capture import declare_slurm_experiment_cleanup
from nvflow.integrations.nemo_gym.generation import (
    ENRICH_MODULE_EVALUATE,
    submit_gym_generation,
)
from nvflow.integrations.nemo_gym.native_contract import NativeGymOutputLayout

from ._helpers import build_trim_cmd


class EvaluateAnswersStage(CapturedExecutionStage):
    """Evaluate answers for correctness and answerability."""

    capture_strategy = "gym_generation"

    def build_tasks(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Execute answer evaluation and filtering."""
        from nvflow.integrations.nemo_skills.commands import run_cmd, wrap_arguments

        declare_slurm_experiment_cleanup([f"{expname}-gen", f"{expname}-gen-render", expname])

        input_file = config["input_file"]
        output_dir = config.get("output_dir")
        output_file = config.get("output_file")
        prompt_template = config.get("prompt_template", config.get("prompt_config", ""))
        generation_key = config.get("generation_key", "evaluate_generation")
        inference_params = config.get("inference_params", {})
        num_repeats = config.get("num_repeats", 1)

        if generation_key != "evaluate_generation":
            console.warning(
                "evaluate_answers currently pins generation field to "
                "'evaluate_generation' (rollout enrich hook is fixed-arg); "
                f"configured generation_key='{generation_key}' is ignored."
            )

        console.status("Evaluating answers for correctness and answerability (NeMo-Gym)")
        console.detail("Input file", input_file)
        console.detail("Output dir", str(output_dir))
        console.detail("Prompt template", prompt_template)
        console.detail("Repeats", str(num_repeats))
        console.blank()

        if output_dir:
            generation_folder = Path(output_dir) / Path(input_file).stem
        else:
            generation_folder = Path(output_file).parent / Path(input_file).stem

        console.detail("Generation folder", str(generation_folder))

        lib_evaluate = "python -m nvflow.stages.sdg.document_grounded.evaluate"
        domain_keep_fields = config.get("domain_keep_fields")

        policy = dict((config.get("models") or {}).get("policy") or {})
        model_path = policy.pop("model", "")
        num_gpus = policy.pop("num_gpus", 8)
        server_nodes = policy.pop("server_nodes", 1)
        server = dict(policy.pop("server", {}))

        console.status("Running LLM evaluation via NeMo-Gym")
        gen_expname = f"{expname}-gen"
        submit_gym_generation(
            cluster=cluster,
            rollout_expname=gen_expname,
            run_after=run_after,
            input_file=input_file,
            output_dir=str(generation_folder),
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
            num_repeats=num_repeats,
            inference_params=inference_params,
            vllm_extra=server,
            extra_record_fields=config.get("extra_record_fields"),
            extra_record_field_mappers=config.get("extra_record_field_mappers"),
            enrich_module=ENRICH_MODULE_EVALUATE,
            force_render=config.get("force_render", False),
        )

        # Parse/filter/trim (single-seed) or trim-only (multi-seed), run under
        # the stage expname so downstream `run_after=[stage_expname]` waits.
        postprocess_output_file: str | None = None
        postprocess_output_dir: str | None = None
        analyzed = NativeGymOutputLayout.collection(generation_folder).analyzed_artifacts
        if num_repeats <= 1:
            generated_file = analyzed.rollouts.as_posix()
            parsed_file = str(generation_folder / "parsed.jsonl")
            final_output = (
                output_file if output_file else str(generation_folder / "evaluated.jsonl")
            )
            postprocess_output_file = final_output
            parse_cmd = (
                f"{lib_evaluate} parse --input_file {generated_file} --output_file {parsed_file}"
            )
            filter_cmd = (
                f"{lib_evaluate} filter --input_file {parsed_file} --output_file {final_output}"
            )
            trim_cmd = build_trim_cmd(
                stage_name="evaluate_answers",
                paths=[final_output],
                domain_keep_fields=domain_keep_fields,
            )
            postprocess_cmd = f"{parse_cmd} && {filter_cmd} && {trim_cmd}"
        else:
            postprocess_output_dir = str(generation_folder)
            # AggregateAnswers still needs _ng_rollout_index to compare
            # repeats. The aggregate output is trimmed at the next boundary.
            postprocess_cmd = f'test -s "{analyzed.rollouts}"'
        with declare_task_outputs(
            output_dir=postprocess_output_dir,
            output_file=postprocess_output_file,
        ):
            run_cmd(
                ctx=wrap_arguments(postprocess_cmd),
                cluster=cluster,
                expname=expname,
                log_dir=f"{generation_folder}/postprocess-logs",
                run_after=[gen_expname],
            )

        console.success(f"Completed Answer Evaluation for: {input_file}")
        if num_repeats > 1:
            console.detail("Parsed outputs in", str(generation_folder))
        else:
            console.detail(
                "Output (correct answers only, with 'answerable' field)", str(output_file)
            )
