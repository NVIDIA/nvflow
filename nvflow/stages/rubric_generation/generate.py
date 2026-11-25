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

"""Shared model-generation stage for all supported rubric methods."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from nvflow.core import console
from nvflow.execution import PlannedExecutionStage
from nvflow.execution.commands import build_python_cmd

from .contracts import METHODS

if TYPE_CHECKING:
    from nvflow.execution import PlanningContext, StagePlan

_CONTAINER_PROMPT_ROOT = Path("/nemo_run/code/nvflow/stages/rubric_generation/prompts")


def default_rubric_prompt(method: str) -> str:
    filename = "simple.yaml" if method == "simple" else "openrubric_contrastive.yaml"
    return str(Path(__file__).with_name("prompts") / filename)


def runtime_rubric_prompt(method: str) -> str:
    """Return the prompt path inside the packaged NeMo-Skills worker."""

    filename = "simple.yaml" if method == "simple" else "openrubric_contrastive.yaml"
    return str(_CONTAINER_PROMPT_ROOT / filename)


def rubric_generation_dir(config: dict[str, Any]) -> str:
    return str(config.get("generation_dir") or f"{config['output_file']}.generation")


def rubric_arguments(config: dict[str, Any]) -> str:
    prompt = config.get("prompt_config") or runtime_rubric_prompt(config["method"])
    key = config.get("generation_key", "rubric_generation")
    inline = config.get(
        "inline_args",
        "++inference.temperature=0.0 ++inference.tokens_to_generate=2048",
    )
    return f"++prompt_config={prompt} ++generation_key={key} {inline}".strip()


def build_rubric_postprocess_command(config: dict[str, Any]) -> str:
    prompt = config.get("prompt_config") or runtime_rubric_prompt(config["method"])
    model = config.get("stage_kwargs", {}).get("model", "unknown")
    return build_python_cmd(
        "nvflow.stages.rubric_generation.postprocess_rubrics",
        input_file=str(Path(rubric_generation_dir(config)) / "output.jsonl"),
        output_file=config["output_file"],
        method=config["method"],
        generation_key=config.get("generation_key", "rubric_generation"),
        prompt_id=str(prompt),
        model_id=str(model),
    )


class GenerateRubricsStage(PlannedExecutionStage):
    """Generate and structurally validate one portable rubric per input record."""

    def validate_config(self, config: dict[str, Any]) -> None:
        for field in ("input_file", "output_file", "method"):
            if not config.get(field):
                raise ValueError(f"'{field}' is required for rubric generation")
        if config["method"] not in METHODS:
            raise ValueError(f"method must be one of {sorted(METHODS)}")
        if not config.get("stage_kwargs", {}).get("model"):
            raise ValueError("stage_kwargs.model is required for rubric generation")

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        from nvflow.execution import declare_task_outputs
        from nvflow.integrations.nemo_skills.commands import generate, wrap_arguments

        console.status(f"Generating rubrics ({config['method']})")
        with declare_task_outputs(output_file=config["output_file"]):
            generate(
                ctx=wrap_arguments(rubric_arguments(config)),
                cluster=cluster,
                input_file=config["input_file"],
                output_dir=rubric_generation_dir(config),
                expname=expname,
                run_after=run_after,
                rerun_done=True,
                postprocess_cmd=build_rubric_postprocess_command(config),
                **config.get("stage_kwargs", {}),
            )

    def plan(self, context: PlanningContext) -> StagePlan:
        from nvflow.execution import declare_task_outputs, plan_generate

        config = dict(context.config)
        with declare_task_outputs(output_file=config["output_file"]):
            return plan_generate(
                context,
                rubric_arguments(config),
                cluster=context.target.cluster,
                input_file=config["input_file"],
                output_dir=rubric_generation_dir(config),
                expname=context.expname,
                run_after=list(context.run_after) or None,
                rerun_done=True,
                postprocess_cmd=build_rubric_postprocess_command(config),
                **config.get("stage_kwargs", {}),
            )
