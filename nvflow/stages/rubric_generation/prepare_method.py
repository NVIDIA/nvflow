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

"""Prepare method-specific inputs for simple and contrastive rubric generation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from nvflow.core import console
from nvflow.execution import PlannedExecutionStage
from nvflow.execution.commands import build_python_cmd

from .contracts import METHODS

if TYPE_CHECKING:
    from nvflow.execution import PlanningContext, StagePlan

STRATEGIES = {"direct", "rank_candidates"}
_CONTAINER_PROMPT_ROOT = Path("/nemo_run/code/nvflow/stages/rubric_generation/prompts")


def default_ranking_prompt() -> str:
    return str(Path(__file__).with_name("prompts") / "rank_answers.yaml")


def runtime_ranking_prompt() -> str:
    """Return the ranking prompt path inside the packaged worker."""

    return str(_CONTAINER_PROMPT_ROOT / "rank_answers.yaml")


def _strategy(config: dict[str, Any]) -> str:
    configured = config.get("strategy")
    if configured:
        return str(configured)
    return "rank_candidates" if config.get("method") == "openrubric_contrastive" else "direct"


def build_direct_command(config: dict[str, Any]) -> str:
    return build_python_cmd(
        "nvflow.stages.rubric_generation.prepare_direct",
        input_file=config["input_file"],
        output_file=config["output_file"],
        method=config["method"],
    )


def ranking_generation_dir(config: dict[str, Any]) -> str:
    return str(config.get("generation_dir") or f"{config['output_file']}.ranking-generation")


def build_ranking_postprocess_command(config: dict[str, Any]) -> str:
    generated_file = str(Path(ranking_generation_dir(config)) / "output.jsonl")
    return build_python_cmd(
        "nvflow.stages.rubric_generation.postprocess_rankings",
        input_file=generated_file,
        output_file=config["output_file"],
        rankings_file=config["rankings_file"],
        audit_file=config["audit_file"],
        generation_key=config.get("generation_key", "ranking_generation"),
    )


def ranking_arguments(config: dict[str, Any]) -> str:
    prompt = config.get("prompt_config") or runtime_ranking_prompt()
    key = config.get("generation_key", "ranking_generation")
    inline = config.get(
        "inline_args",
        "++inference.temperature=0.0 ++inference.tokens_to_generate=1024",
    )
    return f"++prompt_config={prompt} ++generation_key={key} {inline}".strip()


class PrepareRubricMethodStage(PlannedExecutionStage):
    """Pass simple/direct pairs through or rank candidates for OpenRubric-style input."""

    def validate_config(self, config: dict[str, Any]) -> None:
        for field in ("input_file", "output_file", "method"):
            if not config.get(field):
                raise ValueError(f"'{field}' is required for rubric method preparation")
        if config["method"] not in METHODS:
            raise ValueError(f"method must be one of {sorted(METHODS)}")
        strategy = _strategy(config)
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {sorted(STRATEGIES)}")
        if strategy == "rank_candidates":
            if config["method"] != "openrubric_contrastive":
                raise ValueError("rank_candidates is only valid for openrubric_contrastive")
            for field in ("rankings_file", "audit_file"):
                if not config.get(field):
                    raise ValueError(f"'{field}' is required when ranking candidates")
            if not config.get("stage_kwargs", {}).get("model"):
                raise ValueError("stage_kwargs.model is required when ranking candidates")

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        from nvflow.execution import declare_task_outputs

        strategy = _strategy(config)
        console.status(f"Preparing rubric method input ({config['method']}, {strategy})")
        if strategy == "direct":
            from nvflow.integrations.nemo_skills.commands import run_cmd, wrap_arguments

            kwargs: dict[str, Any] = {
                "cluster": cluster,
                "num_gpus": 0,
                "log_dir": config.get("log_dir") or f"{config['output_file']}.logs",
                "expname": expname,
                "run_after": run_after,
            }
            if config.get("container"):
                kwargs["container"] = config["container"]
            if config.get("installation_command"):
                kwargs["installation_command"] = config["installation_command"]
            with declare_task_outputs(output_file=config["output_file"]):
                run_cmd(ctx=wrap_arguments(""), command=build_direct_command(config), **kwargs)
            return

        from nvflow.integrations.nemo_skills.commands import generate, wrap_arguments

        with declare_task_outputs(output_file=config["output_file"]):
            generate(
                ctx=wrap_arguments(ranking_arguments(config)),
                cluster=cluster,
                input_file=config["input_file"],
                output_dir=ranking_generation_dir(config),
                expname=expname,
                run_after=run_after,
                rerun_done=True,
                postprocess_cmd=build_ranking_postprocess_command(config),
                **config.get("stage_kwargs", {}),
            )

    def plan(self, context: PlanningContext) -> StagePlan:
        from nvflow.execution import declare_task_outputs, plan_generate, plan_run_cmd

        config = dict(context.config)
        with declare_task_outputs(output_file=config["output_file"]):
            if _strategy(config) == "direct":
                parameters: dict[str, Any] = {
                    "cluster": context.target.cluster,
                    "num_gpus": 0,
                    "log_dir": config.get("log_dir") or f"{config['output_file']}.logs",
                    "expname": context.expname,
                    "run_after": list(context.run_after) or None,
                }
                if config.get("container"):
                    parameters["container"] = config["container"]
                if config.get("installation_command"):
                    parameters["installation_command"] = config["installation_command"]
                return plan_run_cmd(context, build_direct_command(config), **parameters)
            return plan_generate(
                context,
                ranking_arguments(config),
                cluster=context.target.cluster,
                input_file=config["input_file"],
                output_dir=ranking_generation_dir(config),
                expname=context.expname,
                run_after=list(context.run_after) or None,
                rerun_done=True,
                postprocess_cmd=build_ranking_postprocess_command(config),
                **config.get("stage_kwargs", {}),
            )
