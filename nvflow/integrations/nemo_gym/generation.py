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
"""Plan document-grounded generation through NeMo-Gym."""

from __future__ import annotations

import json
import shlex
from typing import TYPE_CHECKING, Any

from nvflow.execution.contracts import (
    ArtifactDeclaration,
    CommandPlan,
    DependencyPlan,
    ResourceRequest,
    StagePlan,
    TaskPlan,
)

if TYPE_CHECKING:
    from nvflow.execution.filesystem import TargetFilesystem
    from nvflow.execution.planner import PlanningContext


ENRICH_MODULE = "nvflow.stages.sdg.document_grounded.enrich_rollouts"
ENRICH_MODULE_EVALUATE = "nvflow.stages.sdg.document_grounded.enrich_rollouts_evaluate"


def _target_has_nonempty_file(
    cluster: str,
    path: str,
    filesystem: TargetFilesystem | None = None,
) -> bool:
    """Probe a worker path through the selected target's filesystem transport."""
    if filesystem is None:
        from nvflow.integrations.nemo_skills.filesystem import target_filesystem

        filesystem = target_filesystem(cluster)
    return filesystem.is_nonempty_file(path)


def build_responses_render_cmd(
    *,
    input_file: str,
    output_file: str,
    prompt_template: str,
    extra_record_fields: dict[str, Any] | None = None,
    extra_record_field_mappers: dict[str, str] | None = None,
) -> str:
    """Build the deterministic SDG-to-Responses-API render command."""
    parts = [
        "python -m nvflow.stages.sdg.document_grounded.responses_api render_and_convert",
        f"--input_file {shlex.quote(input_file)}",
        f"--output_file {shlex.quote(output_file)}",
        f"--prompt_template {shlex.quote(prompt_template)}",
    ]
    if extra_record_fields:
        parts.append(f"--extra_record_fields {shlex.quote(json.dumps(extra_record_fields))}")
    if extra_record_field_mappers:
        parts.append(
            f"--extra_record_field_mappers {shlex.quote(json.dumps(extra_record_field_mappers))}"
        )
    return " ".join(parts)


def submit_gym_generation(
    *,
    cluster: str,
    rollout_expname: str,
    run_after: list[str] | None,
    input_file: str,
    output_dir: str,
    prompt_template: str,
    gym_config_paths: list[str],
    gym_agent_name: str,
    container: str,
    vllm_container: str | None,
    installation_command: str | None,
    model_path: str,
    num_gpus: int,
    server_nodes: int = 1,
    num_chunks: int = 1,
    num_repeats: int = 1,
    inference_params: dict[str, Any] | None = None,
    vllm_extra: dict[str, Any] | None = None,
    extra_record_fields: dict[str, Any] | None = None,
    extra_record_field_mappers: dict[str, str] | None = None,
    enrich_module: str = ENRICH_MODULE,
    force_render: bool = False,
) -> None:
    """Declare a captured Gym-generation operation.

    Runnable stages invoke this symbol only inside exact execution capture,
    where it is replaced by the backend-neutral planner.
    """
    raise RuntimeError(
        "submit_gym_generation requires exact execution capture; use the workflow planner/runner"
    )


def build_rollout_config(
    *,
    input_file: str,
    output_dir: str,
    gym_config_paths: list[str],
    gym_agent_name: str,
    container: str,
    vllm_container: str | None,
    installation_command: str | None,
    model_path: str,
    num_gpus: int,
    server_nodes: int = 1,
    num_chunks: int = 1,
    num_repeats: int = 1,
    inference_params: dict[str, Any] | None = None,
    vllm_extra: dict[str, Any] | None = None,
    env_key: str = "sdg_format_verification",
) -> dict[str, Any]:
    """Translate SDG generation params into a config for ``rollout()``.

    - ``input_file`` MUST already be in Responses API format (per-row
      ``responses_create_params.input`` + per-row ``verifier``), produced by
      ``responses_api.render_and_convert``.  The per-row prompt and verifier
      live in the data, NOT here.
    - ``inference_params`` become global ``responses_create_params`` overrides.
    - No judge endpoint is declared; format verification uses the policy resource.
    - Gym owns repeat identity; NVFlow retains only outer data chunking.
    """
    # Some knobs are rollout-level (consumed by ``rollout()``), not vLLM serve
    # flags, but they arrive mixed into ``vllm_extra`` from a stage's ``args``.
    #   - num_samples_in_parallel: concurrent requests per server (default 4).
    #   - dependent_jobs: chained resume jobs per chunk so a rollout that doesn't
    #     finish inside the Slurm walltime continues in the next chained job
    #     (default 0).  Essential for big/slow models where one 4h job can't
    #     finish (long-tail generations) -- the chained job resumes the few
    #     remaining samples and exits early once done.
    rollout_level_keys = ("num_samples_in_parallel", "dependent_jobs")
    extra = dict(vllm_extra or {})
    rollout_level = {k: extra.pop(k) for k in rollout_level_keys if k in extra}

    rollout_cfg: dict[str, Any] = {
        "input_data": input_file,
        "models": {
            "policy": {
                "deployment": "nvflow",
                "model": model_path,
                "num_gpus": num_gpus,
                "server_nodes": server_nodes,
                "gym_model": {"vllm_serve_kwargs": extra},
            },
        },
        "responses_create_params": inference_params or {},
        "num_chunks": num_chunks,
        "num_repeats": num_repeats,
    }
    rollout_cfg.update(rollout_level)

    return {
        "output_dir": output_dir,
        "container": container,
        "vllm_container": vllm_container,
        "installation_command": installation_command,
        "rollout": rollout_cfg,
        "environments": {
            env_key: {
                "agent_name": gym_agent_name,
                "config_paths": list(gym_config_paths),
            }
        },
    }


def plan_gym_generation(
    context: PlanningContext,
    *,
    task_index: int,
    **kwargs: Any,
) -> StagePlan:
    """Translate one captured gym-generation submission into planned tasks."""
    from nvflow.integrations.nemo_gym.native_rollout import rollout
    from nvflow.integrations.nemo_skills import stage_plan_from_jobs

    tasks: list[TaskPlan] = []
    output_dir = str(kwargs["output_dir"])
    rollout_expname = str(kwargs["rollout_expname"])
    rapi_file = f"{output_dir}/.responses_api_input.jsonl"
    render_expname = f"{rollout_expname}-render"
    render_exists = False
    if context.live:
        render_exists = _target_has_nonempty_file(
            context.target.cluster,
            rapi_file,
            context.filesystem,
        )
    force_render = bool(kwargs.get("force_render", False))
    rollout_run_after = kwargs.get("run_after")
    if not render_exists or force_render:
        render_command = build_responses_render_cmd(
            input_file=str(kwargs["input_file"]),
            output_file=rapi_file,
            prompt_template=str(kwargs["prompt_template"]),
            extra_record_fields=kwargs.get("extra_record_fields"),
            extra_record_field_mappers=kwargs.get("extra_record_field_mappers"),
        )
        tasks.append(
            TaskPlan(
                task_id=f"{context.stage}.{task_index}",
                stage=context.stage,
                name=render_expname,
                operation="nemo_skills.run_cmd",
                commands=(
                    CommandPlan(
                        command=render_command,
                        container="nemo-skills",
                        runtime_role="nemo-skills",
                    ),
                ),
                resources=(ResourceRequest(gpus_per_node=0),),
                dependencies=tuple(
                    DependencyPlan(str(name), external=True)
                    for name in kwargs.get("run_after") or ()
                ),
                outputs=(
                    ArtifactDeclaration(
                        name="responses_api_input",
                        path=rapi_file,
                    ),
                ),
                parameters={
                    "arguments": render_command,
                    "cluster": kwargs["cluster"],
                    "expname": render_expname,
                    "log_dir": f"{output_dir}/render-logs",
                    "run_after": kwargs.get("run_after"),
                },
            )
        )
        rollout_run_after = [render_expname]

    rollout_config = build_rollout_config(
        input_file=rapi_file,
        output_dir=output_dir,
        gym_config_paths=list(kwargs["gym_config_paths"]),
        gym_agent_name=str(kwargs["gym_agent_name"]),
        container=str(kwargs["container"]),
        vllm_container=kwargs.get("vllm_container"),
        installation_command=kwargs.get("installation_command"),
        model_path=str(kwargs["model_path"]),
        num_gpus=int(kwargs["num_gpus"]),
        server_nodes=int(kwargs.get("server_nodes", 1)),
        num_chunks=int(kwargs.get("num_chunks", 1)),
        num_repeats=int(kwargs.get("num_repeats", 1)),
        inference_params=kwargs.get("inference_params"),
        vllm_extra=kwargs.get("vllm_extra"),
    )
    planned = rollout(
        config=rollout_config,
        cluster=str(kwargs["cluster"]),
        expname=rollout_expname,
        run_after=rollout_run_after,
        enrich_module=kwargs.get("enrich_module", ENRICH_MODULE),
        inspect_live_state=context.live,
        filesystem=context.filesystem,
    )
    contribution = stage_plan_from_jobs(
        context.stage,
        planned,
        output_dir=output_dir,
        task_prefix=f"{context.stage}.gym{task_index + len(tasks)}",
        experiment_name=rollout_expname,
    )
    tasks.extend(contribution.tasks)
    if not tasks:
        return StagePlan.noop(
            context.stage,
            (
                "filesystem inspection selected no gym-generation work"
                if context.live
                else "gym-generation configuration selected no work"
            ),
        )
    return StagePlan(
        stage=context.stage,
        tasks=tuple(tasks),
    )
