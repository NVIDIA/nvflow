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
"""Shared helpers for generic DG-SDG stages."""

import json
import shlex
from typing import Any

from ._schemas import ALWAYS_DROP, STAGE_KEEP


def clean_stale_experiments(cluster: str, expnames: list[str]) -> None:
    """Remove ``<job_dir>/experiments/<expname>/`` for each name in *expnames*.

    ``rollout()`` (reused unmodified from RL) does not clean stale nemo-run
    experiment dirs, but SDG needs it: nemo-run caches the generated bash
    scripts per experiment, so a stale dir makes (a) code edits silently
    no-op (cached scripts re-used; SKILL.md Gotcha #8) and (b) ``run_after``
    resolve to a stale FINISHED experiment, skipping the Slurm dependency
    (Gotcha #1).  Replicated here on the SDG side so RL code stays untouched.
    Idempotent; safe because nemo-run regenerates scripts on next launch and
    we run before any new job is submitted.
    """
    import shutil
    from pathlib import Path

    import nemo_skills.pipeline.utils as pipeline_utils

    cluster_config = pipeline_utils.get_cluster_config(cluster)
    job_dir = cluster_config.get("job_dir")
    if not job_dir:
        return
    root = Path(job_dir) / "experiments"
    if not root.is_dir():
        return
    for expname in expnames:
        target = root / expname
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)


ENRICH_MODULE = "nvflow.lib.sdg.document_grounded.enrich_rollouts"
ENRICH_MODULE_EVALUATE = "nvflow.lib.sdg.document_grounded.enrich_rollouts_evaluate"
ANALYZE_MODULE = "nvflow.lib.sdg.document_grounded.analyze_rollouts"


def submit_gym_generation(
    *,
    cluster: str,
    rollout_expname: str,
    run_after: list[str] | None,
    input_file: str,
    output_dir: str,
    prompt_template: str,
    gym_path: str,
    gym_config_paths: list[str],
    gym_agent_name: str,
    container: str,
    installation_command: str | None,
    model_path: str,
    num_gpus: int,
    server_nodes: int = 1,
    num_chunks: int = 1,
    num_random_seeds: int = 1,
    inference_params: dict[str, Any] | None = None,
    vllm_extra: dict[str, Any] | None = None,
    extra_record_fields: dict[str, Any] | None = None,
    extra_record_field_mappers: dict[str, str] | None = None,
    enrich_module: str = ENRICH_MODULE,
    rerun_done: bool = False,
    gym_uv_venv_dir: str = "",
) -> None:
    """Render SDG JSONL to Responses API, then collect rollouts via ``rollout()``.

    Up to two jobs are submitted:

    1. ``{rollout_expname}-render`` (CPU): ``responses_api render_and_convert``
       turns the flat SDG input into Responses-API rows (per-row prompt under
       ``responses_create_params.input`` + per-row ``verifier`` from
       ``extra_record_fields``).  ``inference_params`` are NOT rendered in --
       they are applied by ``rollout()`` as global ``responses_create_params``
       overrides, keeping ``responses_create_params.input`` stable so the
       content-hash join in ``enrich`` matches input<->output rows.
       SKIPPED when the render output already exists (unless ``rerun_done``):
       re-rendering on resume is wasteful and races a resumed merge's enrich
       (see the guard below).  When skipped, ``rollout()`` inherits the render's
       own ``run_after`` so downstream ordering is preserved.
    2. ``rollout()`` (GPU): chunk + ng_collect_rollouts + per-seed merge, then
       the merge job runs ``enrich`` (restore SDG fields + extract generation)
       and ``analyze`` (sync ``rollout/output-rs*.jsonl`` up to ``output_dir/``).

    The caller is responsible for any per-stage trim / postprocess, submitted
    as a separate ``run_cmd`` under the *stage* expname with
    ``run_after=[rollout_expname]`` (so downstream ``run_after=[stage_expname]``
    waits for trim -> rollout).
    """
    from nemo_skills.pipeline.cli import run_cmd, wrap_arguments

    from nvflow.core import console
    from nvflow.lib.rl.helpers import resolve_host_path
    from nvflow.lib.rl.rollout import rollout

    rapi_file = f"{output_dir}/.responses_api_input.jsonl"
    render_expname = f"{rollout_expname}-render"

    render_cmd_parts = [
        "python -m nvflow.lib.sdg.document_grounded.responses_api render_and_convert",
        f"--input_file {shlex.quote(input_file)}",
        f"--output_file {shlex.quote(rapi_file)}",
        f"--prompt_template {shlex.quote(prompt_template)}",
    ]
    if extra_record_fields:
        payload = json.dumps(extra_record_fields)
        render_cmd_parts.append(f"--extra_record_fields {shlex.quote(payload)}")
    if extra_record_field_mappers:
        payload = json.dumps(extra_record_field_mappers)
        render_cmd_parts.append(f"--extra_record_field_mappers {shlex.quote(payload)}")
    render_cmd = " ".join(render_cmd_parts)

    # Skip re-rendering when the Responses-API input already exists.  The render
    # is a deterministic 1:1 transform of *input_file*, so recomputing it on a
    # resume is pure waste (100s of GB rewrite).  It is also unsafe: the per-seed
    # merge's enrich() reads THIS exact file, and when a seed's chunks are all
    # `.done` the merge loses its (transitive, via chunk jobs) dependency on the
    # render -- it then runs immediately and can race a concurrent render rewrite,
    # reading a half-written file (enrich alignment-check failure).  Skipping the
    # render keeps the input stable for any resumed merge.  Mirrors the
    # skip-if-exists guards on the q-prep / q-verify-prep steps; `rerun_done`
    # forces a fresh render, kept in lock-step with the rollout rerun.
    # NOTE: execute() runs on the orchestrator node, so resolve the container
    # path to its host path before checking existence.
    rapi_host = resolve_host_path(rapi_file)
    rapi_exists = rapi_host.exists() and rapi_host.stat().st_size > 0
    if rapi_exists and not rerun_done:
        console.success("Render skipped (reusing existing Responses-API input)")
        console.detail("Responses-API input", rapi_file)
        rollout_run_after = run_after
    else:
        run_cmd(
            ctx=wrap_arguments(render_cmd),
            cluster=cluster,
            expname=render_expname,
            log_dir=f"{output_dir}/render-logs",
            run_after=run_after,
        )
        rollout_run_after = [render_expname]

    cfg = build_rollout_config(
        input_file=rapi_file,
        output_dir=output_dir,
        gym_path=gym_path,
        gym_config_paths=gym_config_paths,
        gym_agent_name=gym_agent_name,
        container=container,
        installation_command=installation_command,
        model_path=model_path,
        num_gpus=num_gpus,
        server_nodes=server_nodes,
        num_chunks=num_chunks,
        num_random_seeds=num_random_seeds,
        inference_params=inference_params,
        vllm_extra=vllm_extra,
        rerun_done=rerun_done,
        gym_uv_venv_dir=gym_uv_venv_dir,
    )
    rollout(
        config=cfg,
        cluster=cluster,
        expname=rollout_expname,
        run_after=rollout_run_after,
        enrich_module=enrich_module,
        analyze_module=ANALYZE_MODULE,
    )


def parse_stage_kwargs(stage_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Extract normalized fields from a legacy ``args`` / ``ctx_args`` block.

    Returns a dict with ``model_path``, ``num_gpus``, ``server_nodes``,
    ``num_chunks``, ``num_random_seeds``, ``prompt_template``,
    ``generation_key``, ``inference_params`` and ``vllm_extra`` (any remaining
    ``args`` keys that are vLLM serve flags).  Used by the generate_* shims to
    feed both the render step (prompt_template) and :func:`build_rollout_config`.
    """
    args = stage_kwargs.get("args", {}).copy()
    ctx_args = stage_kwargs.get("ctx_args", "")

    model_path = args.pop("model", "")
    num_gpus = args.pop("server_gpus", args.pop("num_gpus", 8))
    server_nodes = args.pop("server_nodes", 1)
    num_chunks = args.pop("num_chunks", 1)
    num_random_seeds = args.pop("num_random_seeds", 1)
    args.pop("server_type", None)
    args.pop("skip_filled", None)

    prompt_template = ""
    generation_key = "generation"
    inference_params: dict[str, Any] = {}
    for part in ctx_args.split():
        if part.startswith("++prompt_config="):
            prompt_template = part.split("=", 1)[1]
        elif part.startswith("++inference."):
            key = part.split("=")[0].replace("++inference.", "")
            val = part.split("=", 1)[1]
            try:
                inference_params[key] = float(val)
            except ValueError:
                inference_params[key] = val
        elif part.startswith("++generation_key="):
            generation_key = part.split("=", 1)[1]

    vllm_extra = {k: v for k, v in args.items() if k != "generation_key"}

    return {
        "model_path": model_path,
        "num_gpus": num_gpus,
        "server_nodes": server_nodes,
        "num_chunks": num_chunks,
        "num_random_seeds": num_random_seeds,
        "prompt_template": prompt_template,
        "generation_key": generation_key,
        "inference_params": inference_params,
        "vllm_extra": vllm_extra,
    }


def build_rollout_config(
    *,
    input_file: str,
    output_dir: str,
    gym_path: str,
    gym_config_paths: list[str],
    gym_agent_name: str,
    container: str,
    installation_command: str | None,
    model_path: str,
    num_gpus: int,
    server_nodes: int = 1,
    num_chunks: int = 1,
    num_random_seeds: int = 1,
    inference_params: dict[str, Any] | None = None,
    vllm_extra: dict[str, Any] | None = None,
    rerun_done: bool = False,
    env_key: str = "sdg_format_verification",
    gym_uv_venv_dir: str = "",
) -> dict[str, Any]:
    """Translate SDG generation params into a config for ``rollout()``.

    ``rollout()`` is reused unmodified (the adapter lives entirely on the SDG
    side).  Notes:

    - ``input_file`` MUST already be in Responses API format (per-row
      ``responses_create_params.input`` + per-row ``verifier``), produced by
      ``responses_api.render_and_convert``.  The per-row prompt and verifier
      live in the data, NOT here.
    - ``inference_params`` (temperature, top_p, max_output_tokens, ...) become
      global ``responses_create_params`` overrides applied by ng_collect.
    - No ``judge_vllm`` is set -> ``determine_judge_mode`` returns
      ``policy_as_judge`` (no judge server).
    - ``environments`` carries the SDG overlay; ``build_config_paths_str``
      prepends the vLLM model config automatically.
    """
    # Some knobs are rollout-level (consumed by ``rollout()``), not vLLM serve
    # flags, but they arrive mixed into ``vllm_extra`` from a stage's ``args`` /
    # ``policy_vllm`` block.  Intercept them here so they reach the ``rollout``
    # config instead of leaking into ``policy_vllm`` -> ``build_vllm_server_args``
    # as invalid CLI flags.
    #   - num_samples_in_parallel: concurrent requests per server (default 4).
    #   - dependent_jobs: chained resume jobs per chunk so a rollout that doesn't
    #     finish inside the Slurm walltime continues in the next chained job
    #     (default 0).  Essential for big/slow models where one 4h job can't
    #     finish (long-tail generations) -- the chained job resumes the few
    #     remaining samples and exits early once done.
    rollout_level_keys = ("num_samples_in_parallel", "dependent_jobs")
    extra = dict(vllm_extra or {})
    rollout_level = {k: extra.pop(k) for k in rollout_level_keys if k in extra}

    policy_vllm: dict[str, Any] = {
        "model_path": model_path,
        "num_gpus": num_gpus,
        "server_nodes": server_nodes,
    }
    policy_vllm.update(extra)

    rollout_cfg: dict[str, Any] = {
        "input_data": input_file,
        "policy_vllm": policy_vllm,
        "responses_create_params": inference_params or {},
        "num_chunks": num_chunks,
        "num_random_seeds": num_random_seeds,
        "rerun_done": rerun_done,
    }
    rollout_cfg.update(rollout_level)

    return {
        "output_dir": output_dir,
        "gym_path": gym_path,
        "gym_uv_venv_dir": gym_uv_venv_dir,
        "container": container,
        "installation_command": installation_command,
        "rollout": rollout_cfg,
        "environments": {
            env_key: {
                "agent_name": gym_agent_name,
                "config_paths": list(gym_config_paths),
            }
        },
    }


def build_trim_cmd(
    *,
    stage_name: str,
    paths: list[str],
    domain_keep_fields: list[str] | None,
    extra_keep_fields: list[str] | None = None,
) -> str:
    """Build the shell command that trims this stage's output JSONL files.

    The returned string invokes ``nvflow.generic_stage.sdg.document_grounded._trim_cli``
    with the keep-list ``(STAGE_KEEP[stage_name] | domain_keep_fields -
    ALWAYS_DROP) | extra_keep_fields`` and the given ``paths`` (files,
    directories, or globs -- the CLI expands them).

    ``extra_keep_fields`` is unioned *after* the ``ALWAYS_DROP`` subtraction, so
    it is the only way to retain a field that is otherwise in ``ALWAYS_DROP``
    (e.g. ``responses_create_params`` on the final ``dgsdg_post_process``
    output, where the Responses-API original form must survive).  Use sparingly.

    The command is meant to be either:
      - appended to ``postprocess_cmd`` for Gym-driven stages
        (``sdg_generate``-based: question gen/verify, answer gen, genselect,
        evaluate), so it runs inside the merge job and the producing stage's
        advertised expname does not need to change; or
      - chained via ``&&`` to the stage's main CPU command for non-Gym stages
        (aggregate, difficulty aggregate, post-process).

    Either way the trim is guaranteed to finish before any downstream stage's
    ``run_after`` clears, with zero extra Slurm overhead.
    """
    if stage_name not in STAGE_KEEP:
        raise KeyError(
            f"build_trim_cmd: stage {stage_name!r} is not in STAGE_KEEP. "
            f"Known stages: {sorted(STAGE_KEEP)}"
        )
    domain = set(domain_keep_fields or [])
    keep = ((STAGE_KEEP[stage_name] | domain) - ALWAYS_DROP) | set(extra_keep_fields or [])
    keep_args = " ".join(sorted(keep))
    paths_arg = " ".join(shlex.quote(p) for p in paths)
    return (
        "python -m nvflow.generic_stage.sdg.document_grounded._trim_cli "
        f"--paths {paths_arg} --keep_fields {keep_args}"
    )
