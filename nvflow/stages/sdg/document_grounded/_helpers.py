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

import shlex
from typing import Any

from ._schemas import ALWAYS_DROP, STAGE_KEEP


def parse_stage_kwargs(stage_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Extract normalized fields from a legacy ``args`` / ``ctx_args`` block.

    Returns a dict with ``model_path``, ``num_gpus``, ``server_nodes``,
    ``num_chunks``, ``num_repeats``, ``prompt_template``,
    ``generation_key``, ``inference_params`` and ``vllm_extra`` (any remaining
    ``args`` keys that are vLLM serve flags).  Used by the generate_* shims to
    feed the NeMo-Gym generation integration.
    """
    args = stage_kwargs.get("args", {}).copy()
    ctx_args = stage_kwargs.get("ctx_args", "")

    model_path = args.pop("model", "")
    num_gpus = args.pop("server_gpus", args.pop("num_gpus", 8))
    server_nodes = args.pop("server_nodes", 1)
    num_chunks = args.pop("num_chunks", 1)
    num_repeats = args.pop("num_repeats", 1)
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
        "num_repeats": num_repeats,
        "prompt_template": prompt_template,
        "generation_key": generation_key,
        "inference_params": inference_params,
        "vllm_extra": vllm_extra,
    }


def build_trim_cmd(
    *,
    stage_name: str,
    paths: list[str],
    domain_keep_fields: list[str] | None,
    extra_keep_fields: list[str] | None = None,
) -> str:
    """Build the shell command that trims this stage's output JSONL files.

    The returned string invokes ``nvflow.stages.sdg.document_grounded._trim_cli``
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
        "python -m nvflow.stages.sdg.document_grounded._trim_cli "
        f"--paths {paths_arg} --keep_fields {keep_args}"
    )
