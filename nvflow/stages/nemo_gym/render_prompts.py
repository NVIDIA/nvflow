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
"""Render prompt templates and their tools into NeMo Gym rows (``nemo_gym.render``)."""

from __future__ import annotations

import json
import re
import shlex
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from nvflow.execution import ArtifactDeclaration, PlannedExecutionStage, declare_task_outputs

if TYPE_CHECKING:
    from nvflow.execution import PlanningContext, StagePlan

WORKER = "python3 -m nvflow.integrations.nemo_gym.workers.render_prompts"
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_RENDER_KEYS = {"input_file", "prompt", "row_fields"}


def _renders(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    renders = config.get("renders")
    if not isinstance(renders, Mapping) or not renders:
        raise ValueError("'renders' must map each output name to an input_file and a prompt")
    result = {}
    for name, spec in renders.items():
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ValueError(f"invalid render name {name!r}")
        if not isinstance(spec, Mapping) or not spec.get("input_file") or not spec.get("prompt"):
            raise ValueError(f"render {name!r} needs 'input_file' and 'prompt'")
        if unknown := sorted(set(spec) - _RENDER_KEYS):
            raise ValueError(f"render {name!r} has unknown option(s): {', '.join(unknown)}")
        row_fields = spec.get("row_fields") or {}
        if not isinstance(row_fields, Mapping):
            raise ValueError(f"render {name!r} 'row_fields' must be a mapping")
        result[name] = {
            "input_file": str(spec["input_file"]),
            "prompt": str(spec["prompt"]),
            "row_fields": dict(row_fields),
        }
    return result


def _command(name: str, spec: Mapping[str, Any], output_dir: str) -> str:
    command = (
        f"{WORKER} --input {shlex.quote(spec['input_file'])} "
        f"--prompt {shlex.quote(spec['prompt'])} "
        f"--output {shlex.quote(f'{output_dir}/{name}.jsonl')}"
    )
    if spec["row_fields"]:
        command += f" --row_fields {shlex.quote(json.dumps(spec['row_fields'], sort_keys=True))}"
    return command


class RenderPromptsStage(PlannedExecutionStage):
    """Write ``NAME.jsonl`` and ``NAME.done`` under ``output_dir`` for each render.

    One CPU task in the Gym container runs every render; each skips itself when
    its ``NAME.done`` matches the current input and template.
    """

    def plan(self, context: PlanningContext) -> StagePlan:
        from nvflow.execution import plan_run_cmd

        config = context.config
        output_dir = str(config["output_dir"]).rstrip("/")
        renders = _renders(config)
        commands = [_command(name, spec, output_dir) for name, spec in renders.items()]
        outputs = [ArtifactDeclaration(name, f"{output_dir}/{name}.jsonl") for name in renders]
        outputs += [
            ArtifactDeclaration(f"{name}_completion", f"{output_dir}/{name}.done")
            for name in renders
        ]
        with declare_task_outputs(*outputs):
            return plan_run_cmd(
                context,
                "set -euo pipefail\n" + "\n".join(commands),
                cluster=context.target.cluster,
                log_dir=f"{output_dir}/logs",
                output_dir=output_dir,
                expname=context.expname,
                run_after=list(context.run_after) or None,
                sbatch_kwargs=None,
                container=str(config.get("container") or "nemo-gym"),
                installation_command=config.get("installation_command"),
            )

    def validate_config(self, config: dict[str, Any]) -> None:
        from nvflow.integrations.nemo_gym.contracts import require_contract_version

        require_contract_version(config.get("contract_version"))
        if not config.get("output_dir"):
            raise ValueError("'output_dir' is required in nemo_gym.render config")
        _renders(config)


__all__ = ["RenderPromptsStage"]
