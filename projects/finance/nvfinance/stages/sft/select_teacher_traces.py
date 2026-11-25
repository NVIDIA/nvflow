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

"""Keep each question's most efficient passing teacher trace, and report on the teacher run."""

from __future__ import annotations

import json
import shlex
from typing import TYPE_CHECKING, Any

from nvflow.execution import ArtifactDeclaration, PlannedExecutionStage, declare_task_outputs
from nvflow.execution.commands import build_python_cmd

if TYPE_CHECKING:
    from nvflow.execution import PlanningContext, StagePlan

WORKER = "nvfinance.utils.sft.select_teacher_traces"
OUTPUTS = {"traces": "traces.jsonl", "questions": "questions.jsonl", "report": "report.json"}
_OPTIONS = {"dataset", "rollouts", "output_dir", "teacher", "shared_cache_dir"}
_REQUIRED = {
    "dataset": "the dataset's name",
    "rollouts": "the folder of the dataset's analyzed rollouts",
    "teacher": "the teacher model's name",
}


class SelectTeacherTracesStage(PlannedExecutionStage):
    """Write one dataset's ``traces.jsonl``, ``questions.jsonl``, and ``report.json``.

    See :mod:`nvfinance.utils.sft.select_teacher_traces`. ``rollouts`` is the
    folder of the dataset's analyzed Gym collection, and every trace and the
    report carry the ``dataset`` and ``teacher`` names. With
    ``shared_cache_dir``, the task first gives the cache folders and files this
    user owns Gym's shared modes, 2775 and 664, so anyone can read them and the
    cache's group can add to them: Gym wrote cache files readable only by their
    owner before it used those modes, and other users' lookups fail on those.
    """

    def plan(self, context: PlanningContext) -> StagePlan:
        from nvflow.execution import plan_run_cmd

        config = dict(context.config)
        output_dir = str(config["output_dir"]).rstrip("/")
        command = build_python_cmd(
            WORKER,
            datasets=json.dumps({str(config["dataset"]): str(config["rollouts"])}),
            output_dir=output_dir,
            teacher=str(config["teacher"]),
        )
        if cache := config.get("shared_cache_dir"):
            command = (
                f'find {shlex.quote(str(cache))} -user "$(id -u)" '
                "'(' -type d -exec chmod 2775 {} + -o -type f -exec chmod 664 {} + ')' "
                f"|| true\n{command}"
            )
        with declare_task_outputs(
            *(ArtifactDeclaration(name, f"{output_dir}/{file}") for name, file in OUTPUTS.items())
        ):
            return plan_run_cmd(
                context,
                command,
                cluster=context.target.cluster,
                container=str(config.get("container") or "nemo-skills"),
                log_dir=f"{output_dir}/logs",
                output_dir=output_dir,
                expname=context.expname,
                run_after=list(context.run_after) or None,
                sbatch_kwargs=None,
            )

    def validate_config(self, config: dict[str, Any]) -> None:
        from nvflow.core.workflow_structure import reject_unknown_options

        reject_unknown_options(config, _OPTIONS, "select_teacher_traces stage")
        for key, meaning in _REQUIRED.items():
            if not isinstance(config.get(key), str) or not config[key]:
                raise ValueError(f"'{key}' must be {meaning}")
        if not config.get("output_dir"):
            raise ValueError("'output_dir' is required in the select_teacher_traces stage")
        cache = config.get("shared_cache_dir")
        if cache is not None and not (isinstance(cache, str) and cache):
            raise ValueError("'shared_cache_dir' must be a folder path")


__all__ = ["SelectTeacherTracesStage"]
