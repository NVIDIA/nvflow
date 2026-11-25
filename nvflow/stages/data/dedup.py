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
"""Remove exact and near-duplicate JSONL records (``data.dedup``)."""

from __future__ import annotations

import json
import shlex
from typing import TYPE_CHECKING, Any

from nvflow.execution import ArtifactDeclaration, PlannedExecutionStage
from nvflow.stages.data.dedup_worker import COMPLETION_NAME, OUTPUT_NAMES, DedupSettings
from nvflow.stages.data.step_plan import (
    Step,
    input_files,
    plan_steps,
    require_live_inputs,
    validate_chunking,
)

if TYPE_CHECKING:
    from nvflow.execution import PlanningContext, StagePlan

SETTINGS_KEYS = (
    "text_fields",
    "text_after",
    "id_field",
    "exact",
    "near",
    "guard_fields",
    "keep",
    "conflict_field",
    "on_conflict",
)
_OPTIONS = {
    *SETTINGS_KEYS,
    "input_file",
    "input_files",
    "output_dir",
    "num_chunks",
    "dependent_jobs",
}


class DedupStage(PlannedExecutionStage):
    """Write ``kept.jsonl``, ``dropped.jsonl``, ``groups.jsonl``, and ``report.json``.

    Each of ``num_chunks`` CPU tasks computes its rows' keys and MinHash
    signatures; a final task groups duplicates across all chunks and publishes
    ``dedup.done``. One chunk runs as a single task.
    """

    def plan(self, context: PlanningContext) -> StagePlan:
        config = context.config
        inputs = input_files(config)
        output_dir = str(config["output_dir"]).rstrip("/")
        require_live_inputs(context, inputs)
        num_chunks = int(config.get("num_chunks", 1))
        settings = json.dumps(
            {key: config[key] for key in SETTINGS_KEYS if key in config}, sort_keys=True
        )
        container = str(config.get("container") or "nemo-skills")

        def command(step: str, *extra: str) -> str:
            return " ".join(
                [
                    "python3 -m nvflow.stages.data.dedup_worker",
                    step,
                    *(shlex.quote(path) for path in inputs),
                    "--output_dir",
                    shlex.quote(output_dir),
                    "--settings",
                    shlex.quote(settings),
                    "--num_chunks",
                    str(num_chunks),
                    *extra,
                ]
            )

        chunks = [
            [
                Step(
                    "signatures",
                    command("signatures", "--chunk", str(chunk)),
                    f"{output_dir}/chunks/{chunk:05d}/signatures.done",
                    container=container,
                )
            ]
            for chunk in range(num_chunks)
        ]
        final = Step(
            "group",
            command("group"),
            f"{output_dir}/{COMPLETION_NAME}",
            container=container,
            outputs=tuple(
                ArtifactDeclaration(name, f"{output_dir}/{filename}")
                for name, filename in OUTPUT_NAMES.items()
            ),
        )
        return plan_steps(
            context,
            chunks=chunks,
            final=final,
            log_dir=f"{output_dir}/logs",
            dependent_jobs=int(config.get("dependent_jobs", 0)),
        )

    def validate_config(self, config: dict[str, Any]) -> None:
        from nvflow.core.workflow_structure import reject_unknown_options

        reject_unknown_options(config, _OPTIONS, "data.dedup")
        if not config.get("output_dir"):
            raise ValueError("'output_dir' is required in data.dedup config")
        input_files(config)
        DedupSettings.from_config(config)
        validate_chunking(config)
