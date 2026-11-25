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
"""Drop records that overlap evaluation sets (``data.decontaminate``)."""

from __future__ import annotations

import json
import shlex
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from nvflow.execution import ArtifactDeclaration, PlannedExecutionStage
from nvflow.stages.data.decontaminate_worker import (
    COMPLETION_NAME,
    OUTPUT_NAMES,
    DecontaminationSettings,
)
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
    "references",
    "exact",
    "ngram",
    "embedding",
    "cache_dir",
)
_OPTIONS = {
    *SETTINGS_KEYS,
    "input_file",
    "input_files",
    "output_dir",
    "num_chunks",
    "dependent_jobs",
}


def _embedding_placement(config: Mapping[str, Any], container: str) -> tuple[str, int] | None:
    """Container and GPU count for the embed step; vLLM defaults to one GPU."""
    embedding = config.get("embedding")
    if not isinstance(embedding, Mapping):
        return None
    on_vllm = embedding.get("backend") == "vllm"
    num_gpus = embedding.get("num_gpus", 1 if on_vllm else 0)
    if isinstance(num_gpus, bool) or not isinstance(num_gpus, int) or num_gpus < 0:
        raise ValueError("'embedding.num_gpus' must be a non-negative integer")
    return str(embedding.get("container") or ("vllm" if on_vllm else container)), num_gpus


def _reference_paths(config: Mapping[str, Any]) -> list[str]:
    return [
        str(value if isinstance(value, str) else value["path"])
        for value in config["references"].values()
    ]


class DecontaminateStage(PlannedExecutionStage):
    """Write ``kept.jsonl``, ``dropped.jsonl``, ``candidates.jsonl``, and ``report.json``.

    Each chunk runs the exact and n-gram layers on CPU, then the embedding layer
    in the embedding backend's container. A final step applies every chunk's
    hits and publishes ``decontaminate.done``. With one chunk and a CPU backend,
    the whole stage is one task; with vLLM it is a CPU task and a GPU task.
    """

    def plan(self, context: PlanningContext) -> StagePlan:
        config = context.config
        inputs = input_files(config)
        output_dir = str(config["output_dir"]).rstrip("/")
        require_live_inputs(context, [*inputs, *_reference_paths(config)])
        num_chunks = int(config.get("num_chunks", 1))
        settings = json.dumps(
            {key: config[key] for key in SETTINGS_KEYS if key in config}, sort_keys=True
        )
        container = str(config.get("container") or "nemo-skills")
        embedding = _embedding_placement(config, container)

        def command(step: str, *extra: str) -> str:
            return " ".join(
                [
                    "python3 -m nvflow.stages.data.decontaminate_worker",
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

        chunks = []
        for chunk in range(num_chunks):
            chunk_dir = f"{output_dir}/chunks/{chunk:05d}"
            steps = [
                Step(
                    "lexical",
                    command("lexical", "--chunk", str(chunk)),
                    f"{chunk_dir}/lexical.done",
                    container=container,
                )
            ]
            if embedding is not None:
                steps.append(
                    Step(
                        "embed",
                        command("embed", "--chunk", str(chunk)),
                        f"{chunk_dir}/embedding.done",
                        container=embedding[0],
                        num_gpus=embedding[1],
                    )
                )
            chunks.append(steps)
        final = Step(
            "merge",
            command("merge"),
            f"{output_dir}/{COMPLETION_NAME}",
            container=container,
            portable=True,
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

        reject_unknown_options(config, _OPTIONS, "data.decontaminate")
        if not config.get("output_dir"):
            raise ValueError("'output_dir' is required in data.decontaminate config")
        input_files(config)
        DecontaminationSettings.from_config(config)
        _embedding_placement(config, "nemo-skills")
        validate_chunking(config)
