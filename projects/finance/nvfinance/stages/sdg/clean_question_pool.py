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
"""Normalize RL and SFT question pools into one file before dedup and decontamination."""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING, Any

from nvflow.execution import ArtifactDeclaration, PlannedExecutionStage, declare_task_outputs
from nvflow.execution.commands import build_python_cmd

if TYPE_CHECKING:
    from nvflow.execution import PlanningContext, StagePlan

WORKER = "nvfinance.utils.sdg.clean_question_pool"
OUTPUTS = {"clean": "clean.jsonl", "rejected": "rejected.jsonl", "report": "report.json"}


def _inputs(config: dict[str, Any]) -> list[dict[str, str]]:
    inputs = config.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("'inputs' must list {path, source} entries")
    for entry in inputs:
        if not isinstance(entry, dict) or not entry.get("path") or not entry.get("source"):
            raise ValueError("each 'inputs' entry needs a 'path' and a 'source' tag")
    return [{"path": str(entry["path"]), "source": str(entry["source"])} for entry in inputs]


class CleanQuestionPoolStage(PlannedExecutionStage):
    """Write ``clean.jsonl``, ``rejected.jsonl``, and ``report.json``.

    See :mod:`nvfinance.utils.sdg.clean_question_pool` for the fields each record
    gains. Every other option overrides one of that module's ``DEFAULTS``.
    """

    def plan(self, context: PlanningContext) -> StagePlan:
        from nvfinance.utils.sdg.clean_question_pool import DEFAULTS
        from nvflow.execution import plan_run_cmd

        config = dict(context.config)
        inputs = _inputs(config)
        output_dir = str(config["output_dir"]).rstrip("/")
        run_after = list(context.run_after) or None
        if context.live and not run_after and context.filesystem is not None:
            paths = [entry["path"] for entry in inputs]
            present = context.filesystem.batch_exists(paths)
            missing = [path for path in paths if not present.get(path)]
            if missing:
                raise ValueError("live clean plan: input files are missing: " + ", ".join(missing))
        settings = {key: config[key] for key in DEFAULTS if key in config}
        command = build_python_cmd(
            WORKER,
            inputs=json.dumps(inputs),
            output_dir=output_dir,
            settings=json.dumps(settings, sort_keys=True),
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
                run_after=run_after,
                sbatch_kwargs=None,
            )

    def validate_config(self, config: dict[str, Any]) -> None:
        from nvfinance.utils.sdg.clean_question_pool import DEFAULTS
        from nvflow.core.workflow_structure import reject_unknown_options

        reject_unknown_options(config, {*DEFAULTS, "inputs", "output_dir"}, "clean stage")
        fallback = config.get("fallback_current_date", DEFAULTS["fallback_current_date"])
        try:
            datetime.strptime(str(fallback), "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError("'fallback_current_date' must be a YYYY-MM-DD date") from exc
        parquet = config.get("sec_metadata_parquet")
        if parquet is not None and not (isinstance(parquet, str) and parquet):
            raise ValueError("'sec_metadata_parquet' must be a file path")
        if not config.get("output_dir"):
            raise ValueError("'output_dir' is required in the clean stage config")
        _inputs(config)
        for key in ("question_fields", "answer_fields", "id_fields", "drop_fields"):
            value = config.get(key, DEFAULTS[key])
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise ValueError(f"'{key}' must be a list of field names")
        if int(config.get("jitter_min_days", 1)) > int(config.get("jitter_max_days", 60)):
            raise ValueError("'jitter_min_days' must not exceed 'jitter_max_days'")
        for key in ("max_rows", "sample_rows"):
            value = config.get(key)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"'{key}' must be a positive integer")
        if config.get("max_rows") is not None and config.get("sample_rows") is not None:
            raise ValueError("set 'max_rows' or 'sample_rows', not both")
        seed = config.get("sample_seed", 0)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("'sample_seed' must be an integer")
