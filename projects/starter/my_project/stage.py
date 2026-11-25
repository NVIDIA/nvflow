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
"""Minimal inspectable generation stage used by the project starter."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from nvflow.core import BaseStage

if TYPE_CHECKING:
    from nvflow.execution import PlanningContext, StageExecution


class GenerateAnswerStage(BaseStage):
    """Generate answers with NeMo-Skills."""

    def validate_config(self, config: dict[str, Any]) -> None:
        for field in ("input_file", "output_dir"):
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
        input_file = Path(config["input_file"])
        if not input_file.exists() and not input_file.is_absolute():
            raise ValueError(f"Input file does not exist: {input_file}")

    def execution(self, context: PlanningContext) -> StageExecution:
        """Build one exact backend-neutral generation graph."""
        from nvflow.execution import StageExecution, plan_generate

        config = dict(context.config)
        arguments = f"++prompt_config={config['prompt_config']} {config.get('inline_args', '')}"
        return StageExecution.from_plan(
            plan_generate(
                context,
                arguments,
                cluster=context.target.cluster,
                input_file=config["input_file"],
                output_dir=config["output_dir"],
                expname=context.expname,
                run_after=list(context.run_after) or None,
                **config.get("stage_kwargs", {}),
            )
        )
