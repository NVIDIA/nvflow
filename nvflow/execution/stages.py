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
"""Exact execution stage helpers owned by the execution layer."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from nvflow.core.base_stage import BaseStage
from nvflow.execution.contracts import StageExecution, StagePlan
from nvflow.execution.planner import PlanningContext


class CapturedExecutionStage(BaseStage):
    """Internal adapter that captures first-party ``build_tasks`` command builders."""

    capture_strategy: ClassVar[Literal["run_cmd", "gym_generation"]]
    capture_empty_reason: ClassVar[str | None] = None

    def build_tasks(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        raise NotImplementedError

    def execution(self, context: PlanningContext) -> StageExecution:
        from nvflow.execution.capture import capture_first_party_execution

        return capture_first_party_execution(self, context)


class PlannedExecutionStage(BaseStage):
    """Stage whose explicit plan builder is already an exact contract."""

    def plan(self, context: PlanningContext) -> StagePlan | None:
        return None

    def execution(self, context: PlanningContext) -> StageExecution:
        contribution = self.plan(context)
        if contribution is None:
            raise ValueError(
                f"Planned stage {type(self).__module__}.{type(self).__name__} "
                "returned no exact contract"
            )
        return StageExecution.from_plan(contribution)


__all__ = ["CapturedExecutionStage", "PlannedExecutionStage"]
