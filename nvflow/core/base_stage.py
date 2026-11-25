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
"""Base stage class that all workflow stages should inherit from."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nvflow.execution.contracts import StageExecution
    from nvflow.execution.planner import PlanningContext
    from nvflow.project.context import ProjectContext


class BaseStage:
    """Base class for all workflow stages.

    All runnable stages must produce an exact ``StageExecution``. Workflows
    select implementations explicitly through their ``uses`` field.

    Example:
        >>> from nvflow.core import BaseStage
        ...
        >>> class SFTStage(BaseStage):
        ...     def execution(self, context):
        ...         return build_sft_execution(context)
    """

    _project_context: ProjectContext | None = None

    def bind_project_context(self, context: ProjectContext) -> None:
        """Bind the project whose resources and source paths the stage uses."""
        self._project_context = context

    @property
    def bound_project_context(self) -> ProjectContext | None:
        """Return the selected project context when invoked by NVFlow."""
        return self._project_context

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate stage configuration (OPTIONAL - override only if needed).

        Most stages don't need custom validation. Only override this if you have
        specific validation requirements beyond checking for required fields.

        Args:
            config: Stage configuration dictionary

        Raises:
            ValueError: If configuration is invalid

        Example:
            >>> def validate_config(self, config):
            ...     if "threshold" in config and config["threshold"] > 1.0:
            ...         raise ValueError("threshold must be <= 1.0")
        """
        # Default implementation: no validation
        return None

    def execution(self, context: PlanningContext) -> StageExecution:
        """Build the exact contract consumed by every execution engine."""
        raise ValueError(
            f"{type(self).__module__}.{type(self).__name__} must implement execution()"
        )

    def get_dependencies(self, config: dict[str, Any]) -> list[str]:
        """Get list of stage dependencies.

        Args:
            config: Stage configuration dictionary

        Returns:
            List of stage names this stage depends on
        """
        return config.get("dependencies", [])
