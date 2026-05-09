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

from abc import ABC, abstractmethod
from typing import Any


class BaseStage(ABC):
    """Base class for all workflow stages.

    All stages must inherit from this class and implement the execute() method.
    Stages are registered using the StageRegistry decorator pattern.

    Example:
        >>> from nvflow.core import BaseStage, StageRegistry
        >>>
        >>> @StageRegistry.register(recipe="finance", workflow="training_sft", stage="sft")
        >>> class SFTStage(BaseStage):
        ...     workflow = "training_sft"
        ...     def execute(self, config, cluster, expname, run_after=None):
        ...         print(f"Running SFT training...")
        ...         # Implementation here
    """

    # Workflow this stage belongs to (optional - for documentation/clarity)
    # Not required by the framework, but helpful for understanding stage organization
    workflow: str = "unknown"

    @abstractmethod
    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Execute the stage.

        This method must be implemented by all stage subclasses.

        Args:
            config: Stage configuration dictionary from the workflow config
            cluster: Cluster name (references cluster_configs/<cluster>.yaml)
            expname: Experiment name for this stage execution
            run_after: List of experiment names that must complete before this stage
                      (used for dependency management in Slurm)

        Example:
            >>> config = {
            ...     "input_dir": "/data/raw",
            ...     "output_dir": "/data/processed",
            ...     "stage_kwargs": {"installation_command": "pip install -q pandas"}
            ... }
            >>> stage.execute(config, "nrt", "my-exp-data-download", run_after=None)
        """
        pass

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

    def get_dependencies(self, config: dict[str, Any]) -> list[str]:
        """Get list of stage dependencies.

        Args:
            config: Stage configuration dictionary

        Returns:
            List of stage names this stage depends on
        """
        return config.get("dependencies", [])
