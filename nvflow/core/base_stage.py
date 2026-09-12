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

    @classmethod
    def submitted_expnames(cls, config: dict[str, Any], expname: str) -> list[str]:
        """Get the experiment names that dependent stages must wait for.

        WorkflowRunner calls this on the stage class, without instantiating it,
        and passes the names as ``run_after`` to every stage that lists this
        stage in ``dependencies``. Each returned name must identify an
        experiment that :meth:`execute` submits for the same ``config`` and
        ``expname``. If :meth:`execute` chains experiments, return the
        terminal experiment of each chain.

        The default follows the convention most stages use: one experiment
        named ``expname``, or, when ``config`` defines ``environments``, one
        experiment per selected environment named ``f"{expname}-{env}"``.
        Override it, as a classmethod, when :meth:`execute` names its
        experiments differently.

        Args:
            config: Stage configuration as passed to :meth:`execute`, including
                the ``_environment`` filter added by the runner
            expname: Experiment name passed to :meth:`execute`

        Returns:
            Experiment names for dependent stages to use as ``run_after``
        """
        environments = config.get("environments")
        if not environments:
            return [expname]
        selected = config.get("_environment")
        if not selected:
            env_names = list(environments)
        else:
            if isinstance(selected, str):
                selected = [selected]
            env_names = [env for env in selected if env in environments]
        return [f"{expname}-{env}" for env in env_names]
