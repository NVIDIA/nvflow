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
"""Registry for managing and discovering workflow stages.

This module provides a hierarchical registry for organizing stages by recipe and workflow.
Stages are organized as: Recipe -> Workflow -> Stage

Example structure:
    finance/
        training_sft/
            - sft
        sdg_secque/
            - generate_answers
        eval/
            - prepare_data
"""

from pathlib import Path
from typing import Any

import yaml

from nvflow.core.base_stage import BaseStage


class StageRegistry:
    """Hierarchical registry for workflow stages.

    Stages are organized in a three-level hierarchy:
    - Recipe (e.g., "finance", "retail")
    - Workflow (e.g., "training_sft", "sdg_basic")
    - Stage (e.g., "baseline", "generate_qa")

    This allows for:
    - Clean YAML configs (short stage names)
    - Natural organization by recipe and workflow
    - Name reusability across different workflows

    Example:
        >>> @StageRegistry.register(recipe="finance", workflow="training_sft", stage="sft")
        >>> class SFTStage(BaseStage):
        ...     def execute(self, config, cluster, expname, run_after=None):
        ...         pass
        >>>
        >>> # Later, retrieve the stage class
        >>> stage_class = StageRegistry.get(recipe="finance", workflow="training_sft", stage="sft")
        >>> stage = stage_class()
    """

    # Hierarchical structure: recipe -> workflow -> stage -> class
    _stages: dict[str, dict[str, dict[str, type[BaseStage]]]] = {}
    # Cache for recipe configurations (loaded from recipe.yaml)
    _recipe_configs: dict[str, dict[str, Any]] = {}

    @classmethod
    def register(cls, recipe: str, workflow: str, stage: str):
        """Decorator to register a stage class in the hierarchy.

        Args:
            recipe: Recipe name (e.g., "finance", "retail")
            workflow: Workflow name within recipe (e.g., "training_sft", "sdg_basic")
            stage: Stage name within workflow (e.g., "baseline", "generate_qa")

        Returns:
            Decorator function

        Raises:
            ValueError: If stage is already registered at this path

        Example:
            >>> @StageRegistry.register(recipe="finance", workflow="training_sft", stage="sft")
            >>> class SFTStage(BaseStage):
            ...     pass
        """

        def decorator(stage_class: type[BaseStage]) -> type[BaseStage]:
            # Create nested structure if it doesn't exist
            if recipe not in cls._stages:
                cls._stages[recipe] = {}
            if workflow not in cls._stages[recipe]:
                cls._stages[recipe][workflow] = {}

            # Check for duplicates
            if stage in cls._stages[recipe][workflow]:
                raise ValueError(
                    f"Stage '{stage}' is already registered at {recipe}.{workflow}.{stage}. "
                    f"Each stage must have a unique name within its workflow."
                )

            # Register the stage
            cls._stages[recipe][workflow][stage] = stage_class
            return stage_class

        return decorator

    @classmethod
    def _get_recipe_config(cls, recipe: str) -> dict[str, Any] | None:
        """Load and cache recipe configuration from recipe.yaml."""
        if recipe in cls._recipe_configs:
            return cls._recipe_configs[recipe]

        # Look for recipe.yaml in nvflow/recipes/{recipe}/
        recipe_dir = Path(__file__).parent.parent / "recipes" / recipe
        for filename in ("recipe.yaml", "recipe.yml"):
            config_path = recipe_dir / filename
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        cls._recipe_configs[recipe] = yaml.safe_load(f)
                    return cls._recipe_configs[recipe]
                except (OSError, yaml.YAMLError):
                    pass  # Fall back to default behavior

        cls._recipe_configs[recipe] = None
        return None

    @classmethod
    def get(cls, recipe: str, workflow: str, stage: str) -> type[BaseStage]:
        """Get a stage class from the hierarchy.

        Args:
            recipe: Recipe name
            workflow: Workflow name
            stage: Stage name

        Returns:
            Stage class

        Raises:
            KeyError: If stage is not found at the specified path
        """
        try:
            return cls._stages[recipe][workflow][stage]
        except KeyError:
            # Provide helpful error message
            if recipe in cls._stages:
                if workflow in cls._stages[recipe]:
                    available = list(cls._stages[recipe][workflow].keys())
                    raise KeyError(
                        f"Stage '{stage}' not found in {recipe}.{workflow}. "
                        f"Available stages: {available}"
                    ) from None
                else:
                    available_workflows = list(cls._stages[recipe].keys())
                    raise KeyError(
                        f"Workflow '{workflow}' not found in recipe '{recipe}'. "
                        f"Available workflows: {available_workflows}"
                    ) from None
            else:
                available_recipes = list(cls._stages.keys())
                raise KeyError(
                    f"Recipe '{recipe}' not found. Available recipes: {available_recipes}"
                ) from None

    @classmethod
    def has(cls, recipe: str, workflow: str, stage: str) -> bool:
        """Check if a stage is registered.

        Args:
            recipe: Recipe name
            workflow: Workflow name
            stage: Stage name

        Returns:
            True if stage is registered, False otherwise
        """
        return (
            recipe in cls._stages
            and workflow in cls._stages[recipe]
            and stage in cls._stages[recipe][workflow]
        )

    @classmethod
    def list_recipes(cls) -> list[str]:
        """List all registered recipes.

        Returns:
            List of recipe names sorted alphabetically
        """
        return sorted(cls._stages.keys())

    @classmethod
    def list_workflows(cls, recipe: str) -> list[str]:
        """List all workflows in a recipe.

        If the recipe has a recipe.yaml with workflow_order defined,
        workflows are returned in that order. Otherwise, they are
        returned in alphabetical order.

        Args:
            recipe: Recipe name

        Returns:
            List of workflow names in defined order (or alphabetically)

        Raises:
            KeyError: If recipe is not found
        """
        if recipe not in cls._stages:
            raise KeyError(f"Recipe '{recipe}' not found. Available recipes: {cls.list_recipes()}")

        registered_workflows = set(cls._stages[recipe].keys())

        # Try to get workflow order from recipe config
        recipe_config = cls._get_recipe_config(recipe)
        if recipe_config and "workflow_order" in recipe_config:
            workflow_order = recipe_config["workflow_order"]
            # Return workflows in defined order, filtering to only registered ones
            ordered = [w for w in workflow_order if w in registered_workflows]
            # Add any registered workflows not in the order (sorted alphabetically)
            remaining = sorted(registered_workflows - set(ordered))
            return ordered + remaining

        # Fallback to alphabetical order
        return sorted(registered_workflows)

    @classmethod
    def list_stages(cls, recipe: str, workflow: str) -> list[str]:
        """List all stages in a workflow.

        Args:
            recipe: Recipe name
            workflow: Workflow name

        Returns:
            List of stage names sorted alphabetically

        Raises:
            KeyError: If recipe or workflow is not found
        """
        if recipe not in cls._stages:
            raise KeyError(f"Recipe '{recipe}' not found. Available recipes: {cls.list_recipes()}")
        if workflow not in cls._stages[recipe]:
            raise KeyError(
                f"Workflow '{workflow}' not found in recipe '{recipe}'. "
                f"Available workflows: {cls.list_workflows(recipe)}"
            )
        return sorted(cls._stages[recipe][workflow].keys())

    @classmethod
    def list_all_stages(cls) -> list[tuple[str, str, str]]:
        """List all registered stages across all recipes and workflows.

        Returns:
            List of (recipe, workflow, stage) tuples sorted alphabetically
        """
        stages = []
        for recipe in cls._stages:
            for workflow in cls._stages[recipe]:
                for stage in cls._stages[recipe][workflow]:
                    stages.append((recipe, workflow, stage))
        return sorted(stages)

    @classmethod
    def clear(cls) -> None:
        """Clear the registry (mainly for testing)."""
        cls._stages = {}
        cls._recipe_configs = {}
