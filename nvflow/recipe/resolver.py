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
"""Resolve inline recipe DAGs to reusable project workflow definitions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

from nvflow.project import (
    ProjectContext,
    ProjectManifestError,
    ProjectResolver,
    WorkflowSelector,
    resolve_project_directory,
    resolve_workflow_in_project,
)
from nvflow.recipe.definition import (
    RecipeDefinition,
    RecipeDefinitionError,
    RecipeSelector,
    RecipeWorkflowNode,
)


@dataclass(frozen=True)
class ResolvedRecipeWorkflow:
    """A recipe node bound to one concrete workflow definition."""

    node: RecipeWorkflowNode
    selector: WorkflowSelector
    config_path: Path
    workflow_type: str
    stage_count: int
    description: str | None = None

    @property
    def dependency_selectors(self) -> tuple[str, ...]:
        """Return project-qualified dependency selectors."""
        return tuple(f"{self.selector.project}:{dependency}" for dependency in self.node.depends_on)


@dataclass(frozen=True)
class ResolvedRecipe:
    """A structurally and referentially valid project recipe."""

    selector: RecipeSelector
    context: ProjectContext
    definition: RecipeDefinition
    workflows: tuple[ResolvedRecipeWorkflow, ...]

    @property
    def workflow_map(self) -> dict[str, ResolvedRecipeWorkflow]:
        """Return resolved workflows by project-relative selector."""
        return {workflow.node.selector: workflow for workflow in self.workflows}

    def topological_workflows(self) -> tuple[ResolvedRecipeWorkflow, ...]:
        """Return resolved nodes in deterministic dependency order."""
        resolved = self.workflow_map
        return tuple(resolved[node.selector] for node in self.definition.topological_workflows())


def resolve_recipe(
    value: str,
    resolver: ProjectResolver,
    *,
    project_dir: str | Path | None = None,
    validate_resources: bool = False,
) -> ResolvedRecipe:
    """Resolve and validate one ``project:recipe`` selector."""
    selector = RecipeSelector.parse(value)
    context = (
        resolve_project_directory(
            resolver,
            project_dir,
            expected_project=selector.project,
        )
        if project_dir is not None
        else resolver.resolve_named(selector.project)
    )
    definition = context.manifest.recipe_map.get(selector.recipe)
    if definition is None:
        available = ", ".join(context.manifest.recipe_map) or "none"
        raise RecipeDefinitionError(
            f"Recipe {selector} was not found in {context.manifest_path}. "
            f"Available recipes: {available}"
        )

    resolved_workflows = tuple(
        _resolve_workflow_node(
            selector,
            context,
            node,
            validate_resources=validate_resources,
        )
        for node in definition.workflows
    )
    return ResolvedRecipe(
        selector=selector,
        context=context,
        definition=definition,
        workflows=resolved_workflows,
    )


def _resolve_workflow_node(
    recipe_selector: RecipeSelector,
    project_context: ProjectContext,
    node: RecipeWorkflowNode,
    *,
    validate_resources: bool,
) -> ResolvedRecipeWorkflow:
    from nvflow.core.workflow_definition import (
        expand_checkpoint_pipeline_stages,
        load_config_with_inheritance,
    )
    from nvflow.core.workflow_structure import (
        description_of,
        resolve_structural_reference,
        stage_name_sequence,
    )

    workflow_selector = WorkflowSelector.parse(f"{recipe_selector.project}:{node.selector}")
    workflow_context = resolve_workflow_in_project(workflow_selector, project_context)
    try:
        source = OmegaConf.load(workflow_context.config_path)
        config = load_config_with_inheritance(
            workflow_context.config_path,
            boundary=project_context.root,
        )
        payload = OmegaConf.to_container(config, resolve=False)
    except (OSError, TypeError, ValueError, OmegaConfBaseException, yaml.YAMLError) as exc:
        raise ProjectManifestError(
            f"Unable to inspect workflow {workflow_selector}: {exc}"
        ) from exc
    if not isinstance(config, DictConfig) or not isinstance(payload, dict):
        raise ProjectManifestError(f"Workflow {workflow_selector} must contain a YAML mapping")
    payload = cast(dict[str, Any], payload)

    workflow = payload.get("workflow")
    if not isinstance(workflow, dict) or not isinstance(workflow.get("name"), str):
        raise ProjectManifestError(f"Workflow {workflow_selector} must declare workflow.name")
    if OmegaConf.select(source, "workflow.runnable") is False:
        raise RecipeDefinitionError(
            f"Recipe node {recipe_selector}.{node.selector} references non-runnable "
            f"workflow template {workflow_selector}"
        )
    cluster = payload.get("cluster")
    if not isinstance(cluster, str) or not cluster:
        raise ProjectManifestError(f"Workflow {workflow_selector} must declare cluster")
    try:
        pipeline_value = resolve_structural_reference(
            payload.get("pipeline_stages"),
            config=payload,
            label=f"pipeline_stages in {workflow_selector}",
        )
        configured_pipeline_stages = stage_name_sequence(
            pipeline_value,
            label=f"pipeline_stages in {workflow_selector}",
        )
    except (TypeError, ValueError) as exc:
        raise ProjectManifestError(f"Workflow {workflow_selector}: {exc}") from exc
    configured_stages = payload.get("stages")
    if not isinstance(configured_stages, dict):
        raise ProjectManifestError(f"Workflow {workflow_selector} must declare stages as a mapping")
    dynamic_stages: set[str] = set()
    for section_name in ("models", "checkpoints"):
        section = payload.get(section_name, {})
        if section is not None and not isinstance(section, dict):
            raise ProjectManifestError(
                f"Workflow {workflow_selector} must declare {section_name} as a mapping"
            )
        if isinstance(section, dict):
            dynamic_stages.update(str(name) for name in section)
    configured_names = set(configured_stages) | dynamic_stages
    missing_configs = [
        stage for stage in configured_pipeline_stages if stage not in configured_names
    ]
    if missing_configs:
        raise ProjectManifestError(
            f"Workflow {workflow_selector} has active stages without configuration: "
            + ", ".join(missing_configs)
        )
    try:
        # Validate checkpoint expansion shape; recipes retain the whole workflow.
        expanded_pipeline_stages = expand_checkpoint_pipeline_stages(
            configured_pipeline_stages,
            payload.get("checkpoints") or {},
        )
    except (TypeError, ValueError) as exc:
        raise ProjectManifestError(f"Workflow {workflow_selector}: {exc}") from exc

    if validate_resources:
        project_context.validate_resources(payload)

    return ResolvedRecipeWorkflow(
        node=node,
        selector=workflow_selector,
        config_path=workflow_context.config_path,
        workflow_type=str(workflow.get("type", "unknown")),
        stage_count=len(expanded_pipeline_stages),
        description=description_of(workflow),
    )
