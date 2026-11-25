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
"""Shared CLI workflow selection and lightweight workflow discovery."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import typer
import yaml

from nvflow.core.workflow_structure import (
    description_of,
    resolve_structural_reference,
    stage_name_sequence,
)
from nvflow.project import (
    ProjectContext,
    ProjectManifestError,
    ProjectResolver,
    WorkflowSelector,
    resolve_project_directory,
    resolve_workflow_selector,
)

_WORKFLOW_SUFFIXES = ("*.yaml", "*.yml")
_REQUIRED_WORKFLOW_KEYS = frozenset({"workflow", "cluster", "pipeline_stages", "stages"})


@dataclass(frozen=True)
class SelectedWorkflow:
    """One config path selected by an explicit path or stable selector."""

    config_path: str | Path
    project_dir: str | Path | None
    display_name: str
    project_context: ProjectContext | None = None


@dataclass(frozen=True)
class WorkflowListing:
    """Lightweight metadata shown by ``nvflow list-workflows``."""

    selector: WorkflowSelector
    workflow_name: str
    workflow_type: str
    template: bool
    stage_count: int = 0
    project_description: str | None = None
    description: str | None = None


def require_workflow(
    *,
    config: str | None,
    workflow: str | None,
    project_dir: str | None,
    project_paths: tuple[str, ...],
) -> SelectedWorkflow:
    """Require one config path or selector for a workflow-loading command."""
    selected = select_workflow(
        config=config,
        workflow=workflow,
        project_dir=project_dir,
        project_paths=project_paths,
    )
    assert selected is not None
    return selected


def select_workflow(
    *,
    config: str | None,
    workflow: str | None,
    project_dir: str | None,
    project_paths: tuple[str, ...],
    required: bool = True,
) -> SelectedWorkflow | None:
    """Select exactly one workflow path using shared CLI semantics."""
    if config and workflow:
        raise typer.BadParameter(
            "Use either --config or --workflow, not both",
            param_hint="--config/--workflow",
        )
    if not config and not workflow:
        if not required:
            return None
        raise typer.BadParameter(
            "One of --config or --workflow is required",
            param_hint="--config/--workflow",
        )
    if config:
        return SelectedWorkflow(
            config_path=config,
            project_dir=project_dir,
            display_name=config,
        )

    assert workflow is not None
    resolver = ProjectResolver(search_paths=project_paths)
    try:
        context = resolve_workflow_selector(
            workflow,
            resolver,
            project_dir=project_dir,
        )
    except ProjectManifestError as exc:
        raise typer.BadParameter(str(exc), param_hint="--workflow") from exc
    return SelectedWorkflow(
        config_path=context.config_path,
        project_dir=context.root,
        display_name=str(WorkflowSelector.parse(workflow)),
        project_context=context,
    )


def discover_workflows(
    *,
    project: str | None,
    project_dir: str | None,
    project_paths: tuple[str, ...],
    include_all: bool,
) -> tuple[WorkflowListing, ...]:
    """Discover workflow YAML without importing project or execution code."""
    resolver = ProjectResolver(search_paths=project_paths)
    try:
        if project_dir is not None:
            context = resolve_project_directory(
                resolver,
                project_dir,
                expected_project=project,
            )
            contexts = (context,)
        else:
            project_names = (project,) if project else tuple(resolver.project_names())
            contexts = tuple(resolver.resolve_named(name) for name in project_names)
    except (ProjectManifestError, TypeError, ValueError) as exc:
        raise typer.BadParameter(str(exc), param_hint="--project/--project-dir") from exc

    try:
        all_listings = [
            listing for context in contexts for listing in _discover_project_workflows(context)
        ]
        _validate_workflow_groups(all_listings)
    except (ProjectManifestError, TypeError, ValueError) as exc:
        raise typer.BadParameter(str(exc), param_hint="--project/--project-dir") from exc
    listings = (
        all_listings
        if include_all
        else [listing for listing in all_listings if not listing.template]
    )
    listings.sort(
        key=lambda listing: (
            listing.selector.project,
            listing.workflow_name,
            str(listing.selector),
        )
    )
    return tuple(listings)


def _validate_workflow_groups(
    listings: list[WorkflowListing],
) -> None:
    """Validate metadata shared by workflow-name groups."""
    types_by_group: dict[tuple[str, str], set[str]] = {}
    for listing in listings:
        key = (listing.selector.project, listing.workflow_name)
        types_by_group.setdefault(key, set()).add(listing.workflow_type)
    conflicts = {
        key: sorted(workflow_types)
        for key, workflow_types in types_by_group.items()
        if len(workflow_types) > 1
    }
    if conflicts:
        details = "; ".join(
            f"{project}.{name}: {', '.join(workflow_types)}"
            for (project, name), workflow_types in sorted(conflicts.items())
        )
        raise ProjectManifestError(
            f"conflicting workflow.type values within workflow.name groups: {details}"
        )


def _discover_project_workflows(
    context: ProjectContext,
) -> tuple[WorkflowListing, ...]:
    workflows_root = context.workflows_root.resolve()
    paths = sorted(
        {
            path
            for pattern in _WORKFLOW_SUFFIXES
            for path in context.workflows_root.rglob(pattern)
            if path.is_file()
        }
    )
    sources: dict[Path, dict[str, Any]] = {}
    candidates: list[Path] = []
    for path in paths:
        try:
            path.resolve().relative_to(workflows_root)
        except ValueError as exc:
            raise ProjectManifestError(
                f"Workflow YAML resolves outside {workflows_root}: {path}"
            ) from exc
        try:
            text = path.read_text(encoding="utf-8")
            raw = yaml.safe_load(text)
            if not isinstance(raw, dict):
                raise TypeError("top-level YAML value must be a mapping")
            if "project" in raw:
                raise ValueError(
                    f"Workflow config {path} must not declare project; "
                    "ownership comes from project.yaml"
                )
            resolved = path.resolve()
            sources[resolved] = raw
            if "_base_" not in raw and not _REQUIRED_WORKFLOW_KEYS <= raw.keys():
                continue
            candidates.append(resolved)
        except (
            OSError,
            TypeError,
            ValueError,
            yaml.YAMLError,
        ) as exc:
            raise ProjectManifestError(f"Unable to inspect workflow YAML {path}: {exc}") from exc

    cache: dict[Path, dict[str, Any]] = {}
    listings = []
    for path in candidates:
        source = sources[path]
        try:
            merged = _load_discovery_config(
                path,
                sources=sources,
                cache=cache,
                boundary=context.root.resolve(),
                stack=(),
            )
        except (
            OSError,
            TypeError,
            ValueError,
            yaml.YAMLError,
        ) as exc:
            raise ProjectManifestError(f"Unable to inspect workflow YAML {path}: {exc}") from exc
        payload = merged
        if not _REQUIRED_WORKFLOW_KEYS <= payload.keys():
            continue
        workflow = payload["workflow"]
        if not isinstance(workflow, dict) or not isinstance(workflow.get("name"), str):
            raise ProjectManifestError(f"Workflow {path} must declare workflow.name")
        pipeline_value = resolve_structural_reference(
            payload["pipeline_stages"],
            config=payload,
            label=f"pipeline_stages in {path}",
        )
        pipeline_stages = stage_name_sequence(
            pipeline_value,
            label=f"pipeline_stages in {path}",
        )
        source_workflow = source.get("workflow")
        template = isinstance(source_workflow, dict) and source_workflow.get("runnable") is False
        relative = path.relative_to(context.workflows_root).with_suffix("")
        listings.append(
            WorkflowListing(
                selector=WorkflowSelector(
                    project=context.identity,
                    relative_path=PurePosixPath(relative.as_posix()),
                ),
                project_description=context.manifest.description,
                workflow_name=workflow["name"],
                workflow_type=str(workflow.get("type", "unknown")),
                template=template,
                stage_count=len(pipeline_stages),
                description=description_of(workflow),
            )
        )
    selector_counts = Counter(str(listing.selector) for listing in listings)
    duplicates = sorted(selector for selector, count in selector_counts.items() if count > 1)
    if duplicates:
        raise ProjectManifestError(
            "Workflow selectors are ambiguous across .yaml and .yml files: " + ", ".join(duplicates)
        )
    return tuple(listings)


def _load_discovery_config(
    path: Path,
    *,
    sources: dict[Path, dict[str, Any]],
    cache: dict[Path, dict[str, Any]],
    boundary: Path,
    stack: tuple[Path, ...],
) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in cache:
        return cache[resolved]
    try:
        resolved.relative_to(boundary)
    except ValueError as exc:
        raise ValueError(f"Workflow base resolves outside project root: {resolved}") from exc
    if resolved in stack:
        cycle = " -> ".join(str(item) for item in (*stack, resolved))
        raise ValueError(f"Cyclic workflow _base_ chain: {cycle}")
    source = sources.get(resolved)
    if source is None:
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError("top-level YAML value must be a mapping")
        if "project" in raw:
            raise ValueError(
                f"Workflow config {resolved} must not declare project; "
                "ownership comes from project.yaml"
            )
        source = raw
        sources[resolved] = source

    base = source.get("_base_")
    if base is None:
        merged = deepcopy(source)
    else:
        if not isinstance(base, str):
            raise TypeError("_base_ must be a relative YAML path")
        if Path(base).is_absolute():
            raise ValueError("_base_ must be a relative YAML path")
        base_path = (resolved.parent / base).resolve()
        if not base_path.is_file():
            raise FileNotFoundError(f"Base config not found: {base} (resolved to {base_path})")
        child = {key: deepcopy(value) for key, value in source.items() if key != "_base_"}
        merged = _merge_discovery_config(
            _load_discovery_config(
                base_path,
                sources=sources,
                cache=cache,
                boundary=boundary,
                stack=(*stack, resolved),
            ),
            child,
        )
    cache[resolved] = merged
    return merged


def _merge_discovery_config(base: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_discovery_config(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged
