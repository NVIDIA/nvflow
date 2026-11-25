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
"""Secret-free structural inspection of one workflow configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from nvflow.core.workflow_definition import (
    load_config_with_inheritance,
    mapping_diff,
    redact_unresolved_config,
    resolve_workflow_config_path,
)
from nvflow.core.workflow_structure import (
    WORKFLOW_STRUCTURAL_KEYS,
    checkpoint_eval_steps,
    checkpoint_stage_name,
    description_of,
    resolve_structural_reference,
    stage_name_sequence,
    validate_unique_stage_names,
)
from nvflow.project import ProjectContext, ProjectResolver

_RUNTIME_DEPENDENCY = "<runtime-dependent>"


@dataclass(frozen=True)
class WorkflowStageStructure:
    """One structurally inspectable stage row."""

    name: str
    dependencies: tuple[str, ...]
    runtime_expanded: bool = False
    uses: str | None = None
    description: str | None = None


@dataclass(frozen=True)
class WorkflowStructure:
    """Unresolved workflow metadata needed by structural CLI commands."""

    project: str
    workflow_name: str
    workflow_type: str
    stage_entries: tuple[WorkflowStageStructure, ...]

    @property
    def runtime_expanded_stages(self) -> tuple[str, ...]:
        return tuple(entry.name for entry in self.stage_entries if entry.runtime_expanded)


def inspect_workflow_structure(
    config_path: str | Path,
    *,
    project_dir: str | Path | None = None,
    project_search_paths: tuple[str | Path, ...] = (),
) -> WorkflowStructure:
    """Load workflow shape without resolving interpolation or importing stages."""
    resolved_path, context, _, merged_payload = _load_unresolved_configs(
        config_path,
        project_dir=project_dir,
        project_search_paths=project_search_paths,
    )

    workflow = merged_payload.get("workflow")
    if not isinstance(workflow, dict) or not isinstance(workflow.get("name"), str):
        raise TypeError(f"Workflow config must declare workflow.name: {resolved_path}")

    stage_entries = _expanded_stage_entries(
        merged_payload,
        config_path=resolved_path,
    )
    return WorkflowStructure(
        project=context.identity,
        workflow_name=workflow["name"],
        workflow_type=str(workflow.get("type", "unknown")),
        stage_entries=stage_entries,
    )


def unresolved_config_view(
    config_path: str | Path,
    *,
    view: str,
    project_dir: str | Path | None = None,
    project_search_paths: tuple[str | Path, ...] = (),
) -> dict[str, Any]:
    """Return a safe source or merged view without resolving runtime values."""
    if view not in {"source", "merged", "diff"}:
        raise ValueError(f"Unresolved config view must be source, merged, or diff, got {view!r}")
    resolved_path, context, source_config, merged_config = _load_unresolved_configs(
        config_path,
        project_dir=project_dir,
        project_search_paths=project_search_paths,
    )
    if view == "source":
        value = source_config
    elif view == "merged":
        value = merged_config
    else:
        base_ref = source_config.get("_base_")
        if not base_ref:
            value = source_config
        else:
            base_path = (resolved_path.parent / str(base_ref)).resolve()
            base_config = _mapping(
                load_config_with_inheritance(base_path, boundary=context.root),
                base_path,
            )
            value = mapping_diff(base_config, merged_config)
    return redact_unresolved_config(value)


def inspect_environment_selectors(
    config_path: str | Path,
    *,
    project_dir: str | Path | None = None,
    project_search_paths: tuple[str | Path, ...] = (),
) -> dict[str, bool]:
    """Return environment names and enabled state without runtime resolution."""
    resolved_path, _context, _source, merged = _load_unresolved_configs(
        config_path,
        project_dir=project_dir,
        project_search_paths=project_search_paths,
    )
    configured = resolve_structural_reference(
        merged.get("environments"),
        config=merged,
        label=f"environments in {resolved_path}",
    )
    if configured is None:
        return {}
    if not isinstance(configured, dict):
        raise TypeError(f"environments in {resolved_path} must be a mapping")
    return {
        str(name): not isinstance(value, dict) or value.get("enabled", True) is not False
        for name, value in configured.items()
    }


def _load_unresolved_configs(
    config_path: str | Path,
    *,
    project_dir: str | Path | None,
    project_search_paths: tuple[str | Path, ...],
) -> tuple[Path, ProjectContext, dict[str, Any], dict[str, Any]]:
    resolver = ProjectResolver(search_paths=project_search_paths)
    resolved_path = resolve_workflow_config_path(
        config_path,
        project_dir=project_dir,
        search_paths=resolver.search_paths,
    )
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    context = resolver.resolve(
        config_path=resolved_path,
        project_dir=project_dir,
    )
    source_payload = _mapping(OmegaConf.load(resolved_path), resolved_path)
    merged_payload = _mapping(
        load_config_with_inheritance(resolved_path, boundary=context.root),
        resolved_path,
    )
    return resolved_path, context, source_payload, merged_payload


def _mapping(config: object, path: Path) -> dict[str, Any]:
    if not isinstance(config, DictConfig):
        raise TypeError(f"Workflow config must contain a mapping: {path}")
    value = OmegaConf.to_container(config, resolve=False)
    if not isinstance(value, dict):
        raise TypeError(f"Workflow config must contain a mapping: {path}")
    return cast(dict[str, Any], value)


def _optional_mapping(value: object, *, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a mapping")
    return cast(dict[str, Any], value)


def _expanded_stage_entries(
    config: dict[str, Any],
    *,
    config_path: Path,
) -> tuple[WorkflowStageStructure, ...]:
    configured_stages = _pipeline_stages(config, config_path=config_path)
    checkpoints = _optional_mapping(
        config.get("checkpoints"),
        label=f"checkpoints in {config_path}",
    )
    result = dict(_optional_mapping(config.get("stages"), label="stages"))
    defaults = {key: value for key, value in config.items() if key not in WORKFLOW_STRUCTURAL_KEYS}

    for name, value in _optional_mapping(config.get("models"), label="models").items():
        if str(name) in result or not isinstance(value, dict):
            continue
        result[str(name)] = {
            **defaults,
            **value,
            "_source_section": "models",
            "_source_name": str(name),
        }

    checkpoint_details: dict[str, tuple[dict[str, Any], list[int | str] | None]] = {}
    for name, value in checkpoints.items():
        checkpoint_name = str(name)
        if not isinstance(value, dict):
            continue
        base = {
            **defaults,
            **value,
            "_source_section": "checkpoints",
            "_source_name": checkpoint_name,
        }
        steps = _concrete_eval_steps(value.get("eval_steps"))
        checkpoint_details[checkpoint_name] = (base, steps)
        if steps is not None:
            for step in steps:
                result.setdefault(
                    checkpoint_stage_name(checkpoint_name, step),
                    {**base, "_step": step},
                )

    entries = []
    for stage in configured_stages:
        if stage not in checkpoints:
            entries.append(
                WorkflowStageStructure(
                    name=stage,
                    dependencies=_dependencies(
                        result.get(stage),
                        stage=stage,
                        config_path=config_path,
                    ),
                    uses=_uses(result.get(stage), stage=stage, config_path=config_path),
                    description=description_of(result.get(stage)),
                )
            )
            continue
        checkpoint = checkpoints[stage]
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Checkpoint {stage!r} must contain a mapping")
        base, steps = checkpoint_details[stage]
        if steps is None:
            entries.append(
                WorkflowStageStructure(
                    name=stage,
                    dependencies=_dependencies(base, stage=stage, config_path=config_path),
                    uses=_uses(base, stage=stage, config_path=config_path),
                    runtime_expanded=True,
                    description=description_of(base),
                )
            )
            continue
        for step in steps:
            expanded = checkpoint_stage_name(stage, step)
            entries.append(
                WorkflowStageStructure(
                    name=expanded,
                    dependencies=_dependencies(
                        result.get(expanded),
                        stage=expanded,
                        config_path=config_path,
                    ),
                    uses=_uses(
                        result.get(expanded),
                        stage=expanded,
                        config_path=config_path,
                    ),
                    description=description_of(result.get(expanded)),
                )
            )
    validate_unique_stage_names([entry.name for entry in entries])
    return tuple(entries)


def _uses(
    config: object,
    *,
    stage: str,
    config_path: Path,
) -> str | None:
    if isinstance(config, str) and "${" in config:
        raise TypeError(
            f"stage {stage!r} in {config_path} must be an explicit mapping; "
            "stage implementation bindings cannot be interpolated"
        )
    if isinstance(config, dict) and isinstance(config.get("substages"), dict):
        return "substages: " + ", ".join(config["substages"])
    if not isinstance(config, dict) or config.get("uses") is None:
        return None
    value = config["uses"]
    if not isinstance(value, str) or not value or "${" in value:
        raise TypeError(f"uses for stage {stage!r} in {config_path} must be a literal stage type")
    if value.startswith("project:"):
        from nvflow.project.stage_type import parse_project_stage_type

        parse_project_stage_type(value)
        return value
    from nvflow.stages.catalog import get_stage_type

    get_stage_type(value)
    return value


def _concrete_eval_steps(value: object) -> list[int | str] | None:
    if isinstance(value, str) and value.startswith("${"):
        return None
    return checkpoint_eval_steps(value)


def _pipeline_stages(
    config: dict[str, Any],
    *,
    config_path: Path,
) -> tuple[str, ...]:
    value = resolve_structural_reference(
        config.get("pipeline_stages"),
        config=config,
        label=f"pipeline_stages in {config_path}",
    )
    return stage_name_sequence(value, label=f"pipeline_stages in {config_path}")


def _dependencies(
    config: object,
    *,
    stage: str,
    config_path: Path,
) -> tuple[str, ...]:
    if not isinstance(config, dict):
        return ()
    value = config.get("dependencies", [])
    if isinstance(value, str) and value.startswith("${"):
        return (_RUNTIME_DEPENDENCY,)
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise TypeError(
            f"dependencies for stage {stage!r} in {config_path} must be a list of non-empty strings"
        )
    return tuple(value)
