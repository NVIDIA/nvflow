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
"""Target-independent workflow loading, expansion, and validation."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypeAlias, cast

from omegaconf import DictConfig, OmegaConf

from nvflow.core.base_stage import BaseStage
from nvflow.core.workflow_structure import (
    EXPANDABLE_WORKFLOW_SECTIONS,
    WORKFLOW_STRUCTURAL_KEYS,
    checkpoint_eval_steps,
    checkpoint_stage_name,
    validate_unique_stage_names,
)
from nvflow.project import ProjectContext, ProjectResolver

ConfigPath: TypeAlias = tuple[str | int, ...]  # noqa: UP040 - mypy compatibility
_ENV_REFERENCE_RE = re.compile(r"\$\{oc\.env:([^,}]+)(?:,[^}]*)?\}")
_SAFE_ENV_REFERENCE_RE = re.compile(
    r"^\$\{oc\.env:[^,}]+(?:,\s*(?:none|null|~)?\s*)?\}$",
    re.IGNORECASE,
)
_SECRET_NAME_RE = re.compile(
    r"(?:^|[._/-])(?:api[_-]?key|credentials?|password|secrets?|tokens?)$",
    re.IGNORECASE,
)
_NON_SECRET_TOKEN_NAME_RE = re.compile(
    r"^(?:(?:max|min|num|total|input|output|prompt|completion|new|reasoning|train|logprob)"
    r"(?:_[a-z0-9]+)*_tokens|token_(?:counts?|budgets?|limits?))$",
    re.IGNORECASE,
)


def _is_secret_name(value: object) -> bool:
    name = str(value)
    return bool(_SECRET_NAME_RE.search(name) and not _NON_SECRET_TOKEN_NAME_RE.fullmatch(name))


def load_config_with_inheritance(
    config_path: Path,
    *,
    boundary: Path | None = None,
    _stack: tuple[Path, ...] = (),
) -> DictConfig:
    """Load a workflow config, merging its ``_base_`` chain child-over-base.

    Interpolations remain unresolved so callers can inspect the authored
    composition or resolve it for execution.
    """
    resolved_path = config_path.resolve()
    if boundary is not None:
        try:
            resolved_path.relative_to(boundary.resolve())
        except ValueError as exc:
            raise ValueError(
                f"Workflow base resolves outside project root: {resolved_path}"
            ) from exc
    if resolved_path in _stack:
        cycle = " -> ".join(str(path) for path in (*_stack, resolved_path))
        raise ValueError(f"Cyclic workflow _base_ chain: {cycle}")

    config = OmegaConf.load(resolved_path)
    if not isinstance(config, DictConfig):
        raise TypeError(f"Workflow config must contain a mapping: {resolved_path}")
    if "project" in config:
        raise ValueError(
            f"Workflow config {resolved_path} must not declare project; "
            "ownership comes from project.yaml"
        )

    if "_base_" in config:
        base_ref = config["_base_"]
        if not isinstance(base_ref, str) or not base_ref:
            raise TypeError(f"_base_ in {resolved_path} must be a non-empty relative path")
        if Path(base_ref).is_absolute():
            raise ValueError(f"_base_ in {resolved_path} must be a relative path")
        base_path = (resolved_path.parent / base_ref).resolve()
        if not base_path.is_file():
            raise FileNotFoundError(f"Base config not found: {base_ref} (resolved to {base_path})")

        base_config = load_config_with_inheritance(
            base_path,
            boundary=boundary,
            _stack=(*_stack, resolved_path),
        )
        child = cast(dict[str, Any], OmegaConf.to_container(config, resolve=False))
        del child["_base_"]
        config = cast(DictConfig, OmegaConf.merge(base_config, OmegaConf.create(child)))

    return config


class WorkflowDefinition:
    """Resolved workflow content independent of a concrete execution target."""

    def __init__(
        self,
        config_path: str | Path,
        project_dir: str | Path | None = None,
        project_search_paths: tuple[str | Path, ...] = (),
    ):
        resolver = ProjectResolver(search_paths=project_search_paths)
        self.config_path = resolve_workflow_config_path(
            config_path,
            project_dir=project_dir,
            search_paths=resolver.search_paths,
        )
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        self.project_context = resolver.resolve(
            config_path=self.config_path,
            project_dir=project_dir,
        )
        self.project = self.project_context.identity

        source = OmegaConf.load(self.config_path)
        if not isinstance(source, DictConfig):
            raise TypeError(f"Workflow config must contain a mapping: {self.config_path}")
        self.source_config = source
        self.merged_config = load_config_with_inheritance(
            self.config_path,
            boundary=self.project_context.root,
        )
        unresolved = OmegaConf.to_container(self.merged_config, resolve=False)
        if not isinstance(unresolved, dict):
            raise TypeError(f"Workflow config must contain a mapping: {self.config_path}")
        _validate_literal_stage_uses(unresolved, self.config_path)
        self.sensitive_paths = _find_env_references(unresolved)
        resolved = OmegaConf.to_container(self.merged_config, resolve=True)
        if not isinstance(resolved, dict):
            raise TypeError(f"Workflow config must contain a mapping: {self.config_path}")
        self._resolved_config = cast(dict[str, Any], resolved)
        self._resolved_config["project"] = self.project
        self._secret_replacements = _secret_replacements(
            self._resolved_config,
            self.sensitive_paths,
        )
        self._secret_replacements.update(_named_secret_replacements(self._resolved_config))
        self.config = copy.deepcopy(self._resolved_config)
        self._expand_dynamic_stages()

        workflow = cast(dict[str, Any], self.config["workflow"])
        self.workflow_name = cast(str, workflow["name"])
        self.workflow_type = cast(str, workflow.get("type", "unknown"))
        self.default_cluster = cast(str, self.config["cluster"])
        self.project_context.validate_resources(self.config)

    @property
    def cluster(self) -> str:
        """Return the workflow's default cluster for compatibility."""
        return self.default_cluster

    @property
    def pipeline_stages(self) -> list[str]:
        """Return the configured stage order."""
        return cast(list[str], self.config["pipeline_stages"])

    def select_active_stages(self, stages: Sequence[str] | None = None) -> list[str]:
        """Return unique requested stages, or the complete active pipeline."""
        active = list(self.pipeline_stages)
        if not stages:
            return active

        selected = list(dict.fromkeys(stages))
        configured = self.config.get("stages", {})
        available = ", ".join(active) or "none"
        for stage in selected:
            if stage not in active:
                if stage in configured:
                    raise ValueError(
                        f"Stage '{stage}' is configured but not active in pipeline_stages. "
                        f"Active stages: {available}"
                    )
                raise ValueError(f"Stage '{stage}' not found. Active stages: {available}")
            if stage not in configured:
                raise ValueError(
                    f"Stage '{stage}' is active but has no configuration in the stages section"
                )
        return selected

    def config_view(
        self,
        view: str = "resolved",
        *,
        stages: Sequence[str] | None = None,
        environment: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Return a safe configuration view for inspection."""
        if stages:
            if view != "resolved":
                raise ValueError("Stage-scoped configuration requires the resolved view")
            selected = self.select_active_stages(stages)
            value = {
                "stages": {
                    stage: _environment_config_view(
                        self.config["stages"][stage],
                        environment,
                    )
                    for stage in selected
                }
            }
            return cast(dict[str, Any], self.redact_runtime_values(value))
        if environment:
            raise ValueError("Environment selection requires a stage-scoped configuration view")

        if view == "source":
            value = OmegaConf.to_container(self.source_config, resolve=False)
        elif view == "merged":
            value = OmegaConf.to_container(self.merged_config, resolve=False)
        elif view == "resolved":
            value = _redact_sensitive(self._resolved_config, self.sensitive_paths)
        elif view == "diff":
            value = self._config_diff()
        else:
            raise ValueError(f"Unknown config view: {view}")
        if not isinstance(value, dict):
            raise TypeError(f"Workflow config must contain a mapping: {self.config_path}")
        if view in {"source", "merged", "diff"}:
            return redact_unresolved_config(cast(dict[str, Any], value))
        return cast(dict[str, Any], self.redact_runtime_values(value))

    def redact_runtime_values(self, value: Any) -> Any:
        """Redact resolved environment values from config or plan output."""
        return _redact_runtime_values(value, self._secret_replacements)

    def _config_diff(self) -> dict[str, Any]:
        base_ref = self.source_config.get("_base_")
        if not base_ref:
            value = OmegaConf.to_container(self.source_config, resolve=False)
            if not isinstance(value, dict):
                raise TypeError(f"Workflow config must contain a mapping: {self.config_path}")
            return cast(dict[str, Any], value)

        base_path = (self.config_path.parent / str(base_ref)).resolve()
        base = OmegaConf.to_container(
            load_config_with_inheritance(base_path, boundary=self.project_context.root),
            resolve=False,
        )
        merged = OmegaConf.to_container(self.merged_config, resolve=False)
        if not isinstance(base, dict) or not isinstance(merged, dict):
            raise TypeError(f"Workflow config must contain a mapping: {self.config_path}")
        return cast(dict[str, Any], mapping_diff(base, merged))

    def _expand_dynamic_stages(self) -> None:
        for section_name in EXPANDABLE_WORKFLOW_SECTIONS:
            section = self.config.get(section_name, {})
            if not section:
                continue

            if section_name == "checkpoints":
                self.config["pipeline_stages"] = expand_checkpoint_pipeline_stages(
                    self.config.get("pipeline_stages", []),
                    section,
                )

            for entry_name, entry_config in section.items():
                if not isinstance(entry_config, dict):
                    continue
                for stage_name, stage_config in self._stage_entries_for(
                    section_name, entry_name, entry_config
                ):
                    if stage_name in self.config.get("stages", {}):
                        continue
                    self.config.setdefault("stages", {})[stage_name] = stage_config

    def _workflow_defaults(self) -> dict[str, Any]:
        return {
            key: value for key, value in self.config.items() if key not in WORKFLOW_STRUCTURAL_KEYS
        }

    def _stage_entries_for(
        self,
        section_name: str,
        entry_name: str,
        entry_config: dict[str, Any],
    ) -> list[tuple[str, dict[str, Any]]]:
        base = {
            **self._workflow_defaults(),
            **entry_config,
            "_source_section": section_name,
            "_source_name": entry_name,
        }
        if section_name != "checkpoints":
            return [(entry_name, base)]

        eval_steps = checkpoint_eval_steps(entry_config.get("eval_steps"))
        return [
            (checkpoint_stage_name(entry_name, step), {**base, "_step": step})
            for step in eval_steps
        ]

    def resolve_stage_class(self, stage_name: str) -> type[BaseStage]:
        """Resolve one workflow-local shared alias or project class coordinate."""
        stage_config = self.config.get("stages", {}).get(stage_name)
        if not isinstance(stage_config, dict):
            raise KeyError(f"Stage {stage_name!r} has no configuration")
        uses = stage_config.get("uses")
        if "substages" in stage_config:
            if uses is not None:
                raise ValueError(f"Stage {stage_name!r} declares both 'uses' and 'substages'")
            from nvflow.stages.substages import SubstagesStage

            return SubstagesStage
        if uses is None:
            raise KeyError(f"Stage {stage_name!r} must declare a literal 'uses' implementation")
        if not isinstance(uses, str) or not uses:
            raise ValueError(f"Stage {stage_name!r}.uses must be a non-empty string")
        return resolve_stage_uses(uses, self.project_context)

    def stage_runtime_config(self, stage_name: str) -> dict[str, Any]:
        """Return one stage config without its declarative ``uses`` and ``description``."""
        config = dict(self.config["stages"][stage_name])
        config.pop("uses", None)
        description = config.pop("description", None)
        if description is not None and not isinstance(description, str):
            raise TypeError(f"Stage {stage_name!r}.description must be a string")
        return config

    def stage_implementation(self, stage_name: str) -> dict[str, Any]:
        """Return versioned implementation identity included in launch plans."""
        from nvflow.core.platform import active_platform_identity

        stage_class = self.resolve_stage_class(stage_name)
        platform = active_platform_identity()
        uses = self.config["stages"][stage_name].get("uses")
        if "substages" in self.config["stages"][stage_name]:
            return {
                "kind": "shared",
                "name": "substages",
                "contract_version": 1,
                "provider": f"{stage_class.__module__}.{stage_class.__name__}",
                "platform": platform,
            }
        if isinstance(uses, str) and not uses.startswith("project:"):
            contract_version = self.config["stages"][stage_name].get("contract_version")
            return {
                "kind": "shared",
                "name": uses,
                "contract_version": contract_version,
                "provider": f"{stage_class.__module__}.{stage_class.__name__}",
                "platform": platform,
            }
        return {
            "kind": "project",
            "name": uses,
            "contract_version": self.config["stages"][stage_name].get("contract_version"),
            "provider": f"{stage_class.__module__}.{stage_class.__name__}",
            "platform": platform,
        }

    def validate_stages(self, stages_to_run: Sequence[str] | None = None) -> list[str]:
        """Validate selected stages and return a dependency-first closure."""
        selected = self.select_active_stages(stages_to_run)
        configured = self.config["stages"]
        ordered: list[str] = []
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(stage: str, upstream: str | None = None) -> None:
            if stage in visiting:
                raise ValueError(f"Workflow contains a stage dependency cycle at {stage!r}")
            if stage in visited:
                return
            if stage not in configured:
                if upstream is None:
                    raise ValueError(f"Stage '{stage}' has no config block in 'stages:'")
                raise ValueError(
                    f"Stage '{upstream}' depends on '{stage}' which has no config block in 'stages:'"
                )
            try:
                self.resolve_stage_class(stage)
            except KeyError as exc:
                if upstream is None:
                    raise ValueError(
                        f"Stage '{stage}' declares no implementation. Set 'uses' to a "
                        "shared stage type or project:python.module:ClassName."
                    ) from exc
                raise ValueError(
                    f"Stage '{upstream}' depends on '{stage}', which declares no 'uses' "
                    "implementation."
                ) from exc
            visiting.add(stage)
            for dependency in configured[stage].get("dependencies", []):
                visit(dependency, stage)
            visiting.remove(stage)
            visited.add(stage)
            ordered.append(stage)

        for stage in selected:
            visit(stage)
        return ordered

    def validate(
        self,
        stages: Sequence[str] | None = None,
        environment: Sequence[str] | None = None,
    ) -> None:
        """Semantically validate the workflow without scheduler submission."""
        required_fields = ["workflow", "cluster", "pipeline_stages", "stages"]
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required field in config: {field}")
        closure = self.validate_stages(stages)
        for stage_name in closure:
            raw_config = self.stage_runtime_config(stage_name)
            launcher_config = self.project_context.render_launcher_paths(raw_config)
            if environment is not None:
                launcher_config["_environment"] = list(environment)
            stage_class = self.resolve_stage_class(stage_name)
            stage = stage_class()
            stage.bind_project_context(self.project_context)
            stage.validate_config(launcher_config)

    def get_expname(self, stage_name: str, stage_config: dict[str, Any]) -> str:
        """Return the stable experiment name for a configured stage."""
        base_name = f"{self.workflow_name}-{stage_name}"
        if self.project_context.is_external:
            base_name = f"{self.project_context.identity}-{base_name}"
        run_name = stage_config.get("run_name")
        return f"{base_name}-{run_name}" if run_name else base_name


def expand_checkpoint_pipeline_stages(
    pipeline_stages: Sequence[str],
    checkpoints: Mapping[str, Any],
) -> list[str]:
    """Expand checkpoint names into the concrete stages used by execution."""
    expanded: list[str] = []
    for stage in pipeline_stages:
        if stage not in checkpoints:
            expanded.append(stage)
            continue
        checkpoint = checkpoints[stage]
        if not isinstance(checkpoint, Mapping):
            raise TypeError(f"Checkpoint {stage!r} must contain a mapping")
        expanded.extend(
            checkpoint_stage_name(stage, step)
            for step in checkpoint_eval_steps(checkpoint.get("eval_steps"))
        )
    validate_unique_stage_names(expanded)
    return expanded


def _validate_literal_stage_uses(config: dict[str, Any], config_path: Path) -> None:
    """Reject interpolated shared implementation identifiers before resolution."""

    def validate(value: object, location: str) -> None:
        if not isinstance(value, str) or not value or "${" in value:
            raise TypeError(
                f"uses for {location} in {config_path} "
                "must be a literal shared alias or project class coordinate"
            )

    if "uses" in config:
        validate(config["uses"], "workflow defaults")
    for section_name in ("stages", *EXPANDABLE_WORKFLOW_SECTIONS):
        section = config.get(section_name)
        if isinstance(section, str) and "${" in section:
            raise TypeError(
                f"{section_name} in {config_path} must be an explicit mapping; "
                "stage implementation bindings cannot be interpolated"
            )
        section = section or {}
        if not isinstance(section, dict):
            continue
        for stage_name, stage_config in section.items():
            if isinstance(stage_config, str) and "${" in stage_config:
                raise TypeError(
                    f"{section_name}.{stage_name} in {config_path} must be an explicit mapping; "
                    "stage implementation bindings cannot be interpolated"
                )
            if not isinstance(stage_config, dict):
                continue
            if "uses" in stage_config:
                validate(stage_config["uses"], f"{section_name}.{stage_name}")
            substages = stage_config.get("substages")
            for name, substage in substages.items() if isinstance(substages, dict) else ():
                if isinstance(substage, dict):
                    validate(substage.get("uses"), f"{section_name}.{stage_name}.substages.{name}")


def resolve_stage_uses(uses: str, project_context: ProjectContext | None) -> type[BaseStage]:
    """Resolve a shared alias or ``project:`` class coordinate to its stage class."""
    if uses.startswith("project:"):
        from nvflow.project.stage_type import resolve_project_stage_type

        if project_context is None:
            raise ValueError(f"{uses!r} needs a selected project")
        return resolve_project_stage_type(project_context, uses)

    from nvflow.stages.catalog import resolve_stage_type

    return resolve_stage_type(uses)


def load_workflow(
    config_path: str | Path,
    project_dir: str | Path | None = None,
    project_search_paths: tuple[str | Path, ...] = (),
) -> WorkflowDefinition:
    """Load one workflow without binding it to an execution target."""
    return WorkflowDefinition(
        config_path=config_path,
        project_dir=project_dir,
        project_search_paths=project_search_paths,
    )


def resolve_workflow_config_path(
    config_path: str | Path,
    *,
    project_dir: str | Path | None,
    search_paths: tuple[Path, ...],
) -> Path:
    path = Path(config_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    if project_dir is not None:
        candidate = Path(project_dir).expanduser() / path
        if candidate.exists():
            return candidate.resolve()
    if path.exists():
        return path.resolve()
    for search_path in search_paths:
        candidates = []
        direct = search_path / path
        if direct.exists():
            candidates.append(direct.resolve())
        if search_path.is_dir():
            candidates.extend(
                candidate.resolve()
                for child in search_path.iterdir()
                if child.is_dir() and (candidate := child / path).exists()
            )
        candidates = list(dict.fromkeys(candidates))
        if len(candidates) > 1:
            raise ValueError(
                f"Workflow path {config_path!r} is ambiguous under {search_path}: "
                + ", ".join(str(candidate) for candidate in candidates)
            )
        if candidates:
            return candidates[0]
    return path.resolve()


def _find_env_references(
    value: Any,
    path: ConfigPath = (),
) -> dict[ConfigPath, tuple[str, ...]]:
    references: dict[ConfigPath, tuple[str, ...]] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            references.update(_find_env_references(child, (*path, str(key))))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            references.update(_find_env_references(child, (*path, index)))
    elif isinstance(value, str):
        names = tuple(match.group(1) for match in _ENV_REFERENCE_RE.finditer(value))
        if names:
            references[path] = names
    return references


def _environment_config_view(
    config: Mapping[str, Any],
    environment: Sequence[str] | None,
) -> dict[str, Any]:
    """Select environment entries and hide an identical internal alias."""
    result = copy.deepcopy(dict(config))
    catalog = result.get("environments")
    if not isinstance(catalog, Mapping):
        return result

    names = list(environment or catalog)
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError("Duplicate environment selection: " + ", ".join(duplicates))
    unknown = [name for name in names if name not in catalog]
    if unknown:
        raise ValueError(
            "Unknown environment(s): " + ", ".join(unknown) + "; available: " + ", ".join(catalog)
        )
    selected = {name: copy.deepcopy(catalog[name]) for name in names}
    aliases = [
        key
        for key, value in result.items()
        if key != "environments" and isinstance(value, Mapping) and value == catalog
    ]
    if aliases:
        result.pop("environments")
        for alias in aliases:
            result[alias] = copy.deepcopy(selected)
    else:
        result["environments"] = selected
    return result


def _redact_sensitive(
    value: Any,
    sensitive_paths: dict[ConfigPath, tuple[str, ...]],
    path: ConfigPath = (),
) -> Any:
    names = sensitive_paths.get(path)
    if names:
        return f"<redacted:oc.env:{','.join(names)}>"
    if isinstance(value, dict):
        return {
            key: _redact_sensitive(child, sensitive_paths, (*path, str(key)))
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [
            _redact_sensitive(child, sensitive_paths, (*path, index))
            for index, child in enumerate(value)
        ]
    return copy.deepcopy(value)


def mapping_diff(base: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    diff: dict[str, Any] = {}
    for key, target_value in target.items():
        if key not in base:
            diff[key] = copy.deepcopy(target_value)
            continue
        base_value = base[key]
        if isinstance(base_value, dict) and isinstance(target_value, dict):
            nested = mapping_diff(base_value, target_value)
            if nested:
                diff[key] = nested
        elif base_value != target_value:
            diff[key] = copy.deepcopy(target_value)
    return diff


def _secret_replacements(
    resolved: dict[str, Any],
    sensitive_paths: dict[ConfigPath, tuple[str, ...]],
) -> dict[str, str]:
    replacements: dict[str, str] = {}
    for path, names in sensitive_paths.items():
        value: Any = resolved
        try:
            for part in path:
                value = value[part]
        except (KeyError, IndexError, TypeError):
            continue
        if isinstance(value, str) and value:
            replacements[value] = f"<redacted:oc.env:{','.join(names)}>"
    return replacements


def _redact_runtime_values(value: Any, replacements: dict[str, str]) -> Any:
    return _redact_runtime_value(value, replacements)


def _redact_runtime_value(
    value: Any,
    replacements: dict[str, str],
    *,
    secret_name: str | None = None,
) -> Any:
    if isinstance(value, dict):
        return {
            key: _redact_runtime_value(
                child,
                replacements,
                secret_name=str(key) if _is_secret_name(key) else secret_name,
            )
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [
            _redact_runtime_value(child, replacements, secret_name=secret_name) for child in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _redact_runtime_value(child, replacements, secret_name=secret_name) for child in value
        )
    if secret_name is not None and value is not None:
        if isinstance(value, str) and value.startswith("<redacted:"):
            return value
        return f"<redacted:{secret_name}>"
    if isinstance(value, str):
        redacted = value
        for secret in sorted(replacements, key=len, reverse=True):
            redacted = redacted.replace(secret, replacements[secret])
        return redacted
    return copy.deepcopy(value)


def _named_secret_replacements(
    value: Any,
    *,
    secret_name: str | None = None,
) -> dict[str, str]:
    replacements: dict[str, str] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_secret_name = str(key) if _is_secret_name(key) else secret_name
            if (
                child_secret_name
                and child is not None
                and not isinstance(child, dict | list | tuple)
                and str(child)
            ):
                replacements[str(child)] = f"<redacted:{child_secret_name}>"
            else:
                replacements.update(
                    _named_secret_replacements(child, secret_name=child_secret_name)
                )
    elif isinstance(value, list):
        for child in value:
            replacements.update(_named_secret_replacements(child, secret_name=secret_name))
    return replacements


def redact_unresolved_config(value: dict[str, Any]) -> dict[str, Any]:
    """Redact literal secrets and secret-valued ``oc.env`` fallbacks."""
    return cast(dict[str, Any], _redact_unresolved_value(value))


def _redact_unresolved_value(
    value: Any,
    *,
    secret_name: str | None = None,
) -> Any:
    if isinstance(value, dict):
        return {
            key: _redact_unresolved_value(
                child,
                secret_name=str(key) if _is_secret_name(key) else secret_name,
            )
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_redact_unresolved_value(child, secret_name=secret_name) for child in value]
    if not isinstance(value, str):
        return f"<redacted:{secret_name}>" if secret_name and value is not None else value

    if "${oc.env:" in value:
        if _SAFE_ENV_REFERENCE_RE.fullmatch(value):
            return value
        return f"<redacted:{secret_name or 'oc.env-fallback'}>"
    if secret_name is None:
        return value
    return f"<redacted:{secret_name}>"
