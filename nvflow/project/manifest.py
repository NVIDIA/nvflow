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
"""Versioned metadata for NVFlow projects."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version
from yaml.nodes import MappingNode

from nvflow.recipe.definition import (
    RecipeDefinition,
    RecipeDefinitionError,
    parse_recipe_definitions,
)

MANIFEST_VERSION = 1
_IDENTITY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_KNOWN_FIELDS = frozenset(
    {
        "manifest_version",
        "project",
        "description",
        "requires_nvflow",
        "python_source_root",
        "recipes",
    }
)


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""

    def construct_mapping(self, node: MappingNode, deep: bool = False) -> dict[Any, Any]:
        self.flatten_mapping(node)
        mapping: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                hash(key)
            except TypeError as exc:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found an unhashable key",
                    key_node.start_mark,
                ) from exc
            if key in mapping:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


class ProjectManifestError(ValueError):
    """Raised when a project manifest violates the public schema."""


@dataclass(frozen=True)
class ProjectManifest:
    """Validated version-one project metadata."""

    project: str
    manifest_version: int = MANIFEST_VERSION
    description: str | None = None
    requires_nvflow: str | None = None
    python_source_root: str = "."
    recipes: tuple[RecipeDefinition, ...] = ()

    @classmethod
    def load(cls, path: Path) -> ProjectManifest:
        """Load and validate ``project.yaml`` at *path*."""
        try:
            with path.open(encoding="utf-8") as manifest_file:
                data = yaml.load(manifest_file, Loader=_UniqueKeyLoader)
        except (OSError, yaml.YAMLError) as exc:
            raise ProjectManifestError(f"Unable to read project manifest {path}: {exc}") from exc

        if not isinstance(data, dict):
            raise ProjectManifestError(f"Project manifest {path} must contain a YAML mapping")

        unknown = sorted(set(data) - _KNOWN_FIELDS)
        if unknown:
            raise ProjectManifestError(
                f"Project manifest {path} has unknown field(s): {', '.join(unknown)}"
            )

        if "manifest_version" not in data:
            raise ProjectManifestError(
                f"Project manifest {path} must declare manifest_version: {MANIFEST_VERSION}"
            )
        version = data["manifest_version"]
        if not isinstance(version, int) or isinstance(version, bool) or version != MANIFEST_VERSION:
            raise ProjectManifestError(
                f"Project manifest {path} uses unsupported manifest_version {version!r}; "
                f"supported version: {MANIFEST_VERSION}"
            )

        project = _required_string(data, "project", path)
        if not _IDENTITY_RE.fullmatch(project):
            raise ProjectManifestError(
                f"Project identity {project!r} in {path} must contain only letters, numbers, "
                "'.', '_' or '-'"
            )

        description = _optional_string(data, "description", path)
        requires_nvflow = _optional_string(data, "requires_nvflow", path)
        python_source_root = data.get("python_source_root", ".")
        if not isinstance(python_source_root, str) or not python_source_root:
            raise ProjectManifestError(
                f"python_source_root in {path} must be a non-empty relative path"
            )
        source_path = PurePosixPath(python_source_root)
        if source_path.is_absolute() or ".." in source_path.parts:
            raise ProjectManifestError(f"python_source_root in {path} must stay inside the project")

        try:
            recipes = parse_recipe_definitions(data.get("recipes"), location=str(path))
        except RecipeDefinitionError as exc:
            raise ProjectManifestError(str(exc)) from exc

        return cls(
            project=project,
            description=description,
            requires_nvflow=requires_nvflow,
            python_source_root=python_source_root,
            recipes=recipes,
        )

    def check_nvflow_version(self, current_version: str) -> None:
        """Reject a manifest whose declared NVFlow range excludes this version."""
        if not self.requires_nvflow:
            return
        try:
            compatible = Version(current_version) in SpecifierSet(self.requires_nvflow)
        except (InvalidSpecifier, InvalidVersion) as exc:
            raise ProjectManifestError(
                f"Invalid requires_nvflow value {self.requires_nvflow!r}"
            ) from exc
        if not compatible:
            raise ProjectManifestError(
                f"Project {self.project!r} requires NVFlow {self.requires_nvflow}; "
                f"current version is {current_version}"
            )

    @property
    def recipe_map(self) -> dict[str, RecipeDefinition]:
        """Return inline recipes by project-scoped identity."""
        return {recipe.name: recipe for recipe in self.recipes}


def _required_string(data: dict[str, Any], field: str, path: Path) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value:
        raise ProjectManifestError(f"{field} in {path} must be a non-empty string")
    return value


def _optional_string(data: dict[str, Any], field: str, path: Path) -> str | None:
    value = data.get(field)
    if value is not None and (not isinstance(value, str) or not value):
        raise ProjectManifestError(f"{field} in {path} must be a non-empty string")
    return value
