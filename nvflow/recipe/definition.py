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
"""Pure manifest contracts for project-scoped recipe DAGs."""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

_IDENTITY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_NODE_FIELDS = frozenset({"depends_on", "optional"})


class RecipeDefinitionError(ValueError):
    """Raised when an inline recipe definition is malformed."""


@dataclass(frozen=True)
class RecipeSelector:
    """Canonical ``project:recipe`` identity."""

    project: str
    recipe: str

    @classmethod
    def parse(cls, value: str) -> RecipeSelector:
        """Parse and validate a project-qualified recipe selector."""
        if value != value.strip():
            raise RecipeDefinitionError("Recipe selector cannot start or end with whitespace")
        project, separator, recipe = value.partition(":")
        if not separator or not _valid_identity(project) or not _valid_identity(recipe):
            raise RecipeDefinitionError("Recipe selector must use 'project:recipe' syntax")
        return cls(project=project, recipe=recipe)

    def __str__(self) -> str:
        return f"{self.project}:{self.recipe}"


@dataclass(frozen=True)
class RecipeWorkflowNode:
    """One workflow configuration referenced by a recipe DAG."""

    selector: str
    depends_on: tuple[str, ...] = ()
    optional: bool = False

    @classmethod
    def from_mapping(
        cls,
        selector: str,
        data: Any,
        *,
        location: str,
    ) -> RecipeWorkflowNode:
        """Build one node from its manifest mapping."""
        _validate_workflow_path(selector, location=f"{location}.{selector}")
        if not isinstance(data, dict):
            raise RecipeDefinitionError(
                f"Recipe workflow node {location}.{selector} must be a mapping"
            )
        unknown = sorted(set(data) - _NODE_FIELDS)
        if unknown:
            raise RecipeDefinitionError(
                f"Recipe workflow node {location}.{selector} has unknown field(s): "
                + ", ".join(unknown)
            )

        optional = data.get("optional", False)
        if not isinstance(optional, bool):
            raise RecipeDefinitionError(f"optional in {location}.{selector} must be a boolean")

        depends_on = _string_list(
            data.get("depends_on", []),
            field="depends_on",
            location=f"{location}.{selector}",
        )
        for dependency in depends_on:
            _validate_workflow_path(
                dependency,
                location=f"{location}.{selector}.depends_on",
            )

        return cls(
            selector=selector,
            depends_on=depends_on,
            optional=optional,
        )


@dataclass(frozen=True)
class RecipeDefinition:
    """A named, project-scoped DAG of reusable workflow configurations."""

    name: str
    workflows: tuple[RecipeWorkflowNode, ...]
    description: str | None = None

    @classmethod
    def from_mapping(
        cls,
        name: str,
        data: Any,
        *,
        location: str,
    ) -> RecipeDefinition:
        """Build and structurally validate one inline recipe definition."""
        _require_identity(name, f"recipe key in {location}")
        if not isinstance(data, dict):
            raise RecipeDefinitionError(f"Recipe {location}.{name} must be a mapping")
        unknown = sorted(set(data) - {"description", "workflows"})
        if unknown:
            raise RecipeDefinitionError(
                f"Recipe {location}.{name} has unknown field(s): " + ", ".join(unknown)
            )

        description = data.get("description")
        if description is not None and (not isinstance(description, str) or not description):
            raise RecipeDefinitionError(
                f"description in {location}.{name} must be a non-empty string"
            )

        workflows = data.get("workflows")
        if not isinstance(workflows, dict) or not workflows:
            raise RecipeDefinitionError(
                f"workflows in {location}.{name} must be a non-empty mapping"
            )
        nodes = tuple(
            RecipeWorkflowNode.from_mapping(
                workflow_selector,
                node_data,
                location=f"{location}.{name}.workflows",
            )
            for workflow_selector, node_data in workflows.items()
        )
        definition = cls(name=name, description=description, workflows=nodes)
        definition.validate_graph()
        return definition

    @property
    def workflow_map(self) -> dict[str, RecipeWorkflowNode]:
        """Return nodes by project-relative workflow selector."""
        return {node.selector: node for node in self.workflows}

    def validate_graph(self) -> None:
        """Reject dangling, self-referential, and cyclic dependencies."""
        nodes = self.workflow_map
        for node in self.workflows:
            for dependency in node.depends_on:
                if dependency == node.selector:
                    raise RecipeDefinitionError(
                        f"Recipe {self.name!r} workflow {node.selector!r} depends on itself"
                    )
                if dependency not in nodes:
                    raise RecipeDefinitionError(
                        f"Recipe {self.name!r} workflow {node.selector!r} depends on "
                        f"unknown workflow selector {dependency!r}"
                    )
        self.topological_workflows()

    def topological_workflows(self) -> tuple[RecipeWorkflowNode, ...]:
        """Return a deterministic dependency order using YAML order as tie-breaker."""
        nodes = self.workflow_map
        indegree = dict.fromkeys(nodes, 0)
        dependents: dict[str, list[str]] = {name: [] for name in nodes}
        for node in self.workflows:
            indegree[node.selector] = len(node.depends_on)
            for dependency in node.depends_on:
                dependents[dependency].append(node.selector)

        ready = deque(node.selector for node in self.workflows if indegree[node.selector] == 0)
        ordered: list[RecipeWorkflowNode] = []
        while ready:
            name = ready.popleft()
            ordered.append(nodes[name])
            for dependent in dependents[name]:
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    ready.append(dependent)

        if len(ordered) != len(nodes):
            cyclic = [name for name in nodes if indegree[name] > 0]
            raise RecipeDefinitionError(
                f"Recipe {self.name!r} contains a dependency cycle involving: " + ", ".join(cyclic)
            )
        return tuple(ordered)


def parse_recipe_definitions(
    data: Any,
    *,
    location: str,
) -> tuple[RecipeDefinition, ...]:
    """Parse the optional top-level recipe map from a project manifest."""
    if data is None:
        return ()
    if not isinstance(data, dict):
        raise RecipeDefinitionError(f"recipes in {location} must be a mapping")
    return tuple(
        RecipeDefinition.from_mapping(name, value, location=f"{location}.recipes")
        for name, value in data.items()
    )


def _string_list(value: Any, *, field: str, location: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise RecipeDefinitionError(f"{field} in {location} must be a list of non-empty strings")
    if len(value) != len(set(value)):
        raise RecipeDefinitionError(f"{field} in {location} must not contain duplicates")
    return tuple(value)


def _validate_workflow_path(value: Any, *, location: str) -> None:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise RecipeDefinitionError(f"{location} must be a relative POSIX workflow selector path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value in {"", ".", ".."}
        or ".." in path.parts
        or path.as_posix() != value
        or path.suffix in {".yaml", ".yml"}
    ):
        raise RecipeDefinitionError(f"{location} must be an extension-free path beneath workflows/")


def _valid_identity(value: str) -> bool:
    return bool(_IDENTITY_RE.fullmatch(value))


def _require_identity(value: Any, label: str) -> None:
    if not isinstance(value, str) or not _valid_identity(value):
        raise RecipeDefinitionError(
            f"{label} {value!r} must contain only letters, numbers, '.', '_' or '-'"
        )
