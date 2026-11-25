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
"""Target-independent planning for project-scoped recipe DAGs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from nvflow.project import ProjectProvenance, ProjectResolver
from nvflow.recipe.resolver import ResolvedRecipe, resolve_recipe


@dataclass(frozen=True)
class RecipeGraphNodePlan:
    """One resolved workflow node in a recipe graph plan."""

    selector: str
    workflow_type: str
    depends_on: tuple[str, ...]
    optional: bool


@dataclass(frozen=True)
class RecipeGraphPlan:
    """Immutable, target-independent plan for one recipe DAG."""

    plan_version: int
    project: str
    recipe: str
    description: str | None
    workflows: tuple[RecipeGraphNodePlan, ...]
    provenance: ProjectProvenance

    @property
    def digest(self) -> str:
        """Return a deterministic digest of the graph and project source."""
        digest_value = self.to_dict(include_digest=False)
        provenance = digest_value["provenance"]
        assert isinstance(provenance, dict)
        # Dirty-tree diagnostics describe local eligibility, not graph identity.
        digest_value["provenance"] = {
            key: provenance[key]
            for key in (
                "commit",
                "source_subpath",
                "packaged_destination",
                "content_digest",
            )
        }
        payload = json.dumps(
            digest_value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return f"sha256:{hashlib.sha256(payload).hexdigest()}"

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        """Convert the graph plan to a JSON-compatible mapping."""
        value = asdict(self)
        value["kind"] = "recipe"
        if include_digest:
            value["digest"] = self.digest
        return value


def build_recipe_graph_plan(resolved: ResolvedRecipe) -> RecipeGraphPlan:
    """Build a graph plan from an already validated recipe."""
    workflows = tuple(
        RecipeGraphNodePlan(
            selector=str(workflow.selector),
            workflow_type=workflow.workflow_type,
            depends_on=workflow.dependency_selectors,
            optional=workflow.node.optional,
        )
        for workflow in resolved.topological_workflows()
    )
    return RecipeGraphPlan(
        plan_version=2,
        project=resolved.selector.project,
        recipe=resolved.selector.recipe,
        description=resolved.definition.description,
        workflows=workflows,
        provenance=resolved.context.source_provenance(strict=False),
    )


def plan_recipe(
    value: str,
    resolver: ProjectResolver,
    *,
    project_dir: str | Path | None = None,
) -> RecipeGraphPlan:
    """Resolve, validate, and plan one recipe without binding a target."""
    return build_recipe_graph_plan(
        resolve_recipe(
            value,
            resolver,
            project_dir=project_dir,
            validate_resources=True,
        )
    )
