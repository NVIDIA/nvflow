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
"""Parse and resolve stable workflow selectors within projects."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

from nvflow.project.context import ProjectContext
from nvflow.project.resolver import ProjectResolutionError, ProjectResolver

_PROJECT_IDENTITY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_MANIFEST_FILENAMES = ("project.yaml", "project.yml")
_WORKFLOW_SUFFIXES = (".yaml", ".yml")


@dataclass(frozen=True)
class WorkflowSelector:
    """Canonical ``project:relative/workflow`` identity."""

    project: str
    relative_path: PurePosixPath

    @classmethod
    def parse(cls, value: str) -> WorkflowSelector:
        """Parse and validate a user-supplied workflow selector."""
        if value != value.strip():
            raise ProjectResolutionError("Workflow selector cannot start or end with whitespace")
        project, separator, workflow = value.partition(":")
        if not separator or not project or not workflow:
            raise ProjectResolutionError(
                "Workflow selector must use 'project:relative/workflow' syntax"
            )
        if not _PROJECT_IDENTITY_RE.fullmatch(project):
            raise ProjectResolutionError(
                f"Invalid project identity in workflow selector: {project!r}"
            )
        if ":" in workflow or "\\" in workflow:
            raise ProjectResolutionError(
                f"Workflow selector path must use relative POSIX syntax: {workflow!r}"
            )

        relative = PurePosixPath(workflow)
        if relative.is_absolute() or workflow in {".", ".."} or ".." in relative.parts:
            raise ProjectResolutionError(
                f"Workflow selector must stay beneath the project workflows directory: {value!r}"
            )
        if relative.suffix in _WORKFLOW_SUFFIXES:
            relative = relative.with_suffix("")
        if not relative.parts:
            raise ProjectResolutionError(f"Workflow selector has no workflow path: {value!r}")
        return cls(project=project, relative_path=relative)

    def __str__(self) -> str:
        return f"{self.project}:{self.relative_path.as_posix()}"


def resolve_project_directory(
    resolver: ProjectResolver,
    project_dir: str | Path,
    *,
    expected_project: str | None = None,
) -> ProjectContext:
    """Resolve one explicitly selected project directory."""
    root = Path(project_dir).expanduser().resolve()
    manifests = [root / name for name in _MANIFEST_FILENAMES if (root / name).is_file()]
    if not manifests:
        raise ProjectResolutionError(f"Project manifest not found in {root}")
    if len(manifests) > 1:
        raise ProjectResolutionError(f"Project {root} contains both project.yaml and project.yml")

    context = resolver.resolve(
        manifests[0],
        project_dir=root,
    )
    if expected_project is not None and context.identity != expected_project:
        raise ProjectResolutionError(
            f"Workflow selector names project {expected_project!r}, but {manifests[0]} "
            f"declares {context.identity!r}"
        )
    return context


def resolve_workflow_selector(
    value: str,
    resolver: ProjectResolver,
    *,
    project_dir: str | Path | None = None,
) -> ProjectContext:
    """Resolve a selector to a validated project context and workflow path."""
    selector = WorkflowSelector.parse(value)
    context = (
        resolve_project_directory(
            resolver,
            project_dir,
            expected_project=selector.project,
        )
        if project_dir is not None
        else resolver.resolve_named(selector.project)
    )
    return resolve_workflow_in_project(selector, context)


def resolve_workflow_in_project(
    selector: WorkflowSelector,
    context: ProjectContext,
) -> ProjectContext:
    """Resolve one parsed selector within an already validated project."""
    if selector.project != context.identity:
        raise ProjectResolutionError(
            f"Workflow selector names project {selector.project!r}, but the selected "
            f"project is {context.identity!r}"
        )
    workflows_root = context.workflows_root.resolve()
    relative = Path(*selector.relative_path.parts)
    candidates = [
        candidate.resolve()
        for suffix in _WORKFLOW_SUFFIXES
        if (candidate := workflows_root / f"{relative}{suffix}").is_file()
    ]
    if not candidates:
        raise ProjectResolutionError(
            f"Workflow {selector} was not found beneath {workflows_root}. "
            f"Run 'nvflow list-workflows --project {selector.project}' to discover selectors."
        )
    if len(candidates) > 1:
        raise ProjectResolutionError(
            f"Workflow selector {selector} is ambiguous: "
            + ", ".join(str(candidate) for candidate in candidates)
        )
    try:
        candidates[0].relative_to(workflows_root)
    except ValueError as exc:
        raise ProjectResolutionError(
            f"Workflow selector {selector} resolves outside {workflows_root}"
        ) from exc
    return replace(context, config_path=candidates[0])
