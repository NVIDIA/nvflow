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
"""Resolved project identity and launcher/worker path behavior."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from nvflow.core.code_root import CODE_ROOT_TOKEN
from nvflow.project.manifest import ProjectManifest, ProjectManifestError
from nvflow.project.resources import PROJECT_URI_RE, project_relative_path

if TYPE_CHECKING:
    from nvflow.project.provenance import ProjectProvenance


class ProjectOrigin(StrEnum):
    """Where a resolved project is owned."""

    REFERENCE = "reference"
    EXTERNAL = "external"


@dataclass(frozen=True)
class ProjectContext:
    """A validated project resolved for one workflow invocation."""

    manifest: ProjectManifest
    root: Path
    manifest_path: Path
    config_path: Path
    packaged_root: PurePosixPath
    origin: ProjectOrigin

    def __post_init__(self) -> None:
        root = self.root.resolve()
        manifest_path = self.manifest_path.resolve()
        config_path = self.config_path.resolve()
        _require_within(manifest_path, root, "manifest")

        source_root = (root / self.manifest.python_source_root).resolve()
        _require_within(source_root, root, "python_source_root")
        if not source_root.is_dir():
            raise ProjectManifestError(
                f"python_source_root does not exist or is not a directory: {source_root}"
            )

        object.__setattr__(self, "root", root)
        object.__setattr__(self, "manifest_path", manifest_path)
        object.__setattr__(self, "config_path", config_path)

    @property
    def identity(self) -> str:
        """Project identity used for provenance and external execution names."""
        return self.manifest.project

    @property
    def is_external(self) -> bool:
        """Whether the project is user-owned outside maintained source roots."""
        return self.origin is ProjectOrigin.EXTERNAL

    @property
    def launcher_python_root(self) -> Path:
        """Python source root used when importing project stage classes."""
        return (self.root / self.manifest.python_source_root).resolve()

    @property
    def worker_python_root(self) -> str:
        """Equivalent Python source root in the packaged worker snapshot."""
        relative = PurePosixPath(self.manifest.python_source_root)
        return str(self.packaged_root / relative)

    @property
    def execution_root(self) -> PurePosixPath:
        """Backend-neutral project root resolved by each worker entrypoint."""
        return PurePosixPath(CODE_ROOT_TOKEN) / "projects" / self.identity

    @property
    def workflows_root(self) -> Path:
        """Directory containing workflow YAML for this project."""
        return self.root / "workflows"

    def launcher_path(self, value: str) -> Path:
        """Resolve a ``project://`` URI for launcher-side reads and validation."""
        relative = project_relative_path(value)
        candidate = (self.root / Path(*relative.parts)).resolve()
        _require_within(candidate, self.root, value)
        if not candidate.exists():
            raise ProjectManifestError(f"Project resource does not exist: {value} ({candidate})")
        return candidate

    def worker_path(self, value: str) -> str:
        """Resolve a ``project://`` URI for a rendered worker command."""
        relative = project_relative_path(value)
        return str(self.packaged_root / relative)

    def launcher_equivalent(self, value: str) -> Path:
        """Map a packaged worker path back to the selected launcher project."""
        worker_path = PurePosixPath(value)
        try:
            relative = worker_path.relative_to(self.packaged_root)
        except ValueError:
            return Path(value)
        candidate = (self.root / Path(*relative.parts)).resolve()
        _require_within(candidate, self.root, value)
        return candidate

    def source_provenance(self, *, strict: bool = False) -> ProjectProvenance:
        """Return read-only Git provenance for planning or submission."""
        from nvflow.project.provenance import inspect_project_source

        return inspect_project_source(
            self.root,
            self.packaged_root,
            strict=strict,
        ).provenance

    def validate_resources(self, value: Any) -> None:
        """Validate every project URI nested in a config without rewriting it."""
        if isinstance(value, str):
            for project_uri in PROJECT_URI_RE.findall(value):
                self.launcher_path(project_uri)
            return
        if isinstance(value, dict):
            for nested in value.values():
                self.validate_resources(nested)
            return
        if isinstance(value, list | tuple):
            for nested in value:
                self.validate_resources(nested)

    def render_execution_paths(self, value: Any) -> Any:
        """Map project URIs to the code-root contract used by exact commands."""
        if isinstance(value, str):
            return PROJECT_URI_RE.sub(
                lambda match: str(self.execution_root / project_relative_path(match.group())),
                value,
            )
        if isinstance(value, dict):
            return {key: self.render_execution_paths(nested) for key, nested in value.items()}
        if isinstance(value, list):
            return [self.render_execution_paths(nested) for nested in value]
        if isinstance(value, tuple):
            return tuple(self.render_execution_paths(nested) for nested in value)
        return value

    def render_launcher_paths(self, value: Any) -> Any:
        """Return a config copy with project URIs mapped to launcher paths."""
        if isinstance(value, str):
            return PROJECT_URI_RE.sub(
                lambda match: str(self.launcher_path(match.group())),
                value,
            )
        if isinstance(value, dict):
            return {key: self.render_launcher_paths(nested) for key, nested in value.items()}
        if isinstance(value, list):
            return [self.render_launcher_paths(nested) for nested in value]
        if isinstance(value, tuple):
            return tuple(self.render_launcher_paths(nested) for nested in value)
        return value


def _require_within(candidate: Path, root: Path, label: str) -> None:
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ProjectManifestError(f"{label} escapes project root {root}: {candidate}") from exc
