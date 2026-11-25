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
"""Deterministic discovery for built-in and external projects."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

from nvflow import __version__
from nvflow.project.context import ProjectContext, ProjectOrigin
from nvflow.project.manifest import ProjectManifest, ProjectManifestError

_MANIFEST_FILENAMES = ("project.yaml", "project.yml")
_PACKAGED_PROJECTS_ROOT = PurePosixPath("/nemo_run/code/projects")


class ProjectResolutionError(ProjectManifestError):
    """Raised when a project cannot be resolved unambiguously."""


class ProjectResolver:
    """Resolve one workflow config to its owning project."""

    def __init__(
        self,
        reference_root: Path | None = None,
        search_paths: Sequence[str | Path] = (),
    ) -> None:
        repository_root = Path(__file__).resolve().parents[2]
        if reference_root is not None:
            self.reference_root = reference_root.resolve()
        elif (repository_root / ".git").exists():
            self.reference_root = (repository_root / "projects").resolve()
        else:
            self.reference_root = repository_root / ".nvflow-reference-projects-unavailable"
        configured = [
            value for value in os.environ.get("NVFLOW_PROJECT_PATH", "").split(os.pathsep) if value
        ]
        self.search_paths = tuple(
            Path(path).expanduser().resolve() for path in (*search_paths, *configured)
        )

    def resolve(
        self,
        config_path: str | Path,
        project_dir: str | Path | None = None,
    ) -> ProjectContext:
        """Resolve *config_path* using an explicit root or bounded discovery."""
        resolved_config = Path(config_path).resolve()
        if project_dir is not None:
            root = Path(project_dir).resolve()
            manifest_path = _manifest_in(root)
        else:
            manifest_path = self._discover_manifest(resolved_config)
            if manifest_path is None:
                raise ProjectResolutionError(
                    f"Unable to discover a project manifest from {resolved_config}"
                )
            root = manifest_path.parent.resolve()

        manifest = ProjectManifest.load(manifest_path)
        origin = self._origin(root, manifest.project)
        self._reject_nested_manifests(root, manifest_path)
        try:
            resolved_config.relative_to(root)
        except ValueError as exc:
            raise ProjectResolutionError(
                f"Workflow config {resolved_config} is outside project root {root}"
            ) from exc

        manifest.check_nvflow_version(__version__)
        return ProjectContext(
            manifest=manifest,
            root=root,
            manifest_path=manifest_path,
            config_path=resolved_config,
            packaged_root=_PACKAGED_PROJECTS_ROOT / manifest.project,
            origin=origin,
        )

    def resolve_reference(
        self,
        project_identity: str,
        config_path: str | Path | None = None,
    ) -> ProjectContext:
        """Resolve a maintained project from a source checkout."""
        return self._resolve_known(
            root=self.reference_root / project_identity,
            project_identity=project_identity,
            config_path=config_path,
            origin=ProjectOrigin.REFERENCE,
        )

    def resolve_named(
        self,
        project_identity: str,
        config_path: str | Path | None = None,
    ) -> ProjectContext:
        """Resolve a named user project before falling back to maintained projects."""
        root = self._search_root(project_identity) or self._current_project_root(project_identity)
        if root is not None:
            return self._resolve_known(
                root=root,
                project_identity=project_identity,
                config_path=config_path,
                origin=ProjectOrigin.EXTERNAL,
            )
        return self.resolve_reference(project_identity, config_path=config_path)

    def project_names(self) -> list[str]:
        """List configured and maintained project identities without importing code."""
        names = set()
        roots = [self.reference_root, *self.search_paths]
        current = self._current_project_root()
        if current is not None:
            roots.append(current)
        for root in roots:
            if not root.is_dir():
                continue
            candidates = [root, *(child for child in root.iterdir() if child.is_dir())]
            for candidate in candidates:
                manifest_path = _optional_manifest_in(candidate)
                if manifest_path is not None:
                    names.add(ProjectManifest.load(manifest_path).project)
        return sorted(names)

    def resolve_current(self) -> ProjectContext:
        """Resolve the project owning the current working directory."""
        root = self._current_project_root()
        if root is None:
            raise ProjectResolutionError(
                f"Current directory is not inside an NVFlow project: {Path.cwd()}"
            )
        manifest = ProjectManifest.load(_manifest_in(root))
        return self._resolve_known(
            root=root,
            project_identity=manifest.project,
            config_path=None,
            origin=self._origin(root, manifest.project),
        )

    def _current_project_root(self, project_identity: str | None = None) -> Path | None:
        manifest_path = self._discover_manifest(Path.cwd() / ".nvflow-project-selection")
        if manifest_path is None:
            return None
        manifest = ProjectManifest.load(manifest_path)
        if project_identity is not None and manifest.project != project_identity:
            return None
        return manifest_path.parent.resolve()

    def _resolve_known(
        self,
        root: Path,
        project_identity: str,
        config_path: str | Path | None,
        origin: ProjectOrigin,
    ) -> ProjectContext:
        root = root.resolve()
        manifest_path = _optional_manifest_in(root)
        if manifest_path is None:
            raise ProjectResolutionError(f"Maintained project not found: {project_identity}")
        manifest = ProjectManifest.load(manifest_path)
        if manifest.project != project_identity:
            raise ProjectResolutionError(
                f"Project directory {root} contains identity {manifest.project!r}, "
                f"expected {project_identity!r}"
            )
        self._reject_nested_manifests(root, manifest_path)
        manifest.check_nvflow_version(__version__)
        effective_config = Path(config_path).resolve() if config_path else manifest_path
        return ProjectContext(
            manifest=manifest,
            root=root,
            manifest_path=manifest_path,
            config_path=effective_config,
            packaged_root=_PACKAGED_PROJECTS_ROOT / project_identity,
            origin=origin,
        )

    def _search_root(self, project_identity: str) -> Path | None:
        for search_path in self.search_paths:
            candidates = []
            roots = [search_path]
            if search_path.is_dir():
                roots.extend(sorted(child for child in search_path.iterdir() if child.is_dir()))
            for candidate in roots:
                manifest_path = _optional_manifest_in(candidate)
                if manifest_path is None:
                    continue
                manifest = ProjectManifest.load(manifest_path)
                if manifest.project == project_identity:
                    candidates.append(candidate.resolve())
            candidates = list(dict.fromkeys(candidates))
            if len(candidates) > 1:
                raise ProjectResolutionError(
                    f"Project {project_identity!r} is ambiguous under search path "
                    f"{search_path}: {', '.join(str(path) for path in candidates)}"
                )
            if candidates:
                return candidates[0]
        return None

    def _discover_manifest(self, config_path: Path) -> Path | None:
        start = config_path.parent
        boundary = _manifest_git_boundary(start) or Path(start.anchor)
        found: list[Path] = []
        current = start
        while True:
            manifest = _optional_manifest_in(current)
            if manifest is not None:
                found.append(manifest)
            if current == boundary or current.parent == current:
                break
            current = current.parent

        if len(found) > 1:
            joined = ", ".join(str(path) for path in found)
            raise ProjectResolutionError(
                f"Nested project manifests are not supported for {config_path}: {joined}"
            )
        if not found:
            return None

        return found[0]

    def _reject_nested_manifests(self, root: Path, selected: Path) -> None:
        nested = sorted(
            path
            for filename in _MANIFEST_FILENAMES
            for path in root.rglob(filename)
            if path.resolve() != selected.resolve()
        )
        if nested:
            raise ProjectResolutionError(
                f"Project {root} contains nested manifest(s): "
                + ", ".join(str(path) for path in nested)
            )

    def _origin(self, root: Path, project_identity: str) -> ProjectOrigin:
        if root == (self.reference_root / project_identity).resolve():
            return ProjectOrigin.REFERENCE
        return ProjectOrigin.EXTERNAL


def _manifest_in(root: Path) -> Path:
    manifest = _optional_manifest_in(root)
    if manifest is None:
        raise ProjectResolutionError(f"Project manifest not found in {root}")
    return manifest


def _optional_manifest_in(root: Path) -> Path | None:
    matches = [root / filename for filename in _MANIFEST_FILENAMES if (root / filename).is_file()]
    if len(matches) > 1:
        raise ProjectResolutionError(f"Project {root} contains both project.yaml and project.yml")
    return matches[0] if matches else None


def _manifest_git_boundary(start: Path) -> Path | None:
    current = start
    while True:
        if (current / ".git").exists():
            return current
        if current.parent == current:
            return None
        current = current.parent
