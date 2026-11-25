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
"""Safe resolution of workflow-local project stage implementations."""

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path

from nvflow.core.base_stage import BaseStage
from nvflow.project.context import ProjectContext
from nvflow.project.manifest import ProjectManifestError

PROJECT_STAGE_PREFIX = "project:"


def parse_project_stage_type(value: str) -> tuple[str, str]:
    """Parse ``project:python.module:ClassName`` into import coordinates."""
    if not value.startswith(PROJECT_STAGE_PREFIX):
        raise ValueError(f"project stage type must start with {PROJECT_STAGE_PREFIX!r}")
    coordinate = value.removeprefix(PROJECT_STAGE_PREFIX)
    module_name, separator, class_name = coordinate.rpartition(":")
    if (
        not separator
        or not module_name
        or not all(part.isidentifier() for part in module_name.split("."))
        or not class_name.isidentifier()
    ):
        raise ValueError(
            "project stage type must use 'project:python.module:BaseStageSubclass' syntax"
        )
    return module_name, class_name


def resolve_project_stage_type(
    context: ProjectContext,
    value: str,
) -> type[BaseStage]:
    """Import one project-owned stage class without scanning other modules."""
    module_name, class_name = parse_project_stage_type(value)
    source_root = context.launcher_python_root
    parts = module_name.split(".")
    for index in range(1, len(parts) + 1):
        existing = sys.modules.get(".".join(parts[:index]))
        if existing is not None:
            _require_project_source(existing, source_root, value)

    sys.path.insert(0, str(source_root))
    try:
        module = importlib.import_module(module_name)
    finally:
        try:
            sys.path.remove(str(source_root))
        except ValueError:  # pragma: no cover - defensive against import hooks
            pass

    _require_project_source(module, source_root, value)
    stage_class = getattr(module, class_name, None)
    if not isinstance(stage_class, type) or not issubclass(stage_class, BaseStage):
        raise ProjectManifestError(f"{value!r} must resolve to a BaseStage subclass")
    _require_path_within(
        Path(inspect.getfile(stage_class)).resolve(),
        source_root,
        value,
    )
    return stage_class


def _require_project_source(module: object, source_root: Path, label: str) -> None:
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        raise ProjectManifestError(f"{label!r} resolved to a module without a source file")
    _require_path_within(Path(module_file).resolve(), source_root, label)


def _require_path_within(path: Path, source_root: Path, label: str) -> None:
    try:
        path.relative_to(source_root)
    except ValueError as exc:
        raise ProjectManifestError(
            f"{label!r} resolves outside project python_source_root {source_root}: {path}"
        ) from exc


__all__ = [
    "PROJECT_STAGE_PREFIX",
    "parse_project_stage_type",
    "resolve_project_stage_type",
]
