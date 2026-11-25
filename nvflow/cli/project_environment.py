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
"""Transparent CLI handoff into an isolated project environment."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

from nvflow.project import ProjectContext
from nvflow.project.environment import (
    ProjectEnvironment,
    ProjectEnvironmentError,
    ProjectEnvironmentManager,
)

DISABLE_PROJECT_ENVIRONMENT = "NVFLOW_DISABLE_PROJECT_ENVIRONMENT"


class EnvironmentManager(Protocol):
    def active_context(self, context: ProjectContext) -> bool: ...

    def check(self, context: ProjectContext) -> ProjectEnvironment | None: ...

    def ensure(self, context: ProjectContext) -> ProjectEnvironment: ...

    def active_for(self, environment: ProjectEnvironment) -> bool: ...

    def activation_environment(
        self,
        environment: ProjectEnvironment,
    ) -> Mapping[str, str]: ...


def handoff_to_project_environment(
    context: ProjectContext,
    *,
    status: Callable[[str], None] | None = None,
    argv: Sequence[str] | None = None,
    manager: EnvironmentManager | None = None,
    prepare: bool = True,
) -> None:
    """Replace this CLI process with the selected project environment."""
    root = getattr(context, "root", None)
    needs_environment = root is not None and (root / "pyproject.toml").is_file()
    if not needs_environment or os.environ.get(DISABLE_PROJECT_ENVIRONMENT) == "1":
        return

    manager = manager or ProjectEnvironmentManager(status=status)
    if manager.active_context(context):
        return
    environment = manager.ensure(context) if prepare else manager.check(context)
    if environment is None:
        return
    if manager.active_for(environment):
        return
    if not environment.nvflow.is_file():
        raise ProjectEnvironmentError(
            f"Prepared project environment has no nvflow executable: {environment.nvflow}"
        )

    arguments = list(sys.argv[1:] if argv is None else argv)
    os.execve(
        str(environment.nvflow),
        [str(environment.nvflow), *arguments],
        dict(manager.activation_environment(environment)),
    )
    raise ProjectEnvironmentError("Project environment handoff returned unexpectedly")
