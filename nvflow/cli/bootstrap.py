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
"""Lightweight project handoff before loading the full Typer CLI."""

from __future__ import annotations

import sys
from collections.abc import Sequence

_WORKFLOW_CODE_COMMANDS = {"plan", "run", "stage-info", "validate"}


def _option(arguments: Sequence[str], name: str, short: str | None = None) -> str | None:
    for index, value in enumerate(arguments):
        if value == name or (short is not None and value == short):
            return arguments[index + 1] if index + 1 < len(arguments) else None
        if value.startswith(f"{name}="):
            return value.split("=", 1)[1]
    return None


def _options(arguments: Sequence[str], name: str) -> tuple[str, ...]:
    values = []
    for index, value in enumerate(arguments):
        if value == name and index + 1 < len(arguments):
            values.append(arguments[index + 1])
        elif value.startswith(f"{name}="):
            values.append(value.split("=", 1)[1])
    return tuple(values)


def _handoff(arguments: Sequence[str]) -> None:
    if not arguments or arguments[0] not in _WORKFLOW_CODE_COMMANDS:
        return
    workflow = _option(arguments, "--workflow", "-w")
    if not workflow:
        return

    try:
        from nvflow.cli.project_environment import handoff_to_project_environment
        from nvflow.project import ProjectResolver, resolve_workflow_selector

        context = resolve_workflow_selector(
            workflow,
            ProjectResolver(search_paths=_options(arguments, "--project-path")),
            project_dir=_option(arguments, "--project-dir"),
        )
        handoff_to_project_environment(
            context,
            argv=arguments,
            status=lambda message: print(f"▶ {message}", file=sys.stderr),
            prepare=False,
        )
    except Exception:
        # The full CLI owns validation, redaction, JSON errors, and help text.
        return


def main() -> None:
    arguments = sys.argv[1:]
    if arguments == ["version"]:
        from nvflow import __version__

        print(f"NVFlow version {__version__}")
        return
    _handoff(arguments)

    from nvflow.cli.main import app

    app()


if __name__ == "__main__":
    main()
