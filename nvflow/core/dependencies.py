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
"""Shared dependency-name resolution for planning and submission."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Any


def resolve_dependency_names(
    dependencies: Sequence[str],
    stage_configs: Mapping[str, Mapping[str, Any]],
    get_expname: Callable[[str, dict[str, Any]], str],
    *,
    environment: Sequence[str] | None = None,
    selected_stages: Collection[str] | None = None,
) -> list[str]:
    """Resolve only dependencies submitted by the current invocation."""
    names: list[str] = []
    for dependency in dependencies:
        if selected_stages is not None and dependency not in selected_stages:
            continue
        dependency_config = dict(stage_configs[dependency])
        base = get_expname(dependency, dependency_config)
        environments = dependency_config.get("environments", {})
        if not environments:
            names.append(base)
            continue
        filter_field = dependency_config.get("environment_execution_filter")
        if filter_field is not None and (not isinstance(filter_field, str) or not filter_field):
            raise ValueError(
                f"stage {dependency!r}.environment_execution_filter must be a non-empty field name"
            )
        enabled_environments = {
            name: config
            for name, config in environments.items()
            if not isinstance(config, Mapping) or config.get("enabled", True)
            if not filter_field or not isinstance(config, Mapping) or config.get(filter_field)
        }
        selected_environments = (
            [name for name in environment if name in enabled_environments]
            if environment
            else list(enabled_environments)
        )
        mode = dependency_config.get(
            "environment_execution",
            "per_environment",
        )
        if mode == "combined":
            if selected_environments:
                names.append(f"{base}-{'+'.join(selected_environments)}")
        elif mode == "per_environment":
            names.extend(f"{base}-{name}" for name in selected_environments)
        else:
            raise ValueError(
                f"stage {dependency!r}.environment_execution must be "
                "'per_environment' or 'combined'"
            )
    return names
