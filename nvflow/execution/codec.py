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
"""Strict JSON decoding for immutable execution contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from nvflow.execution.contracts import (
    ArtifactDeclaration,
    CommandPlan,
    DependencyPlan,
    ExecutionGraph,
    ResourceRequest,
    ServicePlan,
    StageExecution,
    TaskPlan,
)
from nvflow.project.provenance import ProjectProvenance


def execution_graph_from_dict(value: Mapping[str, Any]) -> ExecutionGraph:
    """Decode a graph produced by :meth:`ExecutionGraph.to_dict`."""
    stages_value = _list(value, "stages")
    provenance_value = value.get("provenance")
    if provenance_value is None:
        provenance = None
    else:
        provenance_mapping = _mapping(provenance_value, "provenance")
        provenance = ProjectProvenance(
            commit=_optional_string(provenance_mapping, "commit"),
            source_subpath=_string(provenance_mapping, "source_subpath"),
            packaged_destination=_string(provenance_mapping, "packaged_destination"),
            content_digest=_optional_string(provenance_mapping, "content_digest"),
            clean=_bool(provenance_mapping, "clean"),
            submittable=_bool(provenance_mapping, "submittable"),
            dirty_paths=tuple(str(item) for item in _list(provenance_mapping, "dirty_paths")),
            reason=_optional_string(provenance_mapping, "reason"),
        )
    return ExecutionGraph(
        execution_version=_int(value, "execution_version"),
        project=_string(value, "project"),
        workflow=_string(value, "workflow"),
        target=_string(value, "target"),
        engine=cast(Any, _string(value, "engine")),
        target_fingerprint=_string(value, "target_fingerprint"),
        stages=tuple(_stage(item) for item in stages_value),
        provenance=provenance,
        environments=tuple(str(item) for item in _list(value, "environments")),
    )


def _stage(value: object) -> StageExecution:
    mapping = _mapping(value, "stage")
    return StageExecution(
        stage=_string(mapping, "stage"),
        tasks=tuple(_task(item) for item in _list(mapping, "tasks")),
        reason=_optional_string(mapping, "reason"),
        disposition=cast(Any, _string(mapping, "disposition")),
        implementation=(
            None
            if mapping.get("implementation") is None
            else dict(_mapping(mapping["implementation"], "implementation"))
        ),
    )


def _task(value: object) -> TaskPlan:
    mapping = _mapping(value, "task")
    return TaskPlan(
        task_id=_string(mapping, "task_id"),
        stage=_string(mapping, "stage"),
        name=_string(mapping, "name"),
        operation=_string(mapping, "operation"),
        commands=tuple(
            CommandPlan(**dict(_mapping(item, "command"))) for item in _list(mapping, "commands")
        ),
        resources=tuple(
            ResourceRequest(**dict(_mapping(item, "resource")))
            for item in _list(mapping, "resources")
        ),
        dependencies=tuple(
            DependencyPlan(**dict(_mapping(item, "dependency")))
            for item in _list(mapping, "dependencies")
        ),
        outputs=tuple(
            ArtifactDeclaration(**dict(_mapping(item, "artifact")))
            for item in _list(mapping, "outputs")
        ),
        services=tuple(
            ServicePlan(**dict(_mapping(item, "service"))) for item in _list(mapping, "services")
        ),
        parameters=dict(_mapping(mapping.get("parameters", {}), "parameters")),
    )


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return value


def _list(value: Mapping[str, Any], key: str) -> list[Any]:
    result = value.get(key, [])
    if not isinstance(result, list):
        raise ValueError(f"{key} must be a list")
    return result


def _string(value: Mapping[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str):
        raise ValueError(f"{key} must be a string")
    return result


def _optional_string(value: Mapping[str, Any], key: str) -> str | None:
    result = value.get(key)
    if result is None:
        return None
    if not isinstance(result, str):
        raise ValueError(f"{key} must be a string or null")
    return result


def _int(value: Mapping[str, Any], key: str) -> int:
    result = value.get(key)
    if isinstance(result, bool) or not isinstance(result, int):
        raise ValueError(f"{key} must be an integer")
    return result


def _bool(value: Mapping[str, Any], key: str) -> bool:
    result = value.get(key)
    if not isinstance(result, bool):
        raise ValueError(f"{key} must be a boolean")
    return result


__all__ = ["execution_graph_from_dict"]
