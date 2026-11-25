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
"""Typed facade over native NeMo Gym operation implementations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from nvflow.execution.filesystem import TargetFilesystem

from .contracts import (
    GymContractError,
    RewardConfig,
    RolloutConfig,
    WorkerExtensionKind,
    WorkerExtensionSpec,
)
from .operation_config import parse_reward_config, parse_rollout_config

PlannedJobs = list[dict[str, Any]]

_EXTENSION_KINDS: tuple[WorkerExtensionKind, ...] = ("report",)


@dataclass(frozen=True, slots=True)
class GymOperationContext:
    """Target and planning state shared by typed Gym operations."""

    cluster: str
    expname: str
    run_after: tuple[str, ...] = ()
    inspect_live_state: bool = False
    filesystem: TargetFilesystem | None = None


def parse_worker_extensions(
    value: object,
) -> dict[WorkerExtensionKind, WorkerExtensionSpec]:
    """Parse a closed mapping of versioned packaged-worker extensions."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise GymContractError("extensions must be a mapping")
    unknown = sorted(set(value) - set(_EXTENSION_KINDS))
    if unknown:
        raise GymContractError(f"unknown Gym extension(s): {', '.join(unknown)}")
    result: dict[WorkerExtensionKind, WorkerExtensionSpec] = {}
    for kind in _EXTENSION_KINDS:
        raw = value.get(kind)
        if raw is None:
            continue
        if not isinstance(raw, Mapping):
            raise GymContractError(f"extensions.{kind} must be a mapping")
        unknown_fields = sorted(set(raw) - {"module", "contract_version", "options"})
        if unknown_fields:
            raise GymContractError(
                f"unknown extensions.{kind} field(s): {', '.join(unknown_fields)}"
            )
        module = raw.get("module")
        if not isinstance(module, str):
            raise GymContractError(f"extensions.{kind}.module must be a non-empty string")
        options = raw.get("options", {})
        if not isinstance(options, Mapping):
            raise GymContractError(f"extensions.{kind}.options must be a mapping")
        result[kind] = WorkerExtensionSpec(
            kind=kind,
            module=module,
            contract_version=raw.get("contract_version", 1),
            options=dict(options),
        )
    return result


def normalize_rollout_operation(
    config: Mapping[str, Any],
    extensions: Mapping[WorkerExtensionKind, WorkerExtensionSpec] | None = None,
) -> RolloutConfig:
    """Normalize one selected-environment rollout configuration."""
    report = (extensions or {}).get("report")
    return parse_rollout_config(
        config,
        report_module=None if report is None else report.module,
        report_options=None if report is None else report.options,
    )


def run_rollout_operation(
    operation: RolloutConfig,
    context: GymOperationContext,
) -> PlannedJobs:
    """Plan one normalized rollout operation."""
    from .native_rollout import rollout_operation

    return rollout_operation(
        operation,
        context.cluster,
        context.expname,
        list(context.run_after) or None,
        inspect_live_state=context.inspect_live_state,
        filesystem=context.filesystem,
    )


def normalize_reward_operation(
    config: Mapping[str, Any],
    extensions: Mapping[WorkerExtensionKind, WorkerExtensionSpec] | None = None,
) -> RewardConfig:
    """Normalize one selected-environment reward-recomputation configuration."""
    report = (extensions or {}).get("report")
    return parse_reward_config(
        config,
        report_module=None if report is None else report.module,
        report_options=None if report is None else report.options,
    )


def run_reward_operation(
    operation: RewardConfig,
    context: GymOperationContext,
) -> PlannedJobs:
    """Plan one normalized reward-recomputation operation."""
    from .native_rewards import rejudge_operation

    return rejudge_operation(
        operation,
        context.cluster,
        context.expname,
        list(context.run_after) or None,
        inspect_live_state=context.inspect_live_state,
        filesystem=context.filesystem,
    )


__all__ = [
    "GymOperationContext",
    "normalize_reward_operation",
    "normalize_rollout_operation",
    "parse_worker_extensions",
    "run_reward_operation",
    "run_rollout_operation",
]
