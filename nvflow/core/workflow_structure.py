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
"""Dependency-light structural workflow helpers shared by CLI and runtime."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, cast

EXPANDABLE_WORKFLOW_SECTIONS = ("models", "checkpoints")
WORKFLOW_STRUCTURAL_KEYS = frozenset(
    {
        "project",
        "workflow",
        "cluster",
        "pipeline_stages",
        "stages",
        *EXPANDABLE_WORKFLOW_SECTIONS,
    }
)
_STRUCTURAL_REFERENCE_RE = re.compile(r"^\$\{([A-Za-z0-9_.-]+)\}$")


def checkpoint_eval_steps(value: Any) -> list[int | str]:
    """Normalize concrete checkpoint evaluation steps."""
    if value is None:
        return []
    if isinstance(value, int) and not isinstance(value, bool):
        return [value]
    if isinstance(value, list) and all(
        (isinstance(step, int) and not isinstance(step, bool))
        or (isinstance(step, str) and bool(step))
        for step in value
    ):
        return cast(list[int | str], value)
    raise TypeError(
        "checkpoint eval_steps must resolve to an integer or list of non-empty step names"
    )


def checkpoint_stage_name(checkpoint: str, step: int | str) -> str:
    """Return the concrete stage name for one checkpoint step."""
    return f"{checkpoint}-{step}"


def resolve_structural_reference(
    value: object,
    *,
    config: Mapping[str, Any],
    label: str,
    seen: tuple[str, ...] = (),
) -> object:
    """Resolve config-local references without evaluating runtime resolvers."""
    if isinstance(value, list):
        return [
            resolve_structural_reference(item, config=config, label=label, seen=seen)
            for item in value
        ]
    if not isinstance(value, str) or "${" not in value:
        return value
    match = _STRUCTURAL_REFERENCE_RE.fullmatch(value)
    if match is None or match.group(1).startswith("oc."):
        raise TypeError(f"{label} depends on runtime interpolation and cannot be inspected")
    reference = match.group(1)
    if reference in seen:
        raise TypeError(f"{label} contains an interpolation cycle")
    target: object = config
    for part in reference.split("."):
        if not isinstance(target, Mapping) or part not in target:
            raise TypeError(f"{label} references unknown value {reference!r}")
        target = target[part]
    return resolve_structural_reference(
        target,
        config=config,
        label=label,
        seen=(*seen, reference),
    )


def validate_unique_stage_names(stage_names: Sequence[str]) -> None:
    """Reject duplicate identities in an expanded workflow pipeline."""
    seen: set[str] = set()
    duplicates: list[str] = []
    for stage_name in stage_names:
        if stage_name in seen and stage_name not in duplicates:
            duplicates.append(stage_name)
        seen.add(stage_name)
    if duplicates:
        raise ValueError(
            "Workflow pipeline expands to duplicate stage identities: " + ", ".join(duplicates)
        )


def description_of(config: object) -> str | None:
    """Return the ``description`` a workflow or stage mapping declares, if any."""
    value = config.get("description") if isinstance(config, dict) else None
    return value if isinstance(value, str) else None


PLATFORM_STAGE_KEYS = frozenset(
    {
        "_environment",
        "container",
        "contract_version",
        "dependencies",
        "installation_command",
        "run_name",
        "stage_kwargs",
    }
)


def reject_unknown_options(config: Mapping[str, Any], known: Iterable[str], stage: str) -> None:
    """Fail on a key that neither *stage* nor the platform reads, such as a misspelling."""
    unknown = sorted(set(config) - set(known) - PLATFORM_STAGE_KEYS)
    if unknown:
        raise ValueError(f"unknown {stage} option(s): {', '.join(unknown)}")


def stage_name_sequence(value: object, *, label: str) -> tuple[str, ...]:
    """Validate and return one ordered workflow stage-name sequence."""
    if not isinstance(value, list) or any(
        not isinstance(stage, str) or not stage for stage in value
    ):
        raise TypeError(f"{label} must be a list of non-empty strings")
    result = tuple(value)
    validate_unique_stage_names(result)
    return result
