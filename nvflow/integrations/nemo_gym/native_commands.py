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
"""Shell-safe rendering for native NeMo Gym CLI leaf commands."""

from __future__ import annotations

import json
import os
import re
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

CommandValue: TypeAlias = (  # noqa: UP040 - mypy lacks PEP 695 support
    str | int | float | bool | None | os.PathLike[str]
)

_SHELL_VARIABLE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_HYDRA_OVERRIDE_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*\Z")


@dataclass(frozen=True, slots=True)
class ShellVariable:
    """A trusted bare shell variable supplied by NVFlow's job wrapper."""

    name: str

    def __post_init__(self) -> None:
        if not _SHELL_VARIABLE_NAME.fullmatch(self.name):
            raise ValueError(f"invalid shell variable name: {self.name!r}")


@dataclass(frozen=True, slots=True)
class HydraExpression:
    """A trusted, already-serialized Hydra collection expression."""

    value: str

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("Hydra expression cannot be empty")


ArgumentValue: TypeAlias = CommandValue | ShellVariable  # noqa: UP040
OverrideValue: TypeAlias = (  # noqa: UP040
    CommandValue | ShellVariable | HydraExpression
)
Overrides: TypeAlias = (  # noqa: UP040
    Mapping[str, OverrideValue] | Sequence[tuple[str, OverrideValue]]
)


def shell_variable(name: str) -> ShellVariable:
    """Return a validated reference expanded by the worker shell."""
    return ShellVariable(name)


def hydra_expression(value: str) -> HydraExpression:
    """Mark a generated Hydra collection expression for raw parsing."""
    return HydraExpression(value)


def _literal(value: CommandValue) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, os.PathLike):
        resolved = os.fspath(value)
        if not isinstance(resolved, str):
            raise TypeError("path-like command values must resolve to str")
        return resolved
    if isinstance(value, str | int | float):
        return str(value)
    raise TypeError(f"unsupported command value: {type(value).__name__}")


def _argument(value: ArgumentValue) -> str:
    if isinstance(value, ShellVariable):
        return f'"${{{value.name}}}"'
    return shlex.quote(_literal(value))


def _override(key: str, value: OverrideValue) -> str:
    if not _HYDRA_OVERRIDE_KEY.fullmatch(key):
        raise ValueError(f"invalid Hydra override key: {key!r}")
    if isinstance(value, ShellVariable):
        return f'"+{key}=${{{value.name}}}"'
    if isinstance(value, HydraExpression):
        literal = value.value
    elif isinstance(value, str | os.PathLike):
        literal = json.dumps(os.fspath(value), ensure_ascii=False)
    else:
        literal = _literal(value)
    return shlex.quote(f"+{key}={literal}")


def render_gym_command(
    *command: str,
    args: Sequence[ArgumentValue] = (),
    config_paths: Sequence[str | os.PathLike[str]] = (),
    overrides: Overrides = (),
) -> str:
    """Render one Gym command without duplicating Gym's option schema."""
    if not command or command[0] != "gym" or any(not token for token in command):
        raise ValueError("command must start with non-empty 'gym' argv tokens")

    paths = tuple(os.fspath(path) for path in config_paths)
    if any(not isinstance(path, str) or not path for path in paths):
        raise ValueError("Gym config paths must be non-empty strings")
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate Gym config paths are not allowed")

    items = tuple(sorted(overrides.items())) if isinstance(overrides, Mapping) else tuple(overrides)
    keys = [key for key, _ in items]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate Hydra override keys are not allowed")

    argv = [shlex.quote(token) for token in command]
    for path in paths:
        argv.extend(("--config", shlex.quote(path)))
    argv.extend(_argument(value) for value in args)
    argv.extend(_override(key, value) for key, value in items)
    return " ".join(argv)


__all__ = [
    "ArgumentValue",
    "CommandValue",
    "HydraExpression",
    "OverrideValue",
    "Overrides",
    "ShellVariable",
    "hydra_expression",
    "render_gym_command",
    "shell_variable",
]
