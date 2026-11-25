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
"""Release-qualified Python and Ray runtime identity."""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from importlib import metadata

from packaging.specifiers import SpecifierSet
from packaging.version import Version

PYTHON_VERSION = "3.13.14"
PYTHON_SPEC = ">=3.13.14,<3.14"
RAY_VERSION = "2.58.0"


class RuntimeProfileError(RuntimeError):
    """Launcher or supplied cluster does not match the qualified runtime."""


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    """Comparable identity reported by launcher and supplied clusters."""

    python: str
    ray: str

    def validate(self, *, location: str) -> None:
        if Version(self.python) not in SpecifierSet(PYTHON_SPEC):
            raise RuntimeProfileError(
                f"{location} requires Python {PYTHON_SPEC}, found {self.python}"
            )
        if self.ray != RAY_VERSION:
            raise RuntimeProfileError(f"{location} requires Ray {RAY_VERSION}, found {self.ray}")


def launcher_runtime_profile() -> RuntimeProfile:
    """Return the current launcher identity without importing Ray."""
    try:
        ray_version = metadata.version("ray")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeProfileError(f"Ray execution requires ray[default]=={RAY_VERSION}") from exc
    return RuntimeProfile(
        python=platform.python_version(),
        ray=ray_version,
    )


def validate_launcher_runtime() -> RuntimeProfile:
    """Fail before network access when the launcher runtime is unsupported."""
    profile = launcher_runtime_profile()
    profile.validate(location="NVFlow Ray launcher")
    if not ((3, 13, 14) <= sys.version_info[:3] < (3, 14, 0)):
        raise RuntimeProfileError(f"NVFlow Ray launcher requires CPython {PYTHON_SPEC}")
    return profile


__all__ = [
    "PYTHON_VERSION",
    "PYTHON_SPEC",
    "RAY_VERSION",
    "RuntimeProfile",
    "RuntimeProfileError",
    "launcher_runtime_profile",
    "validate_launcher_runtime",
]
