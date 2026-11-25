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
"""Project-resource transformation for packaged workers."""

from __future__ import annotations

import re
from pathlib import PurePosixPath

from nvflow.project.manifest import ProjectManifestError

PROJECT_URI_PREFIX = "project://"
PROJECT_URI_RE = re.compile(r"project://[A-Za-z0-9_./-]+")


def project_relative_path(value: str) -> PurePosixPath:
    """Validate and return the project-relative path from a project URI."""
    if not value.startswith(PROJECT_URI_PREFIX):
        raise ProjectManifestError(f"Expected a project URI, got {value!r}")
    raw_path = value.removeprefix(PROJECT_URI_PREFIX)
    relative = PurePosixPath(raw_path)
    if not raw_path or relative.is_absolute() or ".." in relative.parts:
        raise ProjectManifestError(f"Project URI must remain inside the project: {value}")
    return relative


def materialize_project_bytes(data: bytes, packaged_destination: PurePosixPath) -> bytes:
    """Rewrite project URIs in UTF-8 resources to their packaged worker root."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return data

    def replace(match: re.Match[str]) -> str:
        return str(packaged_destination / project_relative_path(match.group()))

    return PROJECT_URI_RE.sub(replace, text).encode("utf-8")


__all__ = ["materialize_project_bytes"]
