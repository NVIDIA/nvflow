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
"""Backend-neutral filesystem contracts used during target-aware planning."""

from __future__ import annotations

import fnmatch
from pathlib import Path, PurePosixPath
from typing import Protocol, runtime_checkable


@runtime_checkable
class TargetFilesystem(Protocol):
    """Read-only target filesystem view available to live planning."""

    @property
    def remote(self) -> bool: ...

    def exists(self, path: str) -> bool: ...

    def batch_exists(self, paths: list[str]) -> dict[str, bool]: ...

    def is_nonempty_file(self, path: str) -> bool: ...

    def ls(self, directory: str, pattern: str) -> list[str]: ...

    def count_lines(self, path: str) -> int: ...


@runtime_checkable
class MutableTargetFilesystem(TargetFilesystem, Protocol):
    """Target filesystem operations required by resume-time execution."""

    def rm(self, paths: list[str]) -> None: ...


class LocalTargetFilesystem:
    """Filesystem view for paths mounted identically on launcher and Ray clusters."""

    remote = False

    def mounted(self, path: str) -> bool:
        """Whether this machine has the absolute path's top-level directory.

        Without it, the workload storage is not mounted here, so a path under it
        cannot be inspected and its absence says nothing.
        """
        parts = PurePosixPath(path).parts
        return len(parts) < 2 or parts[0] != "/" or Path("/", parts[1]).exists()

    def exists(self, path: str) -> bool:
        return Path(path).exists()

    def batch_exists(self, paths: list[str]) -> dict[str, bool]:
        return {path: Path(path).exists() for path in paths}

    def is_nonempty_file(self, path: str) -> bool:
        candidate = Path(path)
        return candidate.is_file() and candidate.stat().st_size > 0

    def ls(self, directory: str, pattern: str) -> list[str]:
        root = Path(directory)
        if not root.is_dir():
            return []
        return sorted(str(path) for path in root.iterdir() if fnmatch.fnmatch(path.name, pattern))

    def count_lines(self, path: str) -> int:
        with Path(path).open("rb") as stream:
            return sum(1 for _line in stream)


__all__ = ["LocalTargetFilesystem", "MutableTargetFilesystem", "TargetFilesystem"]
