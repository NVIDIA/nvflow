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
"""Materialize committed Git trees and pinned submodule contents."""

from __future__ import annotations

import subprocess
from pathlib import Path, PurePosixPath


class SnapshotError(RuntimeError):
    """Raised when a source tree cannot produce a deterministic snapshot."""


def materialized_files(
    root: Path,
    *,
    repository_root: Path | None = None,
    check_clean: bool = True,
) -> tuple[PurePosixPath, ...]:
    """Return tracked files, recursively replacing gitlinks with their contents."""
    root = root.resolve()
    return tuple(
        sorted(
            _tracked_files(
                root,
                prefix=PurePosixPath(),
                repository_root=repository_root,
                check_clean=check_clean,
            )
        )
    )


def _tracked_files(
    selected_root: Path,
    *,
    prefix: PurePosixPath,
    repository_root: Path | None = None,
    check_clean: bool = True,
) -> set[PurePosixPath]:
    repository_root = (
        repository_root.resolve()
        if repository_root is not None
        else Path(_git_text(selected_root, "rev-parse", "--show-toplevel").strip()).resolve()
    )
    try:
        selected = selected_root.relative_to(repository_root)
    except ValueError as exc:
        raise SnapshotError(
            f"Selected source {selected_root} is outside Git root {repository_root}"
        ) from exc
    pathspec = selected.as_posix() if selected != Path(".") else "."
    if check_clean:
        staged = subprocess.run(
            [
                "git",
                "-C",
                str(repository_root),
                "diff",
                "--cached",
                "--quiet",
                "HEAD",
                "--",
                pathspec,
            ],
            capture_output=True,
        )
        if staged.returncode == 1:
            raise SnapshotError(f"Source tree has staged changes: {selected_root}")
        if staged.returncode != 0:
            raise SnapshotError(f"Unable to inspect staged changes in {selected_root}")
        dirty = _git_text(
            repository_root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=all",
            "--",
            pathspec,
        ).strip()
        if dirty:
            raise SnapshotError(f"Source tree is dirty: {selected_root}\n{dirty}")

    entries = _git_bytes(
        repository_root,
        "ls-files",
        "--stage",
        "-z",
        "--",
        pathspec,
    ).split(b"\0")
    files: set[PurePosixPath] = set()
    for entry in entries:
        if not entry:
            continue
        metadata, separator, raw_path = entry.partition(b"\t")
        fields = metadata.decode().split()
        if not separator or len(fields) != 3:
            raise SnapshotError(f"Unable to parse Git index entry in {repository_root}: {entry!r}")
        mode, object_id, stage = fields
        if stage != "0":
            raise SnapshotError(f"Unmerged Git index entry in {repository_root}: {raw_path!r}")
        repository_path = PurePosixPath(raw_path.decode())
        relative = (
            repository_path
            if selected == Path(".")
            else repository_path.relative_to(PurePosixPath(selected.as_posix()))
        )
        if relative.is_absolute() or ".." in relative.parts:
            raise SnapshotError(f"Tracked path escapes selected source {selected_root}: {relative}")
        if mode != "160000":
            files.add(prefix / relative)
            continue

        submodule = repository_root.joinpath(*repository_path.parts)
        if not submodule.is_dir():
            raise SnapshotError(f"Submodule is not initialized: {submodule}")
        actual = _git_text(submodule, "rev-parse", "HEAD").strip()
        if actual != object_id:
            raise SnapshotError(
                f"Submodule {submodule} is at {actual}, expected parent gitlink {object_id}"
            )
        files.update(_tracked_files(submodule, prefix=prefix / relative))
    return files


def _git_bytes(root: Path, *args: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
        ).stdout
    except subprocess.CalledProcessError as exc:
        raise SnapshotError(f"Git command failed in {root}: {' '.join(args)}") from exc


def _git_text(root: Path, *args: str) -> str:
    return _git_bytes(root, *args).decode()
