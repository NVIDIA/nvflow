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
"""Read-only source provenance for one selected project."""

from __future__ import annotations

import hashlib
import io
import os
import stat
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from nvflow.project.manifest import ProjectManifestError
from nvflow.project.resources import materialize_project_bytes
from nvflow.project.snapshot import SnapshotError, materialized_files


@dataclass(frozen=True)
class ProjectProvenance:
    """Serializable identity of the project content selected for a run."""

    commit: str | None
    source_subpath: str
    packaged_destination: str
    content_digest: str | None
    clean: bool
    submittable: bool
    dirty_paths: tuple[str, ...] = ()
    reason: str | None = None


@dataclass(frozen=True)
class ProjectSourceSnapshot:
    """Internal Git source details plus their public provenance."""

    git_root: Path | None
    provenance: ProjectProvenance


@dataclass(frozen=True)
class _GitIdentity:
    root: Path
    commit: str
    superproject: Path | None


def inspect_project_source(
    project_root: Path,
    packaged_destination: PurePosixPath,
    *,
    strict: bool = False,
) -> ProjectSourceSnapshot:
    """Describe one project subtree without mutating Git or packaging state."""
    project_root = project_root.resolve()
    git_identity = _git_identity(project_root)
    if git_identity is None:
        message = f"Project must be inside a Git worktree: {project_root}"
        if strict:
            raise ProjectManifestError(message)
        return ProjectSourceSnapshot(
            git_root=None,
            provenance=ProjectProvenance(
                commit=None,
                source_subpath=".",
                packaged_destination=str(packaged_destination),
                content_digest=None,
                clean=False,
                submittable=False,
                reason=message,
            ),
        )
    git_root = git_identity.root
    commit = git_identity.commit

    try:
        relative = project_root.relative_to(git_root)
    except ValueError as exc:  # pragma: no cover - resolved Git roots contain cwd
        raise ProjectManifestError(f"Project root {project_root} is outside {git_root}") from exc

    dirty_paths = (
        *_dirty_paths(git_root, relative),
        *_submodule_gitlink_issues(git_identity),
    )
    materialized: tuple[PurePosixPath, ...] | None = None
    if not dirty_paths:
        try:
            candidate_files = materialized_files(
                project_root,
                repository_root=git_root,
                check_clean=False,
            )
            if candidate_files:
                materialized = candidate_files
        except SnapshotError as exc:
            dirty_paths = (f"project snapshot: {exc}",)
    if strict and dirty_paths:
        raise ProjectManifestError(
            "Project subtree must be clean before submission:\n" + "\n".join(dirty_paths)
        )

    if materialized is not None:
        content_digest = _materialized_tree_digest(
            project_root,
            materialized,
            packaged_destination=packaged_destination,
        )
    else:
        treeish = commit if relative == Path(".") else f"{commit}:{relative.as_posix()}"
        archive_result = subprocess.run(
            ["git", "-C", str(git_root), "archive", "--format=tar", treeish],
            capture_output=True,
        )
        if archive_result.returncode != 0:
            message = "Project subtree is not present in the current commit"
            if strict:
                raise ProjectManifestError(message)
            return ProjectSourceSnapshot(
                git_root=git_root,
                provenance=ProjectProvenance(
                    commit=commit,
                    source_subpath=relative.as_posix(),
                    packaged_destination=str(packaged_destination),
                    content_digest=None,
                    clean=False,
                    submittable=False,
                    dirty_paths=dirty_paths,
                    reason=message,
                ),
            )
        content_digest = _canonical_archive_digest(
            archive_result.stdout,
            packaged_destination=packaged_destination,
        )
    reason = (
        "Project subtree has uncommitted content and cannot be submitted" if dirty_paths else None
    )
    return ProjectSourceSnapshot(
        git_root=git_root,
        provenance=ProjectProvenance(
            commit=commit,
            source_subpath=relative.as_posix(),
            packaged_destination=str(packaged_destination),
            content_digest=content_digest,
            clean=not dirty_paths,
            submittable=not dirty_paths,
            dirty_paths=dirty_paths,
            reason=reason,
        ),
    )


def _canonical_archive_digest(
    archive: bytes,
    *,
    packaged_destination: PurePosixPath,
) -> str:
    digest = hashlib.sha256()
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as source:
        for member in source:
            name = PurePosixPath(member.name).as_posix()
            packaged_name = (packaged_destination / name).as_posix()
            digest.update(packaged_name.encode())
            digest.update(b"\0")
            digest.update(str(member.mode).encode())
            digest.update(b"\0")
            digest.update(member.type)
            digest.update(b"\0")
            if member.issym() or member.islnk():
                digest.update(member.linkname.encode())
            elif member.isfile():
                file_object = source.extractfile(member)
                data = file_object.read() if file_object is not None else b""
                data = materialize_project_bytes(data, packaged_destination)
                digest.update(data)
            digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def _materialized_tree_digest(
    project_root: Path,
    paths: tuple[PurePosixPath, ...],
    *,
    packaged_destination: PurePosixPath,
) -> str:
    digest = hashlib.sha256()
    for relative in paths:
        source = project_root.joinpath(*relative.parts)
        metadata = source.lstat()
        digest.update((packaged_destination / relative).as_posix().encode())
        digest.update(b"\0")
        digest.update(str(stat.S_IMODE(metadata.st_mode)).encode())
        digest.update(b"\0")
        if stat.S_ISLNK(metadata.st_mode):
            digest.update(tarfile.SYMTYPE)
            digest.update(b"\0")
            digest.update(os.readlink(source).encode())
        elif stat.S_ISREG(metadata.st_mode):
            digest.update(tarfile.REGTYPE)
            digest.update(b"\0")
            data = materialize_project_bytes(source.read_bytes(), packaged_destination)
            digest.update(data)
        else:
            raise ProjectManifestError(f"Unsupported tracked project file: {source}")
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def _git_identity(path: Path) -> _GitIdentity | None:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(path),
                "rev-parse",
                "--show-toplevel",
                "HEAD",
                "--show-superproject-working-tree",
            ],
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    lines = result.stdout.splitlines()
    if len(lines) not in {2, 3}:
        return None
    return _GitIdentity(
        root=Path(lines[0]).resolve(),
        commit=lines[1],
        superproject=Path(lines[2]).resolve() if len(lines) == 3 else None,
    )


def _dirty_paths(git_root: Path, relative: Path) -> tuple[str, ...]:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(git_root),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            relative.as_posix(),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(line for line in result.stdout.splitlines() if line)


def _submodule_gitlink_issues(identity: _GitIdentity) -> tuple[str, ...]:
    """Return gitlink mismatches through the complete superproject chain."""
    issues = []
    current = identity
    while current.superproject is not None:
        superproject = current.superproject
        try:
            relative = current.root.relative_to(superproject)
        except ValueError:
            issues.append(f"submodule {current.root} is outside superproject {superproject}")
            break
        expected = subprocess.run(
            ["git", "-C", str(superproject), "ls-tree", "HEAD", "--", relative.as_posix()],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        fields = expected.split(maxsplit=3)
        if len(fields) < 3 or fields[0] != "160000" or fields[1] != "commit":
            issues.append(f"submodule {relative} is not pinned by the parent commit")
        elif fields[2] != current.commit:
            issues.append(
                f"submodule {relative} HEAD {current.commit} "
                f"does not match parent gitlink {fields[2]}"
            )
        parent = _git_identity(superproject)
        if parent is None:
            issues.append(f"unable to inspect superproject {superproject}")
            break
        current = parent
    return tuple(issues)
