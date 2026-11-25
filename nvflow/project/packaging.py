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
"""Submission-scoped packaging for an external project."""

from __future__ import annotations

import base64
import copy
import hashlib
import io
import os
import subprocess
import tarfile
import zipfile
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path, PurePosixPath
from typing import Any

from nvflow.core.code_root import CODE_ROOT_TOKEN
from nvflow.core.platform import wheel_distribution_identity
from nvflow.project.context import ProjectContext
from nvflow.project.manifest import ProjectManifestError
from nvflow.project.provenance import (
    ProjectProvenance,
    inspect_project_source,
)
from nvflow.project.resources import materialize_project_bytes
from nvflow.project.snapshot import materialized_files

try:
    from nemo_run.core.packaging.base import Packager
except ImportError:  # pragma: no cover - lightweight clients do not package jobs

    class Packager:  # type: ignore[no-redef]
        """Fallback base so local manifest tooling does not require NeMo-Run."""


_ACTIVE_PROJECT: ContextVar[ProjectContext | None] = ContextVar(
    "nvflow_active_project",
    default=None,
)
_ACTIVE_PROVENANCE: ContextVar[ProjectProvenance | None] = ContextVar(
    "nvflow_active_project_provenance",
    default=None,
)
_ACTIVE_PLATFORM_SOURCE: ContextVar[str | None] = ContextVar(
    "nvflow_active_platform_source",
    default=None,
)
_ACTIVE_BASE_PACKAGERS: ContextVar[dict[tuple[str, ...], Any] | None] = ContextVar(
    "nvflow_active_base_packagers",
    default=None,
)


@dataclass(frozen=True)
class _PackageSource:
    root: Path
    files: tuple[Path, ...]
    identity: str
    content_digest: str | None = None


@dataclass(frozen=True, slots=True)
class WorkerBundle:
    """Deterministic source archive uploaded as a Ray Jobs working directory."""

    path: Path
    digest: str
    compressed_size: int
    uncompressed_size: int
    project_python_root: str
    project_identity: str
    platform_identity: str


def build_worker_bundle(
    context: ProjectContext,
    provenance: ProjectProvenance,
    platform_source: str,
    destination_dir: Path,
    *,
    max_compressed_bytes: int = 95 * 1024 * 1024,
    max_uncompressed_bytes: int = 2 * 1024 * 1024 * 1024,
) -> WorkerBundle:
    """Build one immutable project/platform ZIP for a supplied Ray cluster."""
    project_root = context.root.resolve()
    before = inspect_project_source(project_root, context.packaged_root, strict=True).provenance
    if (
        before.commit != provenance.commit
        or before.source_subpath != provenance.source_subpath
        or before.packaged_destination != provenance.packaged_destination
        or before.content_digest != provenance.content_digest
    ):
        raise ProjectManifestError(
            "Project source changed after planning; rebuild the execution graph"
        )

    package_root = Path(__file__).resolve().parents[1]
    platform = _platform_package_source(package_root)
    if platform.identity != platform_source:
        raise ProjectManifestError(
            "NVFlow implementation changed after planning: "
            f"planned={platform_source}, packaged={platform.identity}"
        )

    destination = PurePosixPath(str(context.packaged_root)).relative_to(
        PurePosixPath("/nemo_run/code")
    )
    entries: dict[str, tuple[bytes, int]] = {}
    # Worker images omit the nemo_skills package, so the bundle carries the
    # installed copy, as the Slurm code archive does.
    for package in (_nemo_skills_package_source(), platform):
        for source in package.files:
            if source.is_symlink():
                raise ProjectManifestError(
                    f"Installed package source cannot be a symlink: {source}"
                )
            relative = source.relative_to(package.root)
            _add_bundle_entry(
                entries,
                (PurePosixPath(package.root.name) / relative.as_posix()).as_posix(),
                source.read_bytes(),
                source.stat().st_mode,
            )
    # hatch-vcs generates _version.py at install time, so a Git checkout never
    # tracks it; nvflow.core.platform imports it on the cluster.
    version_module = platform.root / "_version.py"
    version_entry = (PurePosixPath(platform.root.name) / "_version.py").as_posix()
    if version_entry not in entries and version_module.is_file():
        _add_bundle_entry(
            entries,
            version_entry,
            version_module.read_bytes(),
            version_module.stat().st_mode,
        )
    for relative in materialized_files(project_root):
        source = project_root.joinpath(*relative.parts)
        if source.is_symlink():
            link = os.readlink(source)
            raise ProjectManifestError(
                f"Worker bundles do not support project symlinks: {relative} -> {link}"
            )
        data = materialize_project_bytes(
            source.read_bytes(),
            PurePosixPath(CODE_ROOT_TOKEN) / destination,
        )
        _add_bundle_entry(
            entries,
            (destination / relative).as_posix(),
            data,
            source.stat().st_mode,
        )

    digest = hashlib.sha256()
    uncompressed_size = 0
    for name, (data, mode) in sorted(entries.items()):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(mode & 0o777).encode())
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
        uncompressed_size += len(data)
    if uncompressed_size > max_uncompressed_bytes:
        raise ProjectManifestError(
            f"Worker bundle is too large uncompressed: {uncompressed_size} > "
            f"{max_uncompressed_bytes} bytes"
        )

    bundle_digest = f"sha256:{digest.hexdigest()}"
    destination_dir.mkdir(parents=True, exist_ok=True)
    final_path = destination_dir / f"nvflow-worker-{digest.hexdigest()[:20]}.zip"
    temporary = final_path.with_suffix(".zip.tmp")
    if not final_path.exists():
        try:
            with zipfile.ZipFile(
                temporary,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            ) as archive:
                for name, (data, mode) in sorted(entries.items()):
                    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.create_system = 3
                    info.external_attr = (mode & 0o777) << 16
                    archive.writestr(info, data)
            if temporary.stat().st_size > max_compressed_bytes:
                raise ProjectManifestError(
                    f"Worker bundle is too large compressed: {temporary.stat().st_size} > "
                    f"{max_compressed_bytes} bytes"
                )
            temporary.replace(final_path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    after = inspect_project_source(project_root, context.packaged_root, strict=True).provenance
    if after.content_digest != before.content_digest or after.commit != before.commit:
        final_path.unlink(missing_ok=True)
        raise ProjectManifestError("Project source changed while building the worker bundle")

    compressed_size = final_path.stat().st_size
    if compressed_size > max_compressed_bytes:
        raise ProjectManifestError(
            f"Worker bundle is too large compressed: {compressed_size} > "
            f"{max_compressed_bytes} bytes"
        )
    return WorkerBundle(
        path=final_path,
        digest=bundle_digest,
        compressed_size=compressed_size,
        uncompressed_size=uncompressed_size,
        project_python_root=(destination / context.manifest.python_source_root).as_posix(),
        project_identity=f"git:{before.commit}:{before.content_digest}",
        platform_identity=platform.identity,
    )


def _add_bundle_entry(
    entries: dict[str, tuple[bytes, int]],
    name: str,
    data: bytes,
    mode: int,
) -> None:
    safe_name = _safe_member_name(name)
    if not safe_name:
        raise ProjectManifestError("Worker bundle entry name cannot be empty")
    if safe_name in entries:
        raise ProjectManifestError(f"Duplicate path in worker bundle: {safe_name}")
    entries[safe_name] = (data, mode)


@contextmanager
def project_submission(
    context: ProjectContext,
    provenance: ProjectProvenance | None = None,
    platform_source: str | None = None,
):
    """Expose one project context only while its stage builds submissions."""
    project_token = _ACTIVE_PROJECT.set(context)
    provenance_token = _ACTIVE_PROVENANCE.set(provenance)
    platform_token = _ACTIVE_PLATFORM_SOURCE.set(platform_source)
    packagers_token = _ACTIVE_BASE_PACKAGERS.set({})
    try:
        yield
    finally:
        _ACTIVE_BASE_PACKAGERS.reset(packagers_token)
        _ACTIVE_PLATFORM_SOURCE.reset(platform_token)
        _ACTIVE_PROVENANCE.reset(provenance_token)
        _ACTIVE_PROJECT.reset(project_token)


def get_submission_base_packager(factory: Any, extra_package_dirs: tuple[str, ...] | None):
    """Construct each NeMo-Skills base packager once per project submission."""
    cache = _ACTIVE_BASE_PACKAGERS.get()
    if cache is None:
        return factory(extra_package_dirs=extra_package_dirs)
    key = tuple(extra_package_dirs or ())
    if key not in cache:
        cache[key] = factory(extra_package_dirs=extra_package_dirs)
    return cache[key]


def apply_project_environment(cluster_config: dict) -> dict:
    """Return a cluster config with the active project import path predeclared."""
    context = _ACTIVE_PROJECT.get()
    if context is None:
        return cluster_config

    configured = list(cluster_config.get("env_vars") or ())
    updated_env = []
    found_pythonpath = False
    for entry in configured:
        if not isinstance(entry, str):
            updated_env.append(entry)
            continue
        name, separator, value = entry.partition("=")
        if name.strip() != "PYTHONPATH":
            updated_env.append(entry)
            continue
        found_pythonpath = True
        existing = value.strip() if separator else os.environ.get("PYTHONPATH", "")
        updated_env.append(f"PYTHONPATH={_project_pythonpath(context, existing)}")
    if not found_pythonpath:
        updated_env.append(f"PYTHONPATH={_project_pythonpath(context, '')}")

    updated = dict(cluster_config)
    updated["env_vars"] = updated_env
    return updated


def apply_project_packager(executor: Any) -> Any:
    """Attach scoped project packaging to an executor when required."""
    context = _ACTIVE_PROJECT.get()
    if context is None:
        return executor

    env_vars = getattr(executor, "env_vars", None)
    if isinstance(env_vars, dict):
        env_vars["PYTHONPATH"] = _project_pythonpath(
            context,
            env_vars.get("PYTHONPATH", ""),
        )

    base_packager = getattr(executor, "packager", None)
    if base_packager is None:
        return executor
    if isinstance(base_packager, ProjectOverlayPackager):
        return executor

    destination = str(context.packaged_root).removeprefix("/nemo_run/code/")
    wrapped = ProjectOverlayPackager(
        base_packager=base_packager,
        project_root=str(context.root),
        destination=destination,
        expected_provenance=_ACTIVE_PROVENANCE.get(),
        expected_platform_source=_ACTIVE_PLATFORM_SOURCE.get(),
    )
    wrapped.debug = getattr(base_packager, "debug", False)
    wrapped.symlink_from_remote_dir = getattr(
        base_packager,
        "symlink_from_remote_dir",
        None,
    )
    executor.packager = wrapped
    return executor


def _project_pythonpath(context: ProjectContext, existing: str) -> str:
    paths = [path for path in existing.split(":") if path] if existing else []
    if context.worker_python_root not in paths:
        paths.insert(0, context.worker_python_root)
    if "/nemo_run/code" not in paths:
        paths.append("/nemo_run/code")
    return ":".join(paths)


@dataclass(kw_only=True)
class ProjectOverlayPackager(Packager):
    """Combine the existing NVFlow package with one committed project subtree."""

    base_packager: Any = None
    project_root: str = ""
    destination: str = ""
    expected_provenance: ProjectProvenance | None = None
    expected_platform_source: str | None = None

    def setup(self):
        """Preserve setup behavior from the existing packager."""
        return self.base_packager.setup()

    def package(self, path: Path, job_dir: str, name: str) -> str:
        """Package verified launcher sources and one committed project subtree."""
        del path
        project_root = Path(self.project_root).resolve()
        packaged_root = PurePosixPath("/nemo_run/code") / PurePosixPath(self.destination)
        snapshot = inspect_project_source(project_root, packaged_root, strict=True)
        provenance = snapshot.provenance
        if snapshot.git_root is None or provenance.commit is None:
            raise ProjectManifestError(f"Unable to package project source: {project_root}")
        expected = self.expected_provenance
        if expected is not None and (
            provenance.commit != expected.commit
            or provenance.source_subpath != expected.source_subpath
            or provenance.packaged_destination != expected.packaged_destination
            or provenance.content_digest != expected.content_digest
        ):
            raise ProjectManifestError(
                "Project source changed after planning; rebuild the launch plan before submission"
            )
        package_root = Path(__file__).resolve().parents[1]
        platform_source = _platform_package_source(package_root)
        if (
            self.expected_platform_source is not None
            and platform_source.identity != self.expected_platform_source
        ):
            raise ProjectManifestError(
                "NVFlow implementation changed after planning: "
                f"planned={self.expected_platform_source}, "
                f"packaged={platform_source.identity}"
            )
        nemo_skills_source = _nemo_skills_package_source()
        platform_digest = _package_source_digest(platform_source)[:12]
        nemo_skills_digest = _package_source_digest(nemo_skills_source)[:12]
        project_digest = str(provenance.content_digest).removeprefix("sha256:")[:12]
        final_archive = (
            Path(job_dir) / f"{name}_{nemo_skills_digest}_{platform_digest}_{project_digest}.tar.gz"
        )
        if final_archive.exists():
            return str(final_archive)

        project_archive = Path(job_dir) / f"{name}_project_{provenance.commit[:12]}.tar.gz"
        try:
            _archive_project_tree(
                project_root,
                project_archive,
            )
            _combine_archives(
                project_archive=project_archive,
                destination=PurePosixPath(self.destination),
                output=final_archive,
                nemo_skills_source=nemo_skills_source,
                platform_source=platform_source,
            )
        finally:
            project_archive.unlink(missing_ok=True)
        return str(final_archive)


def _archive_project_tree(
    project_root: Path,
    output: Path,
) -> None:
    with tarfile.open(output, "w:gz", dereference=False) as archive:
        for relative in materialized_files(project_root):
            source = project_root.joinpath(*relative.parts)
            archive.add(
                source,
                arcname=relative.as_posix(),
                recursive=False,
                filter=_normalize_project_member,
            )


def _normalize_project_member(member: tarfile.TarInfo) -> tarfile.TarInfo:
    member.uid = 0
    member.gid = 0
    member.uname = ""
    member.gname = ""
    member.mtime = 0
    return member


def _combine_archives(
    project_archive: Path,
    destination: PurePosixPath,
    output: Path,
    nemo_skills_source: _PackageSource,
    platform_source: _PackageSource,
) -> None:
    if destination.is_absolute() or ".." in destination.parts:
        raise ProjectManifestError(f"Invalid packaged project destination: {destination}")

    temporary = output.with_suffix(output.suffix + ".tmp")
    names: set[str] = set()
    try:
        with (
            tarfile.open(project_archive, "r:*") as project_tar,
            tarfile.open(temporary, "w:gz") as combined,
        ):
            _add_package_source(combined, names, nemo_skills_source)
            _add_package_source(combined, names, platform_source)

            for member in project_tar:
                relative = _safe_member_name(member.name)
                if not relative:
                    continue
                name = (destination / relative).as_posix()
                _validate_link(member, destination, name)
                data = None
                if member.isfile():
                    file_object = project_tar.extractfile(member)
                    data = file_object.read() if file_object is not None else b""
                    data = materialize_project_bytes(
                        data,
                        PurePosixPath("/nemo_run/code") / destination,
                    )
                _add_member(project_tar, combined, member, name, names, data=data)

        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _add_member(
    source: tarfile.TarFile,
    target: tarfile.TarFile,
    member: tarfile.TarInfo,
    name: str,
    names: set[str],
    *,
    data: bytes | None = None,
) -> None:
    if name in names:
        raise ProjectManifestError(f"Duplicate path in worker package: {name}")
    names.add(name)
    copied = copy.copy(member)
    copied.name = name
    if data is not None:
        copied.size = len(data)
        file_object = io.BytesIO(data)
    else:
        file_object = source.extractfile(member) if member.isfile() else None
    target.addfile(copied, file_object)


def _add_package_source(
    target: tarfile.TarFile,
    names: set[str],
    package: _PackageSource,
) -> None:
    for source in package.files:
        if source.is_symlink():
            raise ProjectManifestError(f"Installed package source cannot be a symlink: {source}")
        relative = source.relative_to(package.root)
        name = (PurePosixPath(package.root.name) / relative.as_posix()).as_posix()
        if name in names:
            raise ProjectManifestError(f"Duplicate path in worker package: {name}")
        names.add(name)
        target.add(source, arcname=name, recursive=False)


def _package_source_digest(package: _PackageSource) -> str:
    if package.content_digest is not None:
        return package.content_digest
    digest = hashlib.sha256()
    digest.update(package.identity.encode())
    digest.update(b"\0")
    for source in package.files:
        if source.is_symlink():
            raise ProjectManifestError(f"Installed package source cannot be a symlink: {source}")
        digest.update(source.relative_to(package.root).as_posix().encode())
        digest.update(b"\0")
        with source.open("rb") as source_file:
            while chunk := source_file.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _platform_package_source(package_root: Path) -> _PackageSource:
    package_root = package_root.resolve()
    tracked = _tracked_package_files(package_root)
    if tracked:
        result = subprocess.run(
            ["git", "-C", str(package_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return _PackageSource(
            root=package_root,
            files=tuple(sorted(tracked)),
            identity=f"git:{result.stdout.strip()}",
        )
    return _wheel_package_source(package_root)


def _wheel_package_source(package_root: Path) -> _PackageSource:
    return _installed_package_source("nvflow", "nvflow", package_root=package_root)


def _nemo_skills_package_source() -> _PackageSource:
    return _installed_package_source("nemo-skills", "nemo_skills")


def _installed_package_source(
    distribution_name: str,
    package_name: str,
    *,
    package_root: Path | None = None,
) -> _PackageSource:
    try:
        distribution = metadata.distribution(distribution_name)
    except metadata.PackageNotFoundError as exc:
        raise ProjectManifestError(
            f"Required worker package is not installed: {distribution_name}"
        ) from exc

    distribution_files = distribution.files
    if distribution_files is None:
        raise ProjectManifestError(f"Installed {distribution_name} distribution has no RECORD file")

    package_root = (
        package_root.resolve()
        if package_root is not None
        else Path(distribution.locate_file(package_name)).resolve()
    )
    recorded_files: list[tuple[metadata.PackagePath, Path]] = []
    for recorded in distribution_files:
        if not recorded.parts or recorded.parts[0] != package_name:
            continue
        if recorded.is_absolute() or ".." in recorded.parts:
            raise ProjectManifestError(
                f"{distribution_name} RECORD path escapes package root: {recorded}"
            )
        if "__pycache__" in recorded.parts or recorded.suffix in {".pyc", ".pyo"}:
            continue

        source = Path(distribution.locate_file(recorded))
        expected = package_root.joinpath(*recorded.parts[1:])
        if source.is_symlink() or source.resolve() != expected.resolve():
            raise ProjectManifestError(
                f"{distribution_name} RECORD path escapes package root: {recorded}"
            )
        if not source.is_file():
            raise ProjectManifestError(f"{distribution_name} RECORD file is missing: {recorded}")
        recorded_files.append((recorded, source.resolve()))

    if not recorded_files:
        raise ProjectManifestError(
            f"Installed {distribution_name} RECORD contains no {package_name} package files"
        )

    identity = wheel_distribution_identity(
        distribution,
        fallback_name=distribution_name,
    )
    recorded_files.sort(key=lambda item: item[1])
    content_digest = hashlib.sha256()
    content_digest.update(identity.encode())
    content_digest.update(b"\0")
    for recorded, source in recorded_files:
        content_digest.update(source.relative_to(package_root).as_posix().encode())
        content_digest.update(b"\0")
        _validate_recorded_file(
            source,
            recorded,
            distribution_name,
            content_digest=content_digest,
        )
        content_digest.update(b"\0")
    return _PackageSource(
        root=package_root,
        files=tuple(source for _, source in recorded_files),
        identity=identity,
        content_digest=content_digest.hexdigest(),
    )


def _validate_recorded_file(
    source: Path,
    recorded: metadata.PackagePath,
    distribution_name: str,
    *,
    content_digest: Any | None = None,
) -> None:
    expected_hash = recorded.hash
    if expected_hash is None:
        raise ProjectManifestError(f"{distribution_name} RECORD has no hash for {recorded}")
    try:
        digest = hashlib.new(expected_hash.mode)
    except ValueError as exc:
        raise ProjectManifestError(
            f"{distribution_name} RECORD uses unsupported hash "
            f"{expected_hash.mode!r} for {recorded}"
        ) from exc
    with source.open("rb") as source_file:
        while chunk := source_file.read(1024 * 1024):
            digest.update(chunk)
            if content_digest is not None:
                content_digest.update(chunk)
    actual_hash = base64.urlsafe_b64encode(digest.digest()).rstrip(b"=").decode()
    if actual_hash != expected_hash.value:
        raise ProjectManifestError(
            f"Installed {distribution_name} file does not match RECORD: {recorded}"
        )
    if recorded.size is None or source.stat().st_size != recorded.size:
        raise ProjectManifestError(
            f"Installed {distribution_name} file size does not match RECORD: {recorded}"
        )


def _tracked_package_files(package_root: Path) -> list[Path]:
    """Return clean Git-tracked platform files, or empty for a wheel install.

    This intentionally remains separate from project provenance: an installed
    wheel can live inside a project worktree, so absence of tracked files under
    ``package_root`` selects RECORD verification instead of failing provenance.
    """
    result = subprocess.run(
        ["git", "-C", str(package_root), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []

    git_root = Path(result.stdout.strip()).resolve()
    relative_root = package_root.relative_to(git_root)
    tracked = subprocess.run(
        [
            "git",
            "-C",
            str(git_root),
            "ls-files",
            "-z",
            "--",
            relative_root.as_posix(),
        ],
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    tracked_files = [
        git_root / path.decode()
        for path in tracked
        if path and (git_root / path.decode()).is_file()
    ]
    if not tracked_files:
        return []

    status = subprocess.run(
        [
            "git",
            "-C",
            str(git_root),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            relative_root.as_posix(),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise ProjectManifestError(f"NVFlow source must be clean before submission:\n{status}")

    return tracked_files


def _safe_member_name(name: str) -> str:
    stripped = name.removeprefix("./")
    path = PurePosixPath(stripped)
    if path.is_absolute() or ".." in path.parts:
        raise ProjectManifestError(f"Unsafe path in worker package: {name}")
    return "" if stripped in {"", "."} else path.as_posix()


def _validate_link(
    member: tarfile.TarInfo,
    destination: PurePosixPath,
    packaged_name: str,
) -> None:
    if member.islnk():
        raise ProjectManifestError(
            f"Hard links are not supported in project packages: {member.name}"
        )
    if not member.issym():
        return
    link = PurePosixPath(member.linkname)
    if link.is_absolute():
        raise ProjectManifestError(f"Absolute link in project package: {member.linkname}")
    resolved = PurePosixPath(packaged_name).parent.joinpath(link)
    if ".." in resolved.parts:
        raise ProjectManifestError(f"Escaping link in project package: {member.linkname}")
    try:
        resolved.relative_to(destination)
    except ValueError as exc:
        raise ProjectManifestError(
            f"Project link leaves packaged destination: {member.linkname}"
        ) from exc
