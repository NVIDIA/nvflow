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
"""Reproduce the exact active NVFlow installation from PEP 610 metadata."""

from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import os
import re
import tomllib
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote, urlparse, urlsplit, urlunsplit
from urllib.request import url2pathname

from nvflow.project.manifest import ProjectManifestError


class ProjectEnvironmentError(ProjectManifestError):
    """Raised when a project launcher environment cannot be prepared."""


@dataclass(frozen=True)
class NVFlowInstallSource:
    """Exact active NVFlow source used by a synthetic project environment."""

    identity: str
    requirement: str
    source: dict[str, str | bool] | None = None
    local_artifact: Path | None = None
    source_directory: Path | None = None
    metadata_path: Path | None = None
    content_hash: str | None = None
    version: str | None = None

    def validate(self) -> None:
        """Fail if a local source can no longer reproduce the active install."""
        if self.local_artifact is not None and not self.local_artifact.is_file():
            raise ProjectEnvironmentError(
                f"Installed NVFlow source artifact is no longer available: {self.local_artifact}"
            )
        if self.local_artifact is not None and self.content_hash is not None:
            _verify_archive_hash(self.local_artifact, self.content_hash)
        if self.source_directory is not None and not self.source_directory.is_dir():
            raise ProjectEnvironmentError(
                f"Installed NVFlow source directory is no longer available: {self.source_directory}"
            )

    def fingerprint_material(self) -> dict[str, str]:
        """Return stable source inputs without exposing credentials."""
        value = {"identity": self.identity}
        if self.version is not None:
            value["version"] = self.version
        if self.metadata_path is not None:
            value["metadata_digest"] = toml_digest(self.metadata_path)
        if self.content_hash is not None:
            value["content_hash"] = self.content_hash
        elif self.local_artifact is not None:
            value["artifact_digest"] = file_digest(self.local_artifact)
        if self.source_directory is not None:
            value["directory_digest"] = directory_digest(self.source_directory)
        return value


class DistributionMetadata(Protocol):
    """Subset of importlib distribution metadata used for source reproduction."""

    metadata: Any
    version: str

    def read_text(self, filename: str) -> str | None: ...


def active_nvflow_install_source(
    distribution: DistributionMetadata | None = None,
) -> NVFlowInstallSource:
    """Describe the exact active NVFlow installation using PEP 610 metadata."""
    if distribution is None:
        try:
            distribution = metadata.distribution("nvflow")
        except metadata.PackageNotFoundError:
            checkout = Path(__file__).resolve().parents[2]
            pyproject = checkout / "pyproject.toml"
            if not pyproject.is_file():
                raise ProjectEnvironmentError(
                    "Unable to identify the active NVFlow installation source"
                ) from None
            return NVFlowInstallSource(
                identity=f"editable:{checkout}",
                requirement="nvflow",
                source={"path": str(checkout), "editable": True},
                metadata_path=pyproject,
            )

    name = distribution.metadata.get("Name", "nvflow")
    version = distribution.version
    direct_url_text = distribution.read_text("direct_url.json")
    if not direct_url_text:
        requirement = f"{name}=={version}"
        return NVFlowInstallSource(
            identity=f"index:{requirement}",
            requirement=requirement,
            version=version,
        )

    try:
        direct_url = json.loads(direct_url_text)
    except json.JSONDecodeError as exc:
        raise ProjectEnvironmentError("Installed NVFlow has invalid direct_url.json") from exc
    if not isinstance(direct_url, dict) or not isinstance(direct_url.get("url"), str):
        raise ProjectEnvironmentError("Installed NVFlow direct_url.json has no source URL")
    url = direct_url["url"]

    vcs_info = direct_url.get("vcs_info")
    if isinstance(vcs_info, dict):
        vcs = vcs_info.get("vcs")
        commit = vcs_info.get("commit_id")
        if not isinstance(vcs, str) or not isinstance(commit, str):
            raise ProjectEnvironmentError(
                "Installed NVFlow VCS metadata must contain vcs and commit_id"
            )
        vcs_url = url if url.startswith(f"{vcs}+") else f"{vcs}+{url}"
        requirement_url = _append_url_fragment(
            _with_vcs_revision(vcs_url, commit),
            subdirectory=direct_url.get("subdirectory"),
        )
        return NVFlowInstallSource(
            identity=f"vcs:{redact_source_credentials(requirement_url)}",
            requirement=f"{name} @ {requirement_url}",
            version=version,
        )

    dir_info = direct_url.get("dir_info")
    if isinstance(dir_info, dict):
        path = _local_file_url(url, label="NVFlow directory")
        pyproject = path / "pyproject.toml"
        if not pyproject.is_file():
            raise ProjectEnvironmentError(f"NVFlow source has no pyproject.toml: {path}")
        source: dict[str, str | bool] = {"path": str(path)}
        kind = "directory"
        source_directory: Path | None = path
        if dir_info.get("editable") is True:
            source["editable"] = True
            kind = "editable"
            source_directory = None
        return NVFlowInstallSource(
            identity=f"{kind}:{path}",
            requirement=name,
            source=source,
            source_directory=source_directory,
            metadata_path=pyproject,
            version=version,
        )

    archive_info = direct_url.get("archive_info")
    if isinstance(archive_info, dict):
        local_artifact = _optional_local_file_url(url)
        content_hash = _archive_content_hash(archive_info)
        if local_artifact is not None and not local_artifact.is_file() and content_hash is None:
            raise ProjectEnvironmentError(
                f"Installed NVFlow source artifact is no longer available: {local_artifact}"
            )
        if local_artifact is not None and local_artifact.is_file() and content_hash is None:
            content_hash = file_digest(local_artifact).replace(":", "=", 1)
        if local_artifact is None and content_hash is None:
            raise ProjectEnvironmentError(
                "Installed remote NVFlow archive has no integrity hash in direct_url.json"
            )
        requirement_url = _append_url_fragment(
            url,
            content_hash=content_hash,
            subdirectory=direct_url.get("subdirectory"),
        )
        return NVFlowInstallSource(
            identity=f"archive:{redact_source_credentials(requirement_url)}",
            requirement=f"{name} @ {requirement_url}",
            local_artifact=local_artifact,
            content_hash=content_hash,
            version=version,
        )

    raise ProjectEnvironmentError(
        "Installed NVFlow direct_url.json has unsupported source metadata"
    )


def file_digest(path: Path) -> str:
    """Return a SHA-256 digest for one environment input file."""
    if path.is_dir():
        raise ProjectEnvironmentError(f"Expected a file while fingerprinting: {path}")
    try:
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256")
    except OSError as exc:
        raise ProjectEnvironmentError(f"Unable to fingerprint {path}: {exc}") from exc
    return f"sha256:{digest.hexdigest()}"


def toml_digest(path: Path) -> str:
    """Digest TOML semantics so comments do not invalidate environments."""
    try:
        value = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ProjectEnvironmentError(f"Unable to read TOML metadata {path}: {exc}") from exc
    payload = json.dumps(
        value,
        default=_toml_json_value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _toml_json_value(value: object) -> dict[str, str]:
    if isinstance(value, date | datetime | time):
        return {
            "toml_type": type(value).__name__,
            "value": value.isoformat(),
        }
    raise TypeError(f"Unsupported TOML value: {value!r}")


def directory_digest(root: Path) -> str:
    """Digest a non-editable local source directory, excluding local caches."""
    digest = hashlib.sha256()
    ignored = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
    }
    try:
        paths = sorted(
            path
            for path in root.rglob("*")
            if not ignored.intersection(path.relative_to(root).parts)
        )
        for path in paths:
            relative = path.relative_to(root).as_posix()
            digest.update(relative.encode())
            digest.update(b"\0")
            digest.update(oct(path.lstat().st_mode & 0o777).encode())
            digest.update(b"\0")
            if path.is_symlink():
                digest.update(b"link\0")
                digest.update(os.readlink(path).encode())
            elif path.is_file():
                digest.update(b"file\0")
                with path.open("rb") as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(chunk)
            elif path.is_dir():
                digest.update(b"dir\0")
            digest.update(b"\0")
    except OSError as exc:
        raise ProjectEnvironmentError(f"Unable to fingerprint NVFlow source {root}: {exc}") from exc
    return f"sha256:{digest.hexdigest()}"


def redact_source_credentials(value: str) -> str:
    """Redact common URL credentials before persisting or displaying a source."""
    redacted = re.sub(r"(://)[^/@\s]+@", r"\1<redacted>@", value)
    return re.sub(
        (
            r"([?&#](?:access_token|api[-_]?key|auth|key|password|private_token|sig|"
            r"signature|token|x-amz-credential|x-amz-security-token|x-amz-signature)=)"
            r"[^&#\s]+"
        ),
        r"\1<redacted>",
        redacted,
        flags=re.IGNORECASE,
    )


def _append_url_fragment(
    url: str,
    *,
    content_hash: str | None = None,
    subdirectory: object = None,
) -> str:
    values = []
    if content_hash:
        values.append(content_hash)
    if isinstance(subdirectory, str) and subdirectory:
        values.append(f"subdirectory={quote(subdirectory, safe='/')}")
    if not values:
        return url
    separator = "&" if "#" in url else "#"
    return f"{url}{separator}{'&'.join(values)}"


def _with_vcs_revision(url: str, revision: str) -> str:
    parsed = urlsplit(url)
    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            f"{parsed.path}@{revision}",
            parsed.query,
            parsed.fragment,
        )
    )


def _local_file_url(url: str, *, label: str) -> Path:
    path = _optional_local_file_url(url)
    if path is None:
        raise ProjectEnvironmentError(f"{label} must use a local file URL: {url}")
    if not path.exists():
        raise ProjectEnvironmentError(f"{label} is no longer available: {path}")
    return path.resolve()


def _optional_local_file_url(url: str) -> Path | None:
    parsed = urlparse(url)
    if parsed.scheme != "file":
        return None
    path = url2pathname(parsed.path)
    if parsed.netloc and parsed.netloc != "localhost":
        path = f"//{parsed.netloc}{path}"
    return Path(path).expanduser().resolve()


def _archive_content_hash(archive_info: dict[str, Any]) -> str | None:
    value = archive_info.get("hash")
    if isinstance(value, str):
        return value
    hashes = archive_info.get("hashes")
    if isinstance(hashes, dict):
        for algorithm in ("sha256", *sorted(hashes)):
            digest = hashes.get(algorithm)
            if isinstance(digest, str):
                return f"{algorithm}={digest}"
    return None


def _verify_archive_hash(path: Path, expected: str) -> None:
    algorithm, separator, digest_value = expected.partition("=")
    if not separator or not algorithm or not digest_value:
        raise ProjectEnvironmentError(f"Invalid NVFlow archive hash metadata: {expected!r}")
    try:
        with path.open("rb") as source:
            actual = hashlib.file_digest(source, algorithm).hexdigest()
    except (OSError, ValueError) as exc:
        raise ProjectEnvironmentError(
            f"Unable to verify NVFlow source artifact {path}: {exc}"
        ) from exc
    if actual != digest_value:
        raise ProjectEnvironmentError(
            f"Installed NVFlow source artifact hash changed: {path} "
            f"({algorithm}={actual}, expected {expected})"
        )
