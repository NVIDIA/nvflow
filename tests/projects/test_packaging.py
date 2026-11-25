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
"""Full-dependency tests for scoped external project packaging."""

import base64
import csv
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import zipfile
from importlib import import_module, metadata
from pathlib import Path, PurePosixPath

import pytest

pytestmark = pytest.mark.requires_nemo
pytest.importorskip("nemo_run")

from nvflow.core import platform as core_platform  # noqa: E402
from nvflow.project import ProjectManifestError, ProjectResolver  # noqa: E402
from nvflow.project import packaging as project_packaging  # noqa: E402
from nvflow.project import provenance as project_provenance  # noqa: E402
from nvflow.project.packaging import (  # noqa: E402
    ProjectOverlayPackager,
    apply_project_environment,
    apply_project_packager,
    build_worker_bundle,
    project_submission,
)
from nvflow.project.provenance import inspect_project_source  # noqa: E402

_REAL_TRACKED_PACKAGE_FILES = project_packaging._tracked_package_files


class _BasePackager:
    def setup(self) -> None:
        return None

    def package(self, *_args, **_kwargs) -> str:
        raise AssertionError("Project packaging must not invoke the untrusted base packager")


class _Executor:
    def __init__(self, packager) -> None:
        self.packager = packager


@pytest.fixture(autouse=True)
def installed_package_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep packaging tests independent of the parent worktree's dirty state."""
    package_init = Path(project_packaging.__file__).parents[1] / "__init__.py"
    monkeypatch.setattr(
        project_packaging,
        "_tracked_package_files",
        lambda _: [package_init],
    )
    nemo_root = tmp_path / "installed" / "nemo_skills"
    nemo_root.mkdir(parents=True)
    nemo_init = nemo_root / "__init__.py"
    nemo_init.write_text("PACKAGE = 'installed-nemo-skills'\n", encoding="utf-8")
    nemo_source = project_packaging._PackageSource(
        root=nemo_root,
        files=(nemo_init,),
        identity="wheel:nemo-skills==test",
    )
    monkeypatch.setattr(project_packaging, "_nemo_skills_package_source", lambda: nemo_source)


def _git_project(tmp_path: Path):
    repository = tmp_path / "repository"
    bundle = repository / "projects" / "demo"
    (bundle / "workflows").mkdir(parents=True)
    (bundle / "assets").mkdir()
    (bundle / "src" / "packaged_project").mkdir(parents=True)
    (bundle / "project.yaml").write_text(
        "manifest_version: 1\nproject: packaging-demo\npython_source_root: src\n",
        encoding="utf-8",
    )
    config = bundle / "workflows" / "noop.yaml"
    config.write_text(
        """
workflow:
  name: noop
  type: custom
cluster: local
pipeline_stages: []
stages: {}
""".lstrip(),
        encoding="utf-8",
    )
    (bundle / "assets" / "message.txt").write_text("packaged\n", encoding="utf-8")
    (bundle / "assets" / "nested.yaml").write_text(
        "message_path: project://assets/message.txt\n",
        encoding="utf-8",
    )
    (bundle / "src" / "packaged_project" / "__init__.py").write_text(
        'VALUE = "worker import succeeded"\n',
        encoding="utf-8",
    )
    (repository / "nvflow").mkdir()
    (repository / "nvflow" / "__init__.py").write_bytes(
        (Path(__file__).parents[2] / "nvflow" / "__init__.py").read_bytes()
    )
    (repository / "unrelated.jsonl").write_text("{}\n", encoding="utf-8")
    (repository / "nemo_skills").mkdir()
    (repository / "nemo_skills" / "__init__.py").write_text(
        "PACKAGE = 'project-owned-must-not-ship'\n",
        encoding="utf-8",
    )

    _git(repository, "init", "-q")
    _git(repository, "add", ".")
    _git(
        repository,
        "-c",
        "user.name=NVFlow Tests",
        "-c",
        "user.email=nvflow-tests@example.com",
        "commit",
        "-qm",
        "fixture",
    )
    return repository, bundle, config


def _git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def test_worker_bundle_is_deterministic_and_external_project_importable(
    tmp_path: Path,
) -> None:
    _repository, project_root, config = _git_project(tmp_path)
    context = ProjectResolver(search_paths=(project_root.parent,)).resolve(config)
    provenance = context.source_provenance(strict=True)
    package_root = Path(project_packaging.__file__).parents[1]
    platform_source = project_packaging._platform_package_source(package_root).identity

    first = build_worker_bundle(
        context,
        provenance,
        platform_source,
        tmp_path / "bundles",
    )
    second = build_worker_bundle(
        context,
        provenance,
        platform_source,
        tmp_path / "bundles",
    )

    assert first == second
    assert first.digest.startswith("sha256:")
    assert first.project_python_root == "projects/packaging-demo/src"
    with zipfile.ZipFile(first.path) as archive:
        names = set(archive.namelist())
        assert "nvflow/__init__.py" in names
        assert "projects/packaging-demo/src/packaged_project/__init__.py" in names
        rendered = archive.read("projects/packaging-demo/assets/nested.yaml").decode()
        nemo_skills_init = archive.read("nemo_skills/__init__.py").decode()
    assert "{{NVFLOW_CODE_ROOT}}/projects/packaging-demo/assets/message.txt" in rendered
    assert nemo_skills_init == "PACKAGE = 'installed-nemo-skills'\n"
    assert not any(name.startswith((".git/", ".venv/")) for name in names)


def test_worker_bundle_imports_the_ray_controller_on_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _repository, project_root, config = _git_project(tmp_path)
    context = ProjectResolver(search_paths=(project_root.parent,)).resolve(config)
    package_root = Path(project_packaging.__file__).parents[1]
    # Like `git ls-files` in a checkout: every module except the generated _version.py.
    monkeypatch.setattr(
        project_packaging,
        "_tracked_package_files",
        lambda root: sorted(
            path
            for path in root.rglob("*.py")
            if path.name != "_version.py" and "__pycache__" not in path.parts
        ),
    )
    bundle = build_worker_bundle(
        context,
        context.source_provenance(strict=True),
        project_packaging._platform_package_source(package_root).identity,
        tmp_path / "bundles",
    )
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(bundle.path) as archive:
        archive.extractall(extracted)

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, nvflow.integrations.ray_jobs.controller as controller; "
            "assert controller.__file__.startswith(sys.argv[1]), controller.__file__",
            str(extracted),
        ],
        cwd=extracted,
        env={**os.environ, "PYTHONPATH": str(extracted)},
        check=True,
    )


def test_worker_bundle_enforces_size_limit(tmp_path: Path) -> None:
    _repository, project_root, config = _git_project(tmp_path)
    context = ProjectResolver(search_paths=(project_root.parent,)).resolve(config)
    provenance = context.source_provenance(strict=True)
    package_root = Path(project_packaging.__file__).parents[1]
    platform_source = project_packaging._platform_package_source(package_root).identity

    with pytest.raises(ProjectManifestError, match="too large uncompressed"):
        build_worker_bundle(
            context,
            provenance,
            platform_source,
            tmp_path / "bundles",
            max_uncompressed_bytes=1,
        )


def _wheel_distribution(
    tmp_path: Path,
    *,
    unhashed_package_file: bool = False,
    vcs_commit: str | None = None,
) -> tuple[Path, metadata.Distribution, Path]:
    site_packages = tmp_path / "site-packages"
    package_root = site_packages / "nvflow"
    package_root.mkdir(parents=True)
    package_files = [
        package_root / "__init__.py",
        package_root / "worker.py",
        package_root / "project" / "packaging.py",
    ]
    package_files[-1].parent.mkdir()
    package_files[0].write_text('__version__ = "1.2.3"\n', encoding="utf-8")
    package_files[1].write_text("WORKER = True\n", encoding="utf-8")
    package_files[2].write_text("WHEEL_PACKAGING = True\n", encoding="utf-8")

    unrelated = site_packages / "unrelated_package" / "secret.py"
    unrelated.parent.mkdir()
    unrelated.write_text("SECRET = True\n", encoding="utf-8")

    dist_info = site_packages / "nvflow-1.2.3.dist-info"
    dist_info.mkdir()
    metadata_file = dist_info / "METADATA"
    metadata_file.write_text(
        "Metadata-Version: 2.3\nName: nvflow\nVersion: 1.2.3\n",
        encoding="utf-8",
    )
    metadata_files = [metadata_file]
    if vcs_commit is not None:
        direct_url = dist_info / "direct_url.json"
        direct_url.write_text(
            json.dumps(
                {
                    "url": "https://example.invalid/nvflow.git",
                    "vcs_info": {
                        "vcs": "git",
                        "commit_id": vcs_commit,
                    },
                }
            ),
            encoding="utf-8",
        )
        metadata_files.append(direct_url)
    record_file = dist_info / "RECORD"

    def record_hash(path: Path) -> str:
        digest = hashlib.sha256(path.read_bytes()).digest()
        return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()

    with record_file.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        for path in [*package_files, unrelated, *metadata_files]:
            relative = path.relative_to(site_packages).as_posix()
            digest = "" if unhashed_package_file and path == package_files[1] else record_hash(path)
            size = "" if not digest else str(path.stat().st_size)
            writer.writerow([relative, f"sha256={digest}" if digest else "", size])
        writer.writerow([record_file.relative_to(site_packages).as_posix(), "", ""])

    return package_root, metadata.Distribution.at(dist_info), unrelated


def test_wheel_platform_source_uses_only_verified_record_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root, distribution, unrelated = _wheel_distribution(tmp_path)
    monkeypatch.setattr(project_packaging, "_tracked_package_files", lambda _: [])
    monkeypatch.setattr(project_packaging.metadata, "distribution", lambda _: distribution)

    source = project_packaging._platform_package_source(package_root)

    assert source.identity == "wheel:nvflow==1.2.3"
    assert {path.relative_to(package_root).as_posix() for path in source.files} == {
        "__init__.py",
        "project/packaging.py",
        "worker.py",
    }
    assert unrelated not in source.files


def test_vcs_wheel_uses_one_identity_for_planning_and_packaging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commit = "a" * 40
    package_root, distribution, _unrelated = _wheel_distribution(
        tmp_path,
        vcs_commit=commit,
    )
    monkeypatch.setattr(project_packaging, "_tracked_package_files", lambda _: [])
    monkeypatch.setattr(
        project_packaging.metadata,
        "distribution",
        lambda _: distribution,
    )
    monkeypatch.setattr(
        core_platform.metadata,
        "distribution",
        lambda _: distribution,
    )
    monkeypatch.setattr(
        core_platform.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1, "", ""),
    )

    core_platform.active_platform_identity.cache_clear()
    try:
        planned = core_platform.active_platform_identity()["source"]
    finally:
        core_platform.active_platform_identity.cache_clear()
    packaged = project_packaging._platform_package_source(package_root).identity

    expected = f"wheel:nvflow==1.2.3@{commit}"
    assert planned == packaged == expected


def test_untracked_wheel_inside_project_git_is_not_treated_as_editable_source(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "project"
    repository.mkdir()
    (repository / "project.yaml").write_text("project: demo\n", encoding="utf-8")
    _git(repository, "init", "-q")
    _git(repository, "add", "project.yaml")
    _git(
        repository,
        "-c",
        "user.name=NVFlow Tests",
        "-c",
        "user.email=nvflow-tests@example.com",
        "commit",
        "-qm",
        "fixture",
    )
    package_root = repository / ".venv" / "site-packages" / "nvflow"
    package_root.mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")

    assert _REAL_TRACKED_PACKAGE_FILES(package_root) == []


def test_tracked_platform_files_support_linked_worktrees(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    package_root = repository / "nvflow"
    package_root.mkdir(parents=True)
    (package_root / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repository, "init", "-q")
    _git(repository, "add", ".")
    _git(
        repository,
        "-c",
        "user.name=NVFlow Tests",
        "-c",
        "user.email=nvflow-tests@example.com",
        "commit",
        "-qm",
        "fixture",
    )
    worktree = tmp_path / "linked-worktree"
    _git(repository, "worktree", "add", "-q", "-b", "linked-worktree", str(worktree))
    linked_package = worktree / "nvflow"

    files = _REAL_TRACKED_PACKAGE_FILES(linked_package)

    assert [path.relative_to(worktree).as_posix() for path in files] == ["nvflow/__init__.py"]
    (linked_package / "__init__.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ProjectManifestError, match="NVFlow source must be clean"):
        _REAL_TRACKED_PACKAGE_FILES(linked_package)


def test_overlay_packager_accepts_verified_wheel_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, bundle, _ = _git_project(tmp_path)
    package_root, distribution, _ = _wheel_distribution(tmp_path / "wheel")
    monkeypatch.setattr(
        project_packaging, "__file__", str(package_root / "project" / "packaging.py")
    )
    monkeypatch.setattr(project_packaging, "_tracked_package_files", lambda _: [])
    monkeypatch.setattr(project_packaging.metadata, "distribution", lambda _: distribution)
    job_dir = tmp_path / "wheel-jobs"
    job_dir.mkdir()
    packager = ProjectOverlayPackager(
        base_packager=_BasePackager(),
        project_root=str(bundle),
        destination="projects/packaging-demo",
    )

    archive = packager.package(repository, str(job_dir), "wheel")
    with tarfile.open(archive, "r:gz") as package:
        members = {member.name for member in package}

    assert "nemo_skills/__init__.py" in members
    assert "nvflow/project/packaging.py" in members
    assert "projects/packaging-demo/project.yaml" in members
    assert "unrelated_package/secret.py" not in members


def test_wheel_platform_source_rejects_modified_record_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root, distribution, _ = _wheel_distribution(tmp_path)
    (package_root / "worker.py").write_text("WORKER = False\n", encoding="utf-8")
    monkeypatch.setattr(project_packaging, "_tracked_package_files", lambda _: [])
    monkeypatch.setattr(project_packaging.metadata, "distribution", lambda _: distribution)

    with pytest.raises(ProjectManifestError, match="does not match RECORD"):
        project_packaging._platform_package_source(package_root)


def test_wheel_platform_source_requires_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root, distribution, _ = _wheel_distribution(
        tmp_path,
        unhashed_package_file=True,
    )
    monkeypatch.setattr(project_packaging, "_tracked_package_files", lambda _: [])
    monkeypatch.setattr(project_packaging.metadata, "distribution", lambda _: distribution)

    with pytest.raises(ProjectManifestError, match="RECORD has no hash"):
        project_packaging._platform_package_source(package_root)


def test_wheel_platform_source_rejects_record_path_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "site-packages" / "nvflow"
    package_root.mkdir(parents=True)

    class EscapingDistribution:
        files = [metadata.PackagePath("nvflow/../escape.py")]
        version = "1.2.3"

        def __init__(self) -> None:
            self.metadata = {"Name": "nvflow"}

        def locate_file(self, recorded: metadata.PackagePath) -> Path:
            return package_root.parent / recorded

    distribution = EscapingDistribution()
    monkeypatch.setattr(project_packaging, "_tracked_package_files", lambda _: [])
    monkeypatch.setattr(project_packaging.metadata, "distribution", lambda _: distribution)

    with pytest.raises(ProjectManifestError, match="RECORD path escapes"):
        project_packaging._platform_package_source(package_root)


def test_package_contains_only_installed_sources_and_selected_project_subtree(
    tmp_path: Path,
) -> None:
    repository, bundle, _ = _git_project(tmp_path)
    (repository / "outside-untracked.jsonl").write_text("{}\n", encoding="utf-8")
    job_dir = tmp_path / "jobs"
    job_dir.mkdir()
    base_packager = _BasePackager()
    packager = ProjectOverlayPackager(
        base_packager=base_packager,
        project_root=str(bundle),
        destination="projects/packaging-demo",
    )

    archive = packager.package(repository, str(job_dir), "external")
    assert not list(job_dir.glob("*_project_*.tar.gz"))
    with tarfile.open(archive, "r:gz") as package:
        members = {member.name for member in package}

    assert "nemo_skills/__init__.py" in members
    assert "nvflow/__init__.py" in members
    assert "projects/packaging-demo/project.yaml" in members
    assert "projects/packaging-demo/assets/message.txt" in members
    assert "unrelated.jsonl" not in members
    assert "outside-untracked.jsonl" not in members
    assert "projects/demo/project.yaml" not in members
    assert not any("__pycache__" in PurePosixPath(name).parts for name in members)
    assert not any(PurePosixPath(name).suffix in {".pyc", ".pyo"} for name in members)

    extraction = tmp_path / "extracted"
    with tarfile.open(archive, "r:gz") as package:
        nested = package.extractfile("projects/packaging-demo/assets/nested.yaml")
        assert nested is not None
        assert (
            nested.read().decode()
            == "message_path: /nemo_run/code/projects/packaging-demo/assets/message.txt\n"
        )
        package.extractall(extraction, filter="data")
    imported = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                f"import sys; sys.path.insert(0, {str(extraction)!r}); "
                "import nemo_skills; print(nemo_skills.PACKAGE)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert imported.stdout.strip() == "installed-nemo-skills"

    sys.path.insert(
        0,
        str(extraction / "projects" / "packaging-demo" / "src"),
    )
    try:
        assert import_module("packaged_project").VALUE == "worker import succeeded"
    finally:
        sys.path.pop(0)
        sys.modules.pop("packaged_project", None)


def test_installed_package_source_requires_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_distribution(name: str):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(project_packaging.metadata, "distribution", missing_distribution)

    with pytest.raises(ProjectManifestError, match="Required worker package is not installed"):
        project_packaging._installed_package_source("nemo-skills", "nemo_skills")


def test_installed_nemo_skills_source_is_record_verified() -> None:
    source = project_packaging._installed_package_source("nemo-skills", "nemo_skills")

    assert source.root.name == "nemo_skills"
    assert source.files
    assert source.identity.startswith("wheel:nemo_skills==")
    assert source.content_digest is not None
    assert project_packaging._package_source_digest(source) == source.content_digest
    assert all(path.is_relative_to(source.root) for path in source.files)


def test_project_provenance_is_deterministic_and_marks_dirty_source(tmp_path: Path) -> None:
    _, bundle, _ = _git_project(tmp_path)
    destination = PurePosixPath("/nemo_run/code/projects/packaging-demo")

    first = inspect_project_source(bundle, destination)
    second = inspect_project_source(bundle, destination)

    assert first.provenance == second.provenance
    assert first.provenance.clean is True
    assert first.provenance.submittable is True
    assert first.provenance.commit
    assert first.provenance.source_subpath == "projects/demo"
    assert first.provenance.content_digest is not None
    assert first.provenance.content_digest.startswith("sha256:")

    (bundle / "assets" / "message.txt").write_text("dirty\n", encoding="utf-8")
    dirty = inspect_project_source(bundle, destination)
    assert dirty.provenance.clean is False
    assert dirty.provenance.submittable is False
    assert dirty.provenance.dirty_paths


def test_clean_project_provenance_uses_three_git_queries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, bundle, _ = _git_project(tmp_path)
    calls = []
    run = project_provenance.subprocess.run

    def tracked_run(command, *args, **kwargs):
        calls.append(command)
        return run(command, *args, **kwargs)

    monkeypatch.setattr(project_provenance.subprocess, "run", tracked_run)

    inspect_project_source(
        bundle,
        PurePosixPath("/nemo_run/code/projects/packaging-demo"),
    )

    assert len(calls) == 3


def test_nested_project_submodule_is_in_provenance_and_archive(tmp_path: Path) -> None:
    dependency = tmp_path / "dependency"
    dependency.mkdir()
    _git(dependency, "init", "-q")
    (dependency / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(dependency, "add", ".")
    _git(
        dependency,
        "-c",
        "user.name=NVFlow Tests",
        "-c",
        "user.email=nvflow-tests@example.com",
        "commit",
        "-qm",
        "dependency",
    )

    root, bundle, _ = _git_project(tmp_path / "parent")
    _git(
        root,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "-q",
        str(dependency),
        "projects/demo/vendor/dependency",
    )
    _git(
        root,
        "-c",
        "user.name=NVFlow Tests",
        "-c",
        "user.email=nvflow-tests@example.com",
        "commit",
        "-qm",
        "pin nested dependency",
    )

    snapshot = inspect_project_source(
        bundle,
        PurePosixPath("/nemo_run/code/projects/packaging-demo"),
    )
    archive = tmp_path / "project.tar.gz"
    project_packaging._archive_project_tree(bundle, archive)

    assert snapshot.provenance.submittable is True
    with tarfile.open(archive, "r:gz") as packaged:
        assert "vendor/dependency/helper.py" in packaged.getnames()


def test_submodule_provenance_requires_parent_gitlink_agreement(tmp_path: Path) -> None:
    source = tmp_path / "project-source"
    (source / "src" / "submodule_project").mkdir(parents=True)
    (source / "project.yaml").write_text(
        "manifest_version: 1\n"
        "project: submodule\n"
        "requires_nvflow: '>=0'\n"
        "python_source_root: src\n",
        encoding="utf-8",
    )
    (source / "pyproject.toml").write_text(
        "[project]\nname='submodule-project'\nversion='0.1.0'\ndependencies=['nvflow>=0']\n",
        encoding="utf-8",
    )
    (source / "src" / "submodule_project" / "__init__.py").write_text("", encoding="utf-8")
    _git(source, "init", "-q")
    _git(source, "add", ".")
    _git(
        source,
        "-c",
        "user.name=NVFlow Tests",
        "-c",
        "user.email=nvflow-tests@example.com",
        "commit",
        "-qm",
        "initial",
    )

    superproject = tmp_path / "superproject"
    superproject.mkdir()
    _git(superproject, "init", "-q")
    _git(
        superproject,
        "-c",
        "user.name=NVFlow Tests",
        "-c",
        "user.email=nvflow-tests@example.com",
        "commit",
        "--allow-empty",
        "-qm",
        "initial",
    )
    _git(
        superproject,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "-q",
        str(source),
        "projects/submodule",
    )
    _git(superproject, "add", ".")
    _git(
        superproject,
        "-c",
        "user.name=NVFlow Tests",
        "-c",
        "user.email=nvflow-tests@example.com",
        "commit",
        "-qm",
        "pin project",
    )

    project = superproject / "projects" / "submodule"
    context = ProjectResolver(reference_root=superproject / "projects").resolve_reference(
        "submodule"
    )
    clean = inspect_project_source(project, PurePosixPath("/nemo_run/code/projects/submodule"))

    assert context.root == project.resolve()
    assert clean.git_root == project.resolve()
    assert clean.provenance.source_subpath == "."
    assert clean.provenance.submittable is True

    outer = tmp_path / "outer"
    outer.mkdir()
    _git(outer, "init", "-q")
    _git(
        outer,
        "-c",
        "user.name=NVFlow Tests",
        "-c",
        "user.email=nvflow-tests@example.com",
        "commit",
        "--allow-empty",
        "-qm",
        "initial",
    )
    _git(
        outer,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "-q",
        str(superproject),
        "catalog",
    )
    _git(
        outer / "catalog",
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "update",
        "--init",
        "--recursive",
    )
    _git(outer, "add", ".")
    _git(
        outer,
        "-c",
        "user.name=NVFlow Tests",
        "-c",
        "user.email=nvflow-tests@example.com",
        "commit",
        "-qm",
        "pin catalog",
    )
    nested_project = outer / "catalog" / "projects" / "submodule"
    nested_clean = inspect_project_source(
        nested_project,
        PurePosixPath("/nemo_run/code/projects/submodule"),
    )
    assert nested_clean.provenance.submittable is True

    (outer / "catalog" / "marker.txt").write_text("advanced\n", encoding="utf-8")
    _git(outer / "catalog", "add", "marker.txt")
    _git(
        outer / "catalog",
        "-c",
        "user.name=NVFlow Tests",
        "-c",
        "user.email=nvflow-tests@example.com",
        "commit",
        "-qm",
        "advance catalog",
    )
    nested_mismatch = inspect_project_source(
        nested_project,
        PurePosixPath("/nemo_run/code/projects/submodule"),
    )
    assert nested_mismatch.provenance.submittable is False
    assert any("submodule catalog HEAD" in path for path in nested_mismatch.provenance.dirty_paths)

    (project / "README.md").write_text("advanced\n", encoding="utf-8")
    _git(project, "add", "README.md")
    _git(
        project,
        "-c",
        "user.name=NVFlow Tests",
        "-c",
        "user.email=nvflow-tests@example.com",
        "commit",
        "-qm",
        "advance submodule",
    )
    mismatched = inspect_project_source(
        project,
        PurePosixPath("/nemo_run/code/projects/submodule"),
    )

    assert mismatched.provenance.submittable is False
    assert any(
        "does not match parent gitlink" in path for path in mismatched.provenance.dirty_paths
    )
    with pytest.raises(ProjectManifestError, match="parent gitlink"):
        inspect_project_source(
            project,
            PurePosixPath("/nemo_run/code/projects/submodule"),
            strict=True,
        )


def test_untracked_project_provenance_is_reported_not_raised(tmp_path: Path) -> None:
    repository, _, _ = _git_project(tmp_path)
    untracked = repository / "projects" / "untracked"
    untracked.mkdir()
    (untracked / "project.yaml").write_text(
        "manifest_version: 1\nproject: untracked\n",
        encoding="utf-8",
    )

    snapshot = inspect_project_source(
        untracked,
        PurePosixPath("/nemo_run/code/projects/untracked"),
    )

    assert snapshot.provenance.submittable is False
    assert snapshot.provenance.content_digest is None
    assert snapshot.provenance.reason == "Project subtree is not present in the current commit"


def test_packaging_rejects_revision_change_after_planning(tmp_path: Path) -> None:
    repository, bundle, _ = _git_project(tmp_path)
    destination = PurePosixPath("/nemo_run/code/projects/packaging-demo")
    planned = inspect_project_source(bundle, destination).provenance
    asset = bundle / "assets" / "message.txt"
    asset.write_text("new revision\n", encoding="utf-8")
    _git(repository, "add", str(asset.relative_to(repository)))
    _git(
        repository,
        "-c",
        "user.name=NVFlow Tests",
        "-c",
        "user.email=nvflow-tests@example.com",
        "commit",
        "-qm",
        "advance",
    )
    job_dir = tmp_path / "revision-jobs"
    job_dir.mkdir()
    packager = ProjectOverlayPackager(
        base_packager=_BasePackager(),
        project_root=str(bundle),
        destination="projects/packaging-demo",
        expected_provenance=planned,
    )

    with pytest.raises(ProjectManifestError, match="changed after planning"):
        packager.package(repository, str(job_dir), "revision")


@pytest.mark.parametrize("change", ["modified", "staged", "untracked"])
def test_package_rejects_dirty_selected_subtree(tmp_path: Path, change: str) -> None:
    repository, bundle, _ = _git_project(tmp_path)
    asset = bundle / "assets" / "message.txt"
    if change == "modified":
        asset.write_text("modified\n", encoding="utf-8")
    elif change == "staged":
        asset.write_text("staged\n", encoding="utf-8")
        _git(repository, "add", str(asset.relative_to(repository)))
    else:
        (bundle / "assets" / "new.txt").write_text("untracked\n", encoding="utf-8")

    job_dir = tmp_path / "jobs"
    job_dir.mkdir()
    packager = ProjectOverlayPackager(
        base_packager=_BasePackager(),
        project_root=str(bundle),
        destination="projects/packaging-demo",
    )
    with pytest.raises(ProjectManifestError, match="must be clean"):
        packager.package(repository, str(job_dir), "dirty")


def test_packager_context_is_scoped_for_reference_and_external_projects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, bundle, config = _git_project(tmp_path)
    external = ProjectResolver().resolve(config, project_dir=bundle)
    reference = ProjectResolver().resolve_reference("finance")
    original = _BasePackager()
    assert external.worker_python_root.endswith("/projects/packaging-demo/src")

    external_executor = _Executor(original)
    monkeypatch.setenv("PYTHONPATH", "/launcher/python")
    with project_submission(external):
        cluster_config = apply_project_environment(
            {"env_vars": ["PYTHONPATH=/custom/python", "OTHER=value"]}
        )
        inherited_config = apply_project_environment({"env_vars": ["PYTHONPATH"]})
        apply_project_packager(external_executor)
    assert isinstance(external_executor.packager, ProjectOverlayPackager)
    assert cluster_config["env_vars"] == [
        "PYTHONPATH=/nemo_run/code/projects/packaging-demo/src:/custom/python:/nemo_run/code",
        "OTHER=value",
    ]
    assert inherited_config["env_vars"] == [
        "PYTHONPATH=/nemo_run/code/projects/packaging-demo/src:/launcher/python:/nemo_run/code"
    ]

    reference_executor = _Executor(original)
    with project_submission(reference):
        apply_project_packager(reference_executor)
    assert isinstance(reference_executor.packager, ProjectOverlayPackager)

    after_scope = _Executor(original)
    apply_project_packager(after_scope)
    assert after_scope.packager is original
