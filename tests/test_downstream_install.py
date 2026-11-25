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
"""Isolated resolution checks for project-owned NVFlow environments."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

from nvflow.project import ProjectResolver
from nvflow.project.environment import ProjectEnvironmentManager
from nvflow.project.install_source import (
    NVFlowInstallSource,
    active_nvflow_install_source,
)

PROJECT_ROOT = Path(__file__).parents[1]
STARTER_ROOT = PROJECT_ROOT / "projects" / "starter"

pytestmark = pytest.mark.requires_nemo


def _copy_clean_starter(destination: Path) -> None:
    """Copy the starter without ignored artifacts from local project runs."""
    shutil.copytree(
        STARTER_ROOT,
        destination,
        ignore=shutil.ignore_patterns(".venv", ".pytest_cache", "__pycache__", "*.pyc"),
    )


def _project_source_files(root: Path) -> set[Path]:
    """Return source files while ignoring normal Python bytecode artifacts."""
    return {
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }


@pytest.fixture(scope="module")
def nvflow_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("downstream-wheel")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return next(output.glob("nvflow-*.whl"))


def test_copied_starter_uv_sync_bootstraps_lightweight_cli(
    tmp_path: Path,
    nvflow_wheel: Path,
) -> None:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required for downstream resolution")
    assert uv is not None

    project_root = tmp_path / "copied-starter"
    _copy_clean_starter(project_root)

    environment = os.environ.copy()
    for key in tuple(environment):
        if key.startswith("UV_") or key in {
            "NVFLOW_CONFIG_DIR",
            "NVFLOW_DISABLE_PROJECT_ENVIRONMENT",
            "NVFLOW_PROJECT_PATH",
            "PYTHONPATH",
            "VIRTUAL_ENV",
        }:
            environment.pop(key)
    environment["UV_NO_PROGRESS"] = "1"
    environment["UV_FIND_LINKS"] = str(nvflow_wheel.parent)
    environment["UV_NO_SOURCES"] = "1"
    environment["NVFLOW_PROJECT_ENV_CACHE"] = str(tmp_path / "managed-cache")

    synced = subprocess.run(
        [uv, "sync", "--no-dev"],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert synced.returncode == 0, synced.stderr
    scripts = "Scripts" if os.name == "nt" else "bin"
    python = project_root / ".venv" / scripts / ("python.exe" if os.name == "nt" else "python")
    lightweight = subprocess.run(
        [
            str(python),
            "-c",
            (
                "from importlib.util import find_spec; import json; "
                "print(json.dumps({name: find_spec(name) is not None "
                "for name in ('nvflow', 'typer', 'click', 'nemo_skills', 'torch')}))"
            ),
        ],
        cwd=project_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(lightweight.stdout) == {
        "nvflow": True,
        "typer": True,
        "click": True,
        "nemo_skills": False,
        "torch": False,
    }
    version = subprocess.run(
        [uv, "run", "--no-sync", "nvflow", "version"],
        cwd=project_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert version.stdout.startswith("NVFlow version ")
    assert not (tmp_path / "managed-cache").exists()

    cold = subprocess.run(
        [uv, "run", "--no-sync", "nvflow", "validate", "--workflow", "starter:generate"],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert cold.returncode == 0, cold.stderr
    warm = subprocess.run(
        [uv, "run", "--no-sync", "nvflow", "validate", "--workflow", "starter:generate"],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert warm.returncode == 0, warm.stderr
    assert "Workflow definition is valid" in cold.stdout
    assert "Preparing full project execution environment" in cold.stderr
    assert "Preparing full project execution environment" not in warm.stderr
    assert (project_root / "uv.lock").is_file()

    stamps = list((tmp_path / "managed-cache").rglob(".nvflow-environment.json"))
    assert len(stamps) == 1
    managed_lock = (stamps[0].parent / "uv.lock").read_text(encoding="utf-8")
    assert 'name = "nemo-skills"' in managed_lock
    assert "17ae86b64d7f75653351664f5d8c9e466faede00" in managed_lock


@pytest.mark.parametrize("installation", ["wheel", "editable"])
def test_external_project_resolves_without_nvflow_root_state(
    tmp_path: Path,
    nvflow_wheel: Path,
    installation: str,
) -> None:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required for downstream resolution")
    assert uv is not None

    external_project_root = tmp_path / "copied-project"
    _copy_clean_starter(external_project_root)
    project_file = external_project_root / "pyproject.toml"
    project_config = tomllib.loads(project_file.read_text(encoding="utf-8"))
    assert set(project_config["tool"]["uv"]) == {"sources"}
    assert set(project_config["tool"]["uv"]["sources"]) == {"nvflow"}
    config = external_project_root / "workflows" / "generate.yaml"
    context = ProjectResolver().resolve(config, project_dir=external_project_root)
    before = _project_source_files(external_project_root)

    environment = os.environ.copy()
    for key in tuple(environment):
        if key.startswith("UV_") or key in {
            "NVFLOW_CONFIG_DIR",
            "NVFLOW_PROJECT_PATH",
            "PYTHONPATH",
            "VIRTUAL_ENV",
        }:
            environment.pop(key)
    environment["UV_NO_PROGRESS"] = "1"
    cache_root = tmp_path / "environment-cache"
    environment["NVFLOW_PROJECT_ENV_CACHE"] = str(cache_root)
    if installation == "wheel":
        source_url = nvflow_wheel.resolve().as_uri()
        source_hash = hashlib.sha256(nvflow_wheel.read_bytes()).hexdigest()
        source = NVFlowInstallSource(
            identity=f"archive:{source_url}#sha256={source_hash}",
            requirement=f"nvflow @ {source_url}#sha256={source_hash}",
            local_artifact=nvflow_wheel,
            content_hash=f"sha256={source_hash}",
        )
    else:
        source = active_nvflow_install_source()

    manager = ProjectEnvironmentManager(
        cache_root=cache_root,
        uv_executable=uv,
        environ=environment,
        source=source,
    )
    if installation == "editable":
        outer_environment = dict(environment)
        outer_environment.pop("NVFLOW_DISABLE_PROJECT_ENVIRONMENT", None)
        scripts = "Scripts" if os.name == "nt" else "bin"
        nvflow = "nvflow.exe" if os.name == "nt" else "nvflow"
        command = [
            str(PROJECT_ROOT / ".venv" / scripts / nvflow),
            "validate",
            "--workflow",
            "starter:generate",
        ]
        cold = subprocess.run(
            command,
            cwd=external_project_root,
            env=outer_environment,
            check=True,
            capture_output=True,
            text=True,
        )
        warm = subprocess.run(
            command,
            cwd=external_project_root,
            env=outer_environment,
            check=True,
            capture_output=True,
            text=True,
        )
        assert "Workflow definition is valid" in cold.stdout
        assert "Preparing full project execution environment" in cold.stderr
        assert "Preparing full project execution environment" not in warm.stderr
        project_environment = manager.check(context)
        assert project_environment is not None
    else:
        project_environment = manager.ensure(context)
        completed = subprocess.run(
            [
                str(project_environment.nvflow),
                "validate",
                "--config",
                str(config),
            ],
            cwd=tmp_path,
            env=manager.activation_environment(project_environment),
            check=True,
            capture_output=True,
            text=True,
        )
        assert "Workflow definition is valid" in completed.stdout
    assert project_environment.lock.is_file()

    versions: dict[str, list[Version]] = defaultdict(list)
    lock = tomllib.loads(project_environment.lock.read_text(encoding="utf-8"))
    for package in lock["package"]:
        if version := package.get("version"):
            versions[canonicalize_name(package["name"])].append(Version(version))
    platform = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    uv_policy = platform["tool"]["uv"]
    policy = [
        Requirement(requirement)
        for requirement in (
            *uv_policy["override-dependencies"],
            *uv_policy.get("constraint-dependencies", []),
        )
    ]
    assert {"httpx", "urllib3"} <= versions.keys()
    for requirement in policy:
        package_versions = versions.get(canonicalize_name(requirement.name), [])
        assert all(requirement.specifier.contains(version) for version in package_versions)
    assert not (external_project_root / ".venv").exists()
    assert not (external_project_root / "uv.lock").exists()
    assert _project_source_files(external_project_root) == before


def test_setuptools_project_environment_does_not_modify_source(tmp_path: Path) -> None:
    external_project_root = tmp_path / "setuptools-project"
    _copy_clean_starter(external_project_root)
    (external_project_root / "pyproject.toml").write_text(
        "[project]\n"
        "name = 'nvflow-project-starter'\n"
        "version = '0.1.0'\n"
        "requires-python = '>=3.13.14,<3.14'\n"
        "dependencies = ['nvflow>=1.1,<2']\n"
        "\n"
        "[build-system]\n"
        "requires = ['setuptools>=68']\n"
        "build-backend = 'setuptools.build_meta'\n"
        "\n"
        "[tool.setuptools.packages.find]\n"
        "where = ['src']\n",
        encoding="utf-8",
    )
    context = ProjectResolver().resolve(
        external_project_root / "workflows" / "generate.yaml",
        project_dir=external_project_root,
    )
    before = {
        path.relative_to(external_project_root)
        for path in external_project_root.rglob("*")
        if path.is_file()
    }
    manager = ProjectEnvironmentManager(cache_root=tmp_path / "cache")

    environment = manager.ensure(context)
    completed = subprocess.run(
        [str(environment.python), "-c", "import my_project"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert {
        path.relative_to(external_project_root)
        for path in external_project_root.rglob("*")
        if path.is_file()
    } == before


@pytest.mark.parametrize(
    ("project", "module", "workflow", "stage"),
    [
        ("finance", "nvfinance", "workflows/download/demo.yaml", "download"),
        (
            "hopchain",
            "nvhopchain",
            "workflows/image_filter/full.yaml",
            "image_filter",
        ),
        ("starter", "my_project", "workflows/generate.yaml", "generate_answer"),
    ],
)
def test_maintained_project_environment_excludes_siblings(
    tmp_path: Path,
    project: str,
    module: str,
    workflow: str,
    stage: str,
) -> None:
    context = ProjectResolver().resolve_reference(project)
    manager = ProjectEnvironmentManager(cache_root=tmp_path / "cache")

    environment = manager.ensure(context)
    completed = subprocess.run(
        [
            str(environment.python),
            "-c",
            (
                "from importlib.util import find_spec; import json; "
                "names=('nvfinance','nvhopchain','my_project'); "
                "print(json.dumps({name: find_spec(name) is not None for name in names}))"
            ),
        ],
        env=manager.activation_environment(environment),
        check=True,
        capture_output=True,
        text=True,
    )
    available = json.loads(completed.stdout)

    assert available == {name: name == module for name in ("nvfinance", "nvhopchain", "my_project")}
    resolved = subprocess.run(
        [
            str(environment.python),
            "-c",
            (
                "from nvflow.core import WorkflowDefinition; "
                f"definition=WorkflowDefinition({str(context.root / workflow)!r}); "
                f"print(definition.resolve_stage_class({stage!r}).__module__)"
            ),
        ],
        env=manager.activation_environment(environment),
        check=True,
        capture_output=True,
        text=True,
    )
    assert resolved.stdout.strip().startswith(module)
