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
"""Verify that the installed NVFlow distribution contains platform code only."""

import os
import subprocess
import sys
import venv
import zipfile
from email.parser import Parser
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from scripts.check_worker_python313 import (
    PROJECT_PACKAGES,
    check_project_imports,
    check_worker_syntax,
)

PROJECT_ROOT = Path(__file__).parents[1]
DEPENDENCY_PATHS = [path for path in sys.path if path and Path(path).name == "site-packages"]


def _venv_python(environment: Path) -> Path:
    directory = "Scripts" if os.name == "nt" else "bin"
    executable = "python.exe" if os.name == "nt" else "python"
    return environment / directory / executable


def test_wheel_contains_project_recipe_contracts_but_no_concrete_projects(
    tmp_path: Path,
) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(tmp_path),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = next(tmp_path.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        members = set(archive.namelist())
        metadata_member = next(name for name in members if name.endswith(".dist-info/METADATA"))
        metadata = Parser().parsestr(archive.read(metadata_member).decode())

    expected = {
        "nvflow/execution/runner.py",
        "nvflow/integrations/nemo_gym/compare.py",
        "nvflow/integrations/nemo_gym/contracts.py",
        "nvflow/integrations/nemo_gym/environments.py",
        "nvflow/integrations/nemo_gym/generation.py",
        "nvflow/integrations/nemo_gym/launcher_fs.py",
        "nvflow/integrations/nemo_gym/native_rewards.py",
        "nvflow/integrations/nemo_gym/native_rollout.py",
        "nvflow/integrations/nemo_gym/native_sharding.py",
        "nvflow/integrations/nemo_gym/operations.py",
        "nvflow/integrations/nemo_gym/postprocess.py",
        "nvflow/integrations/nemo_gym/reward_profiles.py",
        "nvflow/integrations/nemo_gym/workers/build_training_dataset.py",
        "nvflow/integrations/nemo_gym/workers/shuffle_jsonl.py",
        "nvflow/integrations/nemo_gym/workers/validate_dataset_selection.py",
        "nvflow/integrations/nemo_gym/workers/verify.py",
        "nvflow/integrations/nemo_skills/filesystem.py",
        "nvflow/integrations/nemo_skills/planning.py",
        "nvflow/integrations/nemo_skills/commands.py",
        "nvflow/stages/nemo_gym/__init__.py",
        "nvflow/stages/nemo_gym/_common.py",
        "nvflow/stages/nemo_gym/collate_dataset.py",
        "nvflow/stages/nemo_gym/collect_rollouts.py",
        "nvflow/stages/nemo_gym/compare_rollouts.py",
        "nvflow/stages/nemo_gym/prepare_benchmark.py",
        "nvflow/stages/nemo_gym/rejudge_rollouts.py",
        "nvflow/stages/nemo_gym/build_training_dataset.py",
        "nvflow/stages/catalog.py",
        "nvflow/project/__init__.py",
        "nvflow/project/context.py",
        "nvflow/project/manifest.py",
        "nvflow/project/packaging.py",
        "nvflow/project/provenance.py",
        "nvflow/project/resolver.py",
        "nvflow/project/resources.py",
        "nvflow/project/workflow_selector.py",
        "nvflow/recipe/__init__.py",
        "nvflow/recipe/definition.py",
        "nvflow/recipe/planning.py",
        "nvflow/recipe/resolver.py",
        "nvflow/project/environment.py",
        "nvflow/core/platform.py",
        "nvflow/project/install_source.py",
        "nvflow/project/snapshot.py",
        "nvflow/project/stage_type.py",
        "nvflow/project/_environment_policy/overrides.txt",
        "nvflow/project/_environment_policy/constraints.txt",
        "nvflow/project/_environment_policy/execution.txt",
        "nvflow/core/workflow_structure.py",
        "nvflow/cli/bootstrap.py",
        "nvflow/cli/project_environment.py",
        "nvflow/cli/run_workflow.py",
        "nvflow/cli/workflow_inspection.py",
    }
    assert expected <= members
    assert not any(member.startswith("nvflow/execution/adapters/") for member in members)
    assert "nvflow/core/workflow_runner.py" not in members
    assert "nvflow/recipe/packaging.py" not in members
    assert not any(member.startswith(("projects/", "recipes/", "starters/")) for member in members)
    assert not any(member.startswith("tests/fixtures/projects/") for member in members)
    assert not any(
        prefix in member for member in members for prefix in ("nvflow_recipe_", "nvflow_project_")
    )
    requirements = [Requirement(value) for value in metadata.get_all("Requires-Dist", [])]
    by_name = {canonicalize_name(requirement.name): requirement for requirement in requirements}
    assert by_name[canonicalize_name("pyyaml")].marker is None
    for name in ("nemo-skills", "torch"):
        requirement = by_name[canonicalize_name(name)]
        assert requirement.marker is not None
        assert "execution" in str(requirement.marker)

    environment = tmp_path / "platform-venv"
    venv.EnvBuilder(with_pip=True).create(environment)
    python = _venv_python(environment)
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-deps",
            "--no-index",
            str(wheel),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [str(python), "-I", "-c", "import nvflow; print(nvflow.__version__)"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    verified = subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            (
                "import sys; "
                f"sys.path.extend({DEPENDENCY_PATHS!r}); "
                "from pathlib import Path; "
                "import nvflow; "
                "from nvflow.project.packaging import _platform_package_source; "
                "from nvflow.stages.catalog import list_stage_types, resolve_stage_type; "
                "source = _platform_package_source(Path(nvflow.__file__).parent); "
                "resolved = [resolve_stage_type(spec.name) for spec in list_stage_types()]; "
                "print(source.identity, len(source.files), len(resolved))"
            ),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert verified.stdout.startswith("wheel:nvflow==")


def test_reference_project_wheels_build_install_and_import(tmp_path: Path) -> None:
    for project_name, project_root, module_name in PROJECT_PACKAGES:
        wheel_dir = tmp_path / "wheels" / project_name
        wheel_dir.mkdir(parents=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--no-isolation",
                "--outdir",
                str(wheel_dir),
                str(project_root),
            ],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        wheel = next(wheel_dir.glob("*.whl"))

        environment = tmp_path / "venvs" / project_name
        venv.EnvBuilder(with_pip=True).create(environment)
        python = _venv_python(environment)
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-deps",
                "--no-index",
                str(wheel),
            ],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )
        imported = subprocess.run(
            [
                str(python),
                "-I",
                "-c",
                f"import {module_name}; print({module_name}.__file__)",
            ],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )
        assert "site-packages" in imported.stdout


def test_project_packages_keep_python_313_worker_import_boundary() -> None:
    check_project_imports((sys.executable,))


def test_integration_workers_keep_python_313_syntax() -> None:
    check_worker_syntax()
