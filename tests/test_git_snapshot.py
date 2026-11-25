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
"""Tests for history-free snapshots with materialized submodules."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest

from nvflow.project.snapshot import SnapshotError, materialized_files


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _commit(root: Path, message: str) -> None:
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=NVFlow Tests",
        "-c",
        "user.email=nvflow-tests@example.com",
        "commit",
        "-qm",
        message,
    )


def test_materialized_files_expand_clean_pinned_submodule(tmp_path: Path) -> None:
    dependency = tmp_path / "dependency"
    dependency.mkdir()
    _git(dependency, "init", "-q")
    (dependency / "helper.py").write_text("VALUE = 2\n", encoding="utf-8")
    _commit(dependency, "dependency")

    project = tmp_path / "project"
    project.mkdir()
    _git(project, "init", "-q")
    (project / "project.yaml").write_text("project: demo\n", encoding="utf-8")
    (project / "src").mkdir()
    (project / "src" / "stage.py").write_text("VALUE = 1\n", encoding="utf-8")
    _commit(project, "project")
    _git(
        project,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "-q",
        str(dependency),
        "vendor/dependency",
    )
    _commit(project, "pin dependency")

    root = tmp_path / "root"
    root.mkdir()
    _git(root, "init", "-q")
    (root / "README.md").write_text("root\n", encoding="utf-8")
    _commit(root, "root")
    _git(
        root,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "-q",
        str(project),
        "projects/demo",
    )
    _git(
        root / "projects" / "demo",
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "update",
        "--init",
        "--recursive",
    )
    _commit(root, "pin project")

    paths = materialized_files(root)

    assert PurePosixPath("README.md") in paths
    assert PurePosixPath(".gitmodules") in paths
    assert PurePosixPath("projects/demo/project.yaml") in paths
    assert PurePosixPath("projects/demo/src/stage.py") in paths
    assert PurePosixPath("projects/demo/vendor/dependency/helper.py") in paths
    assert PurePosixPath("projects/demo") not in paths

    (root / "README.md").write_text("changed\n", encoding="utf-8")
    with pytest.raises(SnapshotError, match="Source tree is dirty"):
        materialized_files(root)


def test_materialized_files_reject_dirty_or_advanced_submodule(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    _git(project, "init", "-q")
    (project / "project.yaml").write_text("project: demo\n", encoding="utf-8")
    _commit(project, "project")

    root = tmp_path / "root"
    root.mkdir()
    _git(root, "init", "-q")
    _git(
        root,
        "-c",
        "user.name=NVFlow Tests",
        "-c",
        "user.email=nvflow-tests@example.com",
        "commit",
        "--allow-empty",
        "-qm",
        "root",
    )
    _git(
        root,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "-q",
        str(project),
        "projects/demo",
    )
    _commit(root, "pin project")
    checkout = root / "projects" / "demo"

    (checkout / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(SnapshotError, match="dirty"):
        materialized_files(root)
    (checkout / "dirty.txt").unlink()

    (checkout / "advanced.txt").write_text("advanced\n", encoding="utf-8")
    _commit(checkout, "advance")
    with pytest.raises(SnapshotError, match="expected parent gitlink"):
        materialized_files(root)

    _git(root, "add", "projects/demo")
    with pytest.raises(SnapshotError, match="staged changes"):
        materialized_files(root)


def test_materialize_script_excludes_internal_paths(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    _git(root, "init", "-q")
    paths = (
        "keep.txt",
        ".agents/private.txt",
        ".codex/private.txt",
        "tests/golden/recipe_config/golden.txt",
        "tests/test_recipe_config_golden.py",
    )
    for relative in paths:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{relative}\n", encoding="utf-8")
    _commit(root, "tracked files")

    output = tmp_path / "tracked"
    script = Path(__file__).parents[1] / "scripts" / "materialize_git_snapshot.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--root",
            str(root),
            "--output",
            str(output),
            "--exclude",
            ".agents",
            "--exclude",
            ".codex",
            "--exclude",
            "tests/golden/recipe_config",
            "--exclude",
            "tests/test_recipe_config_golden.py",
        ],
        check=True,
    )

    materialized = {
        PurePosixPath(value.decode()) for value in output.read_bytes().split(b"\0") if value
    }
    assert materialized == {PurePosixPath("keep.txt")}
