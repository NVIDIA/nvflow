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
"""Tests for the versioned project manifest and resolver."""

from pathlib import Path

import pytest

from nvflow.project import ProjectManifest, ProjectManifestError, ProjectResolver

PROJECT_ROOT = Path(__file__).parents[2]
REFERENCE_ROOT = PROJECT_ROOT / "projects"
FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "projects"


@pytest.mark.parametrize(
    ("project", "root"),
    [
        ("finance", REFERENCE_ROOT / "finance"),
        ("hopchain", REFERENCE_ROOT / "hopchain"),
        ("starter", REFERENCE_ROOT / "starter"),
    ],
)
def test_project_manifests_use_version_one_contract(project: str, root: Path) -> None:
    manifest = ProjectManifest.load(root / "project.yaml")

    assert manifest.project == project
    assert manifest.manifest_version == 1
    assert manifest.python_source_root == "."


def test_default_discovery_finds_all_projects() -> None:
    expected = sorted(
        path.parent.name for path in (PROJECT_ROOT / "projects").glob("*/project.yaml")
    )
    assert expected
    assert ProjectResolver().project_names() == expected


def test_manifest_rejects_unknown_fields(tmp_path: Path) -> None:
    manifest_path = tmp_path / "project.yaml"
    manifest_path.write_text(
        "manifest_version: 1\nproject: bad\nunexpected: true\n",
        encoding="utf-8",
    )

    with pytest.raises(ProjectManifestError, match="unknown field"):
        ProjectManifest.load(manifest_path)


def test_manifest_version_is_required(tmp_path: Path) -> None:
    manifest_path = tmp_path / "project.yaml"
    manifest_path.write_text("project: bad\n", encoding="utf-8")

    with pytest.raises(ProjectManifestError, match="must declare manifest_version: 1"):
        ProjectManifest.load(manifest_path)


def test_manifest_version_must_be_an_integer(tmp_path: Path) -> None:
    manifest_path = tmp_path / "project.yaml"
    manifest_path.write_text(
        "manifest_version: 1.0\nproject: bad\n",
        encoding="utf-8",
    )

    with pytest.raises(ProjectManifestError, match="unsupported manifest_version"):
        ProjectManifest.load(manifest_path)


def test_manifest_rejects_removed_workflow_order(tmp_path: Path) -> None:
    manifest_path = tmp_path / "project.yaml"
    manifest_path.write_text(
        "manifest_version: 1\nproject: example\nworkflow_order: [training]\n",
        encoding="utf-8",
    )

    with pytest.raises(ProjectManifestError, match="unknown field.*workflow_order"):
        ProjectManifest.load(manifest_path)


def test_manifest_rejects_python_source_escape(tmp_path: Path) -> None:
    manifest_path = tmp_path / "project.yaml"
    manifest_path.write_text(
        "manifest_version: 1\nproject: bad\npython_source_root: ../outside\n",
        encoding="utf-8",
    )

    with pytest.raises(ProjectManifestError, match="inside the project"):
        ProjectManifest.load(manifest_path)


def test_resolver_discovers_external_project_from_workflow() -> None:
    config = FIXTURE_ROOT / "python_project" / "workflows" / "noop.yaml"
    context = ProjectResolver().resolve(config)

    assert context.is_external is True
    assert context.identity == "fixture-python"
    assert context.root == (FIXTURE_ROOT / "python_project").resolve()


def test_resolver_rejects_nested_manifests(tmp_path: Path) -> None:
    outer = tmp_path / "project.yaml"
    inner_root = tmp_path / "nested"
    inner_root.mkdir()
    outer.write_text("manifest_version: 1\nproject: outer\n", encoding="utf-8")
    (inner_root / "project.yaml").write_text(
        "manifest_version: 1\nproject: inner\n",
        encoding="utf-8",
    )
    workflow = inner_root / "workflow.yaml"
    workflow.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ProjectManifestError, match="Nested project manifests"):
        ProjectResolver().resolve(workflow)


def test_project_uri_rejects_symlink_escape(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    workflows = bundle / "workflows"
    workflows.mkdir(parents=True)
    (bundle / "project.yaml").write_text(
        "manifest_version: 1\nproject: symlink-test\n",
        encoding="utf-8",
    )
    workflow = workflows / "workflow.yaml"
    workflow.write_text("{}\n", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")
    (bundle / "escape.txt").symlink_to(outside)

    context = ProjectResolver().resolve(workflow, project_dir=bundle)
    with pytest.raises(ProjectManifestError, match="escapes project root"):
        context.launcher_path("project://escape.txt")


def test_resolver_uses_ordered_named_search_paths(tmp_path: Path) -> None:
    first = tmp_path / "first" / "searched"
    second = tmp_path / "second" / "searched"
    for root, marker in ((first, "first"), (second, "second")):
        (root / "src").mkdir(parents=True)
        (root / "project.yaml").write_text(
            "manifest_version: 1\nproject: searched\npython_source_root: src\n",
            encoding="utf-8",
        )
        (root / "marker.txt").write_text(marker, encoding="utf-8")

    context = ProjectResolver(search_paths=(first.parent, second.parent)).resolve_named("searched")

    assert context.root == first.resolve()
    assert (context.root / "marker.txt").read_text(encoding="utf-8") == "first"


def test_project_context_maps_worker_paths_back_to_launcher() -> None:
    context = ProjectResolver().resolve_reference("finance")

    launcher = context.launcher_equivalent(
        "/nemo_run/code/projects/finance/prompts/secque_template.yaml"
    )

    assert launcher == (REFERENCE_ROOT / "finance" / "prompts" / "secque_template.yaml")
