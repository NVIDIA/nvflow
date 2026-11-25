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
"""Workflow selector parsing, resolution, and lightweight discovery tests."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import typer

from nvflow.cli.workflow_selection import discover_workflows
from nvflow.project import (
    ProjectResolutionError,
    ProjectResolver,
    WorkflowSelector,
    resolve_workflow_selector,
)

PROJECT_ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    ("value", "canonical"),
    [
        ("starter:generate", "starter:generate"),
        (
            "finance:grpo/nemotron_nano_30b_a3b.yaml",
            "finance:grpo/nemotron_nano_30b_a3b",
        ),
        ("team.recipe:training/model.v1", "team.recipe:training/model.v1"),
    ],
)
def test_workflow_selector_normalizes_valid_values(value: str, canonical: str) -> None:
    assert str(WorkflowSelector.parse(value)) == canonical


@pytest.mark.parametrize(
    "value",
    [
        "answer_generation",
        ":answer_generation",
        "example:",
        "example:/absolute",
        "example:../outside",
        "example:nested/../../outside",
        r"example:nested\workflow",
        " starter:generate",
    ],
)
def test_workflow_selector_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ProjectResolutionError):
        WorkflowSelector.parse(value)


def test_reference_workflow_selector_resolves_without_yaml_suffix() -> None:
    resolver = ProjectResolver(reference_root=PROJECT_ROOT / "projects")

    context = resolve_workflow_selector("finance:download/demo", resolver)

    assert context.identity == "finance"
    assert (
        context.config_path
        == (PROJECT_ROOT / "projects/finance/workflows/download/demo.yaml").resolve()
    )


def test_workflow_selector_rejects_missing_and_ambiguous_configs(tmp_path: Path) -> None:
    project = _external_project(tmp_path, "demo")
    workflows = project / "workflows"
    (workflows / "duplicate.yaml").write_text(
        _workflow_yaml(runnable=True),
        encoding="utf-8",
    )
    (workflows / "duplicate.yml").write_text(
        _workflow_yaml(runnable=True),
        encoding="utf-8",
    )
    resolver = ProjectResolver(search_paths=(tmp_path,))

    with pytest.raises(ProjectResolutionError, match="was not found"):
        resolve_workflow_selector("demo:missing", resolver)
    with pytest.raises(ProjectResolutionError, match="ambiguous"):
        resolve_workflow_selector("demo:duplicate", resolver)
    with pytest.raises(typer.BadParameter, match="ambiguous"):
        discover_workflows(
            project="demo",
            project_dir=None,
            project_paths=(str(tmp_path),),
            include_all=False,
        )


def test_external_workflow_selector_uses_configured_search_path(tmp_path: Path) -> None:
    project = _external_project(tmp_path, "external")
    workflow = project / "workflows" / "nested" / "train.yaml"
    workflow.parent.mkdir()
    workflow.write_text("{}\n", encoding="utf-8")

    context = resolve_workflow_selector(
        "external:nested/train",
        ProjectResolver(search_paths=(tmp_path,)),
    )

    assert context.is_external
    assert context.config_path == workflow.resolve()


def test_named_project_resolves_from_arbitrarily_named_search_child(tmp_path: Path) -> None:
    project = _external_project(tmp_path, "external")
    renamed = tmp_path / "team-owned-folder"
    project.rename(renamed)

    context = ProjectResolver(search_paths=(tmp_path,)).resolve_named("external")

    assert context.root == renamed.resolve()


def test_external_workflow_selector_discovers_current_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _external_project(tmp_path, "external")
    workflow = project / "workflows" / "train.yaml"
    workflow.write_text("{}\n", encoding="utf-8")
    monkeypatch.chdir(project / "workflows")

    context = resolve_workflow_selector("external:train", ProjectResolver())

    assert context.is_external
    assert context.config_path == workflow.resolve()


def test_workflow_discovery_is_structural_and_template_aware(
    tmp_path: Path,
) -> None:
    project = _external_project(tmp_path, "external")
    workflows = project / "workflows"
    (workflows / "base.yaml").write_text(
        _workflow_yaml(runnable=False),
        encoding="utf-8",
    )
    (workflows / "child.yaml").write_text("_base_: base.yaml\n", encoding="utf-8")
    (workflows / "presets.yaml").write_text("presets:\n  small: {}\n", encoding="utf-8")
    visible = discover_workflows(
        project=None,
        project_dir=str(project),
        project_paths=(),
        include_all=False,
    )
    with_templates = discover_workflows(
        project=None,
        project_dir=str(project),
        project_paths=(),
        include_all=True,
    )

    assert [str(item.selector) for item in visible] == ["external:child"]
    assert [str(item.selector) for item in with_templates] == [
        "external:base",
        "external:child",
    ]
    assert with_templates[0].template is True
    assert with_templates[1].template is False
    assert all(listing.project_description is None for listing in with_templates)


def test_workflow_discovery_exposes_description_and_alphabetical_order(tmp_path: Path) -> None:
    project = _external_project(tmp_path, "external")
    (project / "project.yaml").write_text(
        "manifest_version: 1\nproject: external\ndescription: Ordered workflow example\n",
        encoding="utf-8",
    )
    workflows = project / "workflows"
    (workflows / "a.yaml").write_text(
        _workflow_yaml(runnable=True, name="first"),
        encoding="utf-8",
    )
    (workflows / "b.yaml").write_text(
        _workflow_yaml(runnable=True, name="second"),
        encoding="utf-8",
    )
    (workflows / "c.yaml").write_text(
        _workflow_yaml(runnable=True, name="zeta"),
        encoding="utf-8",
    )
    (workflows / "d.yaml").write_text(
        _workflow_yaml(runnable=True, name="alpha"),
        encoding="utf-8",
    )

    listings = discover_workflows(
        project=None,
        project_dir=str(project),
        project_paths=(),
        include_all=False,
    )

    assert [listing.workflow_name for listing in listings] == [
        "alpha",
        "first",
        "second",
        "zeta",
    ]
    assert {listing.project_description for listing in listings} == {"Ordered workflow example"}
    assert [listing.stage_count for listing in listings] == [1, 1, 1, 1]


def test_workflow_discovery_rejects_conflicting_types_within_name(
    tmp_path: Path,
) -> None:
    project = _external_project(tmp_path, "external")
    workflows = project / "workflows"
    (workflows / "first.yaml").write_text(
        _workflow_yaml(runnable=True),
        encoding="utf-8",
    )
    (workflows / "second.yaml").write_text(
        _workflow_yaml(runnable=True).replace("type: test", "type: other"),
        encoding="utf-8",
    )

    with pytest.raises(typer.BadParameter, match="conflicting workflow.type"):
        discover_workflows(
            project=None,
            project_dir=str(project),
            project_paths=(),
            include_all=False,
        )


def test_hidden_templates_still_participate_in_project_validation(
    tmp_path: Path,
) -> None:
    project = _external_project(tmp_path, "external")
    workflows = project / "workflows"
    (workflows / "first.yaml").write_text(
        _workflow_yaml(runnable=False),
        encoding="utf-8",
    )
    (workflows / "second.yaml").write_text(
        _workflow_yaml(runnable=False).replace("type: test", "type: other"),
        encoding="utf-8",
    )

    with pytest.raises(typer.BadParameter, match="conflicting workflow.type"):
        discover_workflows(
            project=None,
            project_dir=str(project),
            project_paths=(),
            include_all=False,
        )


def test_workflow_discovery_does_not_import_project_or_provider_code() -> None:
    script = textwrap.dedent(
        """
        import json
        import sys
        from nvflow.cli.workflow_selection import discover_workflows

        before = set(sys.modules)
        discover_workflows(
            project="finance",
            project_dir=None,
            project_paths=(),
            include_all=False,
        )
        loaded = sorted(
            name for name in set(sys.modules) - before
            if name.startswith("nemo_skills")
            or name.startswith("nemo_run")
            or name.startswith("nvflow_recipe_")
            or name.startswith("nvflow_project_")
        )
        print(json.dumps(loaded))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []


def test_workflow_discovery_parses_each_yaml_at_most_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nvflow.cli import workflow_selection

    calls = 0
    safe_load = workflow_selection.yaml.safe_load

    def counted(value):
        nonlocal calls
        calls += 1
        return safe_load(value)

    monkeypatch.setattr(workflow_selection.yaml, "safe_load", counted)
    listings = discover_workflows(
        project="finance",
        project_dir=None,
        project_paths=(),
        include_all=True,
    )
    yaml_files = {
        path
        for pattern in ("*.yaml", "*.yml")
        for path in (PROJECT_ROOT / "projects" / "finance" / "workflows").rglob(pattern)
    }

    assert listings
    assert calls == len(yaml_files)


def test_workflow_discovery_reports_malformed_yaml(tmp_path: Path) -> None:
    project = _external_project(tmp_path, "external")
    malformed = project / "workflows" / "bad.yaml"
    malformed.write_text("workflow: [\n", encoding="utf-8")

    with pytest.raises(typer.BadParameter, match="bad.yaml"):
        discover_workflows(
            project=None,
            project_dir=str(project),
            project_paths=(),
            include_all=False,
        )


def test_workflow_discovery_rejects_authored_project_identity(tmp_path: Path) -> None:
    project = _external_project(tmp_path, "external")
    (project / "workflows" / "workflow.yaml").write_text(
        "project: external\n" + _workflow_yaml(runnable=True),
        encoding="utf-8",
    )

    with pytest.raises(typer.BadParameter, match="must not declare project"):
        discover_workflows(
            project=None,
            project_dir=str(project),
            project_paths=(),
            include_all=False,
        )


def test_workflow_discovery_accepts_flow_style_top_level_mapping(tmp_path: Path) -> None:
    project = _external_project(tmp_path, "external")
    (project / "workflows" / "flow.yaml").write_text(
        "{workflow: {name: flow, type: test}, cluster: test, "
        "pipeline_stages: [one], stages: {one: {}}}\n",
        encoding="utf-8",
    )

    listings = discover_workflows(
        project=None,
        project_dir=str(project),
        project_paths=(),
        include_all=False,
    )

    assert [str(listing.selector) for listing in listings] == ["external:flow"]


def test_workflow_discovery_resolves_config_local_pipeline_reference(tmp_path: Path) -> None:
    project = _external_project(tmp_path, "external")
    (project / "workflows" / "referenced.yaml").write_text(
        "workflow: {name: referenced, type: test}\n"
        "cluster: test\n"
        "stage_order: [noop]\n"
        "pipeline_stages: ${stage_order}\n"
        "stages: {noop: {dependencies: []}}\n",
        encoding="utf-8",
    )

    listings = discover_workflows(
        project=None,
        project_dir=str(project),
        project_paths=(),
        include_all=False,
    )

    assert [str(listing.selector) for listing in listings] == ["external:referenced"]
    assert listings[0].stage_count == 1


def test_workflow_discovery_rejects_cyclic_or_escaping_base(tmp_path: Path) -> None:
    project = _external_project(tmp_path, "external")
    workflows = project / "workflows"
    (workflows / "first.yaml").write_text("_base_: second.yaml\n", encoding="utf-8")
    (workflows / "second.yaml").write_text("_base_: first.yaml\n", encoding="utf-8")

    with pytest.raises(typer.BadParameter, match="Cyclic workflow _base_ chain"):
        discover_workflows(
            project=None,
            project_dir=str(project),
            project_paths=(),
            include_all=False,
        )

    (workflows / "first.yaml").write_text("_base_: ../../outside.yaml\n", encoding="utf-8")
    (workflows / "second.yaml").unlink()
    (tmp_path / "outside.yaml").write_text("{}\n", encoding="utf-8")
    with pytest.raises(typer.BadParameter, match="outside project root"):
        discover_workflows(
            project=None,
            project_dir=str(project),
            project_paths=(),
            include_all=False,
        )


def _external_project(parent: Path, identity: str) -> Path:
    root = parent / identity
    (root / "workflows").mkdir(parents=True)
    (root / "project.yaml").write_text(
        f"manifest_version: 1\nproject: {identity}\n",
        encoding="utf-8",
    )
    return root


def _workflow_yaml(*, runnable: bool, name: str = "training") -> str:
    return (
        "workflow:\n"
        f"  name: {name}\n"
        "  type: test\n"
        f"  runnable: {'true' if runnable else 'false'}\n"
        "cluster: test\n"
        "pipeline_stages: [noop]\n"
        "stages:\n"
        "  noop: {}\n"
    )
