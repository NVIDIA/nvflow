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
"""Tests for the target-independent public workflow definition."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import nvflow
from nvflow.core import BaseStage, WorkflowDefinition, load_workflow
from nvflow.core.workflow_definition import (
    expand_checkpoint_pipeline_stages,
    resolve_workflow_config_path,
)
from nvflow.execution import WorkflowRunner

PROJECT_ROOT = Path(__file__).parents[1]
FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "projects" / "python_project"


def test_workflow_rejects_authored_project_identity(tmp_path: Path) -> None:
    project = tmp_path / "project"
    workflows = project / "workflows"
    workflows.mkdir(parents=True)
    (project / "project.yaml").write_text(
        "manifest_version: 1\nproject: derived\n",
        encoding="utf-8",
    )
    config = workflows / "workflow.yaml"
    config.write_text(
        "project: stale\n"
        "workflow: {name: test, type: test}\n"
        "cluster: test\n"
        "pipeline_stages: [stage]\n"
        "stages: {stage: {dependencies: []}}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must not declare project"):
        load_workflow(config)


def test_workflow_loading_rejects_cyclic_and_escaping_base_files(tmp_path: Path) -> None:
    project = tmp_path / "project"
    workflows = project / "workflows"
    workflows.mkdir(parents=True)
    (project / "project.yaml").write_text(
        "manifest_version: 1\nproject: bounded\n",
        encoding="utf-8",
    )
    first = workflows / "first.yaml"
    second = workflows / "second.yaml"
    first.write_text("_base_: second.yaml\n", encoding="utf-8")
    second.write_text("_base_: first.yaml\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Cyclic workflow _base_ chain"):
        load_workflow(first)

    outside = tmp_path / "outside.yaml"
    outside.write_text("{}\n", encoding="utf-8")
    first.write_text("_base_: ../../outside.yaml\n", encoding="utf-8")
    second.unlink()

    with pytest.raises(ValueError, match="outside project root"):
        load_workflow(first)


def test_explicit_project_dir_precedes_current_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = tmp_path / "current"
    selected = tmp_path / "selected"
    current.mkdir()
    selected.mkdir()
    (current / "workflow.yaml").write_text("current\n", encoding="utf-8")
    expected = selected / "workflow.yaml"
    expected.write_text("selected\n", encoding="utf-8")
    monkeypatch.chdir(current)

    assert (
        resolve_workflow_config_path(
            "workflow.yaml",
            project_dir=selected,
            search_paths=(),
        )
        == expected.resolve()
    )


def test_checkpoint_pipeline_expansion_contract() -> None:
    assert expand_checkpoint_pipeline_stages(
        ["prepare", "checkpoint"],
        {"checkpoint": {"eval_steps": [10, "final"]}},
    ) == ["prepare", "checkpoint-10", "checkpoint-final"]
    assert (
        expand_checkpoint_pipeline_stages(
            ["checkpoint"],
            {"checkpoint": {"eval_steps": None}},
        )
        == []
    )
    assert (
        expand_checkpoint_pipeline_stages(
            ["checkpoint"],
            {"checkpoint": {"eval_steps": []}},
        )
        == []
    )
    assert expand_checkpoint_pipeline_stages(
        ["checkpoint"],
        {"checkpoint": {"eval_steps": 7}},
    ) == ["checkpoint-7"]
    with pytest.raises(TypeError, match="must contain a mapping"):
        expand_checkpoint_pipeline_stages(["checkpoint"], {"checkpoint": None})
    for invalid_steps in (True, "final", [True], [""]):
        with pytest.raises(TypeError, match="checkpoint eval_steps"):
            expand_checkpoint_pipeline_stages(
                ["checkpoint"],
                {"checkpoint": {"eval_steps": invalid_steps}},
            )
    with pytest.raises(ValueError, match="duplicate stage identities"):
        expand_checkpoint_pipeline_stages(
            ["model-large", "model"],
            {
                "model-large": {"eval_steps": ["final"]},
                "model": {"eval_steps": ["large-final"]},
            },
        )


def test_public_loader_matches_runner_definition() -> None:
    config = PROJECT_ROOT / "projects" / "starter" / "workflows" / "generate.yaml"

    definition = nvflow.load_workflow(config)
    runner = WorkflowRunner(str(config))

    assert isinstance(definition, WorkflowDefinition)
    assert definition.config is not runner.config
    assert definition.config == runner.config
    assert definition.project_context.root == runner.project_context.root
    assert definition.pipeline_stages == ["generate_answer"]
    assert definition.get_expname("generate_answer", {}) == ("answer_generation-generate_answer")


def test_core_loader_is_the_public_loader() -> None:
    assert load_workflow is nvflow.load_workflow


def test_external_project_loading_is_idempotent_in_one_process() -> None:
    config = FIXTURE_ROOT / "workflows" / "noop.yaml"
    script = f"""
        import json
        from nvflow import load_workflow
        first = load_workflow({str(config)!r}, project_dir={str(FIXTURE_ROOT)!r})
        second = load_workflow({str(config)!r}, project_dir={str(FIXTURE_ROOT)!r})
        first.validate()
        second.validate()
        print(json.dumps({{
            "same_class": first.resolve_stage_class("noop") is second.resolve_stage_class("noop"),
            "same_root": first.project_context.root == second.project_context.root,
        }}))
    """

    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout.splitlines()[-1]) == {
        "same_class": True,
        "same_root": True,
    }


def test_validate_invokes_stage_configuration_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = FIXTURE_ROOT / "workflows" / "noop.yaml"
    definition = load_workflow(config, project_dir=FIXTURE_ROOT)
    stage_class = definition.resolve_stage_class("noop")

    def reject(_self, _config) -> None:
        raise ValueError("stage configuration rejected")

    monkeypatch.setattr(stage_class, "validate_config", reject)

    with pytest.raises(ValueError, match="stage configuration rejected"):
        definition.validate()


def test_subset_validation_checks_dependencies_but_not_siblings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "subset-project"
    workflows = root / "workflows"
    workflows.mkdir(parents=True)
    (root / "project.yaml").write_text(
        "manifest_version: 1\nproject: subset-project\n",
        encoding="utf-8",
    )
    config = workflows / "subset.yaml"
    config.write_text(
        textwrap.dedent(
            """
            workflow:
              name: subset
              type: test
            cluster: cluster-a
            pipeline_stages:
              - dependency
              - target
              - sibling
            stages:
              dependency:
                label: dependency
                dependencies: []
              target:
                label: target
                dependencies: [dependency]
              sibling:
                label: sibling
                reject: true
                dependencies: []
              inactive:
                label: inactive
                dependencies: []
            """
        ).lstrip(),
        encoding="utf-8",
    )
    calls: list[str] = []
    selections: list[list[str] | None] = []

    class TrackingStage(BaseStage):
        def execute(self, config, cluster, expname, run_after=None):
            return None

        def validate_config(self, config):
            calls.append(config["label"])
            selections.append(config.get("_environment"))
            if config.get("reject"):
                raise ValueError(f"{config['label']} rejected")

    definition = load_workflow(config, project_dir=root)
    monkeypatch.setattr(definition, "resolve_stage_class", lambda _stage: TrackingStage)

    definition.validate(stages=["target", "target"])

    assert calls == ["dependency", "target"]
    assert selections == [None, None]
    calls.clear()
    selections.clear()
    definition.validate(stages=["target"], environment=["dataset-a"])
    assert calls == ["dependency", "target"]
    assert selections == [["dataset-a"], ["dataset-a"]]
    with pytest.raises(ValueError, match="sibling rejected"):
        definition.validate()
    with pytest.raises(ValueError, match="configured but not active"):
        definition.validate(stages=["inactive"])
    with pytest.raises(ValueError, match="not found"):
        definition.validate(stages=["missing"])
    definition.config["stages"]["dependency"]["dependencies"] = ["target"]
    with pytest.raises(ValueError, match="dependency cycle"):
        definition.validate(stages=["target"])


def test_external_project_relative_config_is_resolved_from_project_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    definition = load_workflow(
        "workflows/noop.yaml",
        project_dir=FIXTURE_ROOT,
    )

    assert definition.config_path == (FIXTURE_ROOT / "workflows" / "noop.yaml")


def test_project_search_path_resolves_relative_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    definition = load_workflow(
        "workflows/noop.yaml",
        project_search_paths=(FIXTURE_ROOT,),
    )

    assert definition.project_context.identity == "fixture-python"
