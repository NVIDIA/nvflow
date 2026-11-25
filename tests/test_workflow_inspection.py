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
"""Contracts for unresolved workflow structure inspection."""

from pathlib import Path

import pytest

from nvflow.cli.workflow_inspection import inspect_workflow_structure


def test_interpolated_checkpoint_steps_remain_runtime_expanded(tmp_path: Path) -> None:
    project = tmp_path / "project"
    workflows = project / "workflows"
    workflows.mkdir(parents=True)
    (project / "project.yaml").write_text(
        "manifest_version: 1\nproject: inspection\n",
        encoding="utf-8",
    )
    config = workflows / "checkpoints.yaml"
    config.write_text(
        "workflow:\n"
        "  name: checkpoints\n"
        "  type: evaluation\n"
        "cluster: test\n"
        "steps: [2, 3]\n"
        "stage_order: [prepare, model, model-large]\n"
        "dependencies: [prepare]\n"
        "pipeline_stages: ${stage_order}\n"
        "stages:\n"
        "  prepare:\n"
        "    dependencies: []\n"
        "  model-large:\n"
        "    dependencies: [wrong-static-stage]\n"
        "checkpoints:\n"
        "  model:\n"
        "    eval_steps: [1, final]\n"
        "  model-large:\n"
        "    eval_steps: ${steps}\n",
        encoding="utf-8",
    )

    structure = inspect_workflow_structure(config, project_dir=project)

    assert tuple(entry.name for entry in structure.stage_entries) == (
        "prepare",
        "model-1",
        "model-final",
        "model-large",
    )
    assert structure.runtime_expanded_stages == ("model-large",)
    assert {entry.name: entry.dependencies for entry in structure.stage_entries} == {
        "prepare": (),
        "model-1": ("prepare",),
        "model-final": ("prepare",),
        "model-large": ("prepare",),
    }


def test_checkpoint_collision_is_rejected(tmp_path: Path) -> None:
    project = tmp_path / "project"
    workflows = project / "workflows"
    workflows.mkdir(parents=True)
    (project / "project.yaml").write_text(
        "manifest_version: 1\nproject: inspection\n",
        encoding="utf-8",
    )
    config = workflows / "collisions.yaml"
    config.write_text(
        "workflow: {name: collisions, type: evaluation}\n"
        "cluster: test\n"
        "steps: [2]\n"
        "pipeline_stages: [model-large, model, model-large-final]\n"
        "stages: {}\n"
        "checkpoints:\n"
        "  model:\n"
        "    eval_steps: [large-final]\n"
        "    dependencies: [first]\n"
        "  model-large:\n"
        "    eval_steps: [final]\n"
        "    dependencies: [second]\n"
        "  model-large-final:\n"
        "    eval_steps: ${steps}\n"
        "    dependencies: [runtime-template]\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate stage identities"):
        inspect_workflow_structure(config, project_dir=project)


def test_runtime_pipeline_interpolation_is_not_resolved_or_printed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    workflows = project / "workflows"
    workflows.mkdir(parents=True)
    (project / "project.yaml").write_text(
        "manifest_version: 1\nproject: inspection\n",
        encoding="utf-8",
    )
    config = workflows / "secret-order.yaml"
    config.write_text(
        "workflow: {name: secret-order, type: test}\n"
        "cluster: test\n"
        "stage_order: ['${oc.env:API_TOKEN}']\n"
        "pipeline_stages: ${stage_order}\n"
        "stages: {}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("API_TOKEN", "must-not-appear")

    with pytest.raises(TypeError, match="runtime interpolation") as caught:
        inspect_workflow_structure(config, project_dir=project)

    assert "must-not-appear" not in str(caught.value)


def test_runtime_dependency_interpolation_is_opaque(tmp_path: Path) -> None:
    project = tmp_path / "project"
    workflows = project / "workflows"
    workflows.mkdir(parents=True)
    (project / "project.yaml").write_text(
        "manifest_version: 1\nproject: inspection\n",
        encoding="utf-8",
    )
    config = workflows / "dependencies.yaml"
    config.write_text(
        "workflow: {name: dependencies, type: test}\n"
        "cluster: test\n"
        "pipeline_stages: [stage]\n"
        "stages:\n"
        "  stage:\n"
        "    dependencies: '${oc.env:API_TOKEN,secret-fallback}'\n",
        encoding="utf-8",
    )

    structure = inspect_workflow_structure(config, project_dir=project)

    assert structure.stage_entries[0].dependencies == ("<runtime-dependent>",)
    assert "secret-fallback" not in repr(structure)
