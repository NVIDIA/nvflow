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
"""Integration tests for workflow-local project stage implementations."""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from nvflow.core import WorkflowDefinition
from nvflow.project import ProjectManifestError

PROJECT_ROOT = Path(__file__).parents[2]
FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "projects"
STARTER_ROOT = PROJECT_ROOT / "projects" / "starter"


def _run(script: str) -> dict:
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout.splitlines()[-1])


def test_project_class_imports_only_during_semantic_validation() -> None:
    bundle = FIXTURE_ROOT / "python_project"
    config = bundle / "workflows" / "noop.yaml"
    result = _run(
        f"""
        import json
        import sys
        from nvflow.execution import WorkflowRunner

        runner = WorkflowRunner({str(config)!r}, project_dir={str(bundle)!r})
        imported_before = "fixture_project.stage" in sys.modules
        runner.validate_config()
        print(json.dumps({{
            "imported_before": imported_before,
            "imported_after": "fixture_project.stage" in sys.modules,
            "class_name": runner.definition.resolve_stage_class("noop").__name__,
            "expname": runner.definition.get_expname("noop", {{}}),
            "worker_asset": runner.project_context.worker_path(
                "project://assets/message.txt"
            ),
        }}))
        """
    )

    assert result == {
        "imported_before": False,
        "imported_after": True,
        "class_name": "NoopStage",
        "expname": "fixture-python-external_flow-noop",
        "worker_asset": "/nemo_run/code/projects/fixture-python/assets/message.txt",
    }


def test_copyable_starter_resolves_and_plans_its_project_stage() -> None:
    config = STARTER_ROOT / "workflows" / "generate.yaml"
    result = _run(
        f"""
        import json
        from nvflow.execution import TargetBinding, WorkflowPlanner, WorkflowRunner

        runner = WorkflowRunner({str(config)!r}, project_dir={str(STARTER_ROOT)!r})
        runner.validate_config()
        plan = WorkflowPlanner(
            runner.definition,
            TargetBinding.resolve(runner.definition.cluster, "test-cluster"),
        ).plan()
        print(json.dumps({{
            "class_name": runner.definition.resolve_stage_class("generate_answer").__name__,
            "plan_complete": plan.complete,
            "planned_operation": plan.tasks[0].operation,
            "planned_output": plan.tasks[0].outputs[0].path,
        }}))
        """
    )

    assert result == {
        "class_name": "GenerateAnswerStage",
        "plan_complete": True,
        "planned_operation": "nemo_skills.generate",
        "planned_output": ("/workspace/outputs/starter/generate-answer/output-rs0_chunk_0.jsonl"),
    }


def test_project_stage_must_resolve_inside_python_source_root(tmp_path: Path) -> None:
    project = tmp_path / "project"
    workflows = project / "workflows"
    workflows.mkdir(parents=True)
    (project / "project.yaml").write_text(
        "manifest_version: 1\nproject: boundary\npython_source_root: .\n",
        encoding="utf-8",
    )
    workflow = workflows / "bad.yaml"
    workflow.write_text(
        "workflow: {name: bad}\n"
        "cluster: test\n"
        "pipeline_stages: [bad]\n"
        "stages:\n"
        "  bad:\n"
        "    uses: project:json.decoder:JSONDecoder\n",
        encoding="utf-8",
    )

    definition = WorkflowDefinition(workflow)
    with pytest.raises(ProjectManifestError, match="outside project python_source_root"):
        definition.resolve_stage_class("bad")


def test_project_stage_type_requires_explicit_class_coordinate(tmp_path: Path) -> None:
    project = tmp_path / "project"
    workflows = project / "workflows"
    workflows.mkdir(parents=True)
    (project / "project.yaml").write_text(
        "manifest_version: 1\nproject: syntax\n",
        encoding="utf-8",
    )
    workflow = workflows / "bad.yaml"
    workflow.write_text(
        "workflow: {name: bad}\n"
        "cluster: test\n"
        "pipeline_stages: [bad]\n"
        "stages:\n"
        "  bad:\n"
        "    uses: project:not-a-coordinate\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="project stage type must use"):
        WorkflowDefinition(workflow).resolve_stage_class("bad")
