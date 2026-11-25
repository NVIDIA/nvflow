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
"""Human-readable and machine-readable plan output contracts."""

from __future__ import annotations

import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
from typer.testing import CliRunner

from nvflow.cli.main import _format_plan_warnings, app
from nvflow.cli.plan_output import render_plan_text
from nvflow.execution import (
    CommandPlan,
    DependencyPlan,
    LaunchPlan,
    ResourceRequest,
    StagePlan,
    TaskPlan,
)
from nvflow.execution.inspection import StageOutputInspection
from nvflow.project import ProjectProvenance

PROJECT_ROOT = Path(__file__).parents[2]


@pytest.fixture(autouse=True)
def _target_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "operator-cluster.yaml").write_text(
        f"executor: slurm\njob_dir: {tmp_path / 'jobs'}\n"
        f"mounts:\n  - {tmp_path / 'workspace'}:/workspace\n"
    )
    monkeypatch.setenv("NVFLOW_CONFIG_DIR", str(tmp_path))


def test_live_plan_renderer_separates_filesystem_scope_and_requested_resources() -> None:
    gpu_task = TaskPlan(
        task_id="train.gpu",
        stage="train",
        name="gpu-task",
        operation="train",
        commands=(CommandPlan("python train.py", "trainer"),),
        resources=(
            ResourceRequest(role="policy", nodes=2, gpus_per_node=8),
            ResourceRequest(role="judge", nodes=1, gpus_per_node=2),
        ),
    )
    cpu_task = TaskPlan(
        task_id="train.postprocess",
        stage="train",
        name="postprocess",
        operation="postprocess",
        resources=(ResourceRequest(role="postprocess", nodes=1, gpus_per_node=0),),
        dependencies=(DependencyPlan("train.gpu"),),
    )
    cleanup_task = TaskPlan(
        task_id="train.cleanup",
        stage="train",
        name="cleanup",
        operation="cleanup",
        resources=(ResourceRequest(role="cleanup", gpus_per_node=0),),
        dependencies=(DependencyPlan("train.gpu", condition="after_any"),),
    )
    undeclared_task = TaskPlan(
        task_id="train.undeclared",
        stage="train",
        name="undeclared",
        operation="postprocess",
        resources=(ResourceRequest(),),
    )
    plan = LaunchPlan(
        plan_version=1,
        project="starter",
        workflow="resources",
        target="cluster",
        stages=(
            StagePlan(
                stage="train",
                tasks=(gpu_task, cpu_task, cleanup_task, undeclared_task),
            ),
        ),
        provenance=ProjectProvenance(
            commit="a" * 40,
            source_subpath="projects/starter",
            packaged_destination="/nemo_run/code/projects/starter",
            content_digest="sha256:" + ("b" * 64),
            clean=False,
            submittable=False,
            reason="project source has uncommitted content",
        ),
        planning_mode="live",
        environments=("finance_sec_search",),
    )

    compact = render_plan_text(
        plan,
        output_inspections=(
            StageOutputInspection(
                stage="train",
                presence="exists",
                declared_paths=("/workspace/output",),
                existing_paths=("/workspace/output",),
            ),
        ),
    )
    verbose = render_plan_text(plan, verbose=True)

    assert "Environment: finance_sec_search" in compact
    assert "Mode: LIVE" in compact
    assert "Target config: cluster loaded" in compact
    assert (
        "Filesystem inspection: ENABLED — stages may discover inputs or omit reusable work"
        in compact
    )
    assert "Tasks shown: stage-specific selection after filesystem inspection" in compact
    assert "Source eligibility: BLOCKED — project source has uncommitted content" in compact
    assert "Managed model paths: NOT APPLICABLE" in compact
    assert "Plan coverage: COMPLETE" in compact
    assert "Output presence: 1 stage with existing output; 4 tasks selected" in compact
    assert "● train" in compact
    assert (
        "● an output of the stage's final tasks exists; "
        "may indicate prior activity, not proof of completion."
    ) in compact
    assert "Execution plan ID:" not in compact
    assert "Progress:" not in compact
    assert "1 GPU task · 2 CPU tasks · 1 task with resource undeclared" in compact
    assert "Cumulative requested GPUs: 18 (sum across tasks; not concurrent)" in compact
    assert "Largest resource group: 2 nodes / 16 GPUs" in compact
    assert "GPUs at once" in compact
    assert "Total GPUs" in compact
    assert "Busiest stage: train (18 GPUs at once)" in compact
    assert "Tasks" in compact
    assert "R/C" not in compact
    assert "gpu-task [train]" not in compact
    assert "gpu-task [train]" in verbose
    assert "Plan digest: sha256:" in verbose
    assert "Project provenance: projects/starter@aaaaaaaaaaaa" in verbose
    assert "policy=2 nodes, 8 GPUs/node" in verbose
    assert "dependencies: train.gpu (after_success)" in verbose


def test_plan_renderer_counts_gpus_that_tasks_hold_at_once() -> None:
    def task(stage: str, name: str, gpus: int, *dependencies: DependencyPlan) -> TaskPlan:
        return TaskPlan(
            task_id=f"{stage}.{name}",
            stage=stage,
            name=name,
            operation=stage,
            resources=(ResourceRequest(role="main", nodes=1, gpus_per_node=gpus),),
            dependencies=dependencies,
        )

    def starts_with(upstream: str) -> DependencyPlan:
        return DependencyPlan(upstream, condition="after_start")

    plan = LaunchPlan(
        plan_version=1,
        project="example",
        workflow="grpo",
        target="cluster",
        stages=(
            StagePlan(
                stage="generate",
                tasks=(task("generate", "chunk-0", 8), task("generate", "chunk-1", 8)),
            ),
            StagePlan(
                stage="train",
                tasks=(
                    task("train", "policy-0", 32, DependencyPlan("generate.chunk-1")),
                    task("train", "judge-0", 4, starts_with("train.policy-0")),
                    task(
                        "train",
                        "policy-1",
                        32,
                        DependencyPlan("train.policy-0", condition="after_any"),
                    ),
                    task("train", "judge-1", 4, starts_with("train.policy-1")),
                ),
            ),
        ),
    )

    rendered = render_plan_text(plan)
    gpu_columns = {
        line.split()[0]: line.split()[-2:]
        for line in rendered.splitlines()
        if line.startswith(("generate ", "train "))
    }

    assert gpu_columns == {"generate": ["16", "16"], "train": ["36", "72"]}
    assert "Busiest stage: train (36 GPUs at once)" in rendered


def test_plan_renderer_counts_jobs_only_when_the_engine_reports_them() -> None:
    tasks = tuple(
        TaskPlan(
            task_id=f"checks.{index}",
            stage="checks",
            name=f"check-{index}",
            operation="nemo_skills.run_cmd",
            resources=(ResourceRequest(gpus_per_node=0),),
        )
        for index in range(3)
    )
    plan = LaunchPlan(
        plan_version=1,
        project="example",
        workflow="checks",
        target="cluster",
        stages=(StagePlan(stage="checks", tasks=tasks),),
    )

    rendered = render_plan_text(plan, jobs_per_stage={"checks": 2})

    assert "1 stage · 3 tasks · 2 jobs" in rendered
    table = {line.split()[0]: line.split()[1:3] for line in rendered.splitlines()[-5:-2]}
    assert table["Stage"] == ["Tasks", "Jobs"]
    assert table["checks"] == ["3", "2"]
    assert "Jobs" not in render_plan_text(plan)


def test_live_plan_renderer_reports_when_supported_stages_detect_no_outputs() -> None:
    plan = LaunchPlan(
        plan_version=1,
        project="example",
        workflow="selection",
        target="cluster",
        stages=(
            StagePlan(
                stage="collect",
                tasks=(
                    TaskPlan(
                        task_id="collect.first",
                        stage="collect",
                        name="first",
                        operation="collect",
                    ),
                ),
            ),
        ),
        planning_mode="live",
    )

    rendered = render_plan_text(
        plan,
        output_inspections=(StageOutputInspection(stage="collect", presence="not_declared"),),
    )

    assert "Output presence: 1 stage without declared output; 1 task selected" in rendered


def test_live_plan_renderer_explains_outputs_this_machine_cannot_see() -> None:
    plan = LaunchPlan(
        plan_version=1,
        project="example",
        workflow="ray",
        target="cluster",
        stages=(
            StagePlan(
                stage="train",
                tasks=(TaskPlan(task_id="train.0", stage="train", name="0", operation="train"),),
            ),
        ),
        planning_mode="live",
    )

    rendered = render_plan_text(
        plan,
        output_inspections=(
            StageOutputInspection(
                stage="train",
                presence="unavailable",
                declared_paths=("/workspace/outputs/model",),
                hidden_paths=("/workspace/outputs/model",),
            ),
        ),
    )

    assert "Output presence: 1 stage with unavailable inspection; 1 task selected" in rendered
    assert (
        "unavailable: /workspace is not mounted on this machine, "
        "so outputs under it cannot be checked from here."
    ) in rendered


def test_plan_renderer_uses_singular_resource_labels() -> None:
    task = TaskPlan(
        task_id="generate.one",
        stage="generate",
        name="generate",
        operation="generate",
        resources=(
            ResourceRequest(role="server", nodes=1, gpus_per_node=1),
            ResourceRequest(role="main", nodes=1),
        ),
    )
    plan = LaunchPlan(
        plan_version=1,
        project="example",
        workflow="single",
        target="cluster",
        stages=(StagePlan(stage="generate", tasks=(task,)),),
    )

    rendered = render_plan_text(plan, verbose=True)

    assert "1 stage · 1 task" in rendered
    assert "Largest resource group: 1 node / 1 GPU" in rendered
    assert "Largest resource group: 2 nodes" not in rendered
    assert "resources: server=1 node, 1 GPU/node, main=1 node, GPUs unspecified" in rendered


def test_plan_renderer_describes_opaque_stage_as_partial_plan_coverage() -> None:
    plan = LaunchPlan(
        plan_version=1,
        project="external",
        workflow="legacy",
        target="cluster",
        stages=(StagePlan.opaque("legacy", "legacy execute() has no plan"),),
    )

    rendered = render_plan_text(plan)

    assert "Plan coverage: PARTIAL — 1 opaque stage" in rendered
    assert "opaque — legacy execute() has no plan" in rendered
    assert "runtime completion" not in rendered
    assert "Busiest stage" not in rendered


def test_plan_cli_uses_compact_renderer_without_builder_output() -> None:
    result = CliRunner().invoke(
        app,
        [
            "plan",
            "--workflow",
            "finance:download/demo",
            "--cluster",
            "operator-cluster",
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.stdout.startswith("NVFlow plan: finance.download-sec")
    assert "Mode: STATIC" in result.stdout
    assert (
        "Filesystem inspection: OFF — workflow inputs and outputs are not inspected"
        in result.stdout
    )
    assert "Selection result:" not in result.stdout
    assert "Source eligibility:" in result.stdout
    assert "Largest resource group" in result.stdout
    assert "Use --verbose for task details, output paths, and full provenance" in result.stdout
    assert "download-sec-download-demo [nemo_skills.run_cmd]" not in result.stdout


def test_static_starter_plan_does_not_import_nemo_skills_pipeline() -> None:
    script = """
import sys
from typer.testing import CliRunner
from nvflow.cli.main import app

result = CliRunner().invoke(
    app,
    [
        "plan",
        "--workflow",
        "starter:generate",
        "--cluster",
        "operator-cluster",
    ],
)
if result.exit_code != 0:
    raise SystemExit(result.output)
if "nemo_skills.pipeline" in sys.modules:
    raise SystemExit("static starter planning imported nemo_skills.pipeline")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.requires_nemo
def test_plan_cli_live_mode_reports_filesystem_inspection() -> None:
    config = PROJECT_ROOT / "projects" / "finance" / "workflows" / "download" / "demo.yaml"

    result = CliRunner().invoke(
        app,
        [
            "plan",
            "--config",
            str(config),
            "--cluster",
            "operator-cluster",
            "--live",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Mode: LIVE" in result.stdout
    assert (
        "Filesystem inspection: ENABLED — stages may discover inputs or omit reusable work"
        in result.stdout
    )
    assert "Output presence:" in result.stdout
    assert "Progress:" not in result.stdout


def test_plan_warnings_hide_third_party_deprecations() -> None:
    caught = [
        warnings.WarningMessage(
            DeprecationWarning("old dependency API"),
            DeprecationWarning,
            "/venv/site-packages/dependency/api.py",
            1,
        ),
        warnings.WarningMessage(
            Warning("starlette.middleware.wsgi is deprecated"),
            Warning,
            "/venv/site-packages/fastapi/middleware/wsgi.py",
            1,
        ),
        warnings.WarningMessage(
            UserWarning("configuration needs review"),
            UserWarning,
            "/repo/projects/finance/training.py",
            42,
        ),
        warnings.WarningMessage(
            SyntaxWarning("invalid escape sequence '\\s'"),
            SyntaxWarning,
            "/venv/site-packages/nemo_skills/pipeline/start_server.py",
            80,
        ),
        warnings.WarningMessage(
            SyntaxWarning("project regex needs review"),
            SyntaxWarning,
            "/repo/projects/finance/stage.py",
            12,
        ),
    ]

    assert _format_plan_warnings(caught) == [
        "training: configuration needs review",
        "stage: SyntaxWarning: project regex needs review",
    ]
