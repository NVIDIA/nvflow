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
"""Go/no-go contract tests for simple and service-backed launch plans."""

from __future__ import annotations

import json
import sys
import types
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from my_project.stage import GenerateAnswerStage
from nvfinance.stages.download.download_sec_filings import (
    DownloadSecFilingsStage,
)
from nvfinance.stages.rl.embedded_eval import GymEmbeddedEvalStage
from typer.testing import CliRunner

from nvflow import load_workflow
from nvflow.cli.main import app
from nvflow.execution import (
    ArtifactDeclaration,
    CapturedExecutionStage,
    CommandPlan,
    DependencyPlan,
    ExecutionGraph,
    LaunchPlan,
    PlanningContext,
    ResourceRequest,
    ServicePlan,
    StageExecution,
    StagePlan,
    TargetBinding,
    TaskPlan,
    WorkflowPlanner,
    WorkflowRunner,
    discover_submission_receipt,
    internalize_dependencies,
    plan_run_cmd,
    plan_stage,
)
from nvflow.execution.capture import declare_slurm_experiment_cleanup
from nvflow.execution.operations import task_from_invocation
from nvflow.integrations.nemo_skills import stage_plan_from_jobs
from nvflow.project import ProjectContext, ProjectResolver
from nvflow.project.provenance import ProjectProvenance
from nvflow.stages.nemo_rl.grpo import GRPOStage
from nvflow.stages.nemo_rl.sft import SFTStage

PROJECT_ROOT = Path(__file__).parents[2]


def test_workflow_planner_computes_project_provenance_once(monkeypatch) -> None:
    definition = load_workflow(PROJECT_ROOT / "projects/starter/workflows/generate.yaml")
    context_type = type(definition.project_context)
    original = context_type.source_provenance
    calls = 0

    def tracked_source_provenance(
        context: ProjectContext,
        *,
        strict: bool = False,
    ) -> ProjectProvenance:
        nonlocal calls
        calls += 1
        return original(context, strict=strict)

    monkeypatch.setattr(context_type, "source_provenance", tracked_source_provenance)

    WorkflowPlanner(
        definition,
        TargetBinding.resolve(definition.cluster, "test-cluster"),
    ).plan()

    assert calls == 1


def test_simple_stage_plan_is_deterministic_and_complete() -> None:
    context = PlanningContext(
        stage="download",
        config={
            "tickers": ["NVDA", "AAPL"],
            "forms": ["10-K"],
            "start_year": 2025,
            "end_year": 2025,
            "output_dir": "/workspace/output",
            "sec_identity_email": "user@example.com",
            "sec_identity_company": "Example",
            "stage_kwargs": {"partition": "cpu", "num_nodes": 1},
        },
        launcher_config={
            "tickers": ["NVDA", "AAPL"],
            "forms": ["10-K"],
            "start_year": 2025,
            "end_year": 2025,
            "output_dir": "/workspace/output",
            "sec_identity_email": "user@example.com",
            "sec_identity_company": "Example",
            "stage_kwargs": {"partition": "cpu", "num_nodes": 1},
        },
        target=TargetBinding.resolve("cluster-a"),
        expname="download-sec-download-smoke",
        run_after=(),
    )

    stage_plan = DownloadSecFilingsStage().plan(context)
    first = LaunchPlan(
        plan_version=1,
        project="finance",
        workflow="download-sec",
        target="cluster-a",
        stages=(stage_plan,),
    )
    second = LaunchPlan(
        plan_version=1,
        project="finance",
        workflow="download-sec",
        target="cluster-a",
        stages=(DownloadSecFilingsStage().plan(context),),
    )

    assert first.complete is True
    assert first.digest == second.digest
    assert first.digest == replace(first, planning_mode="live").digest
    assert first.tasks[0].operation == "nemo_skills.run_cmd"
    assert first.tasks[0].resources == (ResourceRequest(nodes=1, partition="cpu"),)
    assert first.tasks[0].outputs == (
        ArtifactDeclaration(
            name="filings",
            path="/workspace/output",
            kind="directory",
        ),
    )


def test_stage_plan_requires_an_explicit_noop_disposition() -> None:
    with pytest.raises(ValueError, match="must contain at least one task"):
        StagePlan(stage="empty")

    noop = StagePlan.noop("empty", "environment filter selected no work")

    assert noop.complete is True
    assert noop.disposition == "noop"
    assert noop.reason == "environment filter selected no work"


def test_launch_plan_rejects_internal_dependency_cycles() -> None:
    first = TaskPlan(
        task_id="stage.first",
        stage="stage",
        name="first",
        operation="test",
        dependencies=(DependencyPlan("stage.second"),),
    )
    second = TaskPlan(
        task_id="stage.second",
        stage="stage",
        name="second",
        operation="test",
        dependencies=(DependencyPlan("stage.first"),),
    )

    with pytest.raises(ValueError, match="dependency cycle"):
        LaunchPlan(
            plan_version=1,
            project="example",
            workflow="cycle",
            target="cluster",
            stages=(StagePlan(stage="stage", tasks=(first, second)),),
        )


def test_dependency_internalization_targets_all_terminal_tasks() -> None:
    first = TaskPlan(
        task_id="upstream.0",
        stage="upstream",
        name="upstream-first",
        operation="shell",
        parameters={"experiment_name": "upstream-exp"},
    )
    second = TaskPlan(
        task_id="upstream.1",
        stage="upstream",
        name="upstream-second",
        operation="shell",
        dependencies=(DependencyPlan("upstream.0"),),
        parameters={"experiment_name": "upstream-exp"},
    )
    fanout = TaskPlan(
        task_id="upstream.2",
        stage="upstream",
        name="upstream-fanout",
        operation="shell",
        parameters={"experiment_name": "upstream-exp"},
    )
    downstream = TaskPlan(
        task_id="downstream.0",
        stage="downstream",
        name="downstream",
        operation="shell",
        dependencies=(DependencyPlan("upstream-exp", external=True),),
    )

    internalized = internalize_dependencies((first, second, fanout, downstream))

    assert internalized[-1].dependencies == (
        DependencyPlan("upstream.1"),
        DependencyPlan("upstream.2"),
    )


def test_dependency_internalization_waits_on_work_not_its_service_or_cleanup() -> None:
    training = TaskPlan(
        task_id="training.policy",
        stage="training",
        name="grpo",
        operation="nemo_rl.grpo",
        parameters={"experiment_name": "grpo"},
    )
    judge = TaskPlan(
        task_id="training.judge",
        stage="training",
        name="grpo-judge",
        operation="service",
        dependencies=(DependencyPlan("training.policy", condition="after_start"),),
        parameters={"experiment_name": "grpo"},
    )
    cleanup = TaskPlan(
        task_id="training.cleanup",
        stage="training",
        name="grpo-judge-cleanup",
        operation="cleanup",
        dependencies=(DependencyPlan("training.policy", condition="after_any"),),
        parameters={"experiment_name": "grpo"},
    )
    evaluation = TaskPlan(
        task_id="eval.0",
        stage="eval",
        name="eval",
        operation="shell",
        dependencies=(DependencyPlan("grpo", external=True),),
    )

    internalized = internalize_dependencies((training, judge, cleanup, evaluation))

    assert internalized[-1].dependencies == (DependencyPlan("training.policy"),)


def test_execution_graph_rejects_unresolved_dependencies() -> None:
    task = TaskPlan(
        task_id="stage.0",
        stage="stage",
        name="stage",
        operation="shell",
        commands=(
            CommandPlan(
                "true",
                "project",
                runtime_role="project",
            ),
        ),
        dependencies=(DependencyPlan("missing-experiment", external=True),),
    )
    implementation = {
        "kind": "project",
        "name": "project.stage:Stage",
        "contract_version": None,
        "provider": "project",
        "platform": {
            "distribution": "nvflow",
            "version": "1.2.0",
            "source": "git:test",
        },
    }

    with pytest.raises(ValueError, match="unresolved selected-stage"):
        ExecutionGraph(
            execution_version=1,
            project="project",
            workflow="workflow",
            target="target",
            engine="ray",
            target_fingerprint="sha256:target",
            stages=(
                StageExecution(
                    stage="stage",
                    tasks=(task,),
                    implementation=implementation,
                ),
            ),
        )


def test_execution_graph_derives_public_plan_and_binds_engine_identity() -> None:
    task = TaskPlan(
        task_id="generate.0",
        stage="generate",
        name="generate",
        operation="shell",
        commands=(
            CommandPlan(
                "python -m project.generate",
                "project",
                runtime_role="project",
            ),
        ),
    )
    implementation = {
        "kind": "project",
        "name": "project.stage:Generate",
        "contract_version": None,
        "provider": "project",
        "platform": {
            "distribution": "nvflow",
            "version": "1.2.0",
            "source": "git:abc",
        },
    }
    graph = ExecutionGraph(
        execution_version=1,
        project="project",
        workflow="generate",
        target="ray-target",
        engine="ray",
        target_fingerprint="sha256:target",
        stages=(
            StageExecution(
                stage="generate",
                tasks=(task,),
                implementation=implementation,
            ),
        ),
    )

    plan = graph.to_plan()

    assert plan.tasks == (task,)
    assert plan.stages[0].implementation == implementation
    assert graph.digest != replace(graph, engine="slurm").digest
    assert graph.digest != replace(graph, target_fingerprint="sha256:other").digest


def test_stage_execution_rejects_display_only_runtime_templates() -> None:
    task = TaskPlan(
        task_id="stage.0",
        stage="stage",
        name="stage",
        operation="shell",
        commands=(
            CommandPlan(
                "runtime.builder(<runtime-bindings>)",
                "project",
                runtime_template=True,
                runtime_role="project",
            ),
        ),
    )

    with pytest.raises(ValueError, match="runtime command templates"):
        StageExecution(stage="stage", tasks=(task,))


@pytest.mark.parametrize(
    "command",
    [
        "python prepare.py && pip install surprise",
        "pip3 install surprise",
        "python -m pip install surprise",
        "uv pip install surprise",
        "uv sync",
        "uv sync --frozen",
        "uv venv .venv",
        "bash -c 'pip install surprise'",
        "gym env prefetch environment",
    ],
)
def test_runtime_package_installs_are_recognized_and_left_to_the_engine(command: str) -> None:
    from nvflow.execution.contracts import installs_packages

    task = TaskPlan(
        task_id="stage.0",
        stage="stage",
        name="stage",
        operation="shell",
        commands=(
            CommandPlan(
                command,
                "project",
                runtime_role="project",
            ),
        ),
    )

    assert installs_packages(task)
    assert StageExecution(stage="stage", tasks=(task,)).tasks == (task,)


def test_execute_only_stage_has_no_compatibility_path() -> None:
    from nvflow.core import BaseStage

    class LegacyStage(BaseStage):
        def execute(self, config, cluster, expname, run_after=None):
            raise AssertionError("legacy submission must never run")

    context = PlanningContext(
        stage="legacy",
        config={},
        launcher_config={},
        target=TargetBinding.resolve("target"),
        expname="legacy",
        run_after=(),
    )

    with pytest.raises(ValueError, match=r"must implement execution\(\)"):
        LegacyStage().execution(context)


def test_captured_run_cmd_uses_explicit_command_keyword() -> None:
    context = PlanningContext(
        stage="command",
        config={},
        launcher_config={},
        target=TargetBinding.resolve("cluster"),
        expname="command",
        run_after=(),
    )

    task = task_from_invocation(
        context,
        index=0,
        operation="nemo_skills.run_cmd",
        positional=(),
        keyword={"command": "python -m project.worker"},
    )

    assert task.commands[0].command == "python -m project.worker"


def test_capture_preserves_declared_slurm_experiment_cleanup() -> None:
    class CleanupStage(CapturedExecutionStage):
        capture_strategy = "run_cmd"

        def build_tasks(self, config, cluster, expname, run_after=None):
            from nvflow.integrations.nemo_skills.commands import run_cmd

            declare_slurm_experiment_cleanup([expname, f"{expname}-child", expname])
            run_cmd(
                command="echo ready",
                cluster=cluster,
                expname=expname,
                log_dir="/logs",
                run_after=run_after,
                container="nemo-skills",
            )

    context = PlanningContext(
        stage="cleanup",
        config={},
        launcher_config={},
        target=TargetBinding.resolve("cluster"),
        expname="cleanup",
        run_after=(),
    )

    execution = CleanupStage().execution(context)

    assert execution.tasks[0].parameters["slurm_cleanup_experiments"] == (
        "cleanup",
        "cleanup-child",
    )


def test_document_grounded_prep_reuses_output_that_a_live_plan_finds(tmp_path: Path) -> None:
    from nvflow.execution.filesystem import LocalTargetFilesystem
    from nvflow.stages.sdg.document_grounded.dg_sdg_preprocess import DGSDGPreprocessStage

    sampled = tmp_path / "out" / "jsonl" / "10-k-data.jsonl"
    sampled.parent.mkdir(parents=True)
    sampled.write_text('{"text": "filing"}\n', encoding="utf-8")
    config = {
        "input_dir": str(tmp_path / "in"),
        "output_dir": str(tmp_path / "out"),
        "preprocess_module": "project.preprocess",
        "distribution_dir": str(tmp_path / "distribution"),
        "extra_args": {"forms": "10-K"},
    }
    context = PlanningContext(
        stage="dg_sdg_preprocess",
        config=config,
        launcher_config=config,
        target=TargetBinding.resolve("cluster"),
        expname="preprocess",
        run_after=(),
        live=True,
        filesystem=LocalTargetFilesystem(),
    )

    assert DGSDGPreprocessStage().execution(context).disposition == "noop"
    assert DGSDGPreprocessStage().execution(replace(context, live=False)).tasks


def test_public_plan_run_cmd_builds_one_inspectable_task() -> None:
    context = PlanningContext(
        stage="command",
        config={},
        launcher_config={},
        target=TargetBinding.resolve("cluster"),
        expname="command",
        run_after=("upstream",),
    )

    plan = plan_run_cmd(
        context,
        "python -m project.worker",
        cluster="cluster",
        expname="command",
        run_after=["upstream"],
        container="nemo-skills",
    )

    assert plan.stage == "command"
    assert len(plan.tasks) == 1
    assert plan.tasks[0].operation == "nemo_skills.run_cmd"
    assert plan.tasks[0].commands[0].command == "python -m project.worker"


def test_run_cmd_does_not_inherit_generation_server_resources() -> None:
    context = PlanningContext(
        stage="prepare",
        config={"stage_kwargs": {"server_gpus": 8, "server_nodes": 2}},
        launcher_config={},
        target=TargetBinding.resolve("cluster"),
        expname="prepare",
        run_after=(),
    )

    task = plan_run_cmd(
        context,
        "python -m project.prepare",
        num_gpus=0,
    ).tasks[0]

    assert task.resources == (ResourceRequest(gpus_per_node=0),)


@pytest.mark.parametrize(
    ("dependency_type", "expected_condition"),
    [
        (None, "after_success"),
        ("afterok", "after_success"),
        ("afterany", "after_any"),
        ("after", "after_start"),
    ],
)
def test_public_plan_preserves_slurm_dependency_condition(
    dependency_type: str | None,
    expected_condition: str,
) -> None:
    context = PlanningContext(
        stage="command",
        config={},
        launcher_config={},
        target=TargetBinding.resolve("cluster"),
        expname="command",
        run_after=("upstream",),
    )
    parameters = {}
    if dependency_type is not None:
        parameters["sbatch_kwargs"] = {"dependency_type": dependency_type}

    task = plan_run_cmd(
        context,
        "python -m project.worker",
        **parameters,
    ).tasks[0]

    assert len(task.dependencies) == 1
    assert task.dependencies[0].condition == expected_condition


def test_launch_plan_to_dict_normalizes_path_parameters() -> None:
    task = TaskPlan(
        task_id="stage.main",
        stage="stage",
        name="stage",
        operation="test",
        parameters={"path": Path("/workspace/output")},
    )
    plan = LaunchPlan(
        plan_version=1,
        project="example",
        workflow="json",
        target="cluster",
        stages=(StagePlan(stage="stage", tasks=(task,)),),
    )

    payload = plan.to_dict()
    json.dumps(payload)
    assert payload["stages"][0]["tasks"][0]["parameters"]["path"] == "/workspace/output"


def test_example_generate_stage_uses_explicit_plan_without_capture_or_submission(
    tmp_path: Path,
    monkeypatch,
) -> None:
    assert not hasattr(GenerateAnswerStage, "execute")
    monkeypatch.setattr(
        "nvflow.execution.capture.capture_first_party_execution",
        lambda *_args, **_kwargs: pytest.fail("planning used capture"),
    )
    input_file = tmp_path / "input.jsonl"
    input_file.write_text("{}\n", encoding="utf-8")
    context = PlanningContext(
        stage="generate_answer",
        config={
            "input_file": str(input_file),
            "output_dir": "/workspace/output",
            "prompt_config": "/workspace/prompt.yaml",
            "stage_kwargs": {"model": "/hf/model", "server_gpus": 1},
        },
        launcher_config={
            "input_file": str(input_file),
            "output_dir": "/workspace/output",
            "prompt_config": "/workspace/prompt.yaml",
            "stage_kwargs": {"model": "/hf/model", "server_gpus": 1},
        },
        target=TargetBinding.resolve("cluster-a"),
        expname="sdg-simple-generate",
        run_after=(),
    )

    stage_plan = plan_stage(GenerateAnswerStage(), context)

    assert stage_plan is not None
    assert stage_plan.complete is True
    assert stage_plan.tasks[0].operation == "nemo_skills.generate"
    assert stage_plan.tasks[0].resources[0].role == "server-0"
    assert stage_plan.tasks[0].parameters["input_file"] == str(input_file)
    server_command = stage_plan.tasks[0].commands[0].command
    assert "nvflow.integrations.model_servers.serve" in server_command
    assert "SLURM_NODEID" not in server_command
    invocations = {task.parameters["generation_invocation"] for task in stage_plan.tasks}
    assert len(invocations) == 1
    assert next(iter(invocations)).startswith("generate_answer:sdg-simple-generate:")


@pytest.mark.parametrize(
    ("scope", "expected_stages"),
    [
        ("smoke", ["download", "build_index"]),
        ("demo", ["download", "build_index", "prefetch_cache"]),
        ("sp500", ["download", "build_index", "prefetch_cache"]),
    ],
)
def test_download_workflow_variant_plans_every_stage(
    monkeypatch,
    scope: str,
    expected_stages: list[str],
) -> None:
    config = PROJECT_ROOT / "projects" / "finance" / "workflows" / "download" / f"{scope}.yaml"
    definition = load_workflow(config)
    assert not hasattr(DownloadSecFilingsStage, "execute")

    class ExistingOutputFilesystem:
        remote = False

        def __init__(self) -> None:
            self.batches: list[tuple[str, ...]] = []

        def exists(self, path: str) -> bool:
            return True

        def batch_exists(self, paths: list[str]) -> dict[str, bool]:
            self.batches.append(tuple(paths))
            return dict.fromkeys(paths, True)

        def is_nonempty_file(self, path: str) -> bool:
            return True

        def ls(self, directory: str, pattern: str) -> list[str]:
            return []

        def count_lines(self, path: str) -> int:
            return 0

    filesystem = ExistingOutputFilesystem()
    original_plan = DownloadSecFilingsStage.plan

    def plan_with_shared_filesystem(self, context):
        assert context.filesystem is (filesystem if context.live else None)
        return original_plan(self, context)

    monkeypatch.setattr(DownloadSecFilingsStage, "plan", plan_with_shared_filesystem)
    planner = WorkflowPlanner(
        definition,
        TargetBinding.resolve(definition.cluster, "operator-cluster"),
        filesystem=filesystem,
    )
    launch_plan = planner.plan()
    live_plan = planner.plan(live=True)
    output_inspections = planner.output_inspections
    batch_count = len(filesystem.batches)
    planner.plan(live=True, inspect_outputs=False)

    assert launch_plan.complete is True
    assert [stage.stage for stage in launch_plan.stages] == expected_stages
    assert launch_plan.tasks[0].name == f"download-sec-download-{scope}"
    assert all(task.operation == "nemo_skills.run_cmd" for task in launch_plan.tasks)
    assert all(task.parameters["cluster"] == "operator-cluster" for task in launch_plan.tasks)
    assert all(f"/finance/{scope}/" in output.path for output in launch_plan.tasks[0].outputs)
    assert live_plan.planning_mode == "live"
    assert live_plan.stages == launch_plan.stages
    assert all(inspection.presence == "exists" for inspection in output_inspections)
    assert len(filesystem.batches) == batch_count
    assert planner.output_inspections == ()


def test_plan_cli_emits_machine_readable_complete_plan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = PROJECT_ROOT / "projects" / "finance" / "workflows" / "download" / "demo.yaml"
    provenance = ProjectResolver().resolve(config).source_provenance()
    monkeypatch.setattr(
        ProjectContext,
        "source_provenance",
        lambda _self, **_kwargs: replace(
            provenance,
            content_digest="sha256:" + ("a" * 64),
            clean=True,
            submittable=True,
            dirty_paths=(),
            reason=None,
        ),
    )
    (tmp_path / "operator-cluster.yaml").write_text("executor: slurm\n")
    monkeypatch.setenv("NVFLOW_CONFIG_DIR", str(tmp_path))

    result = CliRunner().invoke(
        app,
        [
            "plan",
            "--config",
            str(config),
            "--cluster",
            "operator-cluster",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["complete"] is True
    assert payload["plan_version"] == 2
    assert payload["kind"] == "workflow"
    assert "manifest_version" not in payload
    assert payload["project"] == "finance"
    assert "recipe" not in payload
    assert payload["target"] == "operator-cluster"
    assert payload["planning_mode"] == "static"
    assert payload["environments"] == []
    assert all("completed_tasks" not in stage for stage in payload["stages"])
    implementation = payload["stages"][0]["implementation"]
    assert implementation["kind"] == "project"
    assert implementation["provider"].endswith(".DownloadSecFilingsStage")
    assert implementation["platform"]["distribution"] == "nvflow"
    assert implementation["platform"]["version"]
    assert implementation["platform"]["source"].startswith(("git:", "wheel:"))
    assert payload["stages"][0]["tasks"][0]["operation"] == "nemo_skills.run_cmd"
    assert payload["provenance"]["commit"]
    assert payload["provenance"]["source_subpath"] == "projects/finance"
    assert payload["provenance"]["content_digest"].startswith("sha256:")


def test_plan_cli_emits_machine_readable_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "operator-cluster.yaml").write_text("executor: slurm\n")
    monkeypatch.setenv("NVFLOW_CONFIG_DIR", str(tmp_path))
    result = CliRunner().invoke(
        app,
        [
            "plan",
            "--config",
            "does-not-exist.yaml",
            "--cluster",
            "operator-cluster",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload == {
        "complete": False,
        "error": "FileNotFoundError",
        "message": "Workflow planning failed before configuration could be loaded",
    }


@pytest.mark.requires_nemo
def test_submission_receipt_discovers_handles_by_experiment(monkeypatch) -> None:
    monkeypatch.setattr(
        "nemo_skills.pipeline.utils.exp.get_exp_handles",
        lambda name, **_kwargs: [f"slurm:///{name}/123"],
    )
    task = TaskPlan(
        task_id="stage.main",
        stage="stage",
        name="stage-job",
        operation="nemo_skills.run_cmd",
        parameters={"experiment_name": "workflow-stage"},
    )
    plan = LaunchPlan(
        plan_version=1,
        project="example",
        workflow="workflow",
        target="cluster-a",
        stages=(StagePlan(stage="stage", tasks=(task,)),),
    )

    receipt = discover_submission_receipt(plan)

    assert receipt.plan_digest == plan.digest
    assert receipt.tasks[0].experiment == "workflow-stage"
    assert receipt.experiments[0].scheduler_handles == ("slurm:///workflow-stage/123",)
    assert receipt.experiments[0].discovery_complete is True


@pytest.mark.requires_nemo
@pytest.mark.parametrize(
    "captured",
    [("slurm:///workflow-stage/456",), ()],
)
def test_submission_receipt_prefers_authoritative_captured_handles(
    monkeypatch,
    captured: tuple[str, ...],
) -> None:
    monkeypatch.setattr(
        "nemo_skills.pipeline.utils.exp.get_exp_handles",
        lambda *_args, **_kwargs: pytest.fail("receipt reconstructed an experiment"),
    )
    task = TaskPlan(
        task_id="stage.main",
        stage="stage",
        name="stage-job",
        operation="nemo_skills.run_cmd",
        parameters={"experiment_name": "workflow-stage"},
    )
    plan = LaunchPlan(
        plan_version=1,
        project="example",
        workflow="workflow",
        target="cluster-a",
        stages=(StagePlan(stage="stage", tasks=(task,)),),
    )

    receipt = discover_submission_receipt(
        plan,
        captured_handles={"workflow-stage": captured},
    )

    assert receipt.experiments[0].scheduler_handles == captured
    assert receipt.experiments[0].discovery_complete is True


def test_submission_receipt_distinguishes_opaque_plan_from_noop() -> None:
    plan = LaunchPlan(
        plan_version=1,
        project="external",
        workflow="legacy",
        target="cluster",
        stages=(StagePlan.opaque("legacy", "execute-only external stage"),),
    )

    receipt = discover_submission_receipt(plan)

    assert receipt.plan_complete is False
    assert receipt.opaque_stages == ("legacy",)
    assert receipt.tasks == ()


def test_workflow_runner_returns_live_submission_receipt(monkeypatch, tmp_path: Path) -> None:
    from nvflow.integrations.nemo_skills import filesystem as filesystem_module
    from nvflow.integrations.nemo_skills import slurm as sbatch

    experiment = types.SimpleNamespace(
        jobs=[
            types.SimpleNamespace(
                handles=["slurm:///answer_generation-generate_answer/456"],
            )
        ],
    )
    generate_module = types.ModuleType("nemo_skills.pipeline.generate")
    generate_module.__dict__["generate"] = lambda **_kwargs: experiment
    run_cmd_module = types.ModuleType("nemo_skills.pipeline.run_cmd")
    run_cmd_module.__dict__["run_cmd"] = lambda **_kwargs: types.SimpleNamespace(
        jobs=[
            types.SimpleNamespace(handles=["slurm:///answer_generation-generate_answer-merge/457"])
        ],
    )
    exp_module = types.ModuleType("nemo_skills.pipeline.utils.exp")
    exp_module.__dict__["get_exp_handles"] = lambda *_args, **_kwargs: [
        "slurm:///answer_generation-generate_answer/456"
    ]
    utils_module = types.ModuleType("nemo_skills.pipeline.utils")
    pipeline_module = types.ModuleType("nemo_skills.pipeline")
    pipeline_module.__path__ = []  # type: ignore[attr-defined]
    pipeline_module.__dict__["utils"] = utils_module
    nemo_skills_module = types.ModuleType("nemo_skills")
    nemo_skills_module.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nemo_skills", nemo_skills_module)
    monkeypatch.setitem(sys.modules, "nemo_skills.pipeline", pipeline_module)
    monkeypatch.setitem(sys.modules, "nemo_skills.pipeline.generate", generate_module)
    monkeypatch.setitem(sys.modules, "nemo_skills.pipeline.run_cmd", run_cmd_module)
    monkeypatch.setitem(sys.modules, "nemo_skills.pipeline.utils", utils_module)
    monkeypatch.setitem(sys.modules, "nemo_skills.pipeline.utils.exp", exp_module)
    monkeypatch.setattr(sbatch, "apply_sbatch_args_autopatch", lambda: None)
    monkeypatch.setattr(
        "nvflow.execution.engines.slurm._submit_pipeline",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        filesystem_module,
        "target_filesystem",
        lambda _cluster: types.SimpleNamespace(
            batch_exists=lambda paths: dict.fromkeys(paths, False),
        ),
    )
    config = PROJECT_ROOT / "projects" / "starter" / "workflows" / "generate.yaml"
    provenance = ProjectResolver().resolve_reference("starter", config).source_provenance()
    monkeypatch.setattr(
        "nvflow.project.context.ProjectContext.source_provenance",
        lambda _self, **_kwargs: replace(
            provenance,
            clean=True,
            submittable=True,
            dirty_paths=(),
            reason=None,
        ),
    )
    target = tmp_path / "cluster-a.yaml"
    target.write_text("executor: slurm\n")
    receipt = WorkflowRunner(
        str(config),
        cluster="cluster-a",
        target_config_path=target,
    ).run()

    assert receipt.target == "cluster-a"
    assert receipt.tasks[0].experiment == "answer_generation-generate_answer"
    assert receipt.experiments[0].scheduler_handles == (
        "slurm:///answer_generation-generate_answer/456",
        "slurm:///answer_generation-generate_answer-merge/457",
    )


def test_contract_represents_service_backed_grpo_without_scheduler_script() -> None:
    judge = TaskPlan(
        task_id="training.judge",
        stage="training",
        name="grpo-judge",
        operation="service",
        commands=(
            CommandPlan(
                command="python -m vllm.entrypoints.openai.api_server",
                container="nemo-rl",
                role="judge",
            ),
        ),
        resources=(ResourceRequest(role="judge", nodes=1, gpus_per_node=4),),
        services=(
            ServicePlan(
                name="judge-api",
                task_id="training.judge",
                readiness="GET /health",
            ),
        ),
    )
    training = TaskPlan(
        task_id="training.policy",
        stage="training",
        name="grpo-policy",
        operation="nemo_rl.grpo",
        commands=(
            CommandPlan(
                command="python run_grpo_nemo_gym.py --config grpo.yaml",
                container="nemo-rl",
            ),
        ),
        resources=(ResourceRequest(role="policy", nodes=2, gpus_per_node=8),),
        dependencies=(DependencyPlan("training.judge", condition="after_ready"),),
        outputs=(
            ArtifactDeclaration(
                name="checkpoint",
                path="/workspace/output/checkpoints",
                kind="checkpoint",
            ),
        ),
    )
    cleanup = TaskPlan(
        task_id="training.cleanup",
        stage="training",
        name="grpo-cleanup",
        operation="cleanup",
        dependencies=(
            DependencyPlan("training.policy", condition="after_any"),
            DependencyPlan("training.judge", condition="after_any"),
        ),
    )
    launch_plan = LaunchPlan(
        plan_version=1,
        project="finance",
        workflow="grpo",
        target="cluster-a",
        stages=(
            StagePlan(
                stage="training",
                tasks=(judge, training, cleanup),
            ),
        ),
    )

    assert launch_plan.complete is True
    assert [task.task_id for task in launch_plan.tasks] == [
        "training.judge",
        "training.policy",
        "training.cleanup",
    ]
    assert launch_plan.tasks[1].dependencies[0].condition == "after_ready"
    assert launch_plan.tasks[2].dependencies[0].condition == "after_any"
    assert "sbatch" not in str(launch_plan.to_dict()).lower()


@pytest.mark.requires_nemo
def test_nemo_job_conversion_preserves_internal_dependencies() -> None:
    hardware = types.SimpleNamespace(
        num_nodes=1,
        num_gpus=2,
        num_tasks=1,
        partition="batch",
        account=None,
        sbatch_kwargs={},
    )
    first = {
        "name": "rollout-chunk",
        "group": types.SimpleNamespace(
            name="rollout",
            hardware=hardware,
            commands=[
                types.SimpleNamespace(
                    name="client",
                    container="nemo-rl",
                    script=types.SimpleNamespace(inline="python collect.py"),
                )
            ],
        ),
        "dependencies": ["prepare-data"],
    }
    second = {
        "name": "rollout-merge",
        "group": types.SimpleNamespace(
            name="merge",
            hardware=types.SimpleNamespace(
                num_nodes=1,
                num_gpus=0,
                num_tasks=1,
                partition="cpu",
                account=None,
                sbatch_kwargs={"dependency_type": "afterany"},
            ),
            commands=[
                types.SimpleNamespace(
                    name="merge",
                    container="nemo-rl",
                    script=types.SimpleNamespace(inline="python merge.py"),
                )
            ],
        ),
        "dependencies": [first],
    }

    stage_plan = stage_plan_from_jobs(
        "collect_rollouts",
        [first, second],
        output_dir="/workspace/output",
    )

    assert stage_plan.complete is True
    assert stage_plan.tasks[0].dependencies == (DependencyPlan("prepare-data", external=True),)
    assert stage_plan.tasks[1].dependencies == (
        DependencyPlan("collect_rollouts.0", condition="after_any"),
    )
    assert stage_plan.tasks[1].outputs[0].path == "/workspace/output"


@pytest.mark.requires_nemo
def test_direct_slurm_nemo_rl_lowering_preserves_ray_and_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contextlib import nullcontext

    from nvflow.execution.engines.slurm import _submit_nemo_rl

    cluster_config = {
        "executor": "slurm",
        "containers": {
            "nemo-rl": "nemo-rl-image",
            "vllm": "vllm-image",
        },
        "mounts": ["/host/logs:/logs"],
        "ray_template": "ray_enroot.sub.j2",
        "partition": "batch",
        "cpu_partition": "cpu",
        "job_name_prefix": "nvflow-",
    }
    submitted: list[dict[str, Any]] = []
    handles: list[str] = []
    run_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    cleanup_scripts: list[str] = []
    experiment = types.SimpleNamespace(jobs=[])

    def add_task(exp, **kwargs):
        # NeMo-Run returns a job id; the scheduler handle exists only after launch.
        exp.jobs.append(types.SimpleNamespace(id=kwargs["task_name"], handle=""))
        handles.append(kwargs["task_name"])
        submitted.append(kwargs)
        return kwargs["task_name"]

    def run_exp(exp, *args, **kwargs):
        run_calls.append(((exp, *args), kwargs))
        for index, job in enumerate(exp.jobs):
            job.handle = f"slurm:///{123 + index}/master/0"

    monkeypatch.setattr(
        "nvflow.integrations.nemo_skills.load_target_config",
        lambda _cluster: cluster_config,
    )
    monkeypatch.setattr(
        "nemo_skills.pipeline.utils.exp.get_exp",
        lambda *_args, **_kwargs: nullcontext(experiment),
    )
    monkeypatch.setattr("nemo_skills.pipeline.utils.exp.add_task", add_task)
    monkeypatch.setattr("nemo_skills.pipeline.utils.exp.run_exp", run_exp)

    def run_cleanup(*_args, **kwargs):
        cleanup_scripts.append(kwargs["input"])
        return types.SimpleNamespace(
            returncode=0,
            stdout="Submitted",
            stderr="",
        )

    monkeypatch.setattr(
        "nvflow.execution.engines.slurm.subprocess.run",
        run_cleanup,
    )

    first = TaskPlan(
        task_id="training.0",
        stage="training",
        name="sft-0",
        operation="nemo_rl.sft",
        commands=(
            CommandPlan(
                "python train.py",
                "nemo-rl",
                runtime_role="nemo-rl",
            ),
        ),
        resources=(ResourceRequest(nodes=2, gpus_per_node=8),),
        dependencies=(DependencyPlan("prepare.0"),),
        parameters={
            "experiment_name": "sft",
            "ray_application": True,
            "slurm": {"log_dir": "/logs"},
        },
    )
    retry = replace(
        first,
        task_id="training.1",
        name="sft-1",
        dependencies=(DependencyPlan("training.0", condition="after_any"),),
    )
    judge = TaskPlan(
        task_id="judge.0",
        stage="training",
        name="sft-judge-0",
        operation="service",
        commands=(
            CommandPlan(
                "python judge.py",
                "vllm",
                runtime_role="vllm",
            ),
        ),
        resources=(ResourceRequest(gpus_per_node=1),),
        dependencies=(DependencyPlan("training.1", condition="after_start"),),
        parameters={
            "experiment_name": "sft",
            "nemo_rl_group": True,
            "slurm": {"log_dir": "/logs"},
        },
    )
    cleanup = TaskPlan(
        task_id="cleanup.0",
        stage="training",
        name="sft-judge-cleanup-0",
        operation="cleanup",
        commands=(
            CommandPlan(
                "scancel --name=sft-judge-0",
                "host",
                runtime_role="host",
            ),
        ),
        dependencies=(DependencyPlan("training.1", condition="after_any"),),
        parameters={
            "experiment_name": "sft",
            "nemo_rl_group": True,
            "slurm_cleanup_target": "sft-judge-0",
            "slurm_cleanup_target_task": "judge.0",
            "slurm": {"log_dir": "/logs"},
        },
    )

    excluded_jobs: set[str] = set()
    _submit_nemo_rl(
        (first, retry, judge, cleanup),
        cluster="slurm",
        task_experiments={
            "prepare.0": "prepare",
            "training.0": "sft",
            "training.1": "sft",
            "judge.0": "sft",
            "cleanup.0": "sft",
        },
        excluded_jobs=excluded_jobs,
    )

    # Later stages wait only on the final attempt, which runs after every earlier one.
    assert excluded_jobs == {"slurm:///123/master/0", "slurm:///125/master/0"}
    assert submitted[0]["with_ray"] is True
    assert submitted[0]["ray_template"] == "ray_enroot.sub.j2"
    assert submitted[0]["run_after"] == ["prepare"]
    assert submitted[1]["task_dependencies"] == [handles[0]]
    assert submitted[1]["sbatch_kwargs"]["dependency_type"] == "afterany"
    assert submitted[2]["with_ray"] is False
    assert submitted[2]["task_dependencies"] == [handles[1]]
    assert submitted[2]["sbatch_kwargs"]["dependency_type"] == "after"
    assert "export NVFLOW_CHECKPOINT_SAVE_BY=99:23:45:00" in submitted[0]["cmd"]
    assert "NVFLOW_CHECKPOINT_SAVE_BY" not in submitted[2]["cmd"]
    assert submitted[2]["sbatch_kwargs"]["time"] == "100-00:00:00"
    assert "--dependency=afterany:124" in cleanup_scripts[0]
    assert "#SBATCH --output=/host/logs/judge-cleanup-%j.log" in cleanup_scripts[0]
    assert "scancel 125" in cleanup_scripts[0]
    assert len(run_calls) == 1


@pytest.mark.requires_nemo
def test_direct_slurm_multi_node_service_spans_group_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nvflow.execution.engines.slurm import _submit_pipeline

    captured = []
    monkeypatch.setattr(
        "nvflow.integrations.nemo_skills.load_target_config",
        lambda _cluster: {
            "executor": "none",
            "containers": {"sglang": "sglang-image"},
        },
    )
    monkeypatch.setattr(
        "nemo_skills.pipeline.utils.declarative.Pipeline.run",
        lambda self: captured.extend(self.jobs),
    )
    task = TaskPlan(
        task_id="serve.0",
        stage="serve",
        name="serve",
        operation="nemo_skills.pipeline",
        commands=(
            CommandPlan(
                "python -m nvflow.integrations.model_servers.serve "
                "--server-type sglang --model /model --port 8000 "
                "--num-gpus 8 --num-nodes 2",
                "sglang",
                runtime_role="sglang",
                group="service",
                kind="service",
                readiness="after_start",
            ),
        ),
        resources=(
            ResourceRequest(
                role="service",
                nodes=2,
                gpus_per_node=8,
            ),
        ),
        parameters={
            "experiment_name": "serve",
            "group_log_dirs": {"service": "/logs"},
        },
    )

    _submit_pipeline(
        (task,),
        cluster="slurm",
        task_experiments={"serve.0": "serve"},
        excluded_jobs=set(),
    )

    script = captured[0]["group"].commands[0].script
    assert script.span_group_nodes is True


@pytest.mark.requires_nemo
def test_direct_slurm_pipeline_retries_never_hold_back_later_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nvflow.execution.engines.slurm import _submit_pipeline

    monkeypatch.setattr(
        "nvflow.integrations.nemo_skills.load_target_config",
        lambda _cluster: {"executor": "none", "containers": {"nemo-gym": "gym-image"}},
    )

    def run(self):
        # NeMo-Run names every job alike; the scheduler handle exists only after launch.
        return types.SimpleNamespace(
            jobs=[
                types.SimpleNamespace(id="nemo-run", handle=f"slurm:///{201 + index}/master/0")
                for index, _job in enumerate(self.jobs)
            ]
        )

    monkeypatch.setattr("nemo_skills.pipeline.utils.declarative.Pipeline.run", run)

    def collect_task(task_id: str, dependencies: tuple[DependencyPlan, ...] = ()) -> TaskPlan:
        return TaskPlan(
            task_id=task_id,
            stage="collect",
            name=task_id.replace(".", "-"),
            operation="nemo_skills.pipeline",
            commands=(
                CommandPlan(
                    "python collect.py",
                    "nemo-gym",
                    runtime_role="nemo-gym",
                    group="main",
                ),
            ),
            resources=(ResourceRequest(role="main", gpus_per_node=4),),
            dependencies=dependencies,
            parameters={"experiment_name": "collect", "group_log_dirs": {"main": "/logs"}},
        )

    first = collect_task("attempt.0")
    retry = collect_task("attempt.1", (DependencyPlan("attempt.0", condition="after_any"),))
    analyze = collect_task("analyze.0", (DependencyPlan("attempt.1", condition="after_any"),))
    excluded_jobs: set[str] = set()

    _submit_pipeline(
        (first, retry, analyze),
        cluster="slurm",
        task_experiments=dict.fromkeys(("attempt.0", "attempt.1", "analyze.0"), "collect"),
        excluded_jobs=excluded_jobs,
    )

    # Later stages wait only on the analysis, which runs after every attempt.
    assert excluded_jobs == {"slurm:///201/master/0", "slurm:///202/master/0"}


@pytest.mark.requires_nemo
def test_sft_stage_builds_target_independent_training_plan(
    monkeypatch,
) -> None:
    config_path = PROJECT_ROOT / "projects" / "finance" / "workflows" / "sft" / "qwen3_4b.yaml"
    definition = load_workflow(config_path)
    monkeypatch.setattr(
        "nvflow.integrations.nemo_skills.load_target_config",
        lambda _cluster: pytest.fail("SFT planning loaded a Slurm target"),
    )
    raw = definition.config["stages"]["training"]
    context = PlanningContext(
        stage="training",
        config=definition.project_context.render_execution_paths(raw),
        launcher_config=definition.project_context.render_launcher_paths(raw),
        target=TargetBinding.resolve(definition.cluster, "cluster-a"),
        expname="sft-training",
        run_after=("sequence-length-grouping",),
    )

    stage_plan = SFTStage().plan(context)

    assert stage_plan.complete is True
    assert len(stage_plan.tasks) == 1
    assert stage_plan.tasks[0].operation == "nemo_rl.sft"
    assert stage_plan.tasks[0].resources[0].nodes == 1
    assert stage_plan.tasks[0].resources[0].gpus_per_node == 8
    command = stage_plan.tasks[0].commands[0]
    assert "run_sft.py" in command.command
    assert "base64 -d" in command.command
    assert command.display_command is not None
    assert "<nemo-rl-config:sha256:" in command.display_command
    assert "base64 -d" not in command.display_command

    four_gpu_plan = SFTStage().plan(
        replace(
            context,
            engine="slurm",
            gpu_topology={"*": 4},
        )
    )
    assert four_gpu_plan.tasks[0].resources[0].nodes == 2
    assert four_gpu_plan.tasks[0].resources[0].gpus_per_node == 4
    assert "++cluster.gpus_per_node=4" in four_gpu_plan.tasks[0].commands[0].command

    retry_plan = SFTStage().plan(
        replace(
            context,
            config={**context.config, "dependent_jobs": 1},
        )
    )
    assert retry_plan.tasks[1].dependencies == (
        DependencyPlan("training.0", condition="after_any"),
    )

    subset_plan = WorkflowPlanner(
        definition,
        TargetBinding.resolve(definition.cluster, "cluster-a"),
    ).plan(stages=["training", "training"])
    assert [stage.stage for stage in subset_plan.stages] == ["training"]
    assert all(
        not task.dependencies for task in subset_plan.tasks if task.operation == "nemo_rl.sft"
    )


@pytest.mark.requires_nemo
def test_sft_resumption_message_matches_the_planned_jobs(monkeypatch, capfd) -> None:
    config_path = PROJECT_ROOT / "projects" / "finance" / "workflows" / "sft" / "qwen3_4b.yaml"
    definition = load_workflow(config_path)
    monkeypatch.setattr(
        "nvflow.integrations.nemo_skills.load_target_config",
        lambda _cluster: pytest.fail("SFT planning loaded a Slurm target"),
    )
    raw = {
        key: value
        for key, value in definition.config["stages"]["training"].items()
        if key != "dependent_jobs"
    }
    context = PlanningContext(
        stage="training",
        config=definition.project_context.render_execution_paths(raw),
        launcher_config=definition.project_context.render_launcher_paths(raw),
        target=TargetBinding.resolve(definition.cluster, "cluster-a"),
        expname="sft-training",
        run_after=(),
    )
    capfd.readouterr()

    assert len(SFTStage().plan(context).tasks) == 1
    assert "Checkpoint resumption" not in "".join(capfd.readouterr())
    retry_plan = SFTStage().plan(replace(context, config={**context.config, "dependent_jobs": 1}))
    assert len(retry_plan.tasks) == 2
    assert "enabled (1 dependent jobs)" in " ".join("".join(capfd.readouterr()).split())


@pytest.mark.requires_nemo
def test_grpo_stage_builds_service_aware_plan_from_real_workflow(monkeypatch) -> None:
    config_path = PROJECT_ROOT / "projects" / "finance" / "workflows" / "grpo" / "qwen3_4b.yaml"
    definition = load_workflow(config_path)
    cluster_config = {
        "gpus_per_node": 8,
        "partition": "batch",
        "containers": {
            "nemo-rl": "nemo-rl-image",
            "vllm": "vllm-image",
        },
    }
    monkeypatch.setattr(
        "nvflow.integrations.nemo_skills.load_target_config",
        lambda _cluster: cluster_config,
    )
    monkeypatch.setattr(
        "nemo_skills.pipeline.nemo_rl.grpo.get_timeout_str",
        lambda *_args, **_kwargs: "00:30:00",
    )
    raw = definition.config["stages"]["training"]
    context = PlanningContext(
        stage="training",
        config=definition.project_context.render_execution_paths(raw),
        launcher_config=definition.project_context.render_launcher_paths(raw),
        target=TargetBinding.resolve(definition.cluster, "cluster-a"),
        expname="grpo-training",
        run_after=("prepare-data",),
    )

    stage_plan = GRPOStage().plan(context)
    public_plan = LaunchPlan(
        plan_version=1,
        project="finance",
        workflow="grpo",
        target="cluster-a",
        stages=(stage_plan,),
    )

    assert stage_plan.complete is True
    assert any(task.operation == "nemo_rl.grpo" for task in stage_plan.tasks)
    judge_tasks = [task for task in stage_plan.tasks if task.operation == "service"]
    policy_tasks = [task for task in stage_plan.tasks if task.operation == "nemo_rl.grpo"]
    cleanup_tasks = [task for task in stage_plan.tasks if task.operation == "cleanup"]
    assert judge_tasks
    assert cleanup_tasks
    assert all(task.resources[0].gpus_per_node == 0 for task in cleanup_tasks)
    assert all(
        dependency.condition == "after_start"
        for task in judge_tasks
        for dependency in task.dependencies
    )
    assert all(
        dependency.upstream not in {judge.task_id for judge in judge_tasks}
        for task in policy_tasks
        for dependency in task.dependencies
    )
    assert all(
        "sbatch" not in command.command for task in stage_plan.tasks for command in task.commands
    )
    assert any(
        "base64 -d" in command.command
        for task in stage_plan.tasks
        for command in task.commands
        if task.operation == "nemo_rl.grpo"
    )
    assert "base64 -d" not in json.dumps(public_plan.to_dict())

    retry_plan = GRPOStage().plan(
        replace(
            context,
            config={**context.config, "dependent_jobs": 1},
        )
    )
    retry_judges = [task for task in retry_plan.tasks if task.operation == "service"]
    assert len(retry_judges) == 2
    assert retry_judges[0].services[0].readiness.endswith("judge_host_0.txt")
    assert retry_judges[1].services[0].readiness.endswith("judge_host_1.txt")


@pytest.mark.requires_nemo
def test_embedded_eval_composes_checkpoint_plans(monkeypatch) -> None:
    cluster_config = {
        "executor": "none",
        "gpus_per_node": 8,
        "partition": "batch",
        "containers": {
            "nemo-gym": "nemo-gym-image",
            "nemo-rl": "nemo-rl-image",
            "nemo-skills": "nemo-skills-image",
            "vllm": "vllm-image",
        },
    }
    monkeypatch.setattr(
        "nvflow.integrations.nemo_skills.load_target_config",
        lambda _cluster: cluster_config,
    )
    context = PlanningContext(
        stage="eval",
        config={
            "eval_steps": [20],
            "eval_data_root": "/workspace/eval-campaign/step-1-eval-data",
            "checkpoint_path": "/workspace/checkpoints",
            "format": "fsdp",
            "base_output_dir": "/workspace/output",
            "campaign": "embedded.yaml",
            "preset": "qwen3_4b",
            "benchmarks": ["secque"],
            "seeds": 1,
        },
        launcher_config={},
        target=TargetBinding.resolve("cluster-a"),
        expname="sft-eval",
        run_after=("training",),
    )

    stage_plan = GymEmbeddedEvalStage().plan(context)

    assert stage_plan.complete is True
    assert stage_plan.tasks
    assert all(task.stage == "eval" for task in stage_plan.tasks)
    assert not any("prepare" in task.task_id for task in stage_plan.tasks)
    assert any(task.name.endswith("-convert") for task in stage_plan.tasks)
    assert any("step-20" in task.task_id for task in stage_plan.tasks)
