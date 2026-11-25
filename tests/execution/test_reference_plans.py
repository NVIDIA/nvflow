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
"""Representative end-to-end completeness gate for every reference workflow."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

import pytest
from omegaconf import OmegaConf

from nvflow import load_workflow
from nvflow.core import BaseStage
from nvflow.core.code_root import render_code_root
from nvflow.core.workflow_definition import load_config_with_inheritance
from nvflow.execution import ExecutionGraph, StageExecution, TargetBinding, WorkflowPlanner
from nvflow.execution.contracts import completion_task_ids
from nvflow.integrations.nemo_gym.runtime_binding import _parse_request
from nvflow.integrations.ray_jobs.lowering import lower_execution_graph

pytestmark = pytest.mark.requires_nemo

PROJECT_ROOT = Path(__file__).parents[2]


def _runnable_workflows() -> tuple[str, ...]:
    workflows = []
    for path in sorted((PROJECT_ROOT / "projects").glob("*/workflows/**/*.yaml")):
        source = OmegaConf.load(path)
        loaded = load_config_with_inheritance(path)
        if OmegaConf.select(source, "workflow.runnable") is False:
            continue
        config = OmegaConf.to_container(loaded, resolve=False)
        if not isinstance(config, dict):
            continue
        required = {"workflow", "cluster", "pipeline_stages", "stages"}
        if not required <= config.keys():
            continue
        if _contains_missing(config):
            raise AssertionError(
                f"Runnable workflow {path.relative_to(PROJECT_ROOT)} contains mandatory "
                "placeholders; complete it or declare workflow.runnable: false"
            )
        workflows.append(str(path.relative_to(PROJECT_ROOT)))
    return tuple(workflows)


def _contains_missing(value: object) -> bool:
    if value == "???":
        return True
    if isinstance(value, dict):
        return any(_contains_missing(nested) for nested in value.values())
    if isinstance(value, list):
        return any(_contains_missing(nested) for nested in value)
    return False


def _finishing_tasks_declare_outputs(stage) -> bool:
    final = set(completion_task_ids(stage.tasks))
    return any(task.outputs for task in stage.tasks if task.task_id in final)


def _assert_declared_outputs_match_config(definition, plan) -> None:
    for stage_plan in plan.stages:
        stage_name = stage_plan.stage
        config = definition.project_context.render_execution_paths(
            dict(definition.config["stages"][stage_name])
        )
        declared_paths = {output.path for output in stage_plan.outputs}
        configured_output = config.get("output_file")
        if configured_output and stage_name not in OUTPUT_FILE_DECLARATION_EXEMPTIONS:
            assert str(configured_output) in declared_paths, (
                f"{stage_name} does not declare configured output_file {configured_output!r}"
            )

        derived_filename = DERIVED_PRIMARY_OUTPUTS.get(stage_name)
        if derived_filename:
            output_dir = str(config["output_dir"]).rstrip("/")
            expected = f"{output_dir}/{derived_filename}"
            assert expected in declared_paths, (
                f"{stage_name} does not declare primary output {expected!r}"
            )


WORKFLOWS = _runnable_workflows()
EXPECTED_RUNNABLE_WORKFLOW_COUNT = 56
EXPECTED_DOWNLOAD_STAGES = {
    "projects/finance/workflows/download/demo.yaml": ["download", "build_index", "prefetch_cache"],
    "projects/finance/workflows/download/smoke.yaml": ["download", "build_index"],
    "projects/finance/workflows/download/sp500.yaml": ["download", "build_index", "prefetch_cache"],
}
OUTPUT_FILE_DECLARATION_EXEMPTIONS = {
    # The transformer always writes chunk files under output_dir/chunks.
    "data_transformation",
}
DERIVED_PRIMARY_OUTPUTS = {
    "filter_easy_candidates": "kept_output.jsonl",
    "filter_sft_reasoning_traces": "sft_output.jsonl",
    "generate_multihop_queries": "final_output.jsonl",
    "identify_categories": "final_output.jsonl",
    "image_filter": "final_output.jsonl",
    "localize_instances": "final_output.jsonl",
    "visualize_candidate_hopchain_data": "index.html",
    "visualize_final_hopchain_data": "index.html",
    "visualize_reconciled_hopchain_data": "index.html",
}
SUPPORTED_RUNTIME_ROLES = {
    "nemo-gym",
    "nemo-rl",
    "nemo-skills",
    "sglang",
    "vllm",
}


def test_grpo_reward_source_switches_input_and_dependency_together() -> None:
    config = OmegaConf.load(PROJECT_ROOT / "projects/finance/workflows/grpo/base.yaml")
    config.scope = "test"
    config.reward_source_stage = "rejudge"

    split = cast(
        dict[str, Any],
        OmegaConf.to_container(
            config.stages.train_validation_split,
            resolve=True,
        ),
    )

    assert split["input_dir"].endswith("/step-6-rejudge")
    assert split["dependencies"] == ["rejudge"]


def test_reference_workflow_inventory_is_expected() -> None:
    assert len(WORKFLOWS) == EXPECTED_RUNNABLE_WORKFLOW_COUNT
    assert {
        path for path in WORKFLOWS if path.startswith("projects/finance/workflows/download/")
    } == {
        "projects/finance/workflows/download/demo.yaml",
        "projects/finance/workflows/download/smoke.yaml",
        "projects/finance/workflows/download/sp500.yaml",
    }


@pytest.fixture
def planning_cluster(monkeypatch) -> dict:
    cluster = {
        "executor": "none",
        "gpus_per_node": 8,
        "partition": "batch",
        "cpu_partition": "cpu",
        "containers": {
            "nemo-gym": "nemo-gym-image",
            "nemo-rl": "nemo-rl-image",
            "nemo-skills": "nemo-skills-image",
            "sglang": "sglang-image",
            "vllm": "vllm-image",
        },
    }
    monkeypatch.setattr(
        "nvflow.integrations.nemo_skills.load_target_config",
        lambda *_args, **_kwargs: pytest.fail("planning loaded the Slurm target config"),
    )
    monkeypatch.setattr(
        "nemo_skills.pipeline.utils.cluster.get_cluster_config",
        lambda *_args, **_kwargs: cluster,
    )
    monkeypatch.setattr(
        "nemo_skills.pipeline.nemo_rl.grpo.get_timeout_str",
        lambda *_args, **_kwargs: "00:30:00",
    )
    monkeypatch.setenv("SEC_IDENTITY_EMAIL", "nvflow-tests@example.com")
    monkeypatch.setenv("SEC_IDENTITY_COMPANY", "NVFlow Tests")
    return cluster


def test_finance_grpo_planning_does_not_load_slurm_target_schema(planning_cluster: dict) -> None:
    definition = load_workflow(PROJECT_ROOT / "projects/finance/workflows/grpo/qwen3_4b.yaml")

    plan = WorkflowPlanner(
        definition,
        TargetBinding.resolve(definition.cluster, "supplied-ray"),
    ).plan()

    assert plan.complete
    assert any(task.operation == "nemo_rl.grpo" for task in plan.tasks)
    assert all(not dependency.external for task in plan.tasks for dependency in task.dependencies)


def test_grpo_eval_waits_on_training_rather_than_its_judge(planning_cluster: dict) -> None:
    definition = load_workflow(PROJECT_ROOT / "projects/finance/workflows/grpo/qwen3_4b.yaml")

    plan = WorkflowPlanner(
        definition,
        TargetBinding.resolve(definition.cluster, "planning-cluster"),
    ).plan(stages=("training", "eval"), environment=("equivalence_llm_judge",))

    tasks = {task.task_id: task for task in plan.tasks}
    training_upstream = {
        tasks[dependency.upstream].operation
        for task in plan.tasks
        if task.stage == "eval"
        for dependency in task.dependencies
        if tasks[dependency.upstream].stage == "training"
    }
    assert training_upstream == {"nemo_rl.grpo"}


def test_finance_sec_search_trains_and_evaluates_with_its_own_settings(
    planning_cluster: dict,
) -> None:
    definition = load_workflow(PROJECT_ROOT / "projects/finance/workflows/grpo/qwen3_4b.yaml")

    plan = WorkflowPlanner(
        definition,
        TargetBinding.resolve(definition.cluster, "planning-cluster"),
    ).plan(stages=("training", "eval"), environment=("finance_sec_search",))

    training = next(task for task in plan.tasks if task.operation == "nemo_rl.grpo")
    assert "grpo-qwen3-4b-32g-tp2-cp8-seq128k" in training.commands[0].command
    assert (
        sum(resource.nodes * (resource.gpus_per_node or 0) for resource in training.resources) == 32
    )
    eval_commands = " ".join(
        command.command for task in plan.tasks if task.stage == "eval" for command in task.commands
    )
    assert "grpo-qwen3-4b-32g-tp2-cp8-seq128k" in eval_commands
    assert '--hf-overrides \'{"rope_scaling":' in eval_commands


def _assert_commands_run_as_planned(plan) -> None:
    """Check what a Slurm job sees after the engine renders its command."""
    empty_modules = [
        task.task_id
        for task in plan.tasks
        for command in task.commands
        if "python3 -m ''" in command.command
    ]
    assert not empty_modules, f"commands run an empty Python module: {empty_modules}"
    for task in plan.tasks:
        for command in task.commands:
            rendered = render_code_root(command.command, "/nemo_run/code")
            for encoded in re.findall(r"runtime_binding --request '?([\w=-]+)", rendered):
                for path in _parse_request(encoded)["config_paths"]:
                    assert "{{" not in path, f"{task.task_id}: unrendered Gym config path {path}"
                    if path.startswith("/nemo_run/code/"):
                        shipped = PROJECT_ROOT / path.removeprefix("/nemo_run/code/")
                        assert shipped.is_file(), f"{task.task_id}: {path} is not shipped"


@pytest.mark.parametrize("relative_path", WORKFLOWS)
def test_reference_workflow_has_complete_plan(
    relative_path: str,
    planning_cluster: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition = load_workflow(PROJECT_ROOT / relative_path)
    selected = definition.select_active_stages(None)
    assert all(
        definition.resolve_stage_class(stage).execution is not BaseStage.execution
        for stage in selected
    )
    assert all(not hasattr(definition.resolve_stage_class(stage), "execute") for stage in selected)

    plan = WorkflowPlanner(
        definition,
        TargetBinding.resolve(definition.cluster, "planning-cluster"),
    ).plan()

    assert plan.complete is True
    assert plan.stages
    assert all(stage.complete for stage in plan.stages)
    assert all(not dependency.external for task in plan.tasks for dependency in task.dependencies)
    runtime_templates = [
        f"{task.task_id}:{command.role}"
        for task in plan.tasks
        for command in task.commands
        if command.runtime_template
    ]
    assert not runtime_templates, (
        "Executable reference plans cannot retain runtime command templates: "
        + ", ".join(runtime_templates)
    )
    assert all(
        "/nemo_run/code" not in command.command
        and "nemo_skills.inference.server.serve_" not in command.command
        for task in plan.tasks
        for command in task.commands
    )
    absolute_runtime_roles = [
        f"{task.task_id}:{command.container}"
        for task in plan.tasks
        for command in task.commands
        if command.container.startswith("/")
    ]
    assert not absolute_runtime_roles, (
        "Execution commands must use target-routable runtime roles, not image paths: "
        + ", ".join(absolute_runtime_roles)
    )
    monkeypatch.setenv("NVFLOW_CODE_ROOT", "/nemo_run/code")
    _assert_commands_run_as_planned(plan)
    ray_graph = lower_execution_graph(
        ExecutionGraph(
            execution_version=1,
            project=plan.project,
            workflow=plan.workflow,
            target="ray",
            engine="ray",
            target_fingerprint="sha256:test",
            stages=tuple(StageExecution.from_plan(stage) for stage in plan.stages),
            provenance=plan.provenance,
            environments=plan.environments,
        )
    )
    runtime_roles = {
        command.require_runtime_role() for task in ray_graph.tasks for command in task.commands
    }
    assert runtime_roles <= SUPPORTED_RUNTIME_ROLES
    if expected_download_stages := EXPECTED_DOWNLOAD_STAGES.get(relative_path):
        assert [stage.stage for stage in plan.stages] == expected_download_stages
    missing_outputs = [
        stage.stage
        for stage in plan.stages
        if stage.tasks and not _finishing_tasks_declare_outputs(stage)
    ]
    assert not missing_outputs, (
        f"The tasks that finish a stage must declare TaskPlan.outputs: {missing_outputs}"
    )
    _assert_declared_outputs_match_config(definition, plan)
    task_names = {task.name for task in plan.tasks}
    assert all(
        not (dependency.external and dependency.upstream in task_names)
        for task in plan.tasks
        for dependency in task.dependencies
    )


def test_optional_rejudge_has_complete_static_plan(
    planning_cluster: dict,
) -> None:
    definition = load_workflow(PROJECT_ROOT / "projects/finance/workflows/grpo/qwen3_4b.yaml")
    definition.config["pipeline_stages"] = ["rejudge"]

    plan = WorkflowPlanner(
        definition,
        TargetBinding.resolve(definition.cluster, "planning-cluster"),
    ).plan(environment=("equivalence_llm_judge",))

    assert plan.complete is True
    assert plan.planning_mode == "static"
    assert plan.environments == ("equivalence_llm_judge",)
    assert len(plan.tasks) == 3
    assert plan.tasks[0].commands[-1].runtime_template is False
    assert plan.tasks[1].name.endswith("-job-1-analyze-rollouts")
    assert plan.tasks[1].dependencies[0].upstream == plan.tasks[0].task_id
    assert plan.tasks[2].name.endswith("-job-2-build-training-dataset")
    assert plan.tasks[2].dependencies[0].upstream == plan.tasks[1].task_id
    assert plan.stages[0].outputs
    _assert_declared_outputs_match_config(definition, plan)


def test_gym_generation_capture_plan_shape_is_stable(
    planning_cluster: dict,
) -> None:
    definition = load_workflow(
        PROJECT_ROOT / "projects/finance/workflows/sdg/document-grounded-sdg-demo.yaml"
    )

    plan = WorkflowPlanner(
        definition,
        TargetBinding.resolve(definition.cluster, "planning-cluster"),
    ).plan(stages=("evaluate_answers",))

    assert [stage.stage for stage in plan.stages] == ["evaluate_answers"]
    assert len(plan.tasks) == 15
    assert [task.task_id for task in plan.tasks] == [
        "evaluate_answers.0",
        *(f"evaluate_answers.gym1.{index}" for index in range(13)),
        "evaluate_answers.14",
    ]
    assert [task.operation for task in plan.tasks].count("nemo_skills.run_cmd") == 2
    assert [task.operation for task in plan.tasks].count("nemo_skills.pipeline") == 13
    assert plan.tasks[0].name == "document_grounded_sdg-evaluate_answers-gen-render"
    assert [len(task.dependencies) for task in plan.tasks[1:13]] == [1] * 12
    assert all(
        dependency.upstream == "evaluate_answers.0"
        for task in plan.tasks[1:13]
        for dependency in task.dependencies
    )
    assert len(plan.tasks[13].dependencies) == 12
    assert plan.tasks[-1].name == "document_grounded_sdg-evaluate_answers"
    assert [
        (dependency.upstream, dependency.external) for dependency in plan.tasks[-1].dependencies
    ] == [("evaluate_answers.gym1.12", False)]


def test_gym_generation_declares_single_repeat_output_file(
    planning_cluster: dict,
) -> None:
    definition = load_workflow(
        PROJECT_ROOT / "projects/finance/workflows/sdg/document-grounded-sdg-demo.yaml"
    )
    output_file = "/workspace/outputs/evaluate/final_result.jsonl"
    definition.config["stages"]["evaluate_answers"]["output_file"] = output_file
    definition.config["stages"]["evaluate_answers"]["num_repeats"] = 1

    plan = WorkflowPlanner(
        definition,
        TargetBinding.resolve(definition.cluster, "planning-cluster"),
    ).plan(stages=("evaluate_answers",))

    assert [output.path for output in plan.tasks[-1].outputs] == [output_file]


def test_hopchain_localization_declares_chunk_and_final_outputs(
    planning_cluster: dict,
) -> None:
    definition = load_workflow(PROJECT_ROOT / "projects/hopchain/workflows/sdg/full.yaml")
    plan = WorkflowPlanner(
        definition,
        TargetBinding.resolve(definition.cluster, "planning-cluster"),
    ).plan(stages=("localize_instances",))
    config = definition.project_context.render_execution_paths(
        dict(definition.config["stages"]["localize_instances"])
    )
    output_dir = str(config["output_dir"]).rstrip("/")

    assert all(task.outputs for task in plan.tasks)
    assert f"{output_dir}/final_output.jsonl" in {output.path for output in plan.tasks[-1].outputs}
    if len(plan.tasks) > 1:
        assert all(
            any("/temp/chunks/final_output.part-" in output.path for output in task.outputs)
            for task in plan.tasks[:-1]
        )
