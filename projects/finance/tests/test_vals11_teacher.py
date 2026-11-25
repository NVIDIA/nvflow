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

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from nvflow.core.workflow_definition import WorkflowDefinition, load_workflow
from nvflow.execution import TargetBinding, WorkflowPlanner
from nvflow.execution.target import resolve_target

WORKFLOWS = Path(__file__).resolve().parents[1] / "workflows/sdg"
DATASETS = [
    (question_set, dataset) for question_set in ("sft", "rl") for dataset in ("web_off", "web_on")
] + [("sft", "held_out_web_off"), ("sft", "held_out_web_on")]


def _workflow(question_set: str, dataset: str) -> WorkflowDefinition:
    name = dataset.replace("_", "-")
    return load_workflow(WORKFLOWS / f"vals11-teacher-{question_set}-{name}.yaml")


def _teacher_environment(workflow: WorkflowDefinition) -> dict[str, Any]:
    [environment] = workflow.stage_runtime_config("teacher")["environments"].values()
    return dict(environment)


def _environment(question_set: str, dataset: str) -> dict[str, Any]:
    return _teacher_environment(_workflow(question_set, dataset))


@pytest.mark.parametrize(("question_set", "dataset"), DATASETS)
def test_each_run_answers_one_workflow_1_dataset(question_set: str, dataset: str) -> None:
    environment = _environment(question_set, dataset)
    server = environment["gym"]["finance_sec_search_resources_server"]["resources_servers"][
        "finance_sec_search"
    ]

    assert environment["input_data"].endswith(
        f"/workflow-1-prepare/{question_set}/step-2-build-pools/1-render/{dataset}.jsonl"
    )
    assert server["tavily_api_key"] == "${oc.env:TAVILY_API_KEY}"
    assert server["edgar_search_mode"] == "local"
    assert server["local_edgar_index_path"].endswith("/step-1-index/sp500_sec_fts.sqlite")
    assert server["reward_mode"] == "binary"


def test_every_dataset_runs_the_same_environment() -> None:
    environments = [_environment(question_set, dataset) for question_set, dataset in DATASETS]
    for environment in environments:
        del environment["input_data"]

    assert all(environment == environments[0] for environment in environments)


@pytest.mark.parametrize(
    ("question_set", "dataset", "chunks"),
    [
        ("sft", "web_off", 4),
        ("sft", "web_on", 4),
        ("sft", "held_out_web_off", 1),
        ("sft", "held_out_web_on", 1),
    ],
)
def test_held_out_sets_run_in_one_chunk(question_set: str, dataset: str, chunks: int) -> None:
    rollout = _workflow(question_set, dataset).stage_runtime_config("teacher")["rollout"]

    assert rollout["num_chunks"] == chunks


@pytest.mark.parametrize(("question_set", "dataset"), DATASETS)
def test_each_run_validates_and_writes_its_own_traces(question_set: str, dataset: str) -> None:
    workflow = _workflow(question_set, dataset)
    workflow.validate()
    traces = workflow.stage_runtime_config("select_traces")
    teacher = workflow.config["teacher"]

    assert (traces["dataset"], traces["teacher"]) == (dataset, teacher)
    for folder in (traces["output_dir"], traces["rollouts"]):
        assert f"/workflow-2-teacher/{question_set}/{dataset}/{teacher}/" in folder


def test_rejudge_is_off_and_would_grade_the_teachers_answers_with_deepseek() -> None:
    workflow = _workflow("sft", "web_off")
    teacher = workflow.stage_runtime_config("teacher")
    rejudge = workflow.stage_runtime_config("rejudge")
    judge = rejudge["verification"]["models"]["judge"]

    assert workflow.config["pipeline_stages"] == ["teacher", "select_traces"]
    assert rejudge["rollouts_dir"] == teacher["output_dir"]
    assert rejudge["environments"] == teacher["environments"]
    assert rejudge["output_dir"].endswith("/sft/web_off/kimi-k3.1-pro/rejudge/deepseek-v4.1-flash")
    assert judge["model"] == "private/nemotron/deepseek-ai/deepseek-v4.1-flash"
    assert (judge["model_type"], judge["api_key_env"]) == ("vllm_model", "NVCF_API_KEY")
    assert judge["gym_model"]["uses_reasoning_parser"] is True


def test_kimi_teaches_on_two_nodes() -> None:
    workflow = _workflow("sft", "web_off")
    workflow.validate()
    kimi = workflow.config["teacher_model"]
    traces = workflow.stage_runtime_config("select_traces")

    assert workflow.config["teacher"] == "kimi-k3.1-pro"
    assert kimi["model"] == "/hf_models/Kimi-K3.1-Pro"
    assert kimi["server_nodes"] == 2
    assert kimi["gym_model"]["vllm_serve_kwargs"]["distributed_executor_backend"] == "mp"
    assert "/workflow-2-teacher/sft/web_off/kimi-k3.1-pro/" in traces["output_dir"]


@pytest.mark.requires_nemo
def test_kimis_two_nodes_run_without_ray(tmp_path: Path) -> None:
    kimi = _workflow("sft", "web_off")
    target = tmp_path / "site.yaml"
    target.write_text("executor: slurm\n")
    binding = TargetBinding.resolve(kimi.cluster, "site")
    graph = WorkflowPlanner(kimi, binding).execution_graph(
        resolve_target(binding, target), stages=["teacher"], live=False
    )
    tasks = [task for stage in graph.stages for task in stage.tasks]
    commands = "\n".join(command.command for task in tasks for command in task.commands)
    servers = [resource for task in tasks for resource in task.resources if resource.nodes > 1]
    collects = [task for task in tasks if "collect-rollouts" in task.name]

    assert "ray start" not in commands
    assert "--distributed-executor-backend mp" in commands
    assert collects
    assert [resource.nodes for resource in servers] == [2] * len(collects)
