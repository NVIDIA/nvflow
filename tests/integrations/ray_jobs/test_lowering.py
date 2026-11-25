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
"""Ray lowering for heterogeneous service/client task groups."""

import pytest

from nvflow.execution import (
    CommandPlan,
    DependencyPlan,
    ExecutionGraph,
    ResourceRequest,
    ServicePlan,
    StageExecution,
    TaskPlan,
)
from nvflow.integrations.ray_jobs.lowering import lower_execution_graph


def test_heterogeneous_group_lowers_to_service_and_client_jobs() -> None:
    source = TaskPlan(
        task_id="rollout.0",
        stage="rollout",
        name="rollout",
        operation="nemo_skills.pipeline",
        commands=(
            CommandPlan(
                "python -m vllm.entrypoints.openai.api_server",
                "vllm-image",
                role="policy",
                group="policy-group",
                kind="service",
                readiness="after_start",
                runtime_role="vllm",
            ),
            CommandPlan(
                "gym eval run --output out.jsonl",
                "nemo-gym-image",
                role="client",
                group="policy-group",
                runtime_role="nemo-gym",
                ray_application=True,
            ),
        ),
        resources=(
            ResourceRequest(
                role="policy-group",
                gpus_per_node=4,
            ),
        ),
    )
    analyze = TaskPlan(
        task_id="rollout.1",
        stage="rollout",
        name="analyze",
        operation="shell",
        commands=(
            CommandPlan(
                "python analyze.py",
                "nemo-gym",
                runtime_role="nemo-gym",
            ),
        ),
        dependencies=(DependencyPlan("rollout.0"),),
    )
    graph = ExecutionGraph(
        execution_version=1,
        project="project",
        workflow="workflow",
        target="ray",
        engine="ray",
        target_fingerprint="sha256:target",
        stages=(
            StageExecution(
                stage="rollout",
                tasks=(source, analyze),
                implementation=_implementation(),
            ),
        ),
    )

    lowered = lower_execution_graph(graph)

    service, client, downstream = lowered.tasks
    assert service.commands[0].container == "vllm-image"
    assert service.commands[0].runtime_role == "vllm"
    assert service.resources[0].gpus_per_node == 4
    assert service.services[0].consumer_task_id == client.task_id
    assert client.commands[0].container == "nemo-gym-image"
    assert client.commands[0].runtime_role == "nemo-gym"
    assert client.parameters["ray_application"] is True
    assert service.parameters["source_task_name"] == client.parameters["source_task_name"]
    assert client.parameters["source_task_name"] == "rollout"
    assert client.dependencies == (DependencyPlan(service.task_id, condition="after_ready"),)
    assert downstream.dependencies == (DependencyPlan(client.task_id),)


def test_declared_service_readiness_and_consumer_survive_lowering() -> None:
    training = TaskPlan(
        task_id="training.0",
        stage="training",
        name="training",
        operation="nemo_rl.grpo",
        commands=(
            CommandPlan(
                "python train.py",
                "nemo-rl",
                runtime_role="nemo-rl",
            ),
        ),
    )
    judge = TaskPlan(
        task_id="judge.0",
        stage="training",
        name="judge",
        operation="service",
        commands=(
            CommandPlan(
                "python judge.py",
                "vllm",
                kind="service",
                readiness="after_start",
                runtime_role="vllm",
            ),
        ),
        dependencies=(DependencyPlan("training.0", condition="after_start"),),
        services=(
            ServicePlan(
                name="judge",
                task_id="judge.0",
                readiness="files:/shared/judge-host",
                cleanup="on_failure",
                consumer_task_id="training.0",
            ),
        ),
    )
    graph = ExecutionGraph(
        execution_version=1,
        project="project",
        workflow="workflow",
        target="ray",
        engine="ray",
        target_fingerprint="sha256:target",
        stages=(
            StageExecution(
                stage="training",
                tasks=(training, judge),
                implementation=_implementation(),
            ),
        ),
    )

    lowered = lower_execution_graph(graph)
    lowered_judge = next(task for task in lowered.tasks if task.task_id == "judge.0")

    assert lowered_judge.services == judge.services


def test_multiple_ordered_commands_require_explicit_tasks() -> None:
    source = TaskPlan(
        task_id="ordered.0",
        stage="ordered",
        name="ordered",
        operation="shell",
        commands=(
            CommandPlan("first", "worker", runtime_role="worker"),
            CommandPlan("second", "worker", runtime_role="worker"),
        ),
    )
    graph = ExecutionGraph(
        execution_version=1,
        project="project",
        workflow="workflow",
        target="ray",
        engine="ray",
        target_fingerprint="sha256:target",
        stages=(
            StageExecution(
                stage="ordered",
                tasks=(source,),
                implementation=_implementation(),
            ),
        ),
    )

    with pytest.raises(ValueError, match="multiple ordered non-service commands"):
        lower_execution_graph(graph)


def _implementation() -> dict:
    return {
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
