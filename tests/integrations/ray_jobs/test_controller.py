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
"""Deterministic multi-cluster controller behavior."""

from __future__ import annotations

import base64
import json
import shlex
from pathlib import Path
from typing import Any, Literal, cast

import pytest

from nvflow.execution import (
    CommandPlan,
    DependencyPlan,
    ExecutionGraph,
    ResourceRequest,
    StageExecution,
    TaskPlan,
)
from nvflow.integrations.ray_jobs import controller
from nvflow.integrations.ray_jobs.client import RayJobDetails, RayJobState


class FakeClient:
    instances: dict[str, FakeClient] = {}

    def __init__(self, cluster) -> None:
        self.cluster = cluster
        self.jobs: dict[str, RayJobDetails] = {}
        self.stopped: list[str] = []
        self.submissions: list[dict[str, object]] = []
        self.instances[cluster.name] = self

    def find(self, submission_id):
        return self.jobs.get(submission_id)

    def submit(self, *, submission_id, **_kwargs):
        self.submissions.append({"submission_id": submission_id, **_kwargs})
        task_id = (_kwargs.get("metadata") or {}).get("nvflow_task_id")
        self.jobs[submission_id] = RayJobDetails(
            submission_id,
            (RayJobState.RUNNING if task_id == "serve.0" else RayJobState.SUCCEEDED),
        )
        return submission_id

    def status(self, submission_id):
        return self.jobs[submission_id]

    def stop(self, submission_id):
        self.stopped.append(submission_id)
        self.jobs[submission_id] = RayJobDetails(
            submission_id,
            RayJobState.STOPPED,
        )
        return True


def test_controller_runs_cross_cluster_dependencies(
    tmp_path: Path,
    monkeypatch,
) -> None:
    FakeClient.instances = {}
    graph = _graph()
    spec = {
        "run_id": "run-test",
        "execution_digest": graph.digest,
        "graph": graph.to_dict(),
        "project_python_root": "projects/starter",
        "user": "alice",
        "workflow": "demo:train/small",
        "poll_seconds": 0,
        "timeout_seconds": 10,
        "ray_target": {
            "control_cluster": "cpu",
            "artifact_root": str(tmp_path / "artifacts"),
            "clusters": {
                "cpu": {
                    "jobs_url": "http://127.0.0.1:8265",
                },
                "gpu": {
                    "jobs_url": "http://127.0.0.1:8266",
                },
            },
            "routes": {
                "project-cpu": "cpu",
                "project-gpu": "gpu",
            },
        },
    }
    spec_path = tmp_path / ".nvflow" / "run.json"
    spec_path.parent.mkdir()
    spec_path.write_text(json.dumps(spec))
    (tmp_path / ".nvflow" / "worker-bundle.zip").write_bytes(b"zip")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(controller, "RayJobsClient", FakeClient)

    controller.run_controller(spec_path)

    assert len(FakeClient.instances["cpu"].jobs) == 1
    assert len(FakeClient.instances["gpu"].jobs) == 1
    for client in FakeClient.instances.values():
        assert client.submissions[0]["env_vars"] == {}
        assert str(client.submissions[0]["entrypoint"]).startswith("env ")
    submission = FakeClient.instances["gpu"].submissions[0]
    metadata = cast(dict[str, str], submission["metadata"])
    assert (metadata["nvflow_user"], metadata["nvflow_task_name"]) == ("alice", "train")
    assert {
        key: metadata[key]
        for key in (
            "nvflow_workflow",
            "nvflow_stage",
            "nvflow_experiment",
            "nvflow_nodes",
            "nvflow_gpus_per_node",
        )
    } == {
        "nvflow_workflow": "demo:train/small",
        "nvflow_stage": "train",
        "nvflow_experiment": "train",
        "nvflow_nodes": "1",
        "nvflow_gpus_per_node": "1",
    }
    payload = json.loads(base64.urlsafe_b64decode(shlex.split(str(submission["entrypoint"]))[-1]))
    job_id = str(submission["submission_id"]).rsplit("-", 1)[-1]
    assert payload["log_path"] == (
        f"{tmp_path / 'artifacts'}/.nvflow/runs/run-test/logs/main_train_{job_id}_ray.log"
    )
    state = json.loads(
        (tmp_path / "artifacts" / ".nvflow" / "runs" / "run-test" / "state.json").read_text()
    )
    assert state["state"] == "succeeded"


@pytest.mark.parametrize(
    ("command", "operation", "parameters", "expected"),
    [
        (
            CommandPlan("serve", "vllm", role="server-0", group="server-0", kind="service"),
            "service",
            {
                "group_log_dirs": {"server-0": "/out/generation-logs"},
                "source_task_name": "generate-rs0",
            },
            "/out/generation-logs/server_generate-rs0_{job}_ray.log",
        ),
        (
            CommandPlan("generate", "nemo-skills", role="client", group="client"),
            "nemo_skills.generate",
            {
                "group_log_dirs": {"client": "/out/generation-logs"},
                "source_task_name": "generate-rs0",
            },
            "/out/generation-logs/main_generate-rs0_{job}_ray.log",
        ),
        (
            CommandPlan("train", "nemo-rl"),
            "nemo_rl.sft",
            {"log_dir": "/out/training-logs/"},
            "/out/training-logs/ray-{job}-job.log",
        ),
        (
            CommandPlan("prepare", "nemo-skills"),
            "shell",
            {"slurm": {"log_dir": "/out/logs"}},
            "/out/logs/main_task_{job}_ray.log",
        ),
        (
            CommandPlan("prepare", "nemo-skills"),
            "shell",
            {},
            "/artifacts/.nvflow/runs/run/logs/main_task_{job}_ray.log",
        ),
    ],
    ids=["generation-server", "generation-client", "nemo-rl", "slurm-log-dir", "no-log-dir"],
)
def test_log_path_uses_the_slurm_log_directory(
    command: CommandPlan,
    operation: str,
    parameters: dict[str, Any],
    expected: str,
) -> None:
    task = TaskPlan(
        task_id="stage.0",
        stage="stage",
        name="task",
        operation=operation,
        commands=(command,),
        parameters=parameters,
    )
    submission_id = controller._submission_id("run", task.task_id)

    path = controller._log_path(task, submission_id, "/artifacts/", "run")

    assert path == expected.format(job=submission_id.rsplit("-", 1)[-1])


def test_submission_ids_are_deterministic_and_attempt_bound() -> None:
    first = controller._submission_id("run", "stage.task")
    second = controller._submission_id("run", "stage.task")
    retry = controller._submission_id("run", "stage.task", 2)

    assert first == second
    assert first != retry
    assert len(first) <= 128


def test_dependency_decision_requires_every_edge_to_be_satisfied() -> None:
    task = TaskPlan(
        task_id="consumer",
        stage="stage",
        name="consumer",
        operation="shell",
        dependencies=(
            DependencyPlan("failed-branch", condition="after_any"),
            DependencyPlan("running-branch", condition="after_success"),
        ),
    )

    assert (
        controller._dependency_decision(
            task,
            {"running-branch": RayJobState.RUNNING},
            {"failed-branch"},
            set(),
        )
        == "wait"
    )


@pytest.mark.parametrize("condition", ["after_start", "after_ready"])
def test_dependency_decision_blocks_when_required_upstream_never_started(
    condition: Literal["after_start", "after_ready"],
) -> None:
    task = TaskPlan(
        task_id="consumer",
        stage="stage",
        name="consumer",
        operation="shell",
        dependencies=(DependencyPlan("blocked-service", condition=condition),),
    )

    assert (
        controller._dependency_decision(
            task,
            {},
            {"blocked-service"},
            set(),
        )
        == "blocked"
    )


def test_after_ready_accepts_a_service_that_finished_before_it_was_ready() -> None:
    # Rollout servers exit at once when their chunk is already complete.
    task = TaskPlan(
        task_id="client",
        stage="stage",
        name="client",
        operation="shell",
        dependencies=(DependencyPlan("server", condition="after_ready"),),
    )

    assert (
        controller._dependency_decision(task, {"server": RayJobState.SUCCEEDED}, set(), set())
        == "ready"
    )
    assert (
        controller._dependency_decision(task, {"server": RayJobState.FAILED}, set(), set())
        == "blocked"
    )


def test_after_start_accepts_a_terminal_job_that_reached_ray() -> None:
    task = TaskPlan(
        task_id="consumer",
        stage="stage",
        name="consumer",
        operation="shell",
        dependencies=(DependencyPlan("started", condition="after_start"),),
    )

    assert (
        controller._dependency_decision(
            task,
            {"started": RayJobState.FAILED},
            set(),
            set(),
        )
        == "ready"
    )


def test_controller_stops_owned_service_after_consumer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    graph = _service_graph()
    spec = {
        "run_id": "run-service",
        "execution_digest": graph.digest,
        "graph": graph.to_dict(),
        "project_python_root": "projects/starter",
        "poll_seconds": 0,
        "timeout_seconds": 10,
        "ray_target": {
            "control_cluster": "shared",
            "artifact_root": str(tmp_path / "artifacts"),
            "clusters": {
                "shared": {"jobs_url": "http://127.0.0.1:8265"},
            },
            "routes": {"service": "shared", "client": "shared"},
        },
    }
    spec_path = tmp_path / ".nvflow" / "run.json"
    spec_path.parent.mkdir()
    spec_path.write_text(json.dumps(spec))
    (tmp_path / ".nvflow" / "worker-bundle.zip").write_bytes(b"zip")
    FakeClient.instances = {}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(controller, "RayJobsClient", FakeClient)

    controller.run_controller(spec_path)

    assert len(FakeClient.instances["shared"].stopped) == 1


def test_service_readiness_has_a_bounded_timeout(monkeypatch) -> None:
    monkeypatch.setattr(controller.time, "monotonic", lambda: 31.0)
    monkeypatch.setattr(controller, "_readiness_satisfied", lambda _contract: False)

    with pytest.raises(TimeoutError, match="serve.0"):
        controller._refresh_readiness(
            _service_graph(),
            {"serve.0": RayJobState.RUNNING},
            set(),
            {"serve.0": 0.0},
            30.0,
            cast(Any, object()),
        )


def test_after_any_cleanup_runs_after_child_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FailureClient(FakeClient):
        def submit(self, *, submission_id, **kwargs):
            task_id = (kwargs.get("metadata") or {}).get("nvflow_task_id")
            self.jobs[submission_id] = RayJobDetails(
                submission_id,
                (RayJobState.FAILED if task_id == "work.0" else RayJobState.SUCCEEDED),
            )
            return submission_id

    graph = _failure_graph()
    spec = {
        "run_id": "run-failure",
        "execution_digest": graph.digest,
        "graph": graph.to_dict(),
        "project_python_root": "projects/starter",
        "poll_seconds": 0,
        "timeout_seconds": 10,
        "ray_target": {
            "control_cluster": "shared",
            "artifact_root": str(tmp_path / "artifacts"),
            "clusters": {
                "shared": {"jobs_url": "http://127.0.0.1:8265"},
            },
            "routes": {"worker": "shared"},
        },
    }
    spec_path = tmp_path / ".nvflow" / "run.json"
    spec_path.parent.mkdir()
    spec_path.write_text(json.dumps(spec))
    (tmp_path / ".nvflow" / "worker-bundle.zip").write_bytes(b"zip")
    FailureClient.instances = {}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(controller, "RayJobsClient", FailureClient)

    with pytest.raises(RuntimeError, match="failed"):
        controller.run_controller(spec_path)

    assert len(FailureClient.instances["shared"].jobs) == 2


def _graph() -> ExecutionGraph:
    implementation = _implementation()
    prepare = TaskPlan(
        task_id="prepare.0",
        stage="prepare",
        name="prepare",
        operation="shell",
        commands=(CommandPlan("true", "project-cpu", runtime_role="project-cpu"),),
        resources=(ResourceRequest(),),
    )
    train = TaskPlan(
        task_id="train.0",
        stage="train",
        name="train",
        operation="shell",
        commands=(CommandPlan("true", "project-gpu", runtime_role="project-gpu"),),
        resources=(ResourceRequest(gpus_per_node=1),),
        dependencies=(DependencyPlan("prepare.0"),),
    )
    return ExecutionGraph(
        execution_version=1,
        project="project",
        workflow="workflow",
        target="ray",
        engine="ray",
        target_fingerprint="sha256:target",
        stages=(
            StageExecution(
                stage="prepare",
                tasks=(prepare,),
                implementation=implementation,
            ),
            StageExecution(
                stage="train",
                tasks=(train,),
                implementation=implementation,
            ),
        ),
    )


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


def _service_graph() -> ExecutionGraph:
    from nvflow.execution import ServicePlan

    service = TaskPlan(
        task_id="serve.0",
        stage="serve",
        name="serve",
        operation="service",
        commands=(
            CommandPlan(
                "python serve.py",
                "service",
                kind="service",
                readiness="after_start",
                runtime_role="service",
            ),
        ),
        services=(
            ServicePlan(
                name="api",
                task_id="serve.0",
                readiness="after_start",
                consumer_task_id="consume.0",
            ),
        ),
    )
    consumer = TaskPlan(
        task_id="consume.0",
        stage="consume",
        name="consume",
        operation="shell",
        commands=(
            CommandPlan(
                "python consume.py",
                "client",
                runtime_role="client",
            ),
        ),
        dependencies=(DependencyPlan("serve.0", condition="after_ready"),),
    )
    return ExecutionGraph(
        execution_version=1,
        project="project",
        workflow="service",
        target="ray",
        engine="ray",
        target_fingerprint="sha256:target",
        stages=(
            StageExecution(
                stage="serve",
                tasks=(service,),
                implementation=_implementation(),
            ),
            StageExecution(
                stage="consume",
                tasks=(consumer,),
                implementation=_implementation(),
            ),
        ),
    )


def _failure_graph() -> ExecutionGraph:
    work = TaskPlan(
        task_id="work.0",
        stage="work",
        name="work",
        operation="shell",
        commands=(CommandPlan("false", "worker", runtime_role="worker"),),
    )
    cleanup = TaskPlan(
        task_id="cleanup.0",
        stage="cleanup",
        name="cleanup",
        operation="shell",
        commands=(CommandPlan("true", "worker", runtime_role="worker"),),
        dependencies=(DependencyPlan("work.0", condition="after_any"),),
    )
    return ExecutionGraph(
        execution_version=1,
        project="project",
        workflow="failure",
        target="ray",
        engine="ray",
        target_fingerprint="sha256:target",
        stages=(
            StageExecution(
                stage="work",
                tasks=(work,),
                implementation=_implementation(),
            ),
            StageExecution(
                stage="cleanup",
                tasks=(cleanup,),
                implementation=_implementation(),
            ),
        ),
    )
