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
"""Supplied Ray engine submission contracts."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from nvflow.execution import (
    CommandPlan,
    ExecutionGraph,
    RayTarget,
    RegisteredCluster,
    ResolvedTarget,
    ResourceRequest,
    StageExecution,
    TargetBinding,
    TaskPlan,
)
from nvflow.execution.engines.base import ActiveJob, EngineContext
from nvflow.execution.engines.ray import (
    SuppliedRayJobsEngine,
    _resolve_controller_submission,
    _target_payload,
    _validate_capacity,
)
from nvflow.execution.planner import _reject_ray_incompatible_controls
from nvflow.execution.run_records import RunJournal
from nvflow.execution.runtime_profile import RuntimeProfile
from nvflow.integrations.ray_jobs.bundle import ControllerBundle
from nvflow.integrations.ray_jobs.client import (
    RayClusterProbe,
    RayJobDetails,
    RayJobsError,
    RayJobState,
)
from nvflow.integrations.ray_jobs.launcher_access import LauncherAccess, LocalArtifactFiles
from nvflow.project.packaging import WorkerBundle
from nvflow.project.provenance import ProjectProvenance


class FakeClient:
    instances: dict[str, FakeClient] = {}

    def __init__(self, cluster) -> None:
        self.cluster = cluster
        self.probed = False
        self.submitted: list[dict[str, Any]] = []
        self.instances[cluster.name] = self

    def probe(self):
        self.probed = True

    def probe_runtime(self, *, probe_id, **_kwargs):
        self.probe_id = probe_id
        return SimpleNamespace(
            resources={"CPU": 32.0, "GPU": 8.0},
            labels={},
        )

    def find(self, _submission_id):
        return None

    def submit(self, **kwargs):
        self.submitted.append(kwargs)
        return kwargs["submission_id"]


def test_ray_engine_submits_one_controller_and_never_imports_slurm(
    tmp_path: Path,
    monkeypatch,
) -> None:
    graph = _graph()
    target = _target(tmp_path)
    worker_path = tmp_path / "worker.zip"
    worker_path.write_bytes(b"worker")
    controller_path = tmp_path / "controller.zip"
    controller_path.write_bytes(b"controller")
    worker = WorkerBundle(
        path=worker_path,
        digest="sha256:worker",
        compressed_size=6,
        uncompressed_size=6,
        project_python_root="projects/starter",
        project_identity="git:project",
        platform_identity="git:platform",
    )
    monkeypatch.setattr(
        "nvflow.execution.engines.ray.build_worker_bundle",
        lambda *_args, **_kwargs: worker,
    )
    monkeypatch.setattr(
        "nvflow.execution.engines.ray.build_controller_bundle",
        lambda *_args, **_kwargs: ControllerBundle(
            path=controller_path,
            digest="sha256:controller",
            compressed_size=10,
        ),
    )
    monkeypatch.setattr("nvflow.execution.engines.ray.RayJobsClient", FakeClient)
    monkeypatch.setattr(
        "nvflow.execution.engines.ray.validate_launcher_runtime",
        lambda: None,
    )
    monkeypatch.setattr(
        "nvflow.execution.engines.ray.getpass", SimpleNamespace(getuser=lambda: "alice")
    )

    receipt = SuppliedRayJobsEngine().submit(
        graph,
        EngineContext(
            target=target,
            project_context=cast(Any, SimpleNamespace()),
            platform_source="git:platform",
            cache_dir=tmp_path,
        ),
    )

    assert all(client.probed for client in FakeClient.instances.values())
    assert len(FakeClient.instances["control"].submitted) == 1
    controller_submission = FakeClient.instances["control"].submitted[0]
    assert controller_submission["metadata"]["nvflow_user"] == "alice"
    assert controller_submission["env_vars"] == {}
    assert controller_submission["entrypoint"].startswith("env ")
    assert "NVFLOW_RUN_ID=" in controller_submission["entrypoint"]
    assert FakeClient.instances["worker"].submitted == []
    assert receipt.engine == "ray"
    assert receipt.execution_digest == graph.digest
    assert receipt.controller_handle is not None
    assert receipt.controller_handle.startswith("rayjobs://control/")


class RecordingFiles:
    def __init__(self) -> None:
        self.contents: dict[str, str] = {}
        self.reads: list[str] = []
        self.removed: list[str] = []

    def write_text(self, path: str, text: str) -> None:
        self.contents[path] = text

    def read_text(self, path: str) -> str:
        self.reads.append(path)
        if path not in self.contents:
            raise FileNotFoundError(path)
        return self.contents[path]

    def remove(self, path: str) -> None:
        self.removed.append(path)
        self.contents.pop(path, None)


def test_ray_engine_reaches_clusters_and_artifact_root_through_launcher_access(
    tmp_path: Path,
    monkeypatch,
) -> None:
    graph = _graph()
    target = _target(tmp_path)
    ray_target = target.ray
    assert ray_target is not None
    files = RecordingFiles()
    forwarded = {
        name: replace(cluster, jobs_url=f"http://127.0.0.1:{19000 + index}")
        for index, (name, cluster) in enumerate(ray_target.clusters.items())
    }

    @contextmanager
    def tunneled_access(received, names=None):
        assert received is ray_target
        assert names == {"control", "worker"}
        yield LauncherAccess(forwarded, files)

    worker_path = tmp_path / "worker.zip"
    worker_path.write_bytes(b"worker")
    controller_path = tmp_path / "controller.zip"
    controller_path.write_bytes(b"controller")
    run_specs: list[dict[str, Any]] = []
    monkeypatch.setattr("nvflow.execution.engines.ray.launcher_access", tunneled_access)
    monkeypatch.setattr(
        "nvflow.execution.engines.ray.build_worker_bundle",
        lambda *_args, **_kwargs: WorkerBundle(
            path=worker_path,
            digest="sha256:worker",
            compressed_size=6,
            uncompressed_size=6,
            project_python_root="projects/starter",
            project_identity="git:project",
            platform_identity="git:platform",
        ),
    )

    def record_controller_bundle(_worker, run_spec, _destination):
        run_specs.append(run_spec)
        return ControllerBundle(path=controller_path, digest="sha256:c", compressed_size=10)

    monkeypatch.setattr(
        "nvflow.execution.engines.ray.build_controller_bundle",
        record_controller_bundle,
    )
    monkeypatch.setattr("nvflow.execution.engines.ray.RayJobsClient", FakeClient)
    monkeypatch.setattr("nvflow.execution.engines.ray.validate_launcher_runtime", lambda: None)
    FakeClient.instances.clear()

    SuppliedRayJobsEngine().submit(
        graph,
        EngineContext(
            target=target,
            project_context=cast(Any, SimpleNamespace()),
            platform_source="git:platform",
            cache_dir=tmp_path,
        ),
    )

    assert FakeClient.instances["control"].cluster.jobs_url == "http://127.0.0.1:19000"
    probes = [path for path in files.removed if "/.nvflow/probes/launcher-" in path]
    assert len(probes) == 1 and probes[0].startswith(ray_target.artifact_root)
    assert files.contents == {}
    run_id = run_specs[0]["run_id"]
    assert files.reads == [f"{ray_target.artifact_root}/.nvflow/runs/{run_id}/events.jsonl"]
    assert run_specs[0]["ray_target"]["clusters"]["worker"]["jobs_url"] == (
        ray_target.clusters["worker"].controller_jobs_url
    )
    assert len(FakeClient.instances["control"].submitted) == 1


def test_ray_engine_fails_before_submission_for_insufficient_capacity(
    tmp_path: Path,
) -> None:
    graph = _graph()
    target = _target(tmp_path).ray
    assert target is not None

    with pytest.raises(ValueError, match="requires 1 CPUs"):
        _validate_capacity(
            graph,
            target,
            {
                "control": RayClusterProbe(
                    runtime=RuntimeProfile("3.13.14", "2.58.0"),
                    image=None,
                    artifact_visible=True,
                    resources={"CPU": 1},
                ),
                "worker": RayClusterProbe(
                    runtime=RuntimeProfile("3.13.14", "2.58.0"),
                    image=None,
                    artifact_visible=True,
                    resources={"CPU": 0},
                ),
            },
        )


def test_controller_payload_uses_controller_facing_jobs_url() -> None:
    cluster = RegisteredCluster(
        "worker",
        "http://127.0.0.1:18265",
        controller_jobs_url="http://worker.internal:8265",
        credential_env="LAUNCHER_TOKEN",
        controller_credential_env="CONTROLLER_TOKEN",
        allow_insecure=True,
    )
    target = RayTarget(
        control_cluster="worker",
        artifact_root="/shared",
        clusters={"worker": cluster},
        routes={"project": "worker"},
    )

    payload = _target_payload(target, {"worker"})

    assert payload["clusters"]["worker"]["jobs_url"] == "http://worker.internal:8265"
    assert payload["clusters"]["worker"]["credential_env"] == "CONTROLLER_TOKEN"


def test_failed_controller_job_advances_attempt() -> None:
    class Client:
        def find(self, submission_id):
            if submission_id.endswith("-a1"):
                return RayJobDetails(submission_id, RayJobState.FAILED)
            return None

    submission_id, existing = _resolve_controller_submission(
        cast(Any, Client()),
        "run",
        1,
    )

    assert submission_id == "run-controller-a2"
    assert existing is None


def test_capacity_probe_enforces_declared_labels() -> None:
    cluster = RegisteredCluster(
        "worker",
        "http://127.0.0.1:8265",
        labels={"accelerator": "h100"},
    )
    target = RayTarget(
        control_cluster="worker",
        artifact_root="/shared",
        clusters={"worker": cluster},
        routes={"project": "worker"},
    )

    with pytest.raises(ValueError, match="requires label accelerator"):
        _validate_capacity(
            _graph(),
            target,
            {
                "worker": RayClusterProbe(
                    runtime=RuntimeProfile("3.13.14", "2.58.0"),
                    image=None,
                    artifact_visible=True,
                    resources={"CPU": 8},
                    labels={"accelerator": "a100"},
                )
            },
        )


def test_ray_preflight_rejects_unapplied_slurm_controls() -> None:
    graph = _graph()
    task = graph.tasks[0]
    task = replace(
        task,
        resources=(replace(task.resources[0], partition="batch"),),
        parameters={"slurm": {"sbatch_kwargs": "--qos=interactive"}},
    )
    graph = replace(
        graph,
        stages=(replace(graph.stages[0], tasks=(task,)),),
    )

    with pytest.raises(ValueError, match="partition.*sbatch_kwargs"):
        _reject_ray_incompatible_controls(graph.stages)


def test_ray_preflight_rejects_runtime_package_installs() -> None:
    graph = _graph()
    task = graph.tasks[0]
    command = replace(
        task.commands[0], command="pip install datasketch\n" + task.commands[0].command
    )
    graph = replace(
        graph,
        stages=(replace(graph.stages[0], tasks=(replace(task, commands=(command,)),)),),
    )

    with pytest.raises(ValueError, match="runtime package installation"):
        _reject_ray_incompatible_controls(graph.stages)


def test_ray_status_lists_only_the_users_queued_and_running_jobs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    target = _target(tmp_path)
    ray_target = target.ray
    assert ray_target is not None
    listed = {
        "control": (
            RayJobDetails(
                "nvflow-run-controller-a1",
                RayJobState.RUNNING,
                metadata={"nvflow_kind": "controller", "nvflow_user": "alice"},
                start_time=1_000.0,
            ),
            RayJobDetails(
                "nvflow-old-controller-a1",
                RayJobState.SUCCEEDED,
                metadata={"nvflow_kind": "controller", "nvflow_user": "alice"},
            ),
            RayJobDetails(
                "nvflow-runtime-0123",
                RayJobState.RUNNING,
                metadata={"nvflow_kind": "runtime-probe"},
            ),
        ),
        "worker": (
            RayJobDetails(
                "nvflow-run-generate-4567",
                RayJobState.PENDING,
                metadata={"nvflow_task_name": "generate-server-0", "nvflow_user": "alice"},
                start_time=4_600.0,
            ),
            RayJobDetails(
                "nvflow-other-generate-89ab",
                RayJobState.RUNNING,
                metadata={"nvflow_task_name": "generate", "nvflow_user": "bob"},
            ),
        ),
    }

    class ListingClient:
        def __init__(self, cluster) -> None:
            self.cluster = cluster

        def jobs(self):
            return listed[self.cluster.name]

    @contextmanager
    def access(received):
        assert received is ray_target
        yield LauncherAccess(dict(ray_target.clusters), RecordingFiles())

    monkeypatch.setattr("nvflow.execution.engines.ray.launcher_access", access)
    monkeypatch.setattr("nvflow.execution.engines.ray.RayJobsClient", ListingClient)
    monkeypatch.setattr(
        "nvflow.execution.engines.ray.getpass", SimpleNamespace(getuser=lambda: "alice")
    )
    monkeypatch.setattr("nvflow.execution.engines.ray.time", SimpleNamespace(time=lambda: 4_661.0))

    listing = SuppliedRayJobsEngine().active_jobs(target)

    assert listing.jobs == (
        ActiveJob("nvflow-run-controller-a1", "controller", "RUNNING", "1:01:01", "control"),
        ActiveJob("nvflow-run-generate-4567", "generate-server-0", "PENDING", "1:01", "worker"),
    )
    assert listing.unreachable == {}


def test_ray_status_labels_jobs_and_lists_the_steps_a_run_has_not_submitted(
    tmp_path: Path,
    monkeypatch,
) -> None:
    target = _target(tmp_path)
    ray_target = target.ray
    assert ray_target is not None
    graph = _graph()
    submitted = graph.tasks[0]
    waiting = replace(
        submitted,
        task_id="generate.1",
        name="generate-1",
        resources=(ResourceRequest(nodes=2, gpus_per_node=8),),
        parameters={"experiment_name": "starter-generate-bench"},
    )
    blocked = replace(submitted, task_id="generate.2", name="generate-2")
    graph = replace(
        graph,
        stages=(replace(graph.stages[0], tasks=(submitted, waiting, blocked)),),
    )
    journal = RunJournal(ray_target.artifact_root, "nvflow-run")
    journal.initialize(graph.to_dict())
    journal.append(kind="run", state="running")
    journal.append(
        kind="task",
        state="submitted",
        task_id="generate.0",
        attempt=1,
        cluster="worker",
        submission_id="nvflow-run-generate-a1",
    )
    journal.append(kind="task", state="blocked", task_id="generate.2")
    workflow = {"nvflow_workflow": "starter:generate"}
    listed = {
        "control": (
            RayJobDetails(
                "nvflow-run-controller-a1",
                RayJobState.RUNNING,
                metadata={
                    "nvflow_kind": "controller",
                    "nvflow_user": "alice",
                    "nvflow_run_id": "nvflow-run",
                    **workflow,
                },
                start_time=4_600.0,
                driver_node="10.0.0.1",
            ),
        ),
        "worker": (
            RayJobDetails(
                "nvflow-run-generate-a1",
                RayJobState.PENDING,
                message=(
                    "Job has not started yet. It may be waiting for resources "
                    "(CPUs, GPUs, memory, custom resources) to become available."
                ),
                metadata={
                    "nvflow_user": "alice",
                    "nvflow_task_name": "generate",
                    **workflow,
                    "nvflow_stage": "generate",
                    "nvflow_experiment": "generate",
                    "nvflow_nodes": "1",
                },
                start_time=4_600.0,
            ),
        ),
    }

    class ListingClient:
        def __init__(self, cluster) -> None:
            self.cluster = cluster

        def jobs(self):
            return listed[self.cluster.name]

    @contextmanager
    def access(received):
        yield LauncherAccess(dict(received.clusters), LocalArtifactFiles())

    monkeypatch.setattr("nvflow.execution.engines.ray.launcher_access", access)
    monkeypatch.setattr("nvflow.execution.engines.ray.RayJobsClient", ListingClient)
    monkeypatch.setattr(
        "nvflow.execution.engines.ray.getpass", SimpleNamespace(getuser=lambda: "alice")
    )
    monkeypatch.setattr("nvflow.execution.engines.ray.time", SimpleNamespace(time=lambda: 4_661.0))

    listing = SuppliedRayJobsEngine().active_jobs(target)

    labeled: dict[str, Any] = {"workflow": "starter:generate", "stage": "generate"}
    assert listing.jobs == (
        ActiveJob(
            "nvflow-run-controller-a1",
            "controller",
            "RUNNING",
            "1:01",
            "control",
            workflow="starter:generate",
            reason="10.0.0.1",
        ),
        ActiveJob(
            "nvflow-run-generate-a1",
            "generate",
            "PENDING",
            "1:01",
            "worker",
            **labeled,
            experiment="generate",
            reason="Resources",
            nodes=1,
        ),
        ActiveJob(
            "-",
            "generate-1",
            "PENDING",
            "0:00",
            "worker",
            **labeled,
            experiment="starter-generate-bench",
            reason="Dependency",
            nodes=2,
            gpus_per_node=8,
        ),
        ActiveJob(
            "-",
            "generate-2",
            "PENDING",
            "0:00",
            "worker",
            **labeled,
            experiment="generate-2",
            reason="DependencyNeverSatisfied",
            nodes=1,
        ),
    )
    assert listing.unreachable == {}


def test_ray_status_lists_jobs_in_submission_order_across_clusters(
    tmp_path: Path,
    monkeypatch,
) -> None:
    target = _target(tmp_path)
    alice = {"nvflow_user": "alice"}
    listed = {
        "control": (
            RayJobDetails(
                "nvflow-run-prepare-a1",
                RayJobState.RUNNING,
                metadata={**alice, "nvflow_task_name": "prepare"},
                start_time=3_000.0,
            ),
            RayJobDetails(
                "nvflow-run-controller-a1",
                RayJobState.RUNNING,
                metadata={**alice, "nvflow_kind": "controller"},
                start_time=1_000.0,
            ),
        ),
        "worker": (
            RayJobDetails(
                "nvflow-run-generate-a1",
                RayJobState.PENDING,
                metadata={**alice, "nvflow_task_name": "generate"},
                start_time=2_000.0,
            ),
        ),
    }

    class ListingClient:
        def __init__(self, cluster) -> None:
            self.cluster = cluster

        def jobs(self):
            return listed[self.cluster.name]

    @contextmanager
    def access(received):
        yield LauncherAccess(dict(received.clusters), LocalArtifactFiles())

    monkeypatch.setattr("nvflow.execution.engines.ray.launcher_access", access)
    monkeypatch.setattr("nvflow.execution.engines.ray.RayJobsClient", ListingClient)
    monkeypatch.setattr(
        "nvflow.execution.engines.ray.getpass", SimpleNamespace(getuser=lambda: "alice")
    )

    listing = SuppliedRayJobsEngine().active_jobs(target)

    assert [job.name for job in listing.jobs] == ["controller", "generate", "prepare"]


@pytest.mark.parametrize(
    ("journal", "unreachable"),
    [
        (None, {}),
        ("not json", {"nvflow-run journal": "Expecting value: line 1 column 1 (char 0)"}),
    ],
)
def test_ray_status_skips_a_run_journal_it_cannot_read(
    tmp_path: Path,
    monkeypatch,
    journal: str | None,
    unreachable: dict[str, str],
) -> None:
    target = _target(tmp_path)
    ray_target = target.ray
    assert ray_target is not None
    if journal is not None:
        run_root = Path(ray_target.artifact_root) / ".nvflow" / "runs" / "nvflow-run"
        run_root.mkdir(parents=True)
        (run_root / "execution.json").write_text(journal)

    class ListingClient:
        def __init__(self, cluster) -> None:
            self.cluster = cluster

        def jobs(self):
            if self.cluster.name != "control":
                return ()
            return (
                RayJobDetails(
                    "nvflow-run-controller-a1",
                    RayJobState.RUNNING,
                    metadata={
                        "nvflow_kind": "controller",
                        "nvflow_user": "alice",
                        "nvflow_run_id": "nvflow-run",
                    },
                ),
            )

    @contextmanager
    def access(received):
        yield LauncherAccess(dict(received.clusters), LocalArtifactFiles())

    monkeypatch.setattr("nvflow.execution.engines.ray.launcher_access", access)
    monkeypatch.setattr("nvflow.execution.engines.ray.RayJobsClient", ListingClient)
    monkeypatch.setattr(
        "nvflow.execution.engines.ray.getpass", SimpleNamespace(getuser=lambda: "alice")
    )

    listing = SuppliedRayJobsEngine().active_jobs(target)

    assert [job.job_id for job in listing.jobs] == ["nvflow-run-controller-a1"]
    assert listing.unreachable == unreachable


def test_ray_run_labels_its_controller_with_the_workflow(tmp_path: Path, monkeypatch) -> None:
    target = _target(tmp_path)
    run_specs = _stub_submission(tmp_path, monkeypatch)

    SuppliedRayJobsEngine().submit(
        _graph(),
        replace(_context(target, tmp_path), workflow="starter:generate"),
    )

    assert run_specs[0]["workflow"] == "starter:generate"
    controller = FakeClient.instances["control"].submitted[0]
    assert controller["metadata"]["nvflow_workflow"] == "starter:generate"


class _HTTPError(OSError):
    """Like requests.HTTPError: an OSError that carries the error response."""

    def __init__(self) -> None:
        super().__init__("401 Client Error: Unauthorized")
        self.response = object()


@pytest.mark.parametrize(
    ("cause", "reason"),
    [
        (ConnectionError("refused"), "not answering at http://127.0.0.1:8266 (ConnectionError)"),
        (_HTTPError(), "list failed"),
        (RuntimeError("Request failed with status code 401"), "list failed"),
    ],
)
def test_ray_status_reports_an_unreachable_cluster_without_hiding_the_others(
    tmp_path: Path,
    monkeypatch,
    cause: Exception,
    reason: str,
) -> None:
    target = _target(tmp_path)
    ray_target = target.ray
    assert ray_target is not None

    class PartlyReachableClient:
        def __init__(self, cluster) -> None:
            self.cluster = cluster

        def jobs(self):
            if self.cluster.name == "worker":
                raise RayJobsError("list failed") from cause
            return (
                RayJobDetails(
                    "nvflow-run-controller-a1",
                    RayJobState.RUNNING,
                    metadata={"nvflow_kind": "controller", "nvflow_user": "alice"},
                    start_time=4_600.0,
                ),
            )

    @contextmanager
    def access(received):
        yield LauncherAccess(dict(received.clusters), RecordingFiles())

    monkeypatch.setattr("nvflow.execution.engines.ray.launcher_access", access)
    monkeypatch.setattr("nvflow.execution.engines.ray.RayJobsClient", PartlyReachableClient)
    monkeypatch.setattr(
        "nvflow.execution.engines.ray.getpass", SimpleNamespace(getuser=lambda: "alice")
    )
    monkeypatch.setattr("nvflow.execution.engines.ray.time", SimpleNamespace(time=lambda: 4_661.0))

    listing = SuppliedRayJobsEngine().active_jobs(target)

    assert listing.jobs == (
        ActiveJob("nvflow-run-controller-a1", "controller", "RUNNING", "1:01", "control"),
    )
    assert listing.unreachable == {"worker": reason}


def test_ray_run_needs_only_the_clusters_its_jobs_use(tmp_path: Path, monkeypatch) -> None:
    target = _target(tmp_path)
    ray_target = target.ray
    assert ray_target is not None
    idle = RegisteredCluster("idle", None, unset_variable="RAY_IDLE_JOBS_URL")
    target = replace(
        target,
        ray=replace(
            ray_target,
            clusters={**ray_target.clusters, "idle": idle},
            routes={**ray_target.routes, "vllm": "idle"},
        ),
    )
    run_specs = _stub_submission(tmp_path, monkeypatch)

    SuppliedRayJobsEngine().submit(_graph(), _context(target, tmp_path))

    assert set(FakeClient.instances) == {"control", "worker"}
    payload = run_specs[0]["ray_target"]
    assert set(payload["clusters"]) == {"control", "worker"}
    assert payload["routes"] == {"project": "worker"}


def test_ray_run_names_each_needed_cluster_it_cannot_use(tmp_path: Path, monkeypatch) -> None:
    target = _target(tmp_path)
    ray_target = target.ray
    assert ray_target is not None
    worker = RegisteredCluster("worker", None, unset_variable="RAY_WORKER_JOBS_URL")
    target = replace(
        target,
        ray=replace(ray_target, clusters={**ray_target.clusters, "worker": worker}),
    )
    _stub_submission(tmp_path, monkeypatch)

    class RefusedClient(FakeClient):
        def probe(self):
            raise RayJobsError("probe failed") from ConnectionError("refused")

    monkeypatch.setattr("nvflow.execution.engines.ray.RayJobsClient", RefusedClient)

    with pytest.raises(RayJobsError) as raised:
        SuppliedRayJobsEngine().submit(_graph(), _context(target, tmp_path))
    assert str(raised.value) == (
        "This run needs Ray clusters that the launcher cannot use:\n"
        "  control: not answering at http://127.0.0.1:8265 (ConnectionError)\n"
        "  worker: RAY_WORKER_JOBS_URL is not set"
    )
    assert set(FakeClient.instances) == {"control"}
    assert FakeClient.instances["control"].submitted == []


def test_ray_status_names_an_unconfigured_cluster(tmp_path: Path, monkeypatch) -> None:
    target = _target(tmp_path)
    ray_target = target.ray
    assert ray_target is not None
    idle = RegisteredCluster("idle", None, unset_variable="RAY_IDLE_JOBS_URL")
    target = replace(
        target,
        ray=replace(ray_target, clusters={**ray_target.clusters, "idle": idle}),
    )

    class EmptyClient:
        def __init__(self, cluster) -> None:
            self.cluster = cluster

        def jobs(self):
            return ()

    monkeypatch.setattr("nvflow.execution.engines.ray.RayJobsClient", EmptyClient)

    listing = SuppliedRayJobsEngine().active_jobs(target)

    assert listing.jobs == ()
    assert listing.unreachable == {"idle": "RAY_IDLE_JOBS_URL is not set"}


def _stub_submission(tmp_path: Path, monkeypatch) -> list[dict[str, Any]]:
    """Replace bundling and Jobs clients; return the run specs given to the controller."""
    worker_path = tmp_path / "worker.zip"
    worker_path.write_bytes(b"worker")
    controller_path = tmp_path / "controller.zip"
    controller_path.write_bytes(b"controller")
    run_specs: list[dict[str, Any]] = []

    def record_controller_bundle(_worker, run_spec, _destination):
        run_specs.append(run_spec)
        return ControllerBundle(path=controller_path, digest="sha256:c", compressed_size=10)

    monkeypatch.setattr(
        "nvflow.execution.engines.ray.build_worker_bundle",
        lambda *_args, **_kwargs: WorkerBundle(
            path=worker_path,
            digest="sha256:worker",
            compressed_size=6,
            uncompressed_size=6,
            project_python_root="projects/starter",
            project_identity="git:project",
            platform_identity="git:platform",
        ),
    )
    monkeypatch.setattr(
        "nvflow.execution.engines.ray.build_controller_bundle",
        record_controller_bundle,
    )
    monkeypatch.setattr("nvflow.execution.engines.ray.RayJobsClient", FakeClient)
    monkeypatch.setattr("nvflow.execution.engines.ray.validate_launcher_runtime", lambda: None)
    FakeClient.instances.clear()
    return run_specs


def _context(target: ResolvedTarget, tmp_path: Path) -> EngineContext:
    return EngineContext(
        target=target,
        project_context=cast(Any, SimpleNamespace()),
        platform_source="git:platform",
        cache_dir=tmp_path,
    )


def _target(tmp_path: Path) -> ResolvedTarget:
    control = RegisteredCluster("control", "http://127.0.0.1:8265")
    worker = RegisteredCluster("worker", "http://127.0.0.1:8266")
    binding = TargetBinding.resolve("ray")
    return ResolvedTarget(
        binding=binding,
        engine="ray",
        fingerprint="sha256:target",
        source_path=tmp_path / "ray.yaml",
        ray=RayTarget(
            control_cluster="control",
            artifact_root=str(tmp_path / "artifacts"),
            clusters={"control": control, "worker": worker},
            routes={"project": "worker"},
        ),
    )


def _graph() -> ExecutionGraph:
    implementation = {
        "kind": "project",
        "name": "my_project.stage:GenerateAnswerStage",
        "contract_version": None,
        "provider": "my_project",
        "platform": {
            "distribution": "nvflow",
            "version": "1.2.0",
            "source": "git:platform",
        },
    }
    task = TaskPlan(
        task_id="generate.0",
        stage="generate",
        name="generate",
        operation="shell",
        commands=(
            CommandPlan(
                "true",
                "project",
                runtime_role="project",
            ),
        ),
        resources=(ResourceRequest(),),
    )
    return ExecutionGraph(
        execution_version=1,
        project="starter",
        workflow="generate",
        target="ray",
        engine="ray",
        target_fingerprint="sha256:target",
        stages=(
            StageExecution(
                stage="generate",
                tasks=(task,),
                implementation=implementation,
            ),
        ),
        provenance=ProjectProvenance(
            commit="a" * 40,
            source_subpath="projects/starter",
            packaged_destination="/nemo_run/code/projects/starter",
            content_digest="sha256:project",
            clean=True,
            submittable=True,
        ),
    )
