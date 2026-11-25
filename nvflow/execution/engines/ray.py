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
"""Execution through operator-supplied Ray Jobs clusters."""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import secrets
import shlex
import time
from collections.abc import Collection, Mapping
from pathlib import PurePosixPath

from nvflow.execution.codec import execution_graph_from_dict
from nvflow.execution.contracts import ExecutionGraph, TaskPlan
from nvflow.execution.engines.base import ActiveJob, EngineContext, ExecutionEngine, JobListing
from nvflow.execution.job_labels import JobLabel, planned_resources, task_label
from nvflow.execution.receipts import SubmissionReceipt, discover_submission_receipt
from nvflow.execution.run_records import parse_events, run_events_path
from nvflow.execution.runtime_profile import validate_launcher_runtime
from nvflow.execution.target import (
    RayTarget,
    RegisteredCluster,
    ResolvedTarget,
    TargetResolutionError,
)
from nvflow.integrations.ray_jobs.bundle import build_controller_bundle
from nvflow.integrations.ray_jobs.client import (
    RayClusterProbe,
    RayJobDetails,
    RayJobsClient,
    RayJobsError,
    RayJobState,
)
from nvflow.integrations.ray_jobs.launcher_access import (
    ArtifactFiles,
    launcher_access,
    unreachable_reason,
)
from nvflow.integrations.ray_jobs.lowering import lower_execution_graph
from nvflow.project.packaging import build_worker_bundle


class SuppliedRayJobsEngine(ExecutionEngine):
    """Submit one durable controller without owning Ray infrastructure."""

    def submit(
        self,
        graph: ExecutionGraph,
        context: EngineContext,
    ) -> SubmissionReceipt:
        if graph.engine != "ray" or context.target.engine != "ray":
            raise ValueError("SuppliedRayJobsEngine requires a Ray execution graph and target")
        ray_target = context.target.ray
        if ray_target is None:  # defensive: ResolvedTarget already enforces this
            raise ValueError("Resolved Ray target is missing supplied clusters")
        if graph.provenance is None or not graph.provenance.submittable:
            raise ValueError("Ray execution requires clean, submittable project provenance")
        validate_launcher_runtime()
        lowered_graph = lower_execution_graph(graph)
        _validate_routes(lowered_graph, ray_target.routes)
        _reject_serialized_credentials(lowered_graph, ray_target)

        needed = _run_clusters(lowered_graph, ray_target)
        with launcher_access(ray_target, needed) as access:
            clients = {name: RayJobsClient(cluster) for name, cluster in access.clusters.items()}
            unavailable = dict(access.unavailable)
            for name, client in clients.items():
                try:
                    client.probe()
                except RayJobsError as error:
                    unavailable[name] = _unreachable(ray_target.clusters[name], access.via, error)
            if unavailable:
                raise RayJobsError(
                    "This run needs Ray clusters that the launcher cannot use:\n"
                    + "\n".join(
                        f"  {name}: {unavailable[name]}"
                        for name in ray_target.clusters
                        if name in unavailable
                    )
                )
            probe_path, probe_token = _artifact_probe(access.files, ray_target.artifact_root)
            probes = {}
            try:
                for name, client in clients.items():
                    probe_id = hashlib.sha256(
                        f"{context.target.fingerprint}\0{name}".encode()
                    ).hexdigest()[:20]
                    probes[name] = client.probe_runtime(
                        probe_id=probe_id,
                        artifact_path=probe_path,
                        artifact_token=probe_token,
                        files=access.files,
                    )
            finally:
                access.files.remove(probe_path)
            _validate_capacity(lowered_graph, ray_target, probes)

            worker = build_worker_bundle(
                context.project_context,
                graph.provenance,
                context.platform_source,
                context.cache_dir / "worker-bundles",
            )
            run_id = _run_id(lowered_graph)
            user = getpass.getuser()
            run_spec = {
                "run_id": run_id,
                "execution_digest": lowered_graph.digest,
                "source_execution_digest": graph.digest,
                "graph": lowered_graph.to_dict(),
                "ray_target": _target_payload(ray_target, needed),
                "project_python_root": worker.project_python_root,
                "user": user,
                "workflow": context.workflow or "",
            }
            controller_bundle = build_controller_bundle(
                worker,
                run_spec,
                context.cache_dir / "controller-bundles",
            )
            attempt = _controller_attempt(access.files, ray_target.artifact_root, run_id)
            control_client = clients[ray_target.control_cluster]
            controller_submission, existing = _resolve_controller_submission(
                control_client,
                run_id,
                attempt,
            )
            if existing is None:
                controller_environment = {
                    "NVFLOW_RUN_ID": run_id,
                    "NVFLOW_EXECUTION_DIGEST": lowered_graph.digest,
                }
                control_client.submit(
                    submission_id=controller_submission,
                    entrypoint=_shell_environment_entrypoint(
                        controller_environment,
                        [
                            "python",
                            "-m",
                            "nvflow.integrations.ray_jobs.controller",
                            "--spec",
                            ".nvflow/run.json",
                        ],
                    ),
                    working_dir=controller_bundle.path,
                    env_vars={},
                    metadata={
                        "nvflow_kind": "controller",
                        "nvflow_run_id": run_id,
                        "nvflow_user": user,
                        "nvflow_execution_digest": lowered_graph.digest,
                        "nvflow_source_execution_digest": graph.digest,
                        "nvflow_bundle_digest": worker.digest,
                        **JobLabel(workflow=context.workflow or "").to_metadata(),
                    },
                    entrypoint_num_cpus=1,
                )

        return discover_submission_receipt(
            graph.to_plan(planning_mode="static"),
            engine="ray",
            execution_digest=lowered_graph.digest,
            target_fingerprint=context.target.fingerprint,
            controller_handle=(f"rayjobs://{ray_target.control_cluster}/{controller_submission}"),
        )

    def active_jobs(self, target: ResolvedTarget) -> JobListing:
        ray_target = target.ray
        if ray_target is None:  # defensive: ResolvedTarget already enforces this
            raise ValueError("Resolved Ray target is missing supplied clusters")
        # Ray records no job owner, so shared clusters are filtered by the NVFlow tag.
        user = getpass.getuser()
        submitted: list[tuple[RayJobDetails, str]] = []
        unreachable = {}
        with launcher_access(ray_target) as access:
            for name in ray_target.clusters:
                if name in access.unavailable:
                    unreachable[name] = access.unavailable[name]
                    continue
                try:
                    listed = RayJobsClient(access.clusters[name]).jobs()
                except RayJobsError as error:
                    # A stopped or restarting cluster must not hide the other clusters' jobs.
                    unreachable[name] = _unreachable(ray_target.clusters[name], access.via, error)
                    continue
                submitted.extend(
                    (job, name)
                    for job in listed
                    if not job.state.terminal and job.metadata.get("nvflow_user") == user
                )
            # Ray's start_time is when the Job was submitted.
            submitted.sort(
                key=lambda item: (
                    item[0].start_time is None,
                    item[0].start_time or 0.0,
                    item[0].submission_id,
                )
            )
            jobs = [_active_job(job, name) for job, name in submitted]
            controllers = [
                job for job, _ in submitted if job.metadata.get("nvflow_kind") == "controller"
            ]
            # The controller submits a step only once its dependencies finish, so the
            # steps still waiting exist only in the run's journal.
            for controller in controllers:
                run_id = controller.metadata.get("nvflow_run_id")
                if not run_id:
                    continue
                try:
                    jobs.extend(_waiting_steps(controller, run_id, ray_target, access.files))
                except FileNotFoundError:
                    continue
                except (OSError, ValueError, KeyError, TypeError) as error:
                    unreachable[f"{run_id} journal"] = str(error)
        return JobListing(tuple(jobs), unreachable)


def _active_job(job: RayJobDetails, location: str) -> ActiveJob:
    label = JobLabel.from_metadata(job.metadata) or JobLabel()
    running = job.state is RayJobState.RUNNING
    return ActiveJob(
        job_id=job.submission_id,
        name=job.metadata.get("nvflow_task_name") or job.metadata.get("nvflow_kind", ""),
        state=job.state.value.upper(),
        elapsed=_elapsed(job.start_time),
        location=location,
        workflow=label.workflow or None,
        stage=label.stage or None,
        experiment=label.experiment or None,
        reason=job.driver_node if running else _pending_reason(job.message),
        nodes=_metadata_count(job.metadata, "nvflow_nodes"),
        gpus_per_node=_metadata_count(job.metadata, "nvflow_gpus_per_node"),
    )


def _waiting_steps(
    controller: RayJobDetails,
    run_id: str,
    target: RayTarget,
    files: ArtifactFiles,
) -> list[ActiveJob]:
    """Return the steps of a controller's run that it has not submitted yet."""
    events_path = run_events_path(target.artifact_root, run_id)
    graph_path = str(events_path.with_name("execution.json"))
    graph = execution_graph_from_dict(json.loads(files.read_text(graph_path)))
    events = parse_events(files.read_text(str(events_path)), str(events_path))
    task_states = {
        event.task_id: event.state
        for event in events
        if event.kind == "task" and event.task_id is not None
    }
    submitted = {
        event.task_id for event in events if event.kind == "task" and event.state == "submitted"
    }
    workflow = controller.metadata.get("nvflow_workflow") or None
    steps = []
    for task in graph.tasks:
        if task.task_id in submitted:
            continue
        label = task_label(workflow, task)
        nodes, gpus_per_node = planned_resources(task)
        steps.append(
            ActiveJob(
                job_id="-",
                name=task.name,
                state="PENDING",
                elapsed="0:00",
                location=_task_location(target, task),
                workflow=label.workflow or None,
                stage=label.stage or None,
                experiment=label.experiment or None,
                # Slurm's reasons for a job waiting on, or blocked by, its dependencies.
                reason=(
                    "DependencyNeverSatisfied"
                    if task_states.get(task.task_id) == "blocked"
                    else "Dependency"
                ),
                nodes=nodes,
                gpus_per_node=gpus_per_node,
            )
        )
    return steps


def _pending_reason(message: str | None) -> str | None:
    """Shorten Ray's pending message to the Slurm-style reason it gives."""
    if not message:
        return None
    if "waiting for resources" in message:
        return "Resources"
    if "runtime environment" in message:
        return "RuntimeEnv"
    return message


def _metadata_count(metadata: Mapping[str, str], key: str) -> int | None:
    value = metadata.get(key, "")
    return int(value) if value.isdigit() else None


def _task_location(target: RayTarget, task: TaskPlan) -> str:
    roles = {command.require_runtime_role() for command in task.commands}
    if len(roles) != 1:
        return "-"
    try:
        return target.cluster_for_role(next(iter(roles))).name
    except (KeyError, TargetResolutionError):
        return "-"


def _elapsed(started: float | None) -> str:
    """Format elapsed time the way squeue prints it: M:SS, H:MM:SS, or D-HH:MM:SS."""
    seconds = 0 if started is None else max(int(time.time() - started), 0)
    days, seconds = divmod(seconds, 24 * 60 * 60)
    hours, seconds = divmod(seconds, 60 * 60)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return f"{days}-{hours:02}:{minutes:02}:{seconds:02}"
    if hours:
        return f"{hours}:{minutes:02}:{seconds:02}"
    return f"{minutes}:{seconds:02}"


def _validate_routes(
    graph: ExecutionGraph,
    routes: Mapping[str, str],
) -> None:
    route_names = set(routes)
    missing = {}
    for task in graph.tasks:
        roles = {command.require_runtime_role() for command in task.commands}
        unresolved = roles - route_names
        if unresolved:
            missing[task.task_id] = sorted(unresolved)
    if missing:
        details = "; ".join(
            f"{task}: {', '.join(roles)}" for task, roles in sorted(missing.items())
        )
        raise ValueError(f"Supplied Ray target has no route for task runtime roles: {details}")


def _unreachable(cluster: RegisteredCluster, via: str | None, error: RayJobsError) -> str:
    """Name the address when no HTTP response came back; other failures keep their message."""
    cause = error.__cause__
    # requests' exceptions, HTTP errors included, subclass OSError; only an HTTP
    # error carries the response.
    if isinstance(cause, OSError) and getattr(cause, "response", None) is None:
        return unreachable_reason(cluster, via, type(cause).__name__)
    return str(error)


def _run_clusters(graph: ExecutionGraph, target: RayTarget) -> set[str]:
    """Clusters the run's jobs are routed to, plus the one hosting its controller."""
    names = {target.control_cluster}
    for task in graph.tasks:
        for command in task.commands:
            names.add(target.cluster_for_role(command.require_runtime_role()).name)
    return names


def _target_payload(ray_target: RayTarget, names: Collection[str]) -> dict:
    clusters = {}
    for name, cluster in ray_target.clusters.items():
        if name not in names:
            continue
        clusters[name] = {
            "jobs_url": cluster.controller_jobs_url,
            "controller_jobs_url": cluster.controller_jobs_url,
            "credential_env": cluster.controller_credential_env,
            "controller_credential_env": cluster.controller_credential_env,
            "ca_file": cluster.controller_ca_file,
            "controller_ca_file": cluster.controller_ca_file,
            "allow_insecure": cluster.controller_allow_insecure,
            "controller_allow_insecure": cluster.controller_allow_insecure,
            "expected_image": cluster.expected_image,
            "expected_python": cluster.expected_python,
            "expected_ray": cluster.expected_ray,
            "gpus_per_node": cluster.gpus_per_node,
            "labels": dict(cluster.labels),
            "resources": dict(cluster.resources),
        }
    return {
        "control_cluster": ray_target.control_cluster,
        "artifact_root": ray_target.artifact_root,
        "clusters": clusters,
        "routes": {role: name for role, name in ray_target.routes.items() if name in clusters},
    }


def _validate_capacity(
    graph: ExecutionGraph,
    target: RayTarget,
    probes: Mapping[str, RayClusterProbe],
) -> None:
    for name, probe in probes.items():
        cluster = target.clusters[name]
        actual = probe.resources
        actual_labels = probe.labels
        for label, expected in cluster.labels.items():
            if actual_labels.get(label) != expected:
                raise ValueError(
                    f"Supplied Ray cluster {name!r} requires label "
                    f"{label}={expected!r}, reports {actual_labels.get(label)!r}"
                )
        for resource, expected in cluster.resources.items():
            if float(actual.get(resource, 0)) < float(expected):
                raise ValueError(
                    f"Supplied Ray cluster {name!r} declares {expected} {resource} "
                    f"but reports {actual.get(resource, 0)}"
                )
    for task in graph.tasks:
        roles = {command.require_runtime_role() for command in task.commands}
        if len(roles) != 1:
            raise ValueError(f"Lowered Ray task {task.task_id!r} has multiple roles")
        cluster = target.cluster_for_role(next(iter(roles)))
        available = probes[cluster.name].resources
        required_gpu = sum(
            resource.nodes * (resource.gpus_per_node or 0) for resource in task.resources
        )
        ray_application = bool(task.parameters.get("ray_application")) or task.operation.startswith(
            "nemo_rl."
        )
        required_cpu = (
            0
            if ray_application
            else sum(resource.nodes * resource.tasks_per_node for resource in task.resources)
        )
        if required_gpu > float(available.get("GPU", 0)):
            raise ValueError(
                f"Ray task {task.task_id!r} requires {required_gpu} GPUs but supplied "
                f"cluster {cluster.name!r} reports {available.get('GPU', 0)}"
            )
        if required_cpu > float(available.get("CPU", 0)):
            raise ValueError(
                f"Ray task {task.task_id!r} requires {required_cpu} CPUs but supplied "
                f"cluster {cluster.name!r} reports {available.get('CPU', 0)}"
            )


def _run_id(graph: ExecutionGraph) -> str:
    digest = hashlib.sha256(f"{graph.digest}\0{graph.target_fingerprint}".encode()).hexdigest()[:20]
    return f"nvflow-{digest}"


def _artifact_probe(files: ArtifactFiles, artifact_root: str) -> tuple[str, str]:
    root = PurePosixPath(artifact_root)
    if not root.is_absolute():
        raise ValueError("Ray artifact_root must be absolute")
    token = secrets.token_hex(32)
    name = f"launcher-{hashlib.sha256(token.encode()).hexdigest()[:20]}.txt"
    path = str(root / ".nvflow" / "probes" / name)
    files.write_text(path, token + "\n")
    return path, token


def _reject_serialized_credentials(graph: ExecutionGraph, target: RayTarget) -> None:
    serialized = json.dumps(graph.to_dict(), sort_keys=True, default=str)
    for cluster in target.clusters.values():
        for credential_env in {
            cluster.credential_env,
            cluster.controller_credential_env,
        }:
            if credential_env is None:
                continue
            value = os.environ.get(credential_env)
            if value and value in serialized:
                raise ValueError(
                    f"Execution graph contains secret value from {credential_env}; "
                    "use a secret reference"
                )


def _controller_attempt(files: ArtifactFiles, artifact_root: str, run_id: str) -> int:
    path = str(run_events_path(artifact_root, run_id))
    try:
        events = parse_events(files.read_text(path), path)
    except FileNotFoundError:
        events = ()
    run_events = [event for event in events if event.kind == "run"]
    started = sum(event.state == "running" for event in run_events)
    if not run_events:
        return 1
    if run_events[-1].state == "running":
        return max(started, 1)
    if run_events[-1].state == "succeeded":
        return max(started, 1)
    return started + 1


def _resolve_controller_submission(
    client: RayJobsClient,
    run_id: str,
    attempt: int,
) -> tuple[str, RayJobDetails | None]:
    while True:
        submission_id = f"{run_id}-controller-a{attempt}"
        existing = client.find(submission_id)
        if existing is None or existing.state not in {
            RayJobState.FAILED,
            RayJobState.STOPPED,
        }:
            return submission_id, existing
        attempt += 1


def _shell_environment_entrypoint(
    environment: Mapping[str, str],
    command: list[str],
) -> str:
    return shlex.join(
        [
            "env",
            *(f"{name}={value}" for name, value in sorted(environment.items())),
            *command,
        ]
    )


__all__ = ["SuppliedRayJobsEngine"]
