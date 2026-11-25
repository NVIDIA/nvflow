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
"""Durable dependency controller running on a supplied Ray control cluster."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import shlex
import signal
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from nvflow.execution.codec import execution_graph_from_dict
from nvflow.execution.contracts import ExecutionGraph, TaskPlan
from nvflow.execution.job_labels import planned_resources, task_label
from nvflow.execution.run_records import RunJournal
from nvflow.execution.target import RayTarget, RegisteredCluster
from nvflow.integrations.ray_jobs.bundle import RUN_SPEC_PATH, WORKER_BUNDLE_PATH
from nvflow.integrations.ray_jobs.client import RayJobsClient, RayJobsError, RayJobState

_IDENTITY_PART = re.compile(r"[^A-Za-z0-9_.-]+")


def run_controller(spec_path: Path = Path(RUN_SPEC_PATH)) -> None:
    """Execute one supplied-cluster graph to terminal state."""
    specification = _load_spec(spec_path)
    graph = execution_graph_from_dict(specification["graph"])
    if graph.digest != specification.get("execution_digest"):
        raise ValueError("Controller execution graph digest does not match its envelope")
    ray_target = _ray_target(specification["ray_target"])
    run_id = str(specification["run_id"])
    poll_seconds = float(specification.get("poll_seconds", 2.0))
    timeout_seconds = float(specification.get("timeout_seconds", 7 * 24 * 60 * 60))
    readiness_timeout_seconds = float(specification.get("readiness_timeout_seconds", 30 * 60))
    project_python_root = str(specification["project_python_root"])
    user = str(specification.get("user") or "")
    workflow = str(specification.get("workflow") or "")
    bundle_path = (Path.cwd() / WORKER_BUNDLE_PATH).resolve()
    if not bundle_path.is_file():
        raise ValueError(f"Controller worker bundle is missing: {bundle_path}")

    journal = RunJournal(ray_target.artifact_root, run_id)
    journal.initialize(graph.to_dict())
    journal.append(kind="run", state="running")
    clients = {name: RayJobsClient(cluster) for name, cluster in ray_target.clusters.items()}
    submitted: dict[str, tuple[str, str]] = {}
    states: dict[str, RayJobState] = {}
    blocked: set[str] = set()
    ready_services: set[str] = set()
    readiness_started: dict[str, float] = {}
    service_consumers = _service_consumers(graph)
    attempts = _recorded_attempts(journal)
    started = time.monotonic()

    try:
        while True:
            if time.monotonic() - started > timeout_seconds:
                raise TimeoutError(f"Run {run_id!r} exceeded controller timeout")
            _refresh(submitted, states, clients, journal, attempts)
            _refresh_readiness(
                graph,
                states,
                ready_services,
                readiness_started,
                readiness_timeout_seconds,
                journal,
            )
            _stop_finished_services(
                submitted,
                states,
                clients,
                service_consumers,
                journal,
            )
            progress = False
            for task in graph.tasks:
                if task.task_id in submitted or task.task_id in blocked:
                    continue
                decision = _dependency_decision(
                    task,
                    states,
                    blocked,
                    ready_services,
                )
                if decision == "blocked":
                    blocked.add(task.task_id)
                    journal.append(
                        kind="task",
                        state="blocked",
                        task_id=task.task_id,
                        message="an after_success dependency did not succeed",
                    )
                    progress = True
                    continue
                if decision != "ready":
                    continue
                cluster = _cluster_for_task(ray_target, task)
                client = clients[cluster.name]
                attempt = max(attempts.get(task.task_id, 1), 1)
                submission_id = _submission_id(run_id, task.task_id, attempt)
                existing = client.find(submission_id)
                if existing is not None and existing.state in {
                    RayJobState.FAILED,
                    RayJobState.STOPPED,
                }:
                    attempt += 1
                    submission_id = _submission_id(run_id, task.task_id, attempt)
                    existing = client.find(submission_id)
                if existing is None:
                    _clear_service_readiness(task)
                    _submit_task(
                        client,
                        graph,
                        task,
                        cluster,
                        submission_id,
                        bundle_path,
                        project_python_root,
                        run_id,
                        _log_path(task, submission_id, ray_target.artifact_root, run_id),
                        user,
                        workflow,
                    )
                    state = RayJobState.PENDING
                else:
                    state = existing.state
                submitted[task.task_id] = (cluster.name, submission_id)
                states[task.task_id] = state
                attempts[task.task_id] = attempt
                journal.append(
                    kind="task",
                    state="submitted",
                    task_id=task.task_id,
                    attempt=attempt,
                    cluster=cluster.name,
                    submission_id=submission_id,
                )
                progress = True

            if _complete(graph, states, blocked, service_consumers):
                failures = [
                    task_id
                    for task_id, state in states.items()
                    if state in {RayJobState.FAILED, RayJobState.STOPPED}
                    and task_id not in service_consumers
                ]
                final_state = "failed" if failures or blocked else "succeeded"
                journal.append(
                    kind="run",
                    state=final_state,
                    message=(
                        None
                        if final_state == "succeeded"
                        else "failed or blocked tasks: " + ", ".join(sorted([*failures, *blocked]))
                    ),
                )
                if final_state != "succeeded":
                    raise RuntimeError(f"Run {run_id!r} failed")
                return
            if not progress:
                time.sleep(poll_seconds)
    except BaseException as exc:
        _stop_all(submitted, states, clients, journal)
        journal.append(
            kind="run",
            state=("cancelled" if isinstance(exc, KeyboardInterrupt) else "failed"),
            message=f"{type(exc).__name__}: {exc}",
        )
        raise


def _load_spec(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid Ray controller spec: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError("Ray controller spec must be a mapping")
    return value


def _ray_target(value: object) -> RayTarget:
    if not isinstance(value, dict):
        raise ValueError("ray_target must be a mapping")
    clusters_value = value.get("clusters")
    routes = value.get("routes")
    if not isinstance(clusters_value, dict) or not isinstance(routes, dict):
        raise ValueError("ray_target clusters/routes must be mappings")
    clusters = {}
    for name, cluster in clusters_value.items():
        if not isinstance(name, str) or not isinstance(cluster, dict):
            raise ValueError("ray_target cluster entries must be mappings")
        clusters[name] = RegisteredCluster(name=name, **cluster)
    return RayTarget(
        control_cluster=str(value["control_cluster"]),
        artifact_root=str(value["artifact_root"]),
        clusters=clusters,
        routes={str(role): str(cluster) for role, cluster in routes.items()},
    )


def _cluster_for_task(target: RayTarget, task: TaskPlan) -> RegisteredCluster:
    roles = {command.require_runtime_role() for command in task.commands}
    if len(roles) != 1:
        raise ValueError(
            f"Ray task {task.task_id!r} requires exactly one runtime role, got "
            + ", ".join(sorted(roles))
        )
    return target.cluster_for_role(next(iter(roles)))


def _submit_task(
    client: RayJobsClient,
    graph: ExecutionGraph,
    task: TaskPlan,
    cluster: RegisteredCluster,
    submission_id: str,
    bundle_path: Path,
    project_python_root: str,
    run_id: str,
    log_path: str,
    user: str,
    workflow: str = "",
) -> None:
    payload = {
        "task_id": task.task_id,
        "commands": [asdict(command) for command in task.commands],
        "project_python_root": project_python_root,
        "ray_application": _is_ray_application(task),
        "log_path": log_path,
    }
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).decode()
    # Linux limits one argv entry to 128 KiB even when ARG_MAX is larger.
    # Leave room for shell quoting and the rest of the Jobs entrypoint.
    if len(encoded) > 96 * 1024:
        raise ValueError(f"Ray task payload exceeds the safe 96 KiB argument limit: {task.task_id}")
    task_environment = {
        "NVFLOW_RUN_ID": run_id,
        "NVFLOW_TASK_ID": task.task_id,
        "NVFLOW_EXPECTED_CLUSTER": cluster.name,
    }
    entrypoint = shlex.join(
        [
            "env",
            *(f"{name}={value}" for name, value in sorted(task_environment.items())),
            "python",
            "-m",
            "nvflow.integrations.ray_jobs.worker",
            "--payload",
            encoded,
        ]
    )
    cpu, gpu, custom = _entrypoint_resources(task)
    nodes, gpus_per_node = planned_resources(task)
    try:
        client.submit(
            submission_id=submission_id,
            entrypoint=entrypoint,
            working_dir=bundle_path,
            env_vars={},
            metadata={
                "nvflow_run_id": run_id,
                "nvflow_task_id": task.task_id,
                "nvflow_task_name": task.name,
                "nvflow_user": user,
                "nvflow_execution_digest": graph.digest,
                **task_label(workflow, task).to_metadata(),
                **({} if nodes is None else {"nvflow_nodes": str(nodes)}),
                **({} if gpus_per_node is None else {"nvflow_gpus_per_node": str(gpus_per_node)}),
            },
            entrypoint_num_cpus=cpu,
            entrypoint_num_gpus=gpu,
            entrypoint_resources=custom or None,
        )
    except RayJobsError:
        existing = client.find(submission_id)
        if existing is None:
            raise


def _is_ray_application(task: TaskPlan) -> bool:
    return bool(task.parameters.get("ray_application")) or task.operation.startswith("nemo_rl.")


def _log_path(task: TaskPlan, submission_id: str, artifact_root: str, run_id: str) -> str:
    """Return where Slurm would log this task, named with the Ray Job's ID.

    Tasks that declare no log directory log under the run's records instead.
    """
    command = task.commands[0]
    parameters = task.parameters
    directory = (
        (parameters.get("group_log_dirs") or {}).get(command.group)
        or (parameters.get("slurm") or {}).get("log_dir")
        or parameters.get("log_dir")
        or f"{artifact_root.rstrip('/')}/.nvflow/runs/{run_id}/logs"
    )
    job_id = submission_id.rsplit("-", 1)[-1]
    task_name = parameters.get("source_task_name", task.name)
    if command.kind == "service":
        name = f"server_{task_name}_{job_id}_ray.log"
    elif _is_ray_application(task):
        name = f"ray-{job_id}-job.log"
    else:
        name = f"main_{task_name}_{job_id}_ray.log"
    return f"{str(directory).rstrip('/')}/{name}"


def _entrypoint_resources(
    task: TaskPlan,
) -> tuple[float | None, float | None, dict[str, float]]:
    if _is_ray_application(task):
        return (1.0, None, {})
    if not task.resources:
        return (None, None, {})
    if len(task.resources) != 1:
        raise ValueError(
            f"Ray task {task.task_id!r} requires one resource group or ray_application=true"
        )
    resource = task.resources[0]
    if resource.nodes != 1:
        raise ValueError(
            f"Ray task {task.task_id!r} requests multiple nodes without ray_application=true"
        )
    return (
        float(max(resource.tasks_per_node, 1)),
        None if resource.gpus_per_node is None else float(resource.gpus_per_node),
        {},
    )


def _dependency_decision(
    task: TaskPlan,
    states: dict[str, RayJobState],
    blocked: set[str],
    ready_services: set[str],
) -> str:
    for dependency in task.dependencies:
        if dependency.external:
            continue
        if dependency.upstream in blocked:
            if dependency.condition == "after_any":
                continue
            return "blocked"
        state = states.get(dependency.upstream)
        if state is None:
            return "wait"
        if dependency.condition == "after_success":
            if state is RayJobState.SUCCEEDED:
                continue
            if state.terminal:
                return "blocked"
            return "wait"
        if dependency.condition == "after_ready":
            # A service that exits cleanly before it is ready found its work already done.
            if dependency.upstream in ready_services or state is RayJobState.SUCCEEDED:
                continue
            if state.terminal:
                return "blocked"
            return "wait"
        if dependency.condition == "after_start":
            if state is not RayJobState.PENDING and state is not RayJobState.UNKNOWN:
                continue
            return "wait"
        if dependency.condition == "after_any":
            if state.terminal:
                continue
            return "wait"
    return "ready"


def _refresh_readiness(
    graph: ExecutionGraph,
    states: dict[str, RayJobState],
    ready_services: set[str],
    readiness_started: dict[str, float],
    timeout_seconds: float,
    journal: RunJournal,
) -> None:
    for task in graph.tasks:
        if task.task_id in ready_services:
            continue
        state = states.get(task.task_id)
        if state not in {RayJobState.RUNNING, RayJobState.SUCCEEDED}:
            continue
        if not task.services:
            continue
        if not all(_readiness_satisfied(service.readiness) for service in task.services):
            started = readiness_started.setdefault(task.task_id, time.monotonic())
            if time.monotonic() - started > timeout_seconds:
                raise TimeoutError(f"Service task {task.task_id!r} exceeded readiness timeout")
            continue
        ready_services.add(task.task_id)
        journal.append(
            kind="service",
            state="ready",
            task_id=task.task_id,
        )


def _readiness_satisfied(readiness: str) -> bool:
    if readiness == "after_start":
        return True
    if readiness.startswith("files:"):
        paths = [Path(path) for path in readiness.removeprefix("files:").split(",") if path]
        return bool(paths) and all(path.is_file() and path.stat().st_size > 0 for path in paths)
    if readiness.startswith("http://") or readiness.startswith("https://"):
        import urllib.request

        try:
            with urllib.request.urlopen(readiness, timeout=5) as response:
                return 200 <= response.status < 300
        except OSError:
            return False
    raise ValueError(f"Unsupported service readiness contract: {readiness!r}")


def _clear_service_readiness(task: TaskPlan) -> None:
    for service in task.services:
        if not service.readiness.startswith("files:"):
            continue
        for value in service.readiness.removeprefix("files:").split(","):
            if value:
                Path(value).unlink(missing_ok=True)


def _refresh(
    submitted: dict[str, tuple[str, str]],
    states: dict[str, RayJobState],
    clients: dict[str, RayJobsClient],
    journal: RunJournal,
    attempts: dict[str, int],
) -> None:
    for task_id, (cluster, submission_id) in submitted.items():
        previous = states.get(task_id)
        if previous is not None and previous.terminal:
            continue
        details = clients[cluster].status(submission_id)
        if details.state is previous:
            continue
        states[task_id] = details.state
        journal.append(
            kind="task",
            state=details.state.value,
            task_id=task_id,
            attempt=attempts.get(task_id, 1),
            cluster=cluster,
            submission_id=submission_id,
            message=details.message,
        )


def _service_consumers(graph: ExecutionGraph) -> dict[str, tuple[str | None, str]]:
    services: dict[str, tuple[str | None, str]] = {}
    for task in graph.tasks:
        for service in task.services:
            services[service.task_id] = (service.consumer_task_id, service.cleanup)
    return services


def _stop_finished_services(
    submitted: dict[str, tuple[str, str]],
    states: dict[str, RayJobState],
    clients: dict[str, RayJobsClient],
    services: dict[str, tuple[str | None, str]],
    journal: RunJournal,
) -> None:
    for task_id, (consumer, cleanup) in services.items():
        if task_id not in submitted or states.get(task_id, RayJobState.UNKNOWN).terminal:
            continue
        consumer_state = states.get(consumer) if consumer is not None else None
        should_stop = (
            consumer_state is not None
            and consumer_state.terminal
            and (
                cleanup == "always"
                or (cleanup == "on_success" and consumer_state is RayJobState.SUCCEEDED)
                or (cleanup == "on_failure" and consumer_state is not RayJobState.SUCCEEDED)
            )
        )
        if not should_stop:
            continue
        cluster, submission_id = submitted[task_id]
        clients[cluster].stop(submission_id)
        states[task_id] = RayJobState.STOPPED
        journal.append(
            kind="task",
            state="stopped",
            task_id=task_id,
            cluster=cluster,
            submission_id=submission_id,
            message="service cleanup policy",
        )


def _complete(
    graph: ExecutionGraph,
    states: dict[str, RayJobState],
    blocked: set[str],
    services: dict[str, tuple[str | None, str]],
) -> bool:
    for task in graph.tasks:
        if task.task_id in blocked:
            continue
        state = states.get(task.task_id)
        if state is None:
            return False
        if task.task_id in services:
            if not state.terminal:
                return False
        elif not state.terminal:
            return False
    return True


def _stop_all(
    submitted: dict[str, tuple[str, str]],
    states: dict[str, RayJobState],
    clients: dict[str, RayJobsClient],
    journal: RunJournal,
) -> None:
    for task_id, (cluster, submission_id) in reversed(tuple(submitted.items())):
        if states.get(task_id, RayJobState.UNKNOWN).terminal:
            continue
        try:
            clients[cluster].stop(submission_id)
        except RayJobsError as exc:
            journal.append(
                kind="task",
                state="cleanup_unknown",
                task_id=task_id,
                cluster=cluster,
                submission_id=submission_id,
                message=str(exc),
            )


def _submission_id(run_id: str, task_id: str, attempt: int = 1) -> str:
    readable = _IDENTITY_PART.sub("-", task_id).strip("-")[:48] or "task"
    digest = hashlib.sha256(f"{run_id}\0{task_id}\0attempt={attempt}".encode()).hexdigest()[:16]
    return f"{run_id}-{readable}-{digest}"[:128]


def _recorded_attempts(journal: RunJournal) -> dict[str, int]:
    attempts: dict[str, int] = {}
    for event in journal.events():
        if event.kind != "task" or event.task_id is None or event.attempt is None:
            continue
        attempts[event.task_id] = max(attempts.get(event.task_id, 0), event.attempt)
    return attempts


def main() -> None:
    def cancel(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt("NVFlow controller cancellation requested")

    signal.signal(signal.SIGINT, cancel)
    signal.signal(signal.SIGTERM, cancel)
    parser = argparse.ArgumentParser(description="Coordinate one NVFlow Ray workflow")
    parser.add_argument("--spec", type=Path, default=Path(RUN_SPEC_PATH))
    arguments = parser.parse_args()
    run_controller(arguments.spec)


if __name__ == "__main__":
    main()
