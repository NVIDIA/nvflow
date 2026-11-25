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
"""Start, check, and stop Ray-on-Slurm test clusters declared in a Ray target.

TEST INFRASTRUCTURE ONLY. NVFlow never invokes this script; it uses the
clusters as ordinary supplied Ray Jobs endpoints.

Each cluster runs in one Slurm allocation, from the Slurm target's image for
its routed role and with that target's account, partitions, mounts, and
env_vars. As in a Slurm submission, NAME=value entries are set directly and
bare NAME entries come from this shell. These test Jobs endpoints have no
authentication, so credential-like variables are withheld unless start is
given --pass-credentials. Once every started cluster answers from this host,
start writes endpoints.env, which sets the jobs_url variables that the Ray
target reads.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

HARNESS = Path(__file__).resolve().parent
ACKNOWLEDGEMENT = "I_UNDERSTAND_RAY_JOBS_HAS_NO_AUTH"
ENV_REFERENCE = re.compile(r"\$\{oc\.env:([A-Za-z_][A-Za-z0-9_]*)\}")
# Keep in sync with credential_unset in start_cluster.slurm.
CREDENTIAL_NAME = re.compile(r"(TOKEN|API_KEY|SECRET|PASSWORD|CREDENTIAL)$")
# The NeMo-RL and NeMo-Gym images keep Ray in these virtual environments.
RAY_BIN_DIRS = {"nemo-rl": "/opt/nemo_rl_venv/bin", "nemo-gym": "/opt/nemo_gym_venv/bin"}


class HarnessError(Exception):
    """A configuration or cluster problem reported without a traceback."""


@dataclass(frozen=True)
class Cluster:
    name: str
    roles: tuple[str, ...]
    jobs_url_variable: str
    controller_url_variable: str | None
    expected_image: str
    expected_ray: str | None
    gpus_per_node: int


@dataclass(frozen=True)
class RayTarget:
    artifact_root: str
    clusters: dict[str, Cluster]
    state_dir: Path


@dataclass(frozen=True)
class SlurmSettings:
    account: str
    gpu_partition: str
    cpu_partition: str
    containers: dict[str, object]
    mounts: tuple[str, ...]
    env_vars: tuple[str, ...] = ()


def _mapping(path: Path) -> dict[str, Any]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise HarnessError(f"cannot read {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise HarnessError(f"{path} must be a YAML mapping")
    return document


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HarnessError(f"{field} must be a non-empty string")
    if "<" in value or "${" in value:
        raise HarnessError(f"{field} still holds a placeholder: {value!r}")
    return value.strip()


def _variable(value: object, field: str) -> str:
    match = ENV_REFERENCE.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        raise HarnessError(f"{field} must be ${{oc.env:NAME}} so start can publish the endpoint")
    return match.group(1)


def load_ray_target(path: Path) -> RayTarget:
    document = _mapping(path)
    ray = document.get("ray")
    if document.get("executor") != "ray" or not isinstance(ray, dict):
        raise HarnessError(f"{path} is not an executor: ray target")
    artifact_root = _text(ray.get("artifact_root"), "ray.artifact_root")
    if not artifact_root.startswith("/"):
        raise HarnessError("ray.artifact_root must be an absolute path")
    routes = ray.get("routes") or {}
    declared = ray.get("clusters") or {}
    if not isinstance(routes, dict) or not isinstance(declared, dict):
        raise HarnessError("ray.clusters and ray.routes must be mappings")
    clusters = {}
    for name, spec in declared.items():
        field = f"ray.clusters.{name}"
        if not isinstance(spec, dict):
            raise HarnessError(f"{field} must be a mapping")
        roles = tuple(sorted(role for role, routed in routes.items() if routed == name))
        if not roles:
            raise HarnessError(f"{field} has no route in ray.routes")
        gpus_per_node = spec.get("gpus_per_node", 0)
        if not isinstance(gpus_per_node, int) or gpus_per_node < 0:
            raise HarnessError(f"{field}.gpus_per_node must be a non-negative integer")
        controller_url = spec.get("controller_jobs_url")
        clusters[name] = Cluster(
            name=name,
            roles=roles,
            jobs_url_variable=_variable(spec.get("jobs_url"), f"{field}.jobs_url"),
            controller_url_variable=(
                None
                if controller_url is None
                else _variable(controller_url, f"{field}.controller_jobs_url")
            ),
            expected_image=_text(spec.get("expected_image"), f"{field}.expected_image"),
            expected_ray=None if spec.get("expected_ray") is None else str(spec["expected_ray"]),
            gpus_per_node=gpus_per_node,
        )
    if not clusters:
        raise HarnessError("ray.clusters is empty")
    state_dir = Path(artifact_root) / "test-infra" / getpass.getuser() / path.stem
    return RayTarget(artifact_root, clusters, state_dir)


def load_slurm_settings(path: Path) -> SlurmSettings:
    """Read the account, partitions, containers, mounts, and env_vars entries."""
    document = _mapping(path)
    if document.get("executor", "slurm") != "slurm":
        raise HarnessError(f"{path} is not a Slurm target")
    containers = document.get("containers") or {}
    mounts = document.get("mounts") or []
    env_vars = document.get("env_vars") or []
    if not isinstance(containers, dict) or not isinstance(mounts, list):
        raise HarnessError(f"{path}: containers must be a mapping and mounts a list")
    if not isinstance(env_vars, list):
        raise HarnessError(f"{path}: env_vars must be a list")
    checked_mounts = tuple(_text(mount, "mounts entry") for mount in mounts)
    for mount in checked_mounts:
        if "," in mount:
            raise HarnessError(f"mounts entry contains a comma: {mount!r}")
    gpu_partition = _text(document.get("partition"), "partition")
    return SlurmSettings(
        account=_text(document.get("account"), "account"),
        gpu_partition=gpu_partition,
        cpu_partition=_text(document.get("cpu_partition") or gpu_partition, "cpu_partition"),
        containers=containers,
        mounts=checked_mounts,
        env_vars=tuple(str(entry) for entry in env_vars),
    )


def target_environment(slurm: SlurmSettings, *, pass_credentials: bool) -> dict[str, str]:
    """Resolve env_vars as a Slurm submission does, withholding credentials by default."""
    environment = {}
    for entry in slurm.env_vars:
        name, separator, value = entry.partition("=")
        name = name.strip()
        if "=" in value:
            raise HarnessError(f"env_vars entry for {name} has more than one '='")
        if not separator:
            if name not in os.environ:
                continue
            value = os.environ[name]
        if pass_credentials or not CREDENTIAL_NAME.search(name):
            environment[name] = value.strip()
    return environment


def select(target: RayTarget, names: list[str]) -> list[Cluster]:
    unknown = sorted(set(names) - set(target.clusters))
    if unknown:
        raise HarnessError(
            f"unknown cluster {', '.join(unknown)}; the target declares "
            f"{', '.join(target.clusters)}"
        )
    return [target.clusters[name] for name in names] if names else list(target.clusters.values())


def endpoint_path(target: RayTarget, name: str) -> Path:
    return target.state_dir / f"{name}.env"


def job_path(target: RayTarget, name: str) -> Path:
    return target.state_dir / f"{name}.job"


def allocation(
    cluster: Cluster,
    target: RayTarget,
    slurm: SlurmSettings,
    *,
    nodes: int,
    time_limit: str,
    cpus: int,
    memory: str,
    pass_credentials: bool = False,
) -> tuple[list[str], dict[str, str]]:
    """Return the sbatch command and environment that start one cluster."""
    missing = [role for role in cluster.roles if role not in slurm.containers]
    if missing:
        raise HarnessError(f"the Slurm target has no containers entry for {', '.join(missing)}")
    images = {_text(slurm.containers[role], f"containers.{role}") for role in cluster.roles}
    if len(images) > 1:
        raise HarnessError(f"cluster {cluster.name!r} routes roles with different images")
    ray_bin_dirs = {RAY_BIN_DIRS[role] for role in cluster.roles if role in RAY_BIN_DIRS}
    if len(ray_bin_dirs) > 1:
        raise HarnessError(f"cluster {cluster.name!r} routes roles with different Ray paths")
    partition = slurm.gpu_partition if cluster.gpus_per_node else slurm.cpu_partition
    command = [
        "sbatch",
        "--parsable",
        f"--account={slurm.account}",
        f"--partition={partition}",
        f"--nodes={nodes}",
        f"--time={time_limit}",
        f"--job-name=nvflow-ray-{cluster.name}",
        f"--output={target.state_dir / cluster.name}-%j.out",
    ]
    if cluster.gpus_per_node:
        command += ["--exclusive", f"--gpus-per-node={cluster.gpus_per_node}"]
    else:
        # Idle CPU nodes are scarce, and a CPU cluster needs only a few cores. Without
        # --mem, Slurm may reserve all of the node's memory, which blocks sharing.
        command += [f"--cpus-per-task={cpus}", f"--mem={memory}"]
    command += ["--export=ALL", str(HARNESS / "start_cluster.slurm")]
    environment = {
        **target_environment(slurm, pass_credentials=pass_credentials),
        "NVFLOW_TEST_PASS_CREDENTIALS": "1" if pass_credentials else "",
        "ROLE": cluster.name,
        "IMAGE": images.pop(),
        "IMAGE_ID": cluster.expected_image,
        "ENDPOINT_FILE": str(endpoint_path(target, cluster.name)),
        "ARTIFACT_ROOT": target.artifact_root,
        "MOUNTS": ",".join((f"{target.artifact_root}:{target.artifact_root}", *slurm.mounts)),
        "RAY_BIN_DIR": next(iter(ray_bin_dirs), ""),
        "NVFLOW_INSECURE_TEST_ENDPOINT_ACK": ACKNOWLEDGEMENT,
    }
    return command, environment


def _submit(command: list[str], environment: dict[str, str]) -> str:
    result = subprocess.run(
        command, env={**os.environ, **environment}, capture_output=True, text=True
    )
    job_id = result.stdout.strip().split(";")[0]
    if result.returncode != 0 or not job_id.isdigit():
        raise HarnessError(f"sbatch failed: {result.stderr.strip() or result.stdout.strip()}")
    return job_id


def _queue_status(job_id: str) -> tuple[str, str]:
    """Return the job's Slurm state and reason, or ("", "") once it has left the queue."""
    result = subprocess.run(
        ["squeue", "--noheader", f"--jobs={job_id}", "--format=%T|%r"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return ("", "") if "Invalid job id" in result.stderr else ("UNKNOWN", "")
    state, _, reason = result.stdout.strip().partition("|")
    return state, reason


def _queue_state(job_id: str) -> str:
    """Return the job's Slurm state, or "" once it has left the queue."""
    return _queue_status(job_id)[0]


def _cancel(job_id: str) -> None:
    subprocess.run(["scancel", job_id], check=False)


def _ray_version(jobs_url: str) -> str:
    with urllib.request.urlopen(f"{jobs_url}/api/version", timeout=10) as response:
        return str(json.load(response)["ray_version"])


def _recorded_job(target: RayTarget, name: str) -> str | None:
    path = job_path(target, name)
    return path.read_text(encoding="utf-8").strip() if path.is_file() else None


def read_endpoint(path: Path) -> dict[str, str]:
    """Parse the KEY=value lines that start_cluster.slurm writes."""
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key.strip()] = value.strip()
    return values


def write_endpoints(target: RayTarget) -> Path:
    """Export the jobs_url variables of every cluster that has an endpoint."""
    lines = []
    for cluster in target.clusters.values():
        path = endpoint_path(target, cluster.name)
        if not path.is_file():
            continue
        endpoint = read_endpoint(path)
        lines.append(f"export {cluster.jobs_url_variable}={shlex.quote(endpoint['JOBS_URL'])}")
        if cluster.controller_url_variable:
            url = shlex.quote(endpoint["CONTROLLER_JOBS_URL"])
            lines.append(f"export {cluster.controller_url_variable}={url}")
    output = target.state_dir / "endpoints.env"
    output.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return output


def check(target: RayTarget, clusters: list[Cluster]) -> bool:
    """Print whether each cluster answers from this host; return True if all do."""
    ready = True
    for cluster in clusters:
        path = endpoint_path(target, cluster.name)
        if not path.is_file():
            print(f"{cluster.name:<10} not started")
            ready = False
            continue
        endpoint = read_endpoint(path)
        job_id = endpoint.get("SLURM_JOB_ID", "")
        jobs_url = endpoint.get("JOBS_URL", "")
        state = _queue_state(job_id)
        problem = ""
        if state != "RUNNING":
            problem = f"Slurm job is {state or 'gone'}"
        else:
            try:
                version = _ray_version(jobs_url)
            except (OSError, ValueError, KeyError) as exc:
                problem = f"Jobs API unreachable from this host: {exc}"
            else:
                if cluster.expected_ray and version != cluster.expected_ray:
                    problem = f"Ray {version}, expected {cluster.expected_ray}"
        print(f"{cluster.name:<10} job {job_id:<10} {jobs_url}  {problem or 'ready'}")
        ready = ready and not problem
    return ready


def _wait_for_endpoints(
    target: RayTarget, submitted: list[tuple[Cluster, str]], deadline: float
) -> None:
    """Wait for every cluster together, so one that dies is reported at once."""
    waiting = list(submitted)
    reported: dict[str, str] = {}
    while waiting:
        for cluster, job_id in list(waiting):
            path = endpoint_path(target, cluster.name)
            if path.is_file() and path.stat().st_size:
                waiting.remove((cluster, job_id))
                continue
            state, reason = _queue_status(job_id)
            if not state:
                log = target.state_dir / f"{cluster.name}-{job_id}.out"
                raise HarnessError(
                    f"{cluster.name}: Slurm job {job_id} ended before Ray started; see {log}"
                )
            # A job over a partition or QOS limit stays pending forever; its reason says so.
            status = f"{state} ({reason})" if state == "PENDING" and reason else state
            if reported.get(cluster.name) != status:
                reported[cluster.name] = status
                suffix = ", starting Ray" if state == "RUNNING" else ""
                print(f"{cluster.name:<10} {status}{suffix}", flush=True)
            if time.monotonic() > deadline:
                raise HarnessError(
                    f"{cluster.name}: not ready before --timeout; Slurm job is {status}"
                )
        if waiting:
            time.sleep(5)


def start(
    target: RayTarget,
    slurm: SlurmSettings,
    clusters: list[Cluster],
    *,
    nodes: int,
    time_limit: str,
    cpus: int,
    memory: str,
    timeout: float,
    pass_credentials: bool = False,
) -> Path:
    """Start the clusters, wait until all answer, and return endpoints.env."""
    if os.environ.get("NVFLOW_INSECURE_TEST_ENDPOINT_ACK") != ACKNOWLEDGEMENT:
        raise HarnessError(
            f"export NVFLOW_INSECURE_TEST_ENDPOINT_ACK={ACKNOWLEDGEMENT}: "
            "these test Jobs endpoints have no authentication"
        )
    target.state_dir.mkdir(parents=True, exist_ok=True)
    for cluster in clusters:
        running = _recorded_job(target, cluster.name)
        if running and _queue_state(running):
            raise HarnessError(f"{cluster.name} is already Slurm job {running}; stop it first")
    requests = []
    for cluster in clusters:
        command, environment = allocation(
            cluster,
            target,
            slurm,
            nodes=nodes,
            time_limit=time_limit,
            cpus=cpus,
            memory=memory,
            pass_credentials=pass_credentials,
        )
        requests.append((cluster, command, environment))
    submitted: list[tuple[Cluster, str]] = []
    try:
        for cluster, command, environment in requests:
            endpoint_path(target, cluster.name).unlink(missing_ok=True)
            job_id = _submit(command, environment)
            job_path(target, cluster.name).write_text(f"{job_id}\n", encoding="utf-8")
            submitted.append((cluster, job_id))
            print(f"{cluster.name:<10} submitted Slurm job {job_id}")
        _wait_for_endpoints(target, submitted, time.monotonic() + timeout)
        if not check(target, clusters):
            raise HarnessError("the clusters started but do not all answer from this host")
    except BaseException:
        for cluster, job_id in submitted:
            _cancel(job_id)
            job_path(target, cluster.name).unlink(missing_ok=True)
            endpoint_path(target, cluster.name).unlink(missing_ok=True)
        raise
    return write_endpoints(target)


def stop(target: RayTarget, clusters: list[Cluster]) -> None:
    for cluster in clusters:
        job_id = _recorded_job(target, cluster.name)
        if job_id:
            _cancel(job_id)
            print(f"{cluster.name:<10} cancelled Slurm job {job_id}")
        job_path(target, cluster.name).unlink(missing_ok=True)
        endpoint_path(target, cluster.name).unlink(missing_ok=True)
    if target.state_dir.is_dir():
        write_endpoints(target)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Start, check, and stop Ray-on-Slurm test clusters declared in a Ray target."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    start_parser = commands.add_parser("start", help="start clusters and wait until they answer")
    commands.add_parser("check", help="report whether clusters answer from this host")
    commands.add_parser("stop", help="cancel clusters")
    for command_parser in commands.choices.values():
        command_parser.add_argument("ray_target", type=Path, help="NVFlow Ray target YAML")
        command_parser.add_argument(
            "clusters", nargs="*", help="clusters to act on (default: all in the target)"
        )
    start_parser.add_argument(
        "--slurm-target",
        type=Path,
        required=True,
        help="NVFlow Slurm target that supplies the account, partitions, containers, and mounts",
    )
    start_parser.add_argument("--nodes", type=int, default=1, help="nodes per cluster (default 1)")
    start_parser.add_argument(
        "--time", default="02:00:00", help="Slurm time limit (default 02:00:00)"
    )
    start_parser.add_argument(
        "--cpus", type=int, default=16, help="CPUs per CPU-cluster node (default 16)"
    )
    start_parser.add_argument(
        "--mem", default="64G", help="memory per CPU-cluster node (default 64G)"
    )
    start_parser.add_argument(
        "--timeout",
        type=int,
        default=7200,
        help="seconds to wait for readiness, including time queued in Slurm (default 7200)",
    )
    start_parser.add_argument(
        "--pass-credentials",
        action="store_true",
        help=(
            "also pass env_vars whose names end in TOKEN, API_KEY, SECRET, PASSWORD, or "
            "CREDENTIAL; anyone who can reach these unauthenticated Jobs APIs can read them"
        ),
    )
    args = parser.parse_args(argv)
    try:
        target = load_ray_target(args.ray_target)
        clusters = select(target, args.clusters)
        if args.command == "start":
            endpoints = start(
                target,
                load_slurm_settings(args.slurm_target),
                clusters,
                nodes=args.nodes,
                time_limit=args.time,
                cpus=args.cpus,
                memory=args.mem,
                timeout=args.timeout,
                pass_credentials=args.pass_credentials,
            )
            print(f"\nsource {endpoints}")
        elif args.command == "check":
            return 0 if check(target, clusters) else 1
        else:
            stop(target, clusters)
    except HarnessError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
