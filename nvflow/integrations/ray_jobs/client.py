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
"""Small fail-closed wrapper around Ray's Jobs HTTP client."""

from __future__ import annotations

import json
import os
import shlex
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from nvflow.execution.runtime_profile import RAY_VERSION, RuntimeProfile
from nvflow.execution.target import RegisteredCluster
from nvflow.integrations.ray_jobs.launcher_access import ArtifactFiles, LocalArtifactFiles


class RayJobsError(RuntimeError):
    """A supplied Ray Jobs operation failed."""


class RayJobState(str, Enum):
    """Provider-neutral Ray Jobs states used by the controller."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    STOPPED = "stopped"
    UNKNOWN = "unknown"

    @property
    def terminal(self) -> bool:
        return self in {self.SUCCEEDED, self.FAILED, self.STOPPED}


@dataclass(frozen=True, slots=True)
class RayJobDetails:
    """Status snapshot for one deterministic submission."""

    submission_id: str
    state: RayJobState
    message: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)
    start_time: float | None = None
    driver_node: str | None = None


@dataclass(frozen=True, slots=True)
class RayClusterProbe:
    """Runtime identity observed inside one supplied cluster."""

    runtime: RuntimeProfile
    image: str | None
    artifact_visible: bool
    resources: Mapping[str, float] = field(default_factory=dict)
    labels: Mapping[str, str] = field(default_factory=dict)


class RayJobsClient:
    """Submit and inspect jobs without exposing Ray SDK objects to core code."""

    def __init__(
        self,
        cluster: RegisteredCluster,
        *,
        environ: Mapping[str, str] | None = None,
        client_factory: Callable[..., Any] | None = None,
        request_attempts: int = 4,
        retry_seconds: float = 0.5,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if request_attempts < 1:
            raise ValueError("request_attempts must be positive")
        if retry_seconds < 0:
            raise ValueError("retry_seconds must not be negative")
        self.cluster = cluster
        self._environment = os.environ if environ is None else environ
        self._client_factory = client_factory
        self._client: Any | None = None
        self._request_attempts = request_attempts
        self._retry_seconds = retry_seconds
        self._sleep = sleeper

    def probe(self) -> None:
        """Verify that the supplied Jobs endpoint is reachable and authorized."""
        self._request("probe", lambda: self._sdk().list_jobs())

    def probe_runtime(
        self,
        *,
        probe_id: str,
        artifact_path: str | None = None,
        artifact_token: str | None = None,
        files: ArtifactFiles | None = None,
        timeout_seconds: float = 120,
        poll_seconds: float = 1,
    ) -> RayClusterProbe:
        """Run a bounded in-cluster probe for Python, Ray, and image identity."""
        files = LocalArtifactFiles() if files is None else files
        submission_id = f"nvflow-runtime-{probe_id}"
        existing = self.find(submission_id)
        if existing is not None and existing.state.terminal:
            try:
                self._sdk().delete_job(submission_id)
            except Exception as exc:
                raise self._error("delete stale runtime probe", exc) from exc
            existing = None
        acknowledgment = (
            None if artifact_path is None else f"{artifact_path}.{self.cluster.name}.ack"
        )
        if acknowledgment is not None:
            files.remove(acknowledgment)
            artifact_program = (
                "from pathlib import Path\n"
                f"source=Path({artifact_path!r})\n"
                f"ack=Path({acknowledgment!r})\n"
                f"token={artifact_token!r}\n"
                "artifact_visible=source.read_text(encoding='utf-8').strip()==token\n"
                "temporary=Path(str(ack)+'.tmp')\n"
                "with temporary.open('w',encoding='utf-8') as stream:\n"
                "    stream.write(token+'\\n')\n"
                "    stream.flush()\n"
                "    os.fsync(stream.fileno())\n"
                "os.replace(temporary,ack)\n"
            )
        else:
            artifact_program = "artifact_visible=None\n"
        script = (
            "import json,os,platform,ray;"
            "ray.init(address='auto',include_dashboard=False);"
            f"exec({artifact_program!r});"
            "print('NVFLOW_RUNTIME_PROFILE='+json.dumps({"
            "'python':platform.python_version(),"
            "'ray':ray.__version__,"
            "'image':os.environ.get('NVFLOW_IMAGE_ID'),"
            "'artifact_visible':artifact_visible,"
            "'resources':ray.cluster_resources(),"
            "'labels':{k:str(v) for n in ray.nodes() if n.get('Alive') "
            "for k,v in (n.get('Labels') or {}).items()}"
            "},sort_keys=True))"
        )
        if existing is None:
            try:
                self._sdk().submit_job(
                    entrypoint=f"python -c {shlex.quote(script)}",
                    submission_id=submission_id,
                    metadata={"nvflow_kind": "runtime-probe"},
                    entrypoint_num_cpus=0,
                )
            except Exception as exc:
                adopted = self.find(submission_id)
                if adopted is None:
                    raise self._error("submit runtime probe", exc) from exc
        deadline = time.monotonic() + timeout_seconds
        while True:
            details = self.status(submission_id)
            if details.state is RayJobState.SUCCEEDED:
                break
            if details.state in {RayJobState.FAILED, RayJobState.STOPPED}:
                raise RayJobsError(
                    f"Runtime probe failed for supplied cluster {self.cluster.name!r}"
                )
            if time.monotonic() >= deadline:
                self.stop(submission_id)
                raise RayJobsError(
                    f"Runtime probe timed out for supplied cluster {self.cluster.name!r}"
                )
            time.sleep(poll_seconds)
        marker = "NVFLOW_RUNTIME_PROFILE="
        lines = [line for line in self.logs(submission_id).splitlines() if line.startswith(marker)]
        if not lines:
            raise RayJobsError(
                f"Runtime probe returned no identity for cluster {self.cluster.name!r}"
            )
        try:
            payload = json.loads(lines[-1].split(marker, 1)[1])
            result = RayClusterProbe(
                runtime=RuntimeProfile(
                    python=str(payload["python"]),
                    ray=str(payload["ray"]),
                ),
                image=(None if payload.get("image") is None else str(payload["image"])),
                artifact_visible=bool(payload.get("artifact_visible")),
                resources={
                    str(name): float(quantity)
                    for name, quantity in dict(payload.get("resources") or {}).items()
                },
                labels={
                    str(name): str(value)
                    for name, value in dict(payload.get("labels") or {}).items()
                },
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RayJobsError(
                f"Runtime probe returned malformed identity for cluster {self.cluster.name!r}"
            ) from exc
        result.runtime.validate(location=f"Ray cluster {self.cluster.name!r}")
        if result.runtime.python != self.cluster.expected_python:
            raise RayJobsError(
                f"Ray cluster {self.cluster.name!r} Python mismatch: "
                f"expected {self.cluster.expected_python}, "
                f"found {result.runtime.python}"
            )
        if self.cluster.expected_image is not None and result.image != self.cluster.expected_image:
            raise RayJobsError(
                f"Ray cluster {self.cluster.name!r} image mismatch: "
                f"expected {self.cluster.expected_image!r}, found {result.image!r}"
            )
        if artifact_path is not None and not result.artifact_visible:
            raise RayJobsError(
                f"Ray cluster {self.cluster.name!r} cannot read shared artifact probe "
                f"{artifact_path}"
            )
        if acknowledgment is not None:
            try:
                acknowledged = files.read_text(acknowledgment).strip() == artifact_token
            except OSError as exc:
                raise RayJobsError(
                    f"Ray cluster {self.cluster.name!r} did not publish shared "
                    "artifact acknowledgement"
                ) from exc
            finally:
                files.remove(acknowledgment)
            if not acknowledged:
                raise RayJobsError(
                    f"Ray cluster {self.cluster.name!r} published an invalid shared "
                    "artifact acknowledgement"
                )
        return result

    def submit(
        self,
        *,
        submission_id: str,
        entrypoint: str,
        working_dir: Path,
        env_vars: Mapping[str, str],
        metadata: Mapping[str, str],
        entrypoint_num_cpus: float | None = None,
        entrypoint_num_gpus: float | None = None,
        entrypoint_resources: Mapping[str, float] | None = None,
    ) -> str:
        """Submit one immutable local bundle using a deterministic ID."""
        if not submission_id or any(character.isspace() for character in submission_id):
            raise RayJobsError("Ray submission_id must be non-empty and contain no whitespace")
        if not entrypoint.strip():
            raise RayJobsError("Ray entrypoint must be non-empty")
        bundle = working_dir.resolve()
        if not bundle.is_file() or bundle.suffix != ".zip":
            raise RayJobsError(f"Ray working_dir must be an existing local ZIP: {bundle}")
        if env_vars:
            raise RayJobsError(
                "Ray Jobs runtime_env.env_vars is prohibited; export control "
                "variables in the entrypoint"
            )
        runtime_env = {"working_dir": str(bundle)}
        returned = None
        last_error: Exception | None = None
        for attempt in range(self._request_attempts):
            try:
                returned = self._sdk().submit_job(
                    entrypoint=entrypoint,
                    submission_id=submission_id,
                    runtime_env=runtime_env,
                    metadata={str(key): str(value) for key, value in metadata.items()},
                    entrypoint_num_cpus=entrypoint_num_cpus,
                    entrypoint_num_gpus=entrypoint_num_gpus,
                    entrypoint_resources=(
                        None
                        if entrypoint_resources is None
                        else {str(key): float(value) for key, value in entrypoint_resources.items()}
                    ),
                )
                break
            except RayJobsError:
                raise
            except Exception as exc:
                last_error = exc
                existing = self.find(submission_id)
                if existing is not None:
                    returned = submission_id
                    break
                if attempt + 1 < self._request_attempts:
                    self._sleep(self._retry_seconds * (2**attempt))
        if returned is None:
            assert last_error is not None
            raise self._error("submit", last_error) from last_error
        if str(returned) != submission_id:
            raise RayJobsError(
                f"Ray cluster {self.cluster.name!r} returned unexpected submission identity"
            )
        return submission_id

    def status(self, submission_id: str) -> RayJobDetails:
        """Return one normalized job status with Ray's message."""
        value = self._request(
            "status",
            lambda: self._sdk().get_job_info(submission_id),
        )
        return RayJobDetails(
            submission_id=submission_id,
            state=_normalize_state(getattr(value, "status", None)),
            message=getattr(value, "message", None),
        )

    def find(self, submission_id: str) -> RayJobDetails | None:
        """Find an existing deterministic job without treating absence as failure."""
        return next((job for job in self.jobs() if job.submission_id == submission_id), None)

    def jobs(self) -> tuple[RayJobDetails, ...]:
        """Return every Jobs API submission the cluster still reports."""
        listed = self._request("list", lambda: self._sdk().list_jobs())
        details = []
        for job in listed:
            submission_id = getattr(job, "submission_id", None)
            if not submission_id:
                continue
            started = getattr(job, "start_time", None)
            driver_node = getattr(getattr(job, "driver_info", None), "node_ip_address", None)
            details.append(
                RayJobDetails(
                    submission_id=str(submission_id),
                    state=_normalize_state(getattr(job, "status", None)),
                    message=getattr(job, "message", None),
                    metadata={
                        str(key): str(value)
                        for key, value in (getattr(job, "metadata", None) or {}).items()
                    },
                    start_time=None if started is None else started / 1000,
                    driver_node=None if driver_node is None else str(driver_node),
                )
            )
        return tuple(details)

    def stop(self, submission_id: str) -> bool:
        """Request cancellation of one job owned by the current run."""
        return bool(
            self._request(
                "stop",
                lambda: self._sdk().stop_job(submission_id),
            )
        )

    def logs(self, submission_id: str) -> str:
        """Return bounded-by-provider logs for one submitted job."""
        return str(
            self._request(
                "logs",
                lambda: self._sdk().get_job_logs(submission_id),
            )
        )

    def _request(self, operation: str, action: Callable[[], Any]) -> Any:
        """Retry one idempotent Jobs API request with bounded backoff."""
        last_error: Exception | None = None
        for attempt in range(self._request_attempts):
            try:
                return action()
            except RayJobsError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self._request_attempts:
                    self._sleep(self._retry_seconds * (2**attempt))
        assert last_error is not None
        raise self._error(operation, last_error) from last_error

    def _sdk(self) -> Any:
        if self._client is not None:
            return self._client
        if self.cluster.jobs_url is None:
            # Ray takes a missing address from RAY_ADDRESS, inside a Job its own cluster.
            raise RayJobsError(
                f"Ray cluster {self.cluster.name!r} has no jobs_url: "
                f"{self.cluster.unset_variable} is not set"
            )
        factory = self._client_factory
        if factory is None:
            try:
                from ray.job_submission import JobSubmissionClient
            except ImportError as exc:
                raise RayJobsError(
                    f"Ray execution requires ray[default]=={RAY_VERSION} "
                    "in the launcher environment"
                ) from exc
            factory = JobSubmissionClient
        headers: dict[str, str] = {}
        if self.cluster.credential_env is not None:
            token = self._environment.get(self.cluster.credential_env)
            if not token:
                raise RayJobsError(
                    f"Ray cluster {self.cluster.name!r} requires environment variable "
                    f"{self.cluster.credential_env}"
                )
            headers["Authorization"] = f"Bearer {token}"
        # allow_insecure permits a development-only HTTP endpoint; it never
        # disables certificate validation for HTTPS.
        verify: bool | str = True
        if self.cluster.ca_file is not None:
            ca_file = Path(self.cluster.ca_file).expanduser()
            if not ca_file.is_file():
                raise RayJobsError(
                    f"Ray cluster {self.cluster.name!r} CA file does not exist: {ca_file}"
                )
            verify = str(ca_file)
        # Ray's Jobs client prefers RAY_API_SERVER_ADDRESS and RAY_ADDRESS over its
        # address argument, and Ray sets RAY_ADDRESS inside every job, so the
        # controller would otherwise submit every child to its own cluster.
        overrides = {
            name: os.environ.pop(name)
            for name in ("RAY_API_SERVER_ADDRESS", "RAY_ADDRESS")
            if name in os.environ
        }
        try:
            self._client = factory(
                address=self.cluster.jobs_url,
                create_cluster_if_needed=False,
                headers=headers or None,
                verify=verify,
            )
        except Exception as exc:
            raise self._error("initialize", exc) from exc
        finally:
            os.environ.update(overrides)
        return self._client

    def _error(self, operation: str, exc: Exception) -> RayJobsError:
        message = str(exc)
        if self.cluster.credential_env is not None:
            secret = self._environment.get(self.cluster.credential_env)
            if secret:
                message = message.replace(secret, "<redacted>")
        return RayJobsError(
            f"Ray Jobs {operation} failed for supplied cluster {self.cluster.name!r}: "
            f"{type(exc).__name__}: {message}"
        )


def _normalize_state(value: object) -> RayJobState:
    raw = str(getattr(value, "value", value)).strip().lower()
    aliases = {
        "pending": RayJobState.PENDING,
        "running": RayJobState.RUNNING,
        "succeeded": RayJobState.SUCCEEDED,
        "failed": RayJobState.FAILED,
        "stopped": RayJobState.STOPPED,
    }
    return aliases.get(raw, RayJobState.UNKNOWN)
