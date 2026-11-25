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
"""Contracts for the NVFlow-owned supplied Ray Jobs client."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from nvflow.execution.target import RegisteredCluster
from nvflow.integrations.ray_jobs import RayJobsClient, RayJobsError, RayJobState
from nvflow.integrations.ray_jobs.bundle import build_controller_bundle
from nvflow.project.packaging import WorkerBundle


class FakeSDK:
    def __init__(self, **kwargs) -> None:
        self.init_kwargs = kwargs
        self.submissions: list[dict[str, Any]] = []
        self.status_value = SimpleNamespace(value="RUNNING")
        self.status_message: str | None = None

    def list_jobs(self):
        return []

    def submit_job(self, **kwargs):
        self.submissions.append(kwargs)
        return kwargs["submission_id"]

    def get_job_info(self, _submission_id):
        return SimpleNamespace(status=self.status_value, message=self.status_message)

    def stop_job(self, _submission_id):
        return True

    def get_job_logs(self, _submission_id):
        return "worker log"


def _cluster(**overrides: Any) -> RegisteredCluster:
    values: dict[str, Any] = {
        "name": "skills",
        "jobs_url": "https://ray.example.test",
        "credential_env": "RAY_TOKEN",
    }
    values.update(overrides)
    return RegisteredCluster(**values)


def test_client_submits_only_local_immutable_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"zip")
    sdk = FakeSDK()
    client = RayJobsClient(
        _cluster(),
        environ={"RAY_TOKEN": "secret"},
        client_factory=lambda **kwargs: _record_init(sdk, kwargs),
    )

    submission_id = client.submit(
        submission_id="run-task-a",
        entrypoint="python -m worker",
        working_dir=bundle,
        env_vars={},
        metadata={"run_id": "run"},
    )

    assert submission_id == "run-task-a"
    assert sdk.init_kwargs["address"] == "https://ray.example.test"
    assert sdk.init_kwargs["headers"] == {"Authorization": "Bearer secret"}
    assert sdk.init_kwargs["verify"] is True
    submission = sdk.submissions[0]
    assert submission["runtime_env"] == {"working_dir": str(bundle.resolve())}
    assert "pip" not in submission["runtime_env"]


def test_client_rejects_runtime_environment_variables(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"zip")
    client = RayJobsClient(
        _cluster(credential_env=None),
        client_factory=lambda **_kwargs: FakeSDK(),
    )

    with pytest.raises(RayJobsError, match="env_vars is prohibited"):
        client.submit(
            submission_id="run-task-a",
            entrypoint="python worker.py",
            working_dir=bundle,
            env_vars={"TOKEN": "secret"},
            metadata={},
        )


def test_client_normalizes_status_stop_and_logs() -> None:
    sdk = FakeSDK()
    client = RayJobsClient(
        _cluster(credential_env=None),
        client_factory=lambda **kwargs: _record_init(sdk, kwargs),
    )

    assert client.status("job").state is RayJobState.RUNNING
    assert client.stop("job") is True
    assert client.logs("job") == "worker log"


def test_client_status_carries_rays_failure_message() -> None:
    sdk = FakeSDK()
    sdk.status_value = SimpleNamespace(value="FAILED")
    sdk.status_message = "Job supervisor actor failed to start within 900.0 seconds."
    client = RayJobsClient(_cluster(credential_env=None), client_factory=lambda **_kwargs: sdk)

    details = client.status("job")

    assert details.state is RayJobState.FAILED
    assert details.message == "Job supervisor actor failed to start within 900.0 seconds."


def test_client_retries_a_transient_status_failure() -> None:
    class FlakySDK(FakeSDK):
        def __init__(self) -> None:
            super().__init__()
            self.status_calls = 0

        def get_job_info(self, submission_id):
            self.status_calls += 1
            if self.status_calls == 1:
                raise ConnectionError("temporary outage")
            return super().get_job_info(submission_id)

    sdk = FlakySDK()
    sleeps: list[float] = []
    client = RayJobsClient(
        _cluster(credential_env=None),
        client_factory=lambda **_kwargs: sdk,
        request_attempts=3,
        retry_seconds=0.25,
        sleeper=sleeps.append,
    )

    assert client.status("job").state is RayJobState.RUNNING
    assert sdk.status_calls == 2
    assert sleeps == [0.25]


def test_client_retries_submit_with_the_same_deterministic_id(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"zip")

    class FlakySDK(FakeSDK):
        def __init__(self) -> None:
            super().__init__()
            self.submit_calls = 0

        def submit_job(self, **kwargs):
            self.submit_calls += 1
            if self.submit_calls == 1:
                raise ConnectionError("temporary outage")
            return super().submit_job(**kwargs)

    sdk = FlakySDK()
    sleeps: list[float] = []
    client = RayJobsClient(
        _cluster(credential_env=None),
        client_factory=lambda **_kwargs: sdk,
        request_attempts=3,
        retry_seconds=0.25,
        sleeper=sleeps.append,
    )

    assert (
        client.submit(
            submission_id="stable-id",
            entrypoint="true",
            working_dir=bundle,
            env_vars={},
            metadata={},
        )
        == "stable-id"
    )
    assert sdk.submit_calls == 2
    assert sleeps == [0.25]


def test_client_probes_unified_cluster_runtime(tmp_path: Path) -> None:
    artifact = tmp_path / "probe.txt"
    artifact.write_text("shared-token\n")

    class RuntimeSDK(FakeSDK):
        def submit_job(self, **kwargs):
            self.submissions.append(kwargs)
            Path(f"{artifact}.skills.ack").write_text("shared-token\n")
            return kwargs["submission_id"]

        def get_job_info(self, _submission_id):
            return SimpleNamespace(status=SimpleNamespace(value="SUCCEEDED"), message=None)

        def get_job_logs(self, _submission_id):
            return (
                'NVFLOW_RUNTIME_PROFILE={"image":"stale","python":"3.13.14",'
                '"ray":"2.58.0"}\n'
                'NVFLOW_RUNTIME_PROFILE={"artifact_visible":true,'
                '"image":"nemo-rl","python":"3.13.14","ray":"2.58.0"}'
            )

    sdk = RuntimeSDK()
    client = RayJobsClient(
        _cluster(
            credential_env=None,
            expected_image="nemo-rl",
            expected_python="3.13.14",
            expected_ray="2.58.0",
        ),
        client_factory=lambda **kwargs: _record_init(sdk, kwargs),
    )

    result = client.probe_runtime(
        probe_id="target-cluster",
        artifact_path=str(artifact),
        artifact_token="shared-token",
        poll_seconds=0,
    )

    assert result.runtime.python == "3.13.14"
    assert result.runtime.ray == "2.58.0"
    assert result.image == "nemo-rl"
    assert not Path(f"{artifact}.skills.ack").exists()


@pytest.mark.parametrize("published", [True, False])
def test_client_reads_the_acknowledgement_through_launcher_files(published: bool) -> None:
    artifact = "/shared/nvflow/.nvflow/probes/launcher.txt"
    acknowledgement = f"{artifact}.skills.ack"
    contents: dict[str, str] = {}
    removed: list[str] = []

    def read_text(path: str) -> str:
        if path not in contents:
            raise FileNotFoundError(path)
        return contents[path]

    def remove(path: str) -> None:
        removed.append(path)
        contents.pop(path, None)

    files = SimpleNamespace(read_text=read_text, remove=remove)

    class RuntimeSDK(FakeSDK):
        def submit_job(self, **kwargs):
            if published:
                contents[acknowledgement] = "shared-token\n"
            return kwargs["submission_id"]

        def get_job_info(self, _submission_id):
            return SimpleNamespace(status=SimpleNamespace(value="SUCCEEDED"), message=None)

        def get_job_logs(self, _submission_id):
            return (
                'NVFLOW_RUNTIME_PROFILE={"artifact_visible":true,'
                '"image":"nemo-skills","python":"3.13.14","ray":"2.58.0"}'
            )

    client = RayJobsClient(
        _cluster(credential_env=None, expected_image="nemo-skills"),
        client_factory=lambda **_kwargs: RuntimeSDK(),
    )

    def probe():
        return client.probe_runtime(
            probe_id="target-cluster",
            artifact_path=artifact,
            artifact_token="shared-token",
            files=cast(Any, files),
            poll_seconds=0,
        )

    if published:
        assert probe().image == "nemo-skills"
    else:
        with pytest.raises(RayJobsError, match="did not publish"):
            probe()
    assert removed == [acknowledgement, acknowledgement]
    assert contents == {}


def test_client_lists_submissions_with_metadata_and_start_time() -> None:
    class ListingSDK(FakeSDK):
        def list_jobs(self):
            return [
                SimpleNamespace(
                    submission_id="nvflow-run-controller-a1",
                    status=SimpleNamespace(value="RUNNING"),
                    message=None,
                    metadata={"nvflow_user": "alice"},
                    start_time=1_700_000_000_000,
                ),
                SimpleNamespace(submission_id=None, status=SimpleNamespace(value="RUNNING")),
            ]

    client = RayJobsClient(
        _cluster(credential_env=None, jobs_url="http://10.0.0.2:8265", allow_insecure=True),
        client_factory=lambda **_kwargs: ListingSDK(),
    )

    (job,) = client.jobs()

    assert job.submission_id == "nvflow-run-controller-a1"
    assert job.state is RayJobState.RUNNING
    assert job.metadata == {"nvflow_user": "alice"}
    assert job.start_time == 1_700_000_000


def test_client_uses_its_cluster_address_inside_a_ray_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Ray sets RAY_ADDRESS in every job, and its Jobs client prefers it to address.
    monkeypatch.setenv("RAY_ADDRESS", "10.0.0.1:6379")
    monkeypatch.setenv("RAY_API_SERVER_ADDRESS", "http://10.0.0.1:8265")
    seen: dict[str, Any] = {}

    def factory(**kwargs):
        seen["address"] = kwargs["address"]
        seen["environment"] = {
            name: os.environ.get(name) for name in ("RAY_ADDRESS", "RAY_API_SERVER_ADDRESS")
        }
        return FakeSDK()

    client = RayJobsClient(
        _cluster(credential_env=None, jobs_url="http://10.0.0.2:8265", allow_insecure=True),
        client_factory=factory,
    )
    client.status("job")

    assert seen == {
        "address": "http://10.0.0.2:8265",
        "environment": {"RAY_ADDRESS": None, "RAY_API_SERVER_ADDRESS": None},
    }
    assert os.environ["RAY_ADDRESS"] == "10.0.0.1:6379"
    assert os.environ["RAY_API_SERVER_ADDRESS"] == "http://10.0.0.1:8265"


def test_client_requires_credential_reference_without_leaking_value() -> None:
    client = RayJobsClient(
        _cluster(),
        environ={},
        client_factory=lambda **_kwargs: FakeSDK(),
    )

    with pytest.raises(RayJobsError, match="RAY_TOKEN"):
        client.probe()


def test_client_redacts_secret_from_provider_failure() -> None:
    secret = "super-secret-value"

    class BrokenSDK(FakeSDK):
        def list_jobs(self):
            raise RuntimeError(f"request headers contained {secret}")

    client = RayJobsClient(
        _cluster(),
        environ={"RAY_TOKEN": secret},
        client_factory=lambda **kwargs: _record_init(BrokenSDK(), kwargs),
        request_attempts=1,
    )

    with pytest.raises(RayJobsError) as caught:
        client.probe()

    assert secret not in str(caught.value)
    assert "<redacted>" in str(caught.value)


def test_client_rejects_directory_as_working_dir(tmp_path: Path) -> None:
    client = RayJobsClient(
        _cluster(credential_env=None),
        client_factory=lambda **_kwargs: FakeSDK(),
    )

    with pytest.raises(RayJobsError, match="existing local ZIP"):
        client.submit(
            submission_id="job",
            entrypoint="true",
            working_dir=tmp_path,
            env_vars={},
            metadata={},
        )


def test_controller_bundle_contains_references_not_secret_values(
    tmp_path: Path,
    monkeypatch,
) -> None:
    secret = "must-not-be-serialized"
    monkeypatch.setenv("RAY_TOKEN", secret)
    worker_path = tmp_path / "worker.zip"
    with zipfile.ZipFile(worker_path, "w") as archive:
        archive.writestr("nvflow/__init__.py", "")
    worker = WorkerBundle(
        path=worker_path,
        digest="sha256:worker",
        compressed_size=worker_path.stat().st_size,
        uncompressed_size=0,
        project_python_root="projects/demo",
        project_identity="git:project",
        platform_identity="git:platform",
    )

    controller = build_controller_bundle(
        worker,
        {
            "ray_target": {
                "credential_env": "RAY_TOKEN",
            },
            "execution": {"command": "python worker.py"},
        },
        tmp_path,
    )

    assert secret.encode() not in controller.path.read_bytes()
    with zipfile.ZipFile(controller.path) as archive:
        spec = archive.read(".nvflow/run.json").decode()
    assert "RAY_TOKEN" in spec
    assert secret not in spec


def _record_init(sdk: FakeSDK, kwargs: dict) -> FakeSDK:
    sdk.init_kwargs = kwargs
    return sdk
