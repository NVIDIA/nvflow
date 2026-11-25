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
"""Launcher access to supplied Ray clusters, directly or through ssh_tunnel."""

from __future__ import annotations

import logging
import socket
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from nvflow.execution.target import RayTarget, RegisteredCluster
from nvflow.integrations.ray_jobs import launcher_access as access_module
from nvflow.integrations.ray_jobs.launcher_access import (
    LocalArtifactFiles,
    TunnelArtifactFiles,
    launcher_access,
)


class ChannelRefusedError(Exception):
    """Mirrors paramiko's ChannelException, which names the failure in text."""

    def __init__(self, text: str) -> None:
        super().__init__(2, text)
        self.text = text


class ShellTunnel:
    """Stands in for the NeMo-Run tunnel: commands run in a local shell."""

    def __init__(self, refused: dict[tuple[str, int], str] | None = None) -> None:
        self.connected = False
        self.cleaned_up = False
        self.forwards: list[dict[str, Any]] = []
        self.checked: list[tuple[str, int]] = []
        self.refused = refused or {}
        self.session = SimpleNamespace(
            forward_local=self._forward_local,
            transport=SimpleNamespace(
                open_channel=self._open_channel,
                get_log_channel=lambda: "paramiko.transport",
            ),
        )

    def _open_channel(
        self,
        kind: str,
        destination: tuple[str, int],
        source: tuple[str, int],
        timeout: float,
    ) -> SimpleNamespace:
        self.checked.append(destination)
        if destination in self.refused:
            logging.getLogger("paramiko.transport").error(
                "Secsh channel 3 open FAILED: Connection refused: %s", self.refused[destination]
            )
            raise ChannelRefusedError(self.refused[destination])
        return SimpleNamespace(close=lambda: None)

    def connect(self) -> None:
        self.connected = True

    def cleanup(self) -> None:
        self.cleaned_up = True

    def run(self, command: str, hide: bool = True, warn: bool = False) -> SimpleNamespace:
        completed = subprocess.run(["sh", "-c", command], capture_output=True, text=True)
        return SimpleNamespace(
            stdout=completed.stdout,
            stderr=completed.stderr,
            exited=completed.returncode,
            ok=completed.returncode == 0,
        )

    @contextmanager
    def _forward_local(
        self, local_port: int, remote_port: int, remote_host: str, local_host: str
    ) -> Iterator[None]:
        self.forwards.append(
            {"local": (local_host, local_port), "remote": (remote_host, remote_port)}
        )
        with socket.socket() as listener:
            listener.bind((local_host, local_port))
            listener.listen(1)
            yield


def _target(tmp_path: Path, ssh_tunnel: dict[str, Any] | None = None) -> RayTarget:
    return RayTarget(
        control_cluster="skills",
        artifact_root=str(tmp_path / "artifacts"),
        clusters={
            "skills": RegisteredCluster("skills", "http://10.0.0.1:8265", allow_insecure=True),
            "serving": RegisteredCluster("serving", "http://10.0.0.2:8265", allow_insecure=True),
        },
        routes={"nemo-skills": "skills", "vllm": "serving"},
        ssh_tunnel=ssh_tunnel,
    )


def test_without_ssh_tunnel_the_launcher_reaches_clusters_and_files_directly(
    tmp_path: Path,
) -> None:
    target = _target(tmp_path)

    with launcher_access(target) as access:
        assert dict(access.clusters) == dict(target.clusters)
        assert isinstance(access.files, LocalArtifactFiles)
        assert access.via is None


def test_ssh_tunnel_forwards_each_jobs_url_and_keeps_controller_addresses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tunnel = ShellTunnel()
    monkeypatch.setattr(access_module, "_tunnel", lambda settings: tunnel)
    target = _target(tmp_path, ssh_tunnel={"host": "login.example", "user": "operator"})

    with launcher_access(target) as access:
        assert tunnel.connected
        for name, (host, port) in {
            "skills": ("10.0.0.1", 8265),
            "serving": ("10.0.0.2", 8265),
        }.items():
            cluster = access.clusters[name]
            forward = next(item for item in tunnel.forwards if item["remote"] == (host, port))
            assert cluster.jobs_url == f"http://127.0.0.1:{forward['local'][1]}"
            assert forward["local"][0] == "127.0.0.1"
            assert cluster.controller_jobs_url == f"http://{host}:{port}"
        assert isinstance(access.files, TunnelArtifactFiles)
    assert tunnel.cleaned_up


def test_ssh_tunnel_reports_unset_and_refused_clusters_without_forwarding_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    tunnel = ShellTunnel(refused={("10.0.0.2", 8265): "Connect failed"})
    monkeypatch.setattr(access_module, "_tunnel", lambda settings: tunnel)
    target = _target(tmp_path, ssh_tunnel={"host": "login.example", "user": "operator"})
    serving = RegisteredCluster(
        "serving",
        "http://10.0.0.2:8265",
        allow_insecure=True,
        jobs_url_variable="RAY_VLLM_JOBS_URL",
    )
    training = RegisteredCluster("training", None, unset_variable="RAY_TRAINING_JOBS_URL")
    target = RayTarget(
        control_cluster=target.control_cluster,
        artifact_root=target.artifact_root,
        clusters={**target.clusters, "serving": serving, "training": training},
        routes={**target.routes, "nemo-rl": "training"},
        ssh_tunnel=target.ssh_tunnel,
    )

    with launcher_access(target) as access:
        assert set(access.clusters) == {"skills"}
        assert access.via == "login.example"
        assert dict(access.unavailable) == {
            "serving": (
                "not answering at http://10.0.0.2:8265 from login.example (Connect failed); "
                "check RAY_VLLM_JOBS_URL"
            ),
            "training": "RAY_TRAINING_JOBS_URL is not set",
        }
    assert [forward["remote"] for forward in tunnel.forwards] == [("10.0.0.1", 8265)]
    assert "open FAILED" not in caplog.text
    logging.getLogger("paramiko.transport").error("Secsh channel 9 open FAILED: later")
    assert "channel 9 open FAILED" in caplog.text


def test_launcher_access_reaches_only_the_named_clusters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tunnel = ShellTunnel()
    monkeypatch.setattr(access_module, "_tunnel", lambda settings: tunnel)
    target = _target(tmp_path, ssh_tunnel={"host": "login.example", "user": "operator"})

    with launcher_access(target, {"skills"}) as access:
        assert set(access.clusters) == {"skills"}
        assert dict(access.unavailable) == {}
    assert tunnel.checked == [("10.0.0.1", 8265)]


def test_tunnel_files_write_read_and_remove_on_the_tunnel_host(tmp_path: Path) -> None:
    files = TunnelArtifactFiles(ShellTunnel())
    path = str(tmp_path / "it's shared" / ".nvflow" / "probes" / "launcher.txt")

    with pytest.raises(FileNotFoundError):
        files.read_text(path)
    files.write_text(path, "token\n")
    assert Path(path).read_text(encoding="utf-8") == "token\n"
    assert files.read_text(path) == "token\n"
    files.remove(path)
    assert not Path(path).exists()
    with pytest.raises(OSError, match="Cannot read"):
        files.read_text(str(tmp_path))
