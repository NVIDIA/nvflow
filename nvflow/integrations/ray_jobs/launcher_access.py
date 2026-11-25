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
"""Launcher access to supplied Ray clusters and the shared artifact root.

Without ssh_tunnel, the launcher reaches every jobs_url directly and has the
artifact root mounted. With the target's ssh_tunnel, the same block direct
Slurm targets use, it forwards each jobs_url through the tunnel host and runs
its artifact-root file operations there.
"""

from __future__ import annotations

import logging
import os
import shlex
import socket
import time
from collections.abc import Collection, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from urllib.parse import urlparse

from nvflow.execution.target import RayTarget, RegisteredCluster

_MISSING_FILE_STATUS = 3
_CONNECT_TIMEOUT_SECONDS = 10


class ArtifactFiles(Protocol):
    """Files below the shared artifact root, as the launcher reaches them."""

    def write_text(self, path: str, text: str) -> None: ...

    def read_text(self, path: str) -> str:
        """Return the file's text; raise FileNotFoundError when it does not exist."""
        ...

    def remove(self, path: str) -> None: ...


class LocalArtifactFiles:
    """The artifact root is mounted on the launcher."""

    def write_text(self, path: str, text: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")

    def read_text(self, path: str) -> str:
        return Path(path).read_text(encoding="utf-8")

    def remove(self, path: str) -> None:
        Path(path).unlink(missing_ok=True)


class TunnelArtifactFiles:
    """The artifact root is mounted on the ssh_tunnel host."""

    def __init__(self, tunnel: Any) -> None:
        self._tunnel = tunnel

    def write_text(self, path: str, text: str) -> None:
        directory = shlex.quote(str(PurePosixPath(path).parent))
        self._run(f"mkdir -p {directory} && printf %s {shlex.quote(text)} > {shlex.quote(path)}")

    def read_text(self, path: str) -> str:
        quoted = shlex.quote(path)
        result = self._tunnel.run(
            f"if [ -e {quoted} ]; then cat {quoted}; else exit {_MISSING_FILE_STATUS}; fi",
            hide=True,
            warn=True,
        )
        if result.exited == _MISSING_FILE_STATUS:
            raise FileNotFoundError(path)
        if not result.ok:
            raise OSError(f"Cannot read {path} through ssh_tunnel: {result.stderr.strip()}")
        return str(result.stdout)

    def remove(self, path: str) -> None:
        self._run(f"rm -f {shlex.quote(path)}")

    def _run(self, command: str) -> None:
        result = self._tunnel.run(command, hide=True, warn=True)
        if not result.ok:
            raise OSError(f"ssh_tunnel command failed: {result.stderr.strip()}")


@dataclass(frozen=True)
class LauncherAccess:
    """Clusters addressed as the launcher reaches them, and artifact-root files."""

    clusters: Mapping[str, RegisteredCluster]
    files: ArtifactFiles
    # Requested clusters the launcher cannot use, with the reason.
    unavailable: Mapping[str, str] = field(default_factory=dict)
    # The ssh_tunnel host that connects to each jobs_url, when there is one.
    via: str | None = None


def unreachable_reason(cluster: RegisteredCluster, via: str | None, cause: str) -> str:
    """Describe a cluster whose jobs_url does not answer, naming where that URL comes from."""
    source = f" from {via}" if via else ""
    hint = f"; check {cluster.jobs_url_variable}" if cluster.jobs_url_variable else ""
    return f"not answering at {cluster.jobs_url}{source} ({cause}){hint}"


@contextmanager
def launcher_access(
    target: RayTarget,
    names: Collection[str] | None = None,
) -> Iterator[LauncherAccess]:
    """Yield access to the named clusters (default all), through ssh_tunnel when set."""
    configured: dict[str, tuple[RegisteredCluster, str]] = {}
    unavailable: dict[str, str] = {}
    for name, cluster in target.clusters.items():
        if names is not None and name not in names:
            continue
        if cluster.jobs_url is None:
            unavailable[name] = f"{cluster.unset_variable} is not set"
        else:
            configured[name] = (cluster, cluster.jobs_url)
    if target.ssh_tunnel is None:
        clusters = {name: cluster for name, (cluster, _) in configured.items()}
        yield LauncherAccess(clusters, LocalArtifactFiles(), unavailable)
        return
    via = str(target.ssh_tunnel["host"])
    tunnel = _tunnel(target.ssh_tunnel)
    tunnel.connect()
    try:
        with ExitStack() as forwards:
            clusters = {}
            for name, (cluster, jobs_url) in configured.items():
                address = urlparse(jobs_url)
                remote = (str(address.hostname), address.port or 80)
                failure = _connect_failure(tunnel, remote)
                if failure is not None:
                    unavailable[name] = unreachable_reason(cluster, via, failure)
                    continue
                local_port = _free_local_port()
                forwards.enter_context(
                    tunnel.session.forward_local(
                        local_port,
                        remote_port=remote[1],
                        remote_host=remote[0],
                        local_host="127.0.0.1",
                    )
                )
                _wait_for_listener(local_port)
                # controller_jobs_url keeps the cluster-side address.
                clusters[name] = replace(
                    cluster,
                    jobs_url=f"http://127.0.0.1:{local_port}{address.path.rstrip('/')}",
                )
            yield LauncherAccess(clusters, TunnelArtifactFiles(tunnel), unavailable, via)
    finally:
        tunnel.cleanup()


def _tunnel(settings: Mapping[str, Any]) -> Any:
    """Build the NeMo-Run SSH tunnel that direct Slurm builds from the same block."""
    from nemo_run.core.tunnel.client import SSHTunnel

    identity = settings.get("identity")
    options: dict[str, Any] = {
        "host": settings["host"],
        "user": settings["user"],
        "identity": None if identity is None else os.path.expanduser(str(identity)),
        "shell": settings.get("shell"),
        "pre_command": settings.get("pre_command"),
        "job_dir": str(settings.get("job_dir") or ""),
    }
    if settings.get("port") is not None:
        options["port"] = settings["port"]
    return SSHTunnel(**options)


def _connect_failure(tunnel: Any, address: tuple[str, int]) -> str | None:
    """Return why the tunnel host cannot connect to *address*, or None when it can.

    A forward to a refused address leaves its listener accepting connections that
    never get an answer, so the connection is tested before forwarding. Callers
    report the failure with the cluster's name, so paramiko's own log line for it
    is dropped.
    """
    transport = tunnel.session.transport
    transport_log = logging.getLogger(transport.get_log_channel())
    transport_log.addFilter(_not_channel_open_failure)
    try:
        channel = transport.open_channel(
            "direct-tcpip",
            address,
            ("127.0.0.1", 0),
            timeout=_CONNECT_TIMEOUT_SECONDS,
        )
    except Exception as error:
        # paramiko's ChannelException names the failure in .text, e.g. "Connect failed".
        return str(getattr(error, "text", "") or error or type(error).__name__)
    finally:
        transport_log.removeFilter(_not_channel_open_failure)
    channel.close()
    return None


def _not_channel_open_failure(record: logging.LogRecord) -> bool:
    return "open FAILED" not in record.getMessage()


def _free_local_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _wait_for_listener(port: int, timeout_seconds: float = 10) -> None:
    """Wait for the forward's listener, which binds on a background thread."""
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


__all__ = [
    "ArtifactFiles",
    "LauncherAccess",
    "LocalArtifactFiles",
    "TunnelArtifactFiles",
    "launcher_access",
    "unreachable_reason",
]
