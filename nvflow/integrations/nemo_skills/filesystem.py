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
"""Target filesystem backed by NeMo-Skills local or SSH launch configuration."""

from __future__ import annotations

import hashlib
import json
import shlex
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

CONTAINER_CODE_DIR = "/nemo_run/code"
"""Packaged source snapshot inside Slurm containers."""

_BORROWED_TUNNELS: ContextVar[list[Any] | None] = ContextVar(
    "nvflow_borrowed_nemo_tunnels",
    default=None,
)
_COMPLETION_PROBE_SCRIPT = (
    "import hashlib,json,os,sys\n"
    "def sha(p):\n"
    " h=hashlib.sha256()\n"
    " if os.path.isdir(p):\n"
    "  for root,dirs,files in os.walk(p):\n"
    "   dirs.sort()\n"
    "   for name in sorted(files):\n"
    "    q=os.path.join(root,name); h.update(os.path.relpath(q,p).encode()+b'\\0')\n"
    "    with open(q,'rb') as f:\n"
    "     for c in iter(lambda:f.read(1048576),b''): h.update(c)\n"
    " else:\n"
    "  with open(p,'rb') as f:\n"
    "   for c in iter(lambda:f.read(1048576),b''): h.update(c)\n"
    " return h.hexdigest()\n"
    "ps=json.loads(sys.argv[1])\n"
    "out=[]\n"
    "for p in ps:\n"
    " try:\n"
    "  m=json.load(open(p['marker']))\n"
    "  ok=(m.get('kind')=='rollout_completion' and "
    "m.get('schema_version')==1 and m.get('status')=='complete' and "
    "m.get('materialized_inputs_sha256')==sha(p['materialized_inputs']) and "
    "m.get('successes_sha256')==sha(p['successes']) and "
    "m.get('materialized_input_count')==m.get('success_count') and "
    "m.get('materialized_input_count',0)>0 and "
    "(p.get('artifacts') is None or m.get('artifacts_sha256')=="
    "{k:sha(v) for k,v in sorted(p['artifacts'].items())}) and "
    "(p.get('behavior_sha256') is None or "
    "m.get('behavior_sha256') in [p.get('behavior_sha256')]+p.get('accepted_behavior_sha256',[])))\n"
    "  if ok and p.get('shard_manifest'):\n"
    "   s=json.load(open(p['shard_manifest']))\n"
    "   ok=(s.get('kind')=='shard_manifest' and s.get('schema_version')==1 and "
    "s.get('source_sha256')==sha(p['source']) and "
    "s.get('rollout_config_sha256') in "
    "[p.get('manifest_behavior_sha256')]+p.get('accepted_manifest_behavior_sha256',[]))\n"
    "  if ok and p.get('behavior_manifest') and p.get('reference_completion'):\n"
    "   b=json.load(open(p['behavior_manifest']))\n"
    "   c=sha(p['reference_completion'])\n"
    "   ok=any(b=={'kind':'behavior_manifest','schema_version':1,'behavior_sha256':x,"
    "'source_completion_sha256':c} for x in "
    "[p.get('manifest_behavior_sha256',p.get('behavior_sha256'))]"
    "+p.get('accepted_manifest_behavior_sha256',[]))\n"
    " except Exception: ok=False\n"
    " out.append(ok)\n"
    "print('__NVFLOW_COMPLETION__:'+json.dumps(out,separators=(',',':')))"
)


def _marker_response(
    output: str,
    marker: str,
    *,
    allowed: set[str],
    operation: str,
) -> str:
    responses = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if line.startswith(marker):
            responses.append(line.removeprefix(marker).strip())
    if len(responses) != 1 or responses[0] not in allowed:
        raise RuntimeError(f"Target filesystem returned an invalid {operation} result")
    return responses[0]


def launcher_is_remote(cluster_config: dict[str, Any]) -> bool:
    """Return whether target paths require the configured SSH tunnel."""
    return "ssh_tunnel" in cluster_config


class NemoSkillsTargetFilesystem:
    """Cached target filesystem for one NeMo-Skills launch configuration."""

    def __init__(self, cluster_config: dict[str, Any]) -> None:
        self.cluster_config = cluster_config
        self.remote = launcher_is_remote(cluster_config)
        self._exists_cache: dict[str, bool] = {}
        self._nonempty_cache: dict[str, bool] = {}
        self._listing_cache: dict[tuple[str, str], tuple[str, ...]] = {}

    def _tunnel(self):
        from nemo_skills.pipeline.utils.cluster import get_tunnel

        return get_tunnel(self.cluster_config)

    def _unmounted(self, container_path: str) -> str:
        from nemo_skills.pipeline.utils import get_unmounted_path

        return str(get_unmounted_path(self.cluster_config, container_path))

    def _host_path(self, container_path: str) -> Path:
        return Path(self._unmounted(container_path))

    def exists(self, path: str) -> bool:
        """Return whether one worker-visible path exists."""
        normalized = str(path)
        cached = self._exists_cache.get(normalized)
        if cached is not None:
            return cached
        return self.batch_exists([normalized])[normalized]

    def batch_exists(self, paths: list[str]) -> dict[str, bool]:
        """Return path existence, querying only uncached paths in bounded batches."""
        requested = list(dict.fromkeys(str(path) for path in paths))
        if not requested:
            return {}

        uncached = [path for path in requested if path not in self._exists_cache]
        if uncached:
            if self.remote:
                discovered = self._remote_batch_exists(uncached)
            else:
                discovered = {path: self._host_path(path).exists() for path in uncached}
            self._exists_cache.update(discovered)
        return {path: self._exists_cache[path] for path in requested}

    def _remote_batch_exists(self, paths: list[str]) -> dict[str, bool]:
        tunnel = self._tunnel()
        host_paths = {path: self._unmounted(path) for path in paths}
        discovered = remote_batch_exists(tunnel, list(host_paths.values()))
        return {path: discovered[host_path] for path, host_path in host_paths.items()}

    def is_nonempty_file(self, path: str) -> bool:
        """Return whether a regular file exists and contains data."""
        normalized = str(path)
        if normalized in self._nonempty_cache:
            return self._nonempty_cache[normalized]

        if not self.remote:
            host_path = self._host_path(normalized)
            result = host_path.is_file() and host_path.stat().st_size > 0
        else:
            host_path = shlex.quote(self._unmounted(normalized))
            marker = "__NVFLOW_NONEMPTY__:"
            output = (
                self._tunnel()
                .run(
                    f"if [ -f {host_path} ] && [ -s {host_path} ]; "
                    f'then echo "{marker}Y"; else echo "{marker}N"; fi',
                    hide=True,
                    warn=True,
                )
                .stdout
            )
            result = (
                _marker_response(
                    output,
                    marker,
                    allowed={"Y", "N"},
                    operation=f"file probe for {normalized}",
                )
                == "Y"
            )
        self._nonempty_cache[normalized] = result
        if result:
            self._exists_cache[normalized] = True
        return result

    def rm(self, paths: list[str]) -> None:
        """Remove files and invalidate cached filesystem observations."""
        normalized = list(dict.fromkeys(str(path) for path in paths))
        if not normalized:
            return
        if not self.remote:
            for path in normalized:
                self._host_path(path).unlink(missing_ok=True)
        else:
            tunnel = self._tunnel()
            batch_size = 40
            for start in range(0, len(normalized), batch_size):
                group = normalized[start : start + batch_size]
                quoted = " ".join(shlex.quote(self._unmounted(path)) for path in group)
                marker = "__NVFLOW_RM__:"
                output = tunnel.run(
                    f'rm -f {quoted} >/dev/null 2>&1 && echo "{marker}OK" || echo "{marker}ERROR"',
                    hide=True,
                    warn=True,
                ).stdout
                response = _marker_response(
                    output,
                    marker,
                    allowed={"OK", "ERROR"},
                    operation="remove",
                )
                if response == "ERROR":
                    raise RuntimeError("Target filesystem could not remove one or more paths")
        for path in normalized:
            self._exists_cache[path] = False
            self._nonempty_cache[path] = False
        self._listing_cache.clear()

    def ls(self, directory: str, pattern: str) -> list[str]:
        """Return sorted basenames matching a controlled glob."""
        normalized = str(directory)
        cache_key = (normalized, pattern)
        cached = self._listing_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        if not self.remote:
            host_directory = self._host_path(normalized)
            directory_exists = host_directory.exists()
            entries = (
                sorted(path.name for path in host_directory.glob(pattern))
                if directory_exists
                else []
            )
        else:
            host_directory = shlex.quote(self._unmounted(normalized))
            marker = "__NVFLOW_LS__:"
            entry_marker = "__NVFLOW_ENTRY__:"
            output = (
                self._tunnel()
                .run(
                    f'if [ ! -e {host_directory} ]; then echo "{marker}MISSING"; '
                    f'elif [ ! -d {host_directory} ]; then echo "{marker}NOT_DIRECTORY"; '
                    f'elif [ ! -r {host_directory} ]; then echo "{marker}UNREADABLE"; '
                    f"elif cd {host_directory} 2>/dev/null; then "
                    f'for item in {pattern}; do [ -e "$item" ] && '
                    f'printf "{entry_marker}%s\\n" "$item"; done; '
                    f'echo "{marker}OK"; '
                    f'else echo "{marker}UNREADABLE"; fi',
                    hide=True,
                    warn=True,
                )
                .stdout
            )
            response = _marker_response(
                output,
                marker,
                allowed={"OK", "MISSING", "NOT_DIRECTORY", "UNREADABLE"},
                operation=f"directory listing for {normalized}",
            )
            if response == "MISSING":
                directory_exists = False
                entries = []
            elif response != "OK":
                raise RuntimeError(
                    f"Target filesystem cannot list {normalized}: {response.lower()}"
                )
            else:
                directory_exists = True
                entries = sorted(
                    line.strip().removeprefix(entry_marker)
                    for line in output.splitlines()
                    if line.strip().startswith(entry_marker)
                )

        self._listing_cache[cache_key] = tuple(entries)
        self._exists_cache[normalized] = directory_exists
        return list(entries)

    def count_lines(self, path: str) -> int:
        """Count local file lines; callers avoid this for remote targets."""
        host_path = self._host_path(str(path))
        if not host_path.exists():
            return 0
        with host_path.open() as source:
            return sum(1 for _ in source)

    def batch_completion_matches(
        self,
        probes: list[dict[str, Any]],
    ) -> list[bool]:
        """Validate content-hashed completion evidence without mutating it."""
        if not probes:
            return []
        path_fields = {
            "materialized_inputs",
            "successes",
            "marker",
            "behavior_manifest",
            "reference_completion",
            "source",
            "shard_manifest",
        }
        host_probes = []
        for probe in probes:
            host_probe = {
                key: (self._unmounted(value) if key in path_fields and value is not None else value)
                for key, value in probe.items()
            }
            if probe.get("artifacts"):
                host_probe["artifacts"] = {
                    name: self._unmounted(path) for name, path in probe["artifacts"].items()
                }
            host_probes.append(host_probe)

        if not self.remote:
            from nvflow.integrations.nemo_gym.native_sharding import (
                validate_rollout_completion,
            )

            results = []
            for probe in host_probes:
                try:
                    validate_rollout_completion(
                        Path(probe["materialized_inputs"]),
                        Path(probe["successes"]),
                        Path(probe["marker"]),
                        probe["behavior_sha256"],
                        artifacts={
                            name: Path(path)
                            for name, path in (probe.get("artifacts") or {}).items()
                        }
                        or None,
                        source_path=Path(probe["source"]) if probe.get("source") else None,
                        shard_manifests=(
                            [Path(probe["shard_manifest"])] if probe.get("shard_manifest") else ()
                        ),
                        manifest_behavior_sha256=probe.get("manifest_behavior_sha256"),
                        accepted_behavior_sha256=probe.get("accepted_behavior_sha256") or (),
                        accepted_manifest_behavior_sha256=(
                            probe.get("accepted_manifest_behavior_sha256") or ()
                        ),
                    )
                    manifest_path = probe.get("behavior_manifest")
                    reference_path = probe.get("reference_completion")
                    if manifest_path and reference_path:
                        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
                        digest = hashlib.sha256(Path(reference_path).read_bytes()).hexdigest()
                        behaviors = [
                            probe.get("manifest_behavior_sha256", probe["behavior_sha256"]),
                            *(probe.get("accepted_manifest_behavior_sha256") or ()),
                        ]
                        if not any(
                            manifest
                            == {
                                "kind": "behavior_manifest",
                                "schema_version": 1,
                                "behavior_sha256": behavior,
                                "source_completion_sha256": digest,
                            }
                            for behavior in behaviors
                        ):
                            raise ValueError("behavior manifest mismatch")
                except (OSError, ValueError):
                    results.append(False)
                else:
                    results.append(True)
            return results

        output = (
            self._tunnel()
            .run(
                f"python3 -c {shlex.quote(_COMPLETION_PROBE_SCRIPT)} "
                f"{shlex.quote(json.dumps(host_probes, separators=(',', ':')))}",
                hide=True,
                warn=True,
            )
            .stdout
        )
        payloads = [
            line.removeprefix("__NVFLOW_COMPLETION__:")
            for line in output.splitlines()
            if line.startswith("__NVFLOW_COMPLETION__:")
        ]
        if len(payloads) != 1:
            raise RuntimeError("Target filesystem returned invalid completion evidence")
        result = json.loads(payloads[0])
        if (
            not isinstance(result, list)
            or len(result) != len(probes)
            or not all(isinstance(value, bool) for value in result)
        ):
            raise RuntimeError("Target filesystem returned invalid completion evidence")
        return result


class LazyTargetFilesystem:
    """Create a target filesystem only when a live stage first uses it."""

    def __init__(self, factory: Callable[[], NemoSkillsTargetFilesystem]) -> None:
        self._factory = factory
        self._resolved: NemoSkillsTargetFilesystem | None = None

    def _get(self) -> NemoSkillsTargetFilesystem:
        if self._resolved is None:
            self._resolved = self._factory()
        return self._resolved

    @property
    def remote(self) -> bool:
        return self._get().remote

    def exists(self, path: str) -> bool:
        return self._get().exists(path)

    def batch_exists(self, paths: list[str]) -> dict[str, bool]:
        return self._get().batch_exists(paths)

    def is_nonempty_file(self, path: str) -> bool:
        return self._get().is_nonempty_file(path)

    def rm(self, paths: list[str]) -> None:
        self._get().rm(paths)

    def ls(self, directory: str, pattern: str) -> list[str]:
        return self._get().ls(directory, pattern)

    def count_lines(self, path: str) -> int:
        return self._get().count_lines(path)

    def batch_completion_matches(
        self,
        probes: list[dict[str, str | None]],
    ) -> list[bool]:
        return self._get().batch_completion_matches(probes)


def remote_batch_exists(tunnel: Any, paths: list[str]) -> dict[str, bool]:
    """Check remote host paths in bounded, safely quoted shell batches."""
    result: dict[str, bool] = {}
    batch_size = 40
    for start in range(0, len(paths), batch_size):
        group = paths[start : start + batch_size]
        commands = []
        for index, path in enumerate(group):
            host_path = shlex.quote(path)
            commands.append(
                f'if [ -e {host_path} ]; then echo "{index}:Y"; else echo "{index}:N"; fi'
            )
        output = tunnel.run("; ".join(commands), hide=True, warn=True).stdout
        parsed: dict[int, bool] = {}
        for raw_line in output.splitlines():
            key, separator, value = raw_line.strip().partition(":")
            response = value.strip()
            if separator and key.isdigit() and response in {"Y", "N"}:
                parsed[int(key)] = response == "Y"
        if set(parsed) != set(range(len(group))):
            raise RuntimeError("Target filesystem returned an incomplete path probe result")
        result.update({path: parsed[index] for index, path in enumerate(group)})
    return result


def check_remote_mount_directories(
    directories: list[str] | str,
    cluster_config: dict[str, Any] | None,
    exit_on_failure: bool = True,
) -> None:
    """Batch NeMo-Skills mount checks while preserving failure semantics."""
    if cluster_config is None:
        raise ValueError("Cluster config is not provided.")
    requested = [directories] if isinstance(directories, str) else list(directories)
    if not requested or cluster_config.get("executor") != "slurm":
        return

    from nemo_skills.pipeline.utils.cluster import get_tunnel
    from nemo_skills.pipeline.utils.mounts import get_unmounted_path, is_mounted_filepath

    host_paths = [
        (
            str(get_unmounted_path(cluster_config, path))
            if is_mounted_filepath(cluster_config, path)
            else str(path)
        )
        for path in requested
    ]
    tunnel = get_tunnel(cluster_config)
    borrowed = _BORROWED_TUNNELS.get()
    if borrowed is not None and all(existing is not tunnel for existing in borrowed):
        borrowed.append(tunnel)
    try:
        discovered = remote_batch_exists(tunnel, host_paths)
    finally:
        if borrowed is None:
            tunnel.cleanup()
    missing = [path for path in host_paths if not discovered[path]]
    if missing and exit_on_failure:
        details = "\n".join(f"{path} DOES NOT exist at source destination" for path in missing)
        raise FileNotFoundError(
            "Some files or directories do not exist at the source location for mounting !!"
            f"\n\n{details}"
        )


@contextmanager
def submission_tunnel_scope() -> Iterator[None]:
    """Retain borrowed cached tunnels until one workflow submission completes."""
    tunnels: list[Any] = []
    token = _BORROWED_TUNNELS.set(tunnels)
    try:
        yield
    finally:
        _BORROWED_TUNNELS.reset(token)
        for tunnel in tunnels:
            tunnel.cleanup()


# Compatibility name used by existing project integrations.
LauncherFS = NemoSkillsTargetFilesystem


def target_filesystem(cluster: str) -> NemoSkillsTargetFilesystem:
    """Create one cached filesystem for a selected NeMo-Skills target."""
    from nvflow.integrations.nemo_skills import load_target_config

    return NemoSkillsTargetFilesystem(load_target_config(cluster))


def lazy_target_filesystem(cluster: str) -> LazyTargetFilesystem:
    """Return a filesystem proxy that defers target configuration loading."""
    return LazyTargetFilesystem(lambda: target_filesystem(cluster))


__all__ = [
    "CONTAINER_CODE_DIR",
    "LazyTargetFilesystem",
    "check_remote_mount_directories",
    "remote_batch_exists",
    "LauncherFS",
    "NemoSkillsTargetFilesystem",
    "launcher_is_remote",
    "lazy_target_filesystem",
    "submission_tunnel_scope",
    "target_filesystem",
]
