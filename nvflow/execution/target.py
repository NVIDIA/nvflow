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
"""Operator-selected execution target binding."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Literal
from urllib.parse import urlparse

import yaml
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from nvflow.execution.runtime_profile import PYTHON_SPEC, PYTHON_VERSION, RAY_VERSION

ExecutionEngineKind = Literal["slurm", "ray"]
TARGET_CONFIG_DIR_ENV = "NVFLOW_CONFIG_DIR"
_ENV_REFERENCE = re.compile(r"\$\{oc\.env:([A-Za-z_][A-Za-z0-9_]*)\}\Z")
_IDENTITY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


class TargetResolutionError(ValueError):
    """A selected target cannot be normalized into an execution contract."""


def _immutable_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


@dataclass(frozen=True)
class TargetBinding:
    """One concrete cluster selected independently of workflow identity."""

    cluster: str
    source: Literal["override", "workflow"]

    @classmethod
    def resolve(cls, workflow_default: str, override: str | None = None) -> TargetBinding:
        """Resolve explicit operator selection before the workflow default."""
        cluster = override if override is not None else workflow_default
        if not cluster or not cluster.strip():
            raise ValueError("Execution cluster must be a non-empty name")
        return cls(
            cluster=cluster,
            source="override" if override is not None else "workflow",
        )


@dataclass(frozen=True, slots=True)
class RegisteredCluster:
    """One externally owned Ray Jobs cluster declared by an operator target."""

    name: str
    jobs_url: str | None
    controller_jobs_url: str | None = None
    credential_env: str | None = None
    controller_credential_env: str | None = None
    ca_file: str | None = None
    controller_ca_file: str | None = None
    allow_insecure: bool = False
    controller_allow_insecure: bool | None = None
    expected_image: str | None = None
    expected_python: str = PYTHON_VERSION
    expected_ray: str = RAY_VERSION
    gpus_per_node: int | None = None
    labels: Mapping[str, str] = field(default_factory=dict)
    resources: Mapping[str, float] = field(default_factory=dict)
    jobs_url_variable: str | None = None
    # A cluster whose address variable is unset has no URLs; only runs that need
    # it fail, naming this variable.
    unset_variable: str | None = None

    def __post_init__(self) -> None:
        if not _IDENTITY.fullmatch(self.name):
            raise TargetResolutionError(f"Invalid Ray cluster identity: {self.name!r}")
        if self.jobs_url is None and self.unset_variable is None:
            raise TargetResolutionError(f"Ray cluster {self.name!r} requires jobs_url")
        controller_jobs_url = self.controller_jobs_url or self.jobs_url
        controller_credential_env = self.controller_credential_env or self.credential_env
        controller_ca_file = self.controller_ca_file or self.ca_file
        controller_allow_insecure = (
            self.allow_insecure
            if self.controller_allow_insecure is None
            else self.controller_allow_insecure
        )
        object.__setattr__(self, "controller_jobs_url", controller_jobs_url)
        object.__setattr__(
            self,
            "controller_credential_env",
            controller_credential_env,
        )
        object.__setattr__(self, "controller_ca_file", controller_ca_file)
        object.__setattr__(
            self,
            "controller_allow_insecure",
            controller_allow_insecure,
        )
        for field_name, url, allow_insecure in (
            ("jobs_url", self.jobs_url, self.allow_insecure),
            (
                "controller_jobs_url",
                controller_jobs_url,
                controller_allow_insecure,
            ),
        ):
            if url is None:
                continue
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise TargetResolutionError(
                    f"Ray cluster {self.name!r} {field_name} must be an http(s) URL"
                )
            if parsed.username or parsed.password:
                raise TargetResolutionError(
                    f"Ray cluster {self.name!r} {field_name} must not contain credentials"
                )
            if (
                parsed.scheme == "http"
                and not allow_insecure
                and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            ):
                raise TargetResolutionError(
                    f"Ray cluster {self.name!r} requires https or allow_insecure: true"
                )
        if self.credential_env is not None and not self.credential_env.isidentifier():
            raise TargetResolutionError(f"Ray cluster {self.name!r} has invalid credential_env")
        if controller_credential_env is not None and not controller_credential_env.isidentifier():
            raise TargetResolutionError(
                f"Ray cluster {self.name!r} has invalid controller_credential_env"
            )
        if Version(self.expected_python) not in SpecifierSet(PYTHON_SPEC):
            raise TargetResolutionError(
                f"Ray cluster {self.name!r} Python {self.expected_python} is outside {PYTHON_SPEC}"
            )
        if self.expected_ray != RAY_VERSION:
            raise TargetResolutionError(
                f"Ray cluster {self.name!r} must declare Ray {RAY_VERSION}, got {self.expected_ray}"
            )
        if self.gpus_per_node is not None and (
            not isinstance(self.gpus_per_node, int)
            or isinstance(self.gpus_per_node, bool)
            or self.gpus_per_node < 1
        ):
            raise TargetResolutionError(
                f"Ray cluster {self.name!r} gpus_per_node must be a positive integer"
            )
        object.__setattr__(self, "labels", _immutable_mapping(self.labels))
        object.__setattr__(self, "resources", _immutable_mapping(self.resources))


@dataclass(frozen=True, slots=True)
class RayTarget:
    """Supplied-cluster topology consumed by the Ray Jobs engine."""

    control_cluster: str
    artifact_root: str
    clusters: Mapping[str, RegisteredCluster]
    routes: Mapping[str, str]
    ssh_tunnel: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        clusters = dict(self.clusters)
        routes = dict(self.routes)
        if self.ssh_tunnel is not None:
            for key in ("host", "user"):
                value = self.ssh_tunnel.get(key)
                if not isinstance(value, str) or not value:
                    raise TargetResolutionError(f"ssh_tunnel.{key} must be a non-empty string")
            for cluster in clusters.values():
                # A forwarded 127.0.0.1 address cannot pass the endpoint's TLS check.
                if cluster.jobs_url is not None and urlparse(cluster.jobs_url).scheme != "http":
                    raise TargetResolutionError(
                        f"Ray cluster {cluster.name!r} jobs_url must be http to use "
                        "ssh_tunnel; reach https endpoints directly"
                    )
            object.__setattr__(self, "ssh_tunnel", _immutable_mapping(self.ssh_tunnel))
        if self.control_cluster not in clusters:
            raise TargetResolutionError(
                f"Ray control_cluster {self.control_cluster!r} is not declared"
            )
        artifact_root = PurePosixPath(self.artifact_root)
        if not artifact_root.is_absolute() or ".." in artifact_root.parts:
            raise TargetResolutionError("ray.artifact_root must be an absolute safe path")
        for role, cluster in routes.items():
            if not _IDENTITY.fullmatch(role):
                raise TargetResolutionError(f"Invalid Ray runtime role: {role!r}")
            if cluster not in clusters:
                raise TargetResolutionError(
                    f"Ray route {role!r} references unknown cluster {cluster!r}"
                )
        object.__setattr__(self, "clusters", MappingProxyType(clusters))
        object.__setattr__(self, "routes", MappingProxyType(routes))

    @property
    def controller(self) -> RegisteredCluster:
        """Return the externally owned cluster hosting the run controller."""
        return self.clusters[self.control_cluster]

    def cluster_for_role(self, role: str) -> RegisteredCluster:
        """Resolve one exact runtime role without fallback or heuristics."""
        cluster_name = self.routes.get(role)
        if cluster_name is None:
            raise TargetResolutionError(f"No supplied Ray cluster route for runtime role {role!r}")
        return self.clusters[cluster_name]


@dataclass(frozen=True, slots=True)
class ResolvedTarget:
    """One immutable operator target with engine-specific state."""

    binding: TargetBinding
    engine: ExecutionEngineKind
    fingerprint: str
    source_path: Path
    ray: RayTarget | None = None
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.engine == "ray" and self.ray is None:
            raise TargetResolutionError("Ray target is missing supplied-cluster configuration")
        if self.engine == "slurm" and self.ray is not None:
            raise TargetResolutionError("Slurm target cannot contain resolved Ray state")
        object.__setattr__(self, "raw", _immutable_mapping(self.raw))


def load_target_document(path: Path) -> dict[str, Any]:
    """Load one target mapping without importing a scheduler integration."""
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TargetResolutionError(f"Could not read target configuration {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise TargetResolutionError(f"Target configuration is not valid YAML: {path}") from exc
    if not isinstance(value, dict):
        raise TargetResolutionError(f"Target configuration must contain a YAML mapping: {path}")
    return value


def preflight_target_config(
    target: str,
    *,
    environ: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> Path:
    """Locate and validate one named target without importing either engine."""
    if not target.strip():
        raise TargetResolutionError("Target name must not be empty.")
    if target != target.strip():
        raise TargetResolutionError("Target name must not contain surrounding whitespace.")
    if Path(target).name != target or target in {".", ".."}:
        raise TargetResolutionError(
            f"Invalid target name {target!r}; pass a configuration name, not a path."
        )
    environment = os.environ if environ is None else environ
    working_directory = Path.cwd() if cwd is None else cwd
    configured_directory = environment.get(TARGET_CONFIG_DIR_ENV, "").strip()
    config_directory = (
        Path(configured_directory).expanduser()
        if configured_directory
        else working_directory / "cluster_configs"
    )
    if not config_directory.is_absolute():
        config_directory = working_directory / config_directory
    config_path = config_directory / f"{target}.yaml"
    if not config_path.is_file():
        raise TargetResolutionError(
            f"Target configuration {target!r} was not found at {config_path}. "
            f"Create it from cluster_configs/template-slurm.yaml or "
            f"cluster_configs/template-ray.yaml, or set {TARGET_CONFIG_DIR_ENV}. "
            "See docs/operations/cluster-configuration.md."
        )
    load_target_document(config_path)
    return config_path.resolve()


def resolve_target(
    binding: TargetBinding,
    source_path: Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> ResolvedTarget:
    """Normalize a target file into direct-Slurm or supplied-Ray state."""
    environment = os.environ if environ is None else environ
    document = load_target_document(source_path)
    executor = document.get("executor", "slurm")
    if executor not in {"slurm", "ray"}:
        raise TargetResolutionError(
            f"Target {binding.cluster!r} executor must be 'slurm' or 'ray', got {executor!r}"
        )
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"), default=str).encode()
    fingerprint = f"sha256:{hashlib.sha256(canonical).hexdigest()}"
    if executor == "slurm":
        return ResolvedTarget(
            binding=binding,
            engine="slurm",
            fingerprint=fingerprint,
            source_path=source_path.resolve(),
            raw=document,
        )

    ray_value = document.get("ray")
    if not isinstance(ray_value, dict):
        raise TargetResolutionError(f"Ray target {binding.cluster!r} requires a ray mapping")
    control_cluster = ray_value.get("control_cluster")
    artifact_root = ray_value.get("artifact_root")
    clusters_value = ray_value.get("clusters")
    routes_value = ray_value.get("routes")
    if not isinstance(control_cluster, str) or not control_cluster:
        raise TargetResolutionError("ray.control_cluster must be a non-empty string")
    if not isinstance(artifact_root, str) or not artifact_root:
        raise TargetResolutionError("ray.artifact_root must be a non-empty string")
    if not isinstance(clusters_value, dict) or not clusters_value:
        raise TargetResolutionError("ray.clusters must be a non-empty mapping")
    if not isinstance(routes_value, dict) or not routes_value:
        raise TargetResolutionError("ray.routes must be a non-empty mapping")

    clusters: dict[str, RegisteredCluster] = {}
    for name, cluster_value in clusters_value.items():
        if not isinstance(name, str) or not isinstance(cluster_value, dict):
            raise TargetResolutionError("ray.clusters entries must be mappings")
        jobs_url, jobs_url_variable = _environment_reference(
            cluster_value.get("jobs_url"),
            environment,
            f"ray.clusters.{name}.jobs_url",
        )
        controller_jobs_url, controller_jobs_url_variable = (
            (None, None)
            if cluster_value.get("controller_jobs_url") is None
            else _environment_reference(
                cluster_value["controller_jobs_url"],
                environment,
                f"ray.clusters.{name}.controller_jobs_url",
            )
        )
        if jobs_url is None:
            unset_variable = jobs_url_variable
        elif controller_jobs_url is None:
            unset_variable = controller_jobs_url_variable
        else:
            unset_variable = None
        if unset_variable is not None:
            jobs_url = controller_jobs_url = None
        labels = cluster_value.get("labels") or {}
        resources = cluster_value.get("resources") or {}
        if not isinstance(labels, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in labels.items()
        ):
            raise TargetResolutionError(f"ray.clusters.{name}.labels must map strings to strings")
        if not isinstance(resources, dict) or not all(
            isinstance(key, str)
            and isinstance(value, int | float)
            and not isinstance(value, bool)
            and value >= 0
            for key, value in resources.items()
        ):
            raise TargetResolutionError(
                f"ray.clusters.{name}.resources must map strings to non-negative numbers"
            )
        ca_file = _optional_string(cluster_value.get("ca_file"))
        if ca_file is not None:
            ca_path = Path(ca_file).expanduser()
            if not ca_path.is_absolute():
                ca_path = source_path.resolve().parent / ca_path
            ca_file = str(ca_path.resolve())
        controller_ca_file = _optional_string(cluster_value.get("controller_ca_file"))
        if controller_ca_file is not None and not Path(controller_ca_file).is_absolute():
            raise TargetResolutionError(f"ray.clusters.{name}.controller_ca_file must be absolute")
        expected_image = _optional_string(cluster_value.get("expected_image"))
        if expected_image is None:
            raise TargetResolutionError(f"ray.clusters.{name}.expected_image is required")
        gpus_per_node = cluster_value.get("gpus_per_node")
        if gpus_per_node is not None and (
            not isinstance(gpus_per_node, int)
            or isinstance(gpus_per_node, bool)
            or gpus_per_node < 1
        ):
            raise TargetResolutionError(
                f"ray.clusters.{name}.gpus_per_node must be a positive integer"
            )
        clusters[name] = RegisteredCluster(
            name=name,
            jobs_url=jobs_url,
            controller_jobs_url=controller_jobs_url,
            credential_env=_optional_string(cluster_value.get("credential_env")),
            controller_credential_env=_optional_string(
                cluster_value.get("controller_credential_env")
            ),
            ca_file=ca_file,
            controller_ca_file=controller_ca_file,
            allow_insecure=bool(cluster_value.get("allow_insecure", False)),
            controller_allow_insecure=(
                bool(cluster_value["controller_allow_insecure"])
                if "controller_allow_insecure" in cluster_value
                else None
            ),
            expected_image=expected_image,
            expected_python=(
                _optional_string(cluster_value.get("expected_python")) or PYTHON_VERSION
            ),
            expected_ray=_optional_string(cluster_value.get("expected_ray")) or RAY_VERSION,
            gpus_per_node=gpus_per_node,
            labels=labels,
            resources={str(key): float(value) for key, value in resources.items()},
            jobs_url_variable=jobs_url_variable,
            unset_variable=unset_variable,
        )
    if not all(
        isinstance(role, str) and isinstance(cluster, str) for role, cluster in routes_value.items()
    ):
        raise TargetResolutionError("ray.routes must map runtime-role strings to cluster strings")
    ssh_tunnel = document.get("ssh_tunnel")
    if ssh_tunnel is not None and not isinstance(ssh_tunnel, dict):
        raise TargetResolutionError("ssh_tunnel must be a mapping")
    ray_target = RayTarget(
        control_cluster=control_cluster,
        artifact_root=artifact_root,
        clusters=clusters,
        routes=routes_value,
        ssh_tunnel=ssh_tunnel,
    )
    # Transport details (URLs, ssh_tunnel, credential names, CA paths) are excluded
    # so run identity survives reconnecting through another tunnel or launcher.
    ray_identity = {
        "executor": "ray",
        "control_cluster": ray_target.control_cluster,
        "artifact_root": ray_target.artifact_root,
        "clusters": {
            name: {
                "expected_image": cluster.expected_image,
                "expected_python": cluster.expected_python,
                "expected_ray": cluster.expected_ray,
                "gpus_per_node": cluster.gpus_per_node,
                "labels": dict(cluster.labels),
                "resources": dict(cluster.resources),
            }
            for name, cluster in ray_target.clusters.items()
        },
        "routes": dict(ray_target.routes),
    }
    canonical = json.dumps(
        ray_identity,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    fingerprint = f"sha256:{hashlib.sha256(canonical).hexdigest()}"
    return ResolvedTarget(
        binding=binding,
        engine="ray",
        fingerprint=fingerprint,
        source_path=source_path.resolve(),
        ray=ray_target,
        raw=document,
    )


def _environment_reference(
    value: object,
    environment: Mapping[str, str],
    path: str,
) -> tuple[str | None, str | None]:
    """Return the value and the variable it references; the value is None when unset."""
    if not isinstance(value, str) or not value:
        raise TargetResolutionError(f"{path} must be a non-empty string")
    match = _ENV_REFERENCE.fullmatch(value)
    if match is None:
        return value, None
    name = match.group(1)
    return environment.get(name) or None, name


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise TargetResolutionError("Optional target values must be non-empty strings")
    return value
