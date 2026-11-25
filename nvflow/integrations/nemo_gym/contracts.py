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
"""Dependency-light contracts for shared NeMo Gym operations and stages."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath
from typing import Any, Literal

from .native_contract import NativeGymArtifactPaths, NativeGymOutputLayout

GYM_STAGE_CONTRACT_VERSION = 1
_IDENTITY = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*\Z")
# vLLM settings that size, place, or schedule a server without changing what it
# generates, so a resumed run may change them.
RESIZABLE_VLLM_SERVE_KWARGS = frozenset(
    {
        "compilation_config",
        "cpu_offload_gb",
        "cuda_graph_sizes",
        "data_parallel_size",
        "decode_context_parallel_size",
        "disable_log_requests",
        "disable_log_stats",
        "distributed_executor_backend",
        "enable_chunked_prefill",
        "enable_expert_parallel",
        "enable_log_requests",
        "enable_prefix_caching",
        "enforce_eager",
        "gpu_memory_utilization",
        "max_num_batched_tokens",
        "max_num_seqs",
        "max_seq_len_to_capture",
        "no_enable_flashinfer_autotune",
        "pipeline_parallel_size",
        "swap_space",
        "tensor_parallel_size",
    }
)
# Gym settings that only decide how long Gym waits, so a resumed run may change them.
OPERATIONAL_GYM_OVERRIDES = frozenset({"model_endpoint_readiness_timeout_seconds"})


def result_gym_overrides(overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Return the Gym overrides that can change results."""
    return {key: value for key, value in overrides.items() if key not in OPERATIONAL_GYM_OVERRIDES}


class GymContractError(ValueError):
    """Raised when a shared Gym contract is ambiguous or unsafe."""


class NativeRolloutConfigError(GymContractError):
    """Raised when a native rollout configuration is ambiguous or unsafe."""


@dataclass(frozen=True, slots=True)
class GymDatasetRef:
    """One project-selected dataset consumed or produced by Gym."""

    path: str
    name: str = "train"
    split: str = "train"
    license: str = "TBD"

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (self.path, self.name, self.split, self.license)
        ):
            raise GymContractError("Gym dataset fields must be non-empty strings")


@dataclass(frozen=True, slots=True)
class GymBenchmarkSpec:
    """One native Gym benchmark preparation request."""

    benchmark: str
    prepared_file: str
    output_file: str
    prompt_config: str | None
    overrides: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (self.benchmark, self.prepared_file, self.output_file)
        ):
            raise GymContractError(
                "Gym benchmark requires selector, prepared_file, and output_file"
            )
        selector_parts = self.benchmark.split("/")
        if len(selector_parts) > 2 or any(
            not _IDENTITY.fullmatch(part) or part in {".", ".."} for part in selector_parts
        ):
            raise GymContractError(
                "Gym benchmark must be a safe <name> or <name>/<config> selector"
            )
        if self.prompt_config is not None and (
            not isinstance(self.prompt_config, str) or not self.prompt_config
        ):
            raise GymContractError("Gym benchmark prompt_config must be a non-empty string or null")
        if not all(isinstance(value, str) and value for value in self.overrides):
            raise GymContractError("Gym benchmark overrides must be non-empty strings")
        output = PurePosixPath(self.output_file)
        if output.is_absolute() or ".." in output.parts or output.as_posix() in {"", "."}:
            raise GymContractError("Gym benchmark output_file must be a relative path")


@dataclass(frozen=True, slots=True)
class GymEnvironmentSpec:
    """Minimal project binding for one Gym environment and dataset."""

    name: str
    input_data: str
    output_name: str
    config_paths: tuple[str, ...]
    agent_name: str
    source_data: str | None = None
    enabled: bool = True
    models: Mapping[str, Any] = field(default_factory=dict)
    responses_create_params: Mapping[str, Any] = field(default_factory=dict)
    gym: Mapping[str, Any] = field(default_factory=dict)
    blade: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_identity(self.name, "environment name")
        require_identity(self.output_name, "environment output_name")
        if not isinstance(self.input_data, str):
            raise GymContractError(f"environment {self.name!r}.input_data must be a string")
        if not self.config_paths or not all(
            isinstance(path, str) and path for path in self.config_paths
        ):
            raise GymContractError(f"environment {self.name!r} requires config_paths")
        if not isinstance(self.agent_name, str) or not self.agent_name:
            raise GymContractError(f"environment {self.name!r} requires agent_name")
        if self.source_data is not None and (
            not isinstance(self.source_data, str) or not self.source_data
        ):
            raise GymContractError(
                f"environment {self.name!r}.source_data must be a non-empty string or null"
            )
        for label, value in (
            ("models", self.models),
            ("responses_create_params", self.responses_create_params),
            ("gym", self.gym),
            ("blade", self.blade),
        ):
            if not isinstance(value, Mapping):
                raise GymContractError(f"environment {self.name!r}.{label} must be a mapping")


WorkerExtensionKind = Literal["report"]


@dataclass(frozen=True, slots=True)
class WorkerExtensionSpec:
    """Versioned packaged worker module invoked by a shared Gym stage."""

    kind: WorkerExtensionKind
    module: str
    contract_version: int = 1
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind != "report":
            raise GymContractError(f"unsupported Gym worker extension kind: {self.kind!r}")
        if (
            isinstance(self.contract_version, bool)
            or not isinstance(self.contract_version, int)
            or self.contract_version != 1
        ):
            raise GymContractError(
                f"unsupported {self.kind} extension contract_version "
                f"{self.contract_version}; expected 1"
            )
        if not self.module or not all(part.isidentifier() for part in self.module.split(".")):
            raise GymContractError(f"{self.kind} extension module must be a dotted Python module")
        if not isinstance(self.options, Mapping) or not all(
            isinstance(key, str) for key in self.options
        ):
            raise GymContractError(f"{self.kind} extension options must map names to values")


@dataclass(frozen=True, slots=True)
class ModelEndpoint:
    """One Gym model role, independent of who owns its process."""

    role: str
    deployment: str
    model: str | None = None
    model_type: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    server_type: str | None = None
    container: str | None = None
    num_gpus: int = 0
    server_nodes: int = 1
    gym_model: dict[str, Any] = field(default_factory=dict)

    def behavior(self) -> dict[str, Any]:
        """Return the settings that can change results.

        A server's GPUs and nodes, how vLLM sizes and schedules its work, and the
        server's environment are left out, so a resumed run may resize its servers.
        """
        settings = asdict(self)
        for name in ("num_gpus", "server_nodes"):
            settings.pop(name)
        gym_model = settings["gym_model"]
        gym_model.pop("vllm_serve_env_vars", None)
        serve_kwargs = gym_model.get("vllm_serve_kwargs")
        if isinstance(serve_kwargs, Mapping):
            gym_model["vllm_serve_kwargs"] = {
                key: value
                for key, value in serve_kwargs.items()
                if key not in RESIZABLE_VLLM_SERVE_KWARGS
            }
        return settings

    def legacy_behavior(self) -> dict[str, Any]:
        """Return the settings fingerprints covered before resumes could resize servers."""
        return asdict(self)

    @property
    def nvflow_deployed(self) -> bool:
        return self.deployment == "nvflow"

    @property
    def gym_deployed(self) -> bool:
        return self.deployment == "gym"

    @property
    def uses_workflow_vllm_container(self) -> bool:
        """Whether this endpoint uses the workflow's vLLM container key."""
        return self.nvflow_deployed and self.server_type == "vllm" and self.container is None


@dataclass(frozen=True, slots=True)
class RolloutConfig:
    """Normalized native Gym rollout operation."""

    output_dir: str
    input_data: str
    client_container: str
    dataset_selection_container: str
    vllm_container: str | None
    installation_command: str | None
    config_paths: tuple[str, ...]
    aggregation_config_paths: tuple[str, ...]
    agent_name: str
    num_repeats: int
    num_chunks: int
    dependent_jobs: int
    max_num_samples: int | None
    concurrency: int | None
    responses_create_params: dict[str, Any]
    gym_overrides: dict[str, Any]
    policy: ModelEndpoint
    judge: ModelEndpoint | None
    enrich_module: str | None
    report_module: str | None
    blade_command: str | None
    dataset_selection: dict[str, Any]
    allow_empty_input: bool = False
    report_options: dict[str, Any] = field(default_factory=dict)

    @property
    def layout(self) -> NativeGymOutputLayout:
        return NativeGymOutputLayout.collection(self.output_dir)


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Normalized independent reward-recomputation operation."""

    output_dir: str
    source: NativeGymArtifactPaths
    artifacts: NativeGymArtifactPaths
    analyzed_artifacts: NativeGymArtifactPaths
    behavior_manifest: PurePosixPath
    rejudge_marker: PurePosixPath
    client_container: str
    dataset_selection_container: str
    vllm_container: str | None
    installation_command: str | None
    config_paths: tuple[str, ...]
    agent_name: str
    mode: str
    concurrency: int
    dependent_jobs: int
    judge: ModelEndpoint
    gym_overrides: dict[str, Any]
    report_module: str | None
    blade_command: str | None
    dataset_source: str | None
    source_responses_create_params: dict[str, Any]
    dataset_selection: dict[str, Any]
    report_options: dict[str, Any] = field(default_factory=dict)
    # A unique line starts each answer's last user message, so a hosted judge's
    # prompt cache can't return an identical earlier request's entry past it.
    unique_prompts: bool = False

    @property
    def layout(self) -> NativeGymOutputLayout:
        return NativeGymOutputLayout.rejudge(self.output_dir)


def require_contract_version(value: object) -> None:
    """Validate the shared stage contract version."""
    if isinstance(value, bool) or not isinstance(value, int) or value != GYM_STAGE_CONTRACT_VERSION:
        raise GymContractError(
            f"contract_version must be {GYM_STAGE_CONTRACT_VERSION}, got {value!r}"
        )


def require_identity(value: object, label: str) -> None:
    """Reject values unsafe for task IDs and output path components."""
    if (
        not isinstance(value, str)
        or not value
        or not _IDENTITY.fullmatch(value)
        or value in {".", ".."}
    ):
        raise GymContractError(f"{label} must be a safe non-empty identity, got {value!r}")


__all__ = [
    "GYM_STAGE_CONTRACT_VERSION",
    "GymContractError",
    "GymBenchmarkSpec",
    "GymDatasetRef",
    "GymEnvironmentSpec",
    "ModelEndpoint",
    "NativeRolloutConfigError",
    "RewardConfig",
    "RolloutConfig",
    "WorkerExtensionKind",
    "WorkerExtensionSpec",
    "require_contract_version",
    "require_identity",
]
