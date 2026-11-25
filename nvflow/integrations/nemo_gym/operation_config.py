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
"""Dependency-light parsing for native NeMo Gym operation configuration."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import PurePosixPath
from typing import Any

from nvflow.integrations.model_servers.serve import SERVER_ENV_NAMES

from .contracts import (
    ModelEndpoint,
    NativeRolloutConfigError,
    RewardConfig,
    RolloutConfig,
)
from .native_contract import NativeGymOutputLayout
from .reward_profiles import parse_dataset_selection_spec

_MODEL_CONFIGS = {
    "vllm_model": "responses_api_models/vllm_model/configs/vllm_model.yaml",
    "local_vllm_model": "responses_api_models/local_vllm_model/configs/local_vllm_model.yaml",
    "openai_model": "responses_api_models/openai_model/configs/openai_model.yaml",
    "inference_provider": (
        "responses_api_models/inference_provider/configs/inference_provider.yaml"
    ),
}
_DEPLOYMENT_MODEL_TYPES = {
    "nvflow": {"vllm_model"},
    "external": {"vllm_model", "openai_model", "inference_provider"},
    "gym": {"local_vllm_model"},
}
_DEFAULT_MODEL_TYPE = {
    "nvflow": "vllm_model",
    "external": "vllm_model",
    "gym": "local_vllm_model",
}
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def _mapping(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise NativeRolloutConfigError(f"'{path}' must be a mapping")
    return dict(value)


def parse_server_environment(value: object, path: str) -> dict[str, Any]:
    """Return a server's ``vllm_serve_env_vars``: variable names to strings or numbers."""
    environment = _mapping(value, path)
    invalid = sorted(str(name) for name in environment if not _ENV_NAME.fullmatch(str(name)))
    if invalid:
        raise NativeRolloutConfigError(f"'{path}' has invalid variable names: {', '.join(invalid)}")
    if SERVER_ENV_NAMES in environment:
        raise NativeRolloutConfigError(f"'{path}' can't set {SERVER_ENV_NAMES}; NVFlow sets it")
    for name, item in environment.items():
        if isinstance(item, bool):
            raise NativeRolloutConfigError(
                f"'{path}.{name}' must be a string or number, not bool; quote it to keep the text"
            )
        if not isinstance(item, str | int | float):
            raise NativeRolloutConfigError(
                f"'{path}.{name}' must be a string or number, not {type(item).__name__}"
            )
    return environment


def _positive_int(value: object, path: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise NativeRolloutConfigError(f"'{path}' must be a {qualifier} integer")
    return value


def _integer(value: object, path: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise NativeRolloutConfigError(f"'{path}' must be an integer >= {minimum}")
    return value


def _optional_string(value: object, path: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise NativeRolloutConfigError(f"'{path}' must be a non-empty string")
    return value


def _selected_environment(
    config: Mapping[str, Any],
    *,
    operation: str,
) -> dict[str, Any]:
    environments = _mapping(config.get("environments"), "environments")
    if len(environments) != 1:
        raise NativeRolloutConfigError(f"{operation} planning requires exactly one environment")
    return _mapping(next(iter(environments.values())), "environments.<name>")


def _parse_blade_command(config: Mapping[str, Any]) -> str | None:
    blade = _mapping(config.get("blade", {}), "blade")
    unknown = sorted(set(blade) - {"enabled", "command"})
    if unknown:
        raise NativeRolloutConfigError(f"unknown blade field(s): {', '.join(unknown)}")
    enabled = blade.get("enabled", False)
    if not isinstance(enabled, bool):
        raise NativeRolloutConfigError("'blade.enabled' must be a boolean")
    command = _optional_string(blade.get("command"), "blade.command")
    if enabled and command is None:
        raise NativeRolloutConfigError("'blade.command' is required when 'blade.enabled' is true")
    return command if enabled else None


def parse_model_endpoint(
    role: str,
    raw: object,
    *,
    path_prefix: str = "rollout.models",
) -> ModelEndpoint:
    """Parse one model endpoint without importing execution dependencies."""
    path = f"{path_prefix}.{role}"
    value = _mapping(raw, path)
    allowed = {
        "deployment",
        "model",
        "model_type",
        "base_url",
        "api_key_env",
        "server_type",
        "container",
        "num_gpus",
        "server_nodes",
        "gym_model",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise NativeRolloutConfigError(f"unknown {path} field(s): {', '.join(unknown)}")

    deployment = value.get("deployment")
    valid_deployments = {"external", "gym", "nvflow"}
    if deployment not in valid_deployments:
        raise NativeRolloutConfigError(
            f"'{path}.deployment' must be one of: {', '.join(sorted(valid_deployments))}"
        )
    model = _optional_string(value.get("model"), f"{path}.model")
    if model is None:
        raise NativeRolloutConfigError(f"'{path}.model' is required")
    model_type = _optional_string(
        value.get("model_type", _DEFAULT_MODEL_TYPE[deployment]),
        f"{path}.model_type",
    )
    assert model_type is not None
    if model_type not in _DEPLOYMENT_MODEL_TYPES[deployment]:
        raise NativeRolloutConfigError(
            f"model type {model_type!r} is incompatible with {role} deployment {deployment!r}"
        )

    base_url = _optional_string(value.get("base_url"), f"{path}.base_url")
    api_key_env = _optional_string(value.get("api_key_env"), f"{path}.api_key_env")
    if api_key_env is not None and not _ENV_NAME.fullmatch(api_key_env):
        raise NativeRolloutConfigError(f"'{path}.api_key_env' must be a shell variable name")
    server_type = _optional_string(value.get("server_type"), f"{path}.server_type")
    container = _optional_string(value.get("container"), f"{path}.container")
    num_gpus = _positive_int(
        value.get("num_gpus", 0),
        f"{path}.num_gpus",
        allow_zero=True,
    )
    server_nodes = _positive_int(
        value.get("server_nodes", 1),
        f"{path}.server_nodes",
    )
    gym_model = _mapping(value.get("gym_model", {}), f"{path}.gym_model")
    vllm_serve_kwargs = _mapping(
        gym_model.get("vllm_serve_kwargs", {}),
        f"{path}.gym_model.vllm_serve_kwargs",
    )
    vllm_serve_env_vars = parse_server_environment(
        gym_model.get("vllm_serve_env_vars", {}),
        f"{path}.gym_model.vllm_serve_env_vars",
    )

    if model_type in {"openai_model", "inference_provider"} and deployment != "external":
        raise NativeRolloutConfigError(f"model type {model_type} requires deployment=external")
    if deployment == "gym":
        if container is not None:
            raise NativeRolloutConfigError(
                f"Gym-deployed {role} uses the workflow container and cannot define container"
            )
        server_type = server_type or "vllm"
        if server_type != "vllm":
            raise NativeRolloutConfigError("deployment=gym supports only server_type=vllm")
        if base_url is not None:
            raise NativeRolloutConfigError(f"Gym-deployed {role} cannot define base_url")
        if num_gpus == 0 or server_nodes != 1:
            raise NativeRolloutConfigError(
                f"Gym-deployed {role} currently requires num_gpus > 0 and server_nodes=1"
            )
        uses_reasoning_parser = gym_model.get("uses_reasoning_parser", False)
        if not isinstance(uses_reasoning_parser, bool):
            raise NativeRolloutConfigError(
                f"'{path}.gym_model.uses_reasoning_parser' must be a boolean"
            )
        has_reasoning_parser = bool(vllm_serve_kwargs.get("reasoning_parser"))
        if uses_reasoning_parser != has_reasoning_parser:
            raise NativeRolloutConfigError(
                f"Gym-deployed {role} must set gym_model.uses_reasoning_parser="
                f"{str(has_reasoning_parser).lower()} to match "
                "gym_model.vllm_serve_kwargs.reasoning_parser"
            )
        if "tensor_parallel_size" not in vllm_serve_kwargs:
            raise NativeRolloutConfigError(
                f"Gym-deployed {role} requires gym_model.vllm_serve_kwargs.tensor_parallel_size"
            )
        if role == "judge":
            required_model_fields = {
                "return_token_id_information",
                "uses_reasoning_parser",
                "vllm_serve_env_vars",
            }
            missing_model_fields = sorted(required_model_fields - gym_model.keys())
            required_serve_fields = {
                "data_parallel_size",
                "pipeline_parallel_size",
            }
            missing_serve_fields = sorted(required_serve_fields - vllm_serve_kwargs.keys())
            if missing_model_fields or missing_serve_fields:
                missing = missing_model_fields + [
                    f"vllm_serve_kwargs.{field}" for field in missing_serve_fields
                ]
                raise NativeRolloutConfigError(
                    "Gym-deployed judge gym_model is incomplete; missing: " + ", ".join(missing)
                )
            strategy = vllm_serve_env_vars.get("VLLM_RAY_DP_PACK_STRATEGY")
            if not isinstance(strategy, str) or not strategy:
                raise NativeRolloutConfigError(
                    "Gym-deployed judge requires a non-empty "
                    "gym_model.vllm_serve_env_vars.VLLM_RAY_DP_PACK_STRATEGY"
                )
        topology = 1
        for field_name in (
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "data_parallel_size",
        ):
            topology *= _positive_int(
                vllm_serve_kwargs.get(field_name, 1),
                f"{path}.gym_model.vllm_serve_kwargs.{field_name}",
            )
        if topology != num_gpus:
            raise NativeRolloutConfigError(
                f"Gym-deployed {role} topology reserves {topology} GPUs but "
                f"{path}.num_gpus is {num_gpus}"
            )
        gym_model = {**gym_model, "vllm_serve_kwargs": vllm_serve_kwargs}
    elif deployment == "external":
        if base_url is None:
            raise NativeRolloutConfigError(f"deployment=external {role} requires base_url")
        if num_gpus or vllm_serve_kwargs or server_nodes != 1:
            raise NativeRolloutConfigError(
                f"external {role} cannot reserve GPUs or define vllm_serve_kwargs"
            )
        if server_type or container:
            raise NativeRolloutConfigError(
                f"external {role} cannot define server_type or container"
            )
    else:
        if base_url is not None:
            raise NativeRolloutConfigError(f"NVFlow-deployed {role} cannot define base_url")
        server_type = server_type or "vllm"
        if server_type not in {"vllm", "sglang"}:
            raise NativeRolloutConfigError(
                f"NVFlow-deployed {role} server_type must be vllm or sglang"
            )
        if num_gpus == 0:
            raise NativeRolloutConfigError(f"NVFlow-deployed {role} requires num_gpus > 0")

    return ModelEndpoint(
        role=role,
        deployment=deployment,
        model=model,
        model_type=model_type,
        base_url=base_url,
        api_key_env=api_key_env,
        server_type=server_type,
        container=container,
        num_gpus=num_gpus,
        server_nodes=server_nodes,
        gym_model=gym_model,
    )


def parse_rollout_config(
    config: Mapping[str, Any],
    *,
    enrich_module: str | None = None,
    report_module: str | None = None,
    report_options: Mapping[str, Any] | None = None,
) -> RolloutConfig:
    """Parse the native rollout contract for one selected environment."""
    for removed, replacement in (
        ("profile_selection", "dataset_selection"),
        ("profile_selection_container", "dataset_selection_container"),
    ):
        if removed in config:
            raise NativeRolloutConfigError(f"'{removed}' was removed; use '{replacement}'")
    if "postprocess_container" in config:
        raise NativeRolloutConfigError(
            "'postprocess_container' was removed; use 'dataset_selection_container'"
        )
    if "filter_container" in config:
        raise NativeRolloutConfigError(
            "'filter_container' was removed; use 'dataset_selection_container'"
        )
    if "prepare_data_dir" in config:
        raise NativeRolloutConfigError(
            "'prepare_data_dir' was removed; set environments.<name>.input_data"
        )
    if "filter" in config:
        raise NativeRolloutConfigError(
            "'filter' was removed; use the shared 'dataset_selection' contract"
        )
    rollout = _mapping(config.get("rollout"), "rollout")
    removed = sorted(
        {
            "num_random_seeds",
            "starting_seed",
            "rerun_done",
            "policy_vllm",
            "judge_vllm",
        }
        & rollout.keys()
    )
    if removed:
        raise NativeRolloutConfigError(
            "legacy Gym rollout fields are unsupported; migrate: " + ", ".join(removed)
        )
    allowed = {
        "input_data",
        "allow_empty_input",
        "num_repeats",
        "num_chunks",
        "dependent_jobs",
        "max_num_samples",
        "num_samples_in_parallel",
        "responses_create_params",
        "models",
    }
    unknown = sorted(set(rollout) - allowed)
    if unknown:
        raise NativeRolloutConfigError(f"unknown rollout field(s): {', '.join(unknown)}")

    models = _mapping(rollout.get("models"), "rollout.models")
    if set(models) - {"policy", "judge"} or "policy" not in models:
        raise NativeRolloutConfigError(
            "'rollout.models' requires policy and accepts only policy/judge roles"
        )
    policy = parse_model_endpoint("policy", models["policy"])
    judge = parse_model_endpoint("judge", models["judge"]) if "judge" in models else None

    env_config = _selected_environment(config, operation="native rollout")
    agent_name = _optional_string(
        env_config.get("agent_name"),
        "environments.<name>.agent_name",
    )
    if agent_name is None:
        raise NativeRolloutConfigError("'environments.<name>.agent_name' is required")
    environment_paths = list(env_config.get("config_paths") or [])
    config_paths = tuple(
        dict.fromkeys([_MODEL_CONFIGS[policy.model_type or ""], *environment_paths])
    )
    aggregation_config_paths = tuple(
        dict.fromkeys([_MODEL_CONFIGS["vllm_model"], *environment_paths])
    )

    input_data = _optional_string(rollout.get("input_data"), "rollout.input_data")
    if input_data is None:
        raise NativeRolloutConfigError("'rollout.input_data' is required")
    output_dir = _optional_string(config.get("output_dir"), "output_dir")
    client_container = _optional_string(config.get("container"), "container")
    if output_dir is None or client_container is None:
        raise NativeRolloutConfigError("'output_dir' and 'container' are required")
    dataset_selection_container = (
        _optional_string(
            config.get("dataset_selection_container"),
            "dataset_selection_container",
        )
        or client_container
    )
    vllm_container = _optional_string(
        config.get("vllm_container"),
        "vllm_container",
    )
    if vllm_container is None and any(
        endpoint is not None and endpoint.uses_workflow_vllm_container
        for endpoint in (policy, judge)
    ):
        raise NativeRolloutConfigError(
            "'vllm_container' is required for an NVFlow-deployed vLLM endpoint"
        )
    installation_command = _optional_string(
        config.get("installation_command"),
        "installation_command",
    )

    max_num_samples_raw = rollout.get("max_num_samples")
    max_num_samples = (
        None
        if max_num_samples_raw is None
        else _positive_int(max_num_samples_raw, "rollout.max_num_samples")
    )
    concurrency_raw = rollout.get("num_samples_in_parallel")
    concurrency = (
        None
        if concurrency_raw is None
        else _positive_int(concurrency_raw, "rollout.num_samples_in_parallel")
    )
    responses = _mapping(
        rollout.get("responses_create_params", {}),
        "rollout.responses_create_params",
    )
    num_repeats = _positive_int(
        rollout.get("num_repeats"),
        "rollout.num_repeats",
    )
    dataset_selection_raw = _mapping(
        config.get("dataset_selection", {}),
        "dataset_selection",
    )
    dataset_selection = (
        asdict(parse_dataset_selection_spec(dataset_selection_raw)) if dataset_selection_raw else {}
    )
    if num_repeats < 2 and dataset_selection:
        raise NativeRolloutConfigError("dataset selection requires 'rollout.num_repeats' >= 2")
    allow_empty_input = rollout.get("allow_empty_input", False)
    if not isinstance(allow_empty_input, bool):
        raise NativeRolloutConfigError("'rollout.allow_empty_input' must be true or false")
    blade_command = _parse_blade_command(config)
    if allow_empty_input and (dataset_selection or blade_command):
        raise NativeRolloutConfigError(
            "'rollout.allow_empty_input' cannot be combined with dataset selection or blade"
        )
    gym_overrides = _mapping(
        env_config.get("gym", {}),
        "environments.<name>.gym",
    )
    unsupported_cardinality = sorted({"agent_map", "fan_out", "num_repeats"} & gym_overrides.keys())
    if unsupported_cardinality:
        raise NativeRolloutConfigError(
            "native rollout cardinality is owned by rollout.num_repeats and "
            "one selected agent; remove Gym override(s): " + ", ".join(unsupported_cardinality)
        )
    return RolloutConfig(
        output_dir=output_dir,
        input_data=input_data,
        client_container=client_container,
        dataset_selection_container=dataset_selection_container,
        vllm_container=vllm_container,
        installation_command=installation_command,
        config_paths=config_paths,
        aggregation_config_paths=aggregation_config_paths,
        agent_name=agent_name,
        num_repeats=num_repeats,
        num_chunks=_positive_int(
            rollout.get("num_chunks", 1),
            "rollout.num_chunks",
        ),
        dependent_jobs=_positive_int(
            rollout.get("dependent_jobs", 0),
            "rollout.dependent_jobs",
            allow_zero=True,
        ),
        max_num_samples=max_num_samples,
        concurrency=concurrency,
        responses_create_params=responses,
        gym_overrides=gym_overrides,
        policy=policy,
        judge=judge,
        enrich_module=enrich_module,
        report_module=report_module,
        blade_command=blade_command,
        dataset_selection=dataset_selection,
        allow_empty_input=allow_empty_input,
        report_options=dict(report_options or {}),
    )


def parse_reward_config(
    config: Mapping[str, Any],
    *,
    report_module: str | None = None,
    report_options: Mapping[str, Any] | None = None,
) -> RewardConfig:
    """Parse the reward-recomputation contract for one selected environment."""
    for removed, replacement in (
        ("profile_selection", "dataset_selection"),
        ("profile_selection_container", "dataset_selection_container"),
    ):
        if removed in config:
            raise NativeRolloutConfigError(f"'{removed}' was removed; use '{replacement}'")
    if "postprocess_container" in config:
        raise NativeRolloutConfigError(
            "'postprocess_container' was removed; use 'dataset_selection_container'"
        )
    if "filter_container" in config:
        raise NativeRolloutConfigError(
            "'filter_container' was removed; use 'dataset_selection_container'"
        )
    if "filter" in config:
        raise NativeRolloutConfigError(
            "'filter' was removed; use the shared 'dataset_selection' contract"
        )
    rejudge = _mapping(config.get("rejudge"), "rejudge")
    allowed = {
        "mode",
        "num_samples_in_parallel",
        "dependent_jobs",
        "unique_prompts",
        "models",
    }
    unknown = sorted(set(rejudge) - allowed)
    if unknown:
        raise NativeRolloutConfigError(f"unknown rejudge field(s): {', '.join(unknown)}")
    mode = rejudge.get("mode")
    if mode not in {"native", "fallback"}:
        raise NativeRolloutConfigError("'rejudge.mode' must be 'native' or 'fallback'")
    unique_prompts = rejudge.get("unique_prompts", False)
    if not isinstance(unique_prompts, bool):
        raise NativeRolloutConfigError("'rejudge.unique_prompts' must be true or false")
    if unique_prompts and mode != "fallback":
        raise NativeRolloutConfigError(
            "'rejudge.unique_prompts' applies to fallback mode, where NVFlow sends each answer"
        )
    models = _mapping(rejudge.get("models"), "rejudge.models")
    if set(models) != {"judge"}:
        raise NativeRolloutConfigError("'rejudge.models' must contain exactly the judge role")
    judge_config = _mapping(models["judge"], "rejudge.models.judge")
    if judge_config.get("reuse") == "policy":
        raise NativeRolloutConfigError("reward recomputation requires a dedicated judge endpoint")
    judge = parse_model_endpoint(
        "judge",
        judge_config,
        path_prefix="rejudge.models",
    )

    env_config = _selected_environment(config, operation="reward")
    agent_name = _optional_string(
        env_config.get("agent_name"),
        "environments.<name>.agent_name",
    )
    if agent_name is None:
        raise NativeRolloutConfigError("'environments.<name>.agent_name' is required")
    paths = [_MODEL_CONFIGS[judge.model_type or ""]]
    paths.extend(env_config.get("config_paths") or [])
    source_dir = _optional_string(config.get("rollouts_dir"), "rollouts_dir")
    output_dir = _optional_string(config.get("output_dir"), "output_dir")
    client_container = _optional_string(config.get("container"), "container")
    if source_dir is None or output_dir is None or client_container is None:
        raise NativeRolloutConfigError("'rollouts_dir', 'output_dir', and 'container' are required")
    dataset_selection_container = (
        _optional_string(
            config.get("dataset_selection_container"),
            "dataset_selection_container",
        )
        or client_container
    )
    vllm_container = _optional_string(
        config.get("vllm_container"),
        "vllm_container",
    )
    if vllm_container is None and judge.uses_workflow_vllm_container:
        raise NativeRolloutConfigError(
            "'vllm_container' is required for an NVFlow-deployed vLLM endpoint"
        )
    installation_command = _optional_string(
        config.get("installation_command"),
        "installation_command",
    )
    if PurePosixPath(source_dir) == PurePosixPath(output_dir):
        raise NativeRolloutConfigError("rejudge output_dir must differ from rollouts_dir")
    source_layout = NativeGymOutputLayout.collection(source_dir)
    layout = NativeGymOutputLayout.rejudge(output_dir)
    source = source_layout.analyzed_artifacts
    artifacts = layout.initial_artifacts

    dataset_selection_raw = _mapping(
        config.get("dataset_selection", {}),
        "dataset_selection",
    )
    dataset_source = dataset_selection_raw.pop("source", None)
    if dataset_source is not None and (not isinstance(dataset_source, str) or not dataset_source):
        raise NativeRolloutConfigError("'dataset_selection.source' must be a path")
    if dataset_source is not None and not dataset_selection_raw:
        raise NativeRolloutConfigError(
            "'dataset_selection.source' requires an explicit selection policy"
        )
    dataset_selection = (
        asdict(parse_dataset_selection_spec(dataset_selection_raw)) if dataset_selection_raw else {}
    )
    if dataset_selection and dataset_source is None:
        raise NativeRolloutConfigError(
            "'dataset_selection.source' is required for rejudged dataset selection"
        )
    return RewardConfig(
        output_dir=output_dir,
        source=source,
        artifacts=artifacts,
        analyzed_artifacts=layout.analyzed_artifacts,
        behavior_manifest=layout.initial_behavior_manifest,
        rejudge_marker=layout.initial_artifacts.completion,
        client_container=client_container,
        dataset_selection_container=dataset_selection_container,
        vllm_container=vllm_container,
        installation_command=installation_command,
        config_paths=tuple(dict.fromkeys(paths)),
        agent_name=agent_name,
        mode=mode,
        concurrency=_integer(
            rejudge.get("num_samples_in_parallel", 64),
            "rejudge.num_samples_in_parallel",
            minimum=1,
        ),
        dependent_jobs=_integer(
            rejudge.get("dependent_jobs", 0),
            "rejudge.dependent_jobs",
            minimum=0,
        ),
        unique_prompts=unique_prompts,
        judge=judge,
        gym_overrides=_mapping(
            env_config.get("gym", {}),
            "environments.<name>.gym",
        ),
        report_module=report_module,
        blade_command=_parse_blade_command(config),
        dataset_source=dataset_source,
        source_responses_create_params=_mapping(
            config.get("source_responses_create_params", {}),
            "source_responses_create_params",
        ),
        dataset_selection=dataset_selection,
        report_options=dict(report_options or {}),
    )


__all__ = [
    "parse_model_endpoint",
    "parse_reward_config",
    "parse_rollout_config",
]
