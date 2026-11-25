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
"""NeMo-Gym environment configuration helpers."""

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from yaml.nodes import MappingNode

from nvflow.integrations.nemo_gym.contracts import (
    GymContractError,
    GymEnvironmentSpec,
    require_contract_version,
    require_identity,
)
from nvflow.integrations.nemo_gym.native_contract import NativeGymOutputLayout

VLLM_MODEL_FOR_TRAINING = "responses_api_models/vllm_model/configs/vllm_model_for_training.yaml"


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""

    def construct_mapping(self, node: MappingNode, deep: bool = False) -> dict[Any, Any]:
        self.flatten_mapping(node)
        mapping: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                hash(key)
            except TypeError as exc:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found an unhashable key",
                    key_node.start_mark,
                ) from exc
            if key in mapping:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


@dataclass(frozen=True, slots=True)
class EnvironmentCatalog:
    """One deterministic, explicitly indexed project environment catalog."""

    environments: dict[str, dict[str, Any]]
    digest: str
    files: tuple[Path, ...]


def _catalog_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        value = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise GymContractError(f"unable to read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GymContractError(f"{label} {path} must contain a YAML mapping")
    if not all(isinstance(key, str) and key for key in value):
        raise GymContractError(f"{label} {path} keys must be non-empty strings")
    return value


def _require_catalog_path(path: Path, boundary: Path, label: str) -> None:
    try:
        path.relative_to(boundary)
    except ValueError as exc:
        raise GymContractError(
            f"{label} resolves outside catalog boundary {boundary}: {path}"
        ) from exc


def load_environment_catalog(
    index_path: str | Path,
    *,
    boundary: str | Path | None = None,
) -> EnvironmentCatalog:
    """Load an explicit environment-file index without scanning directories."""
    index = Path(index_path).expanduser().resolve()
    catalog_root = index.parent
    resolved_boundary = Path(boundary).expanduser().resolve() if boundary else catalog_root
    _require_catalog_path(index, resolved_boundary, "environment catalog index")

    index_payload = _catalog_mapping(index, "environment catalog index")
    unknown_index_fields = sorted(set(index_payload) - {"environments"})
    if unknown_index_fields:
        raise GymContractError(
            "unknown environment catalog index field(s): " + ", ".join(unknown_index_fields)
        )
    references = index_payload.get("environments")
    if not isinstance(references, Mapping) or not references:
        raise GymContractError(
            "environment catalog index requires a non-empty environments mapping"
        )

    environments: dict[str, dict[str, Any]] = {}
    sources = [index]
    digest_entries: list[dict[str, Any]] = []
    for identity, reference in references.items():
        require_identity(identity, "environment catalog identity")
        if not isinstance(reference, str) or not reference:
            raise GymContractError(
                f"environment catalog entry {identity!r} must name a non-empty relative YAML file"
            )
        relative = PurePosixPath(reference)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.suffix not in {".yaml", ".yml"}
        ):
            raise GymContractError(
                f"environment catalog entry {identity!r} must stay within the catalog "
                "directory and name a .yaml or .yml file"
            )
        environment_path = (catalog_root / Path(*relative.parts)).resolve()
        _require_catalog_path(environment_path, catalog_root, f"environment {identity!r}")
        _require_catalog_path(environment_path, resolved_boundary, f"environment {identity!r}")
        payload = _catalog_mapping(environment_path, f"environment {identity!r}")
        declared_identity = payload.pop("environment", None)
        if declared_identity != identity:
            raise GymContractError(
                f"environment file {environment_path} declares {declared_identity!r}; "
                f"catalog key is {identity!r}"
            )
        if not payload:
            raise GymContractError(f"environment {identity!r} has no configuration")
        environments[identity] = payload
        sources.append(environment_path)
        digest_entries.append(
            {
                "environment": identity,
                "file": relative.as_posix(),
                "config": payload,
            }
        )

    try:
        encoded = json.dumps(
            {"schema_version": 1, "environments": digest_entries},
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    except (TypeError, ValueError) as exc:
        raise GymContractError(
            "environment catalog must contain finite JSON-compatible data"
        ) from exc
    return EnvironmentCatalog(
        environments=environments,
        digest=f"sha256:{hashlib.sha256(encoded).hexdigest()}",
        files=tuple(sources),
    )


def resolve_environments(config: dict[str, Any]) -> dict[str, Any]:
    """Return the enabled environments to process based on the ``_environment`` filter.

    ``config["_environment"]`` may be a single name (``str``), a list of
    names (``list[str]``), or ``None``.  When set, only the matching
    entries are returned (preserving order).  When ``None``, all
    environments from ``config["environments"]`` are returned.  Either way,
    environments with ``enabled: false`` are left out.

    Raises ``ValueError`` if a requested environment doesn't exist.
    """
    envs = config.get("environments", {})
    if not envs:
        raise ValueError("'environments' dict is required but missing from config")
    selected = config.get("_environment")
    names = list(envs)
    if selected:
        names = [selected] if isinstance(selected, str) else list(selected)
        for name in names:
            if name not in envs:
                available = ", ".join(envs.keys())
                raise ValueError(f"Unknown environment '{name}'. Available: {available}")
    resolved = {}
    for name in names:
        environment = envs[name]
        enabled = environment.get("enabled", True) if isinstance(environment, Mapping) else True
        if not isinstance(enabled, bool):
            raise GymContractError(f"environment {name!r}.enabled must be a boolean")
        if enabled:
            resolved[name] = environment
    return resolved


def training_environments(config: dict[str, Any]) -> dict[str, Any]:
    """Return the selected environments that have training data.

    When exactly one remains it trains alone, and its ``training`` and ``eval``
    mappings apply.
    """
    if not config.get("environments"):
        return {}
    return {
        name: environment
        for name, environment in resolve_environments(config).items()
        if environment.get("raw_train_data")
    }


def resolve_scoped_environments(config: dict[str, Any]) -> dict[str, Any]:
    """Like ``resolve_environments`` but tolerant of out-of-scope selections.

    Intended for stages whose ``environments`` is intentionally a *subset* of
    the workflow's full benchmark catalog (e.g. ``prefetch_cache`` scoped to the
    agentic benchmark only).  When the ``_environment`` filter targets names
    outside this stage's scope -- e.g. a QA-only ``-e secque`` run -- those
    names are simply dropped (intersection), so the stage processes nothing
    instead of raising ``ValueError``.

    Unlike ``resolve_environments``, this never raises on an unknown name: it
    returns the intersection of the requested names with this stage's
    ``environments`` (preserving the selection order), or all environments when
    no filter is set.
    """
    envs = config.get("environments", {})
    selected = config.get("_environment")
    if not selected:
        return envs
    names = [selected] if isinstance(selected, str) else list(selected)
    return {name: envs[name] for name in names if name in envs}


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-merge mappings; ``None`` explicitly deletes a key and lists replace."""
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if value is None:
            merged.pop(key, None)
        elif isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def selected_environments(
    config: Mapping[str, Any],
    *,
    require_input: bool = True,
) -> list[GymEnvironmentSpec]:
    """Return enabled, explicitly bound environments after ``-e`` selection."""
    require_contract_version(config.get("contract_version"))
    configured = config.get("environments")
    if not isinstance(configured, Mapping) or not configured:
        raise GymContractError("environments must be a non-empty mapping")
    selection = config.get("_environment")
    if selection is not None:
        valid_selection = (
            bool(selection)
            if isinstance(selection, str)
            else isinstance(selection, list | tuple)
            and bool(selection)
            and all(isinstance(name, str) and name for name in selection)
        )
        if not valid_selection:
            raise GymContractError("_environment must be a name or sequence of names")
    raw_environments = resolve_environments(dict(config))
    result: list[GymEnvironmentSpec] = []
    for name, value in raw_environments.items():
        if not isinstance(value, Mapping):
            raise GymContractError(f"environment {name!r} must be a mapping")
        obsolete = sorted(
            {"agent_type", "resources_server_name", "resources_server_inner_name"} & set(value)
        )
        if obsolete:
            raise GymContractError(
                f"environment {name!r} must let Gym resolve topology; remove: "
                + ", ".join(obsolete)
            )
        if "agent_config_overlay" in value:
            raise GymContractError(
                f"environment {name!r}.agent_config_overlay was removed; "
                "each Gym job writes its own dataset config, so delete it"
            )
        config_paths = value.get("config_paths") or ()
        if isinstance(config_paths, str) or not isinstance(config_paths, list | tuple):
            raise GymContractError(f"environment {name!r}.config_paths must be a sequence of paths")
        environment = GymEnvironmentSpec(
            name=name,
            input_data=value.get("input_data") or "",
            source_data=value.get("source_data"),
            output_name=name if value.get("output_name") is None else value["output_name"],
            config_paths=tuple(config_paths),
            agent_name=value.get("agent_name") or "",
            models={} if value.get("models") is None else value["models"],
            responses_create_params=(
                {}
                if value.get("responses_create_params") is None
                else value["responses_create_params"]
            ),
            gym={} if value.get("gym") is None else value["gym"],
            blade={} if value.get("blade") is None else value["blade"],
        )
        if require_input and not environment.input_data:
            raise GymContractError(f"environment {name!r} requires input_data")
        result.append(environment)
    outputs = [environment.output_name for environment in result]
    if len(outputs) != len(set(outputs)):
        raise GymContractError("selected environments must have unique output_name values")
    return result


def environment_payload(
    source: Mapping[str, Any],
    environment: GymEnvironmentSpec,
) -> dict[str, Any]:
    """Preserve project fields while applying the minimal Gym binding."""
    raw = dict((source.get("environments") or {})[environment.name])
    raw["enabled"] = environment.enabled
    raw["input_data"] = environment.input_data
    if environment.source_data:
        raw["source_data"] = environment.source_data
    raw["output_name"] = environment.output_name
    raw["config_paths"] = list(environment.config_paths)
    raw["agent_name"] = environment.agent_name
    raw["models"] = dict(environment.models)
    raw["responses_create_params"] = dict(environment.responses_create_params)
    raw["gym"] = dict(environment.gym)
    return raw


def collect_target_config(
    config: Mapping[str, Any],
    environment: GymEnvironmentSpec,
) -> dict[str, Any]:
    """Resolve one declarative environment into the native rollout envelope."""
    raw_rollout = config.get("rollout")
    if not isinstance(raw_rollout, Mapping):
        raise GymContractError("rollout must be a mapping")
    base_rollout = dict(raw_rollout)
    configured_models = base_rollout.get("models")
    raw_base_models = {} if configured_models is None else configured_models
    if not isinstance(raw_base_models, Mapping):
        raise GymContractError("rollout.models must be a mapping")
    base_models = dict(raw_base_models)
    unknown_roles = sorted((set(base_models) | set(environment.models)) - {"policy", "judge"})
    if unknown_roles:
        raise GymContractError(f"unknown model role(s): {', '.join(unknown_roles)}")
    configured_responses = base_rollout.get("responses_create_params")
    base_responses = {} if configured_responses is None else configured_responses
    if not isinstance(base_responses, Mapping):
        raise GymContractError("rollout.responses_create_params must be a mapping")
    merged_models: dict[str, Any] = {}
    for role in ("policy", "judge"):
        configured_base = base_models.get(role)
        configured_override = environment.models.get(role)
        base_model = {} if configured_base is None else configured_base
        override = {} if configured_override is None else configured_override
        if role in base_models or role in environment.models:
            if not isinstance(base_model, Mapping) or not isinstance(override, Mapping):
                raise GymContractError(
                    f"environment {environment.name!r}.models.{role} must be a mapping"
                )
            merged_models[role] = deep_merge(base_model, override)

    target = {
        key: copy.deepcopy(value)
        for key, value in config.items()
        if key not in {"contract_version", "extensions"}
    }
    target["output_dir"] = f"{str(config['output_dir']).rstrip('/')}/{environment.output_name}"
    raw_environment = environment_payload(config, environment)
    gym = dict(raw_environment.get("gym") or {})
    if gym.get("observability_enabled") or "model_call_capture_dir" in gym:
        gym["model_call_capture_dir"] = (
            NativeGymOutputLayout.collection(target["output_dir"]).initial_outputs_dir
            / "captures"
            / "model-calls"
        ).as_posix()
        raw_environment["gym"] = gym
    target["environments"] = {environment.name: raw_environment}
    target["rollout"] = {
        **base_rollout,
        "models": merged_models,
        "responses_create_params": {
            **dict(base_responses),
            **dict(environment.responses_create_params),
        },
        "input_data": environment.input_data,
    }
    configured_blade = config.get("blade")
    raw_blade = {} if configured_blade is None else configured_blade
    if not isinstance(raw_blade, Mapping):
        raise GymContractError("blade must be a mapping")
    blade = dict(raw_blade)
    if environment.blade:
        blade = deep_merge(blade, environment.blade)
    target["blade"] = blade
    return target


__all__ = [
    "EnvironmentCatalog",
    "VLLM_MODEL_FOR_TRAINING",
    "collect_target_config",
    "deep_merge",
    "environment_payload",
    "load_environment_catalog",
    "resolve_environments",
    "resolve_scoped_environments",
    "selected_environments",
    "training_environments",
]
