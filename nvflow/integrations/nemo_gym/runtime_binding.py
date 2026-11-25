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
"""Resolve opaque Gym config paths inside the pinned Gym runtime.

This is an internal worker adapter, not an authored workflow configuration
surface.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import re
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_RUNTIME_MODULE = "nvflow.integrations.nemo_gym.runtime_binding"
# Base64 hides request paths from the engines' code-root rendering, so the worker
# substitutes its own NVFLOW_CODE_ROOT after decoding. The worker runs in the Gym
# container on the standard library alone, so it keeps its own copy of the token.
_CODE_ROOT_TOKEN = "{{NVFLOW_CODE_ROOT}}"
_MISSING_INTERPOLATION = re.compile(r"Interpolation key '([^']+)' not found")
_MAX_DISCOVERY_PLACEHOLDERS = 32


class GymBindingError(ValueError):
    """The composed Gym config cannot provide one unambiguous binding."""


def _discovery_placeholder(key: str) -> str:
    """Return an inert scalar for topology-only Gym config composition."""

    return "http://127.0.0.1:1/v1" if key.casefold().endswith("base_url") else ""


@dataclass(frozen=True, slots=True)
class GymRuntimeBinding:
    """Runtime identities derived from Gym's fully composed configuration."""

    agent_name: str
    agent_implementation: str
    resources_server_instance: str | None
    resources_server_implementation: str | None
    has_judge_model_server: bool


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GymBindingError(f"{label} must be a mapping")
    return value


def _identity(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise GymBindingError(f"{label} must be a non-empty string")
    return value


def _implementation(instance: Mapping[str, Any], server_type: str, label: str) -> str:
    implementations = _mapping(instance.get(server_type), f"{label}.{server_type}")
    if len(implementations) != 1:
        raise GymBindingError(f"{label}.{server_type} must contain exactly one implementation")
    return _identity(next(iter(implementations)), f"{label} implementation")


def binding_from_resolved_config(
    resolved: Mapping[str, Any],
    *,
    agent_name: str,
) -> GymRuntimeBinding:
    """Follow one selected agent's route in a Gym-resolved config mapping."""
    selected_value = resolved.get(agent_name)
    if not isinstance(selected_value, Mapping) or not isinstance(
        selected_value.get("responses_api_agents"), Mapping
    ):
        available = sorted(
            str(name)
            for name, value in resolved.items()
            if isinstance(value, Mapping) and isinstance(value.get("responses_api_agents"), Mapping)
        )
        suffix = f"; available: {', '.join(available)}" if available else ""
        raise GymBindingError(f"unknown Gym agent instance {agent_name!r}{suffix}")
    selected_agent = selected_value
    agent_implementation = _implementation(
        selected_agent,
        "responses_api_agents",
        f"Gym agent instance {agent_name!r}",
    )
    agent_config = _mapping(
        _mapping(
            selected_agent["responses_api_agents"],
            f"Gym agent instance {agent_name!r}.responses_api_agents",
        )[agent_implementation],
        f"Gym agent {agent_name!r}/{agent_implementation!r}",
    )

    available_resources = {
        str(name): value
        for name, value in resolved.items()
        if isinstance(value, Mapping) and isinstance(value.get("resources_servers"), Mapping)
    }
    resource_reference = agent_config.get("resources_server")
    resources_server_instance: str | None = None
    if isinstance(resource_reference, Mapping) and resource_reference.get("name"):
        resources_server_instance = _identity(
            resource_reference["name"],
            f"Gym agent {agent_name!r} resources_server.name",
        )

    resources_server_implementation: str | None = None
    has_judge_model_server = False
    if resources_server_instance is not None:
        resource_instance = available_resources.get(resources_server_instance)
        if resource_instance is None:
            raise GymBindingError(
                f"Gym agent {agent_name!r} references unknown resources-server instance "
                f"{resources_server_instance!r}"
            )
        resources_server_implementation = _implementation(
            resource_instance,
            "resources_servers",
            f"Gym resources-server instance {resources_server_instance!r}",
        )
        resource_config = _mapping(
            _mapping(
                resource_instance["resources_servers"],
                f"Gym resources-server instance {resources_server_instance!r}.resources_servers",
            )[resources_server_implementation],
            "Gym resources server "
            f"{resources_server_instance!r}/{resources_server_implementation!r}",
        )
        has_judge_model_server = "judge_model_server" in resource_config

    return GymRuntimeBinding(
        agent_name=agent_name,
        agent_implementation=agent_implementation,
        resources_server_instance=resources_server_instance,
        resources_server_implementation=resources_server_implementation,
        has_judge_model_server=has_judge_model_server,
    )


def _resolved_gym_config(
    request: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Use Gym's parser lazily so importing NVFlow never requires Gym locally."""
    try:
        from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
        from omegaconf import OmegaConf
    except ImportError as exc:  # pragma: no cover - exercised only in runtime images
        raise GymBindingError(
            "runtime Gym binding resolution requires NeMo Gym in the selected container"
        ) from exc

    config_paths = request["config_paths"]
    # Binding inspection happens before managed model endpoints exist. Supply
    # inert model instances so Gym can validate agent/resource references while
    # still using its real composition and inheritance rules.
    runtime_models: dict[str, Any] = {}
    judge_model_name = request.get("judge_model_name")
    if isinstance(judge_model_name, str) and judge_model_name != "policy_model":
        runtime_models[judge_model_name] = {
            "responses_api_models": {
                "dummy_model": {
                    "entrypoint": "app.py",
                }
            }
        }
    config_overrides = request.get("config_overrides", {})
    if not isinstance(config_overrides, Mapping):
        raise GymBindingError("config_overrides must be a mapping")
    initial = OmegaConf.merge(
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        runtime_models,
        {
            "config_paths": config_paths,
            # Unrelated runtime-only leaves may remain intentionally incomplete;
            # the actual Gym command performs strict validation immediately after.
            "error_on_almost_servers": False,
        },
        config_overrides,
    )
    unresolved: set[str] = set()
    discovery_overrides = OmegaConf.create({})
    for _ in range(_MAX_DISCOVERY_PLACEHOLDERS + 1):
        try:
            resolved = GlobalConfigDictParser().parse(
                GlobalConfigDictParserConfig(
                    initial_global_config_dict=initial,
                    skip_load_from_cli=True,
                    offline=True,
                )
            )
            break
        except Exception as exc:
            match = _MISSING_INTERPOLATION.search(str(exc))
            key = match.group(1) if match else None
            if key is None or key in unresolved or len(unresolved) >= _MAX_DISCOVERY_PLACEHOLDERS:
                raise GymBindingError(f"unable to resolve Gym config_paths: {exc}") from exc
            placeholder = _discovery_placeholder(key)
            OmegaConf.update(
                initial,
                key,
                placeholder,
                merge=True,
                force_add=True,
            )
            # Persist only otherwise-unbound leaves in the generated overlay.
            # Gym must compose the same provider config again in its actors,
            # where even unused optional services are interpolation-checked.
            # Managed runtime endpoints still override their actual model
            # routes below; these inert values merely make unused leaves valid.
            OmegaConf.update(
                discovery_overrides,
                key,
                placeholder,
                merge=True,
                force_add=True,
            )
            unresolved.add(key)
    else:  # pragma: no cover - loop always breaks or raises
        raise GymBindingError("unable to resolve Gym config_paths")
    value = OmegaConf.to_container(resolved, resolve=False)
    placeholders = OmegaConf.to_container(discovery_overrides, resolve=False)
    return (
        _mapping(value, "resolved Gym configuration"),
        _mapping(placeholders, "Gym discovery overrides"),
    )


def _merge_overlay(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge generated leaves without mutating either input."""
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _merge_overlay(current, value)
        else:
            merged[key] = value
    return merged


def _atomic_text(path: str, content: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, destination)


def _overlay(
    binding: GymRuntimeBinding,
    *,
    datasets: object,
    judge_model_name: object,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if datasets:
        if not isinstance(datasets, list) or not all(
            isinstance(item, Mapping) for item in datasets
        ):
            raise GymBindingError("datasets must be a sequence of mappings")
        payload[binding.agent_name] = {
            "responses_api_agents": {
                binding.agent_implementation: {
                    "datasets": datasets,
                }
            }
        }
    if judge_model_name is not None:
        model_name = _identity(judge_model_name, "judge_model_name")
        if binding.has_judge_model_server:
            assert binding.resources_server_instance is not None
            assert binding.resources_server_implementation is not None
            payload.setdefault(binding.resources_server_instance, {}).setdefault(
                "resources_servers",
                {},
            ).setdefault(binding.resources_server_implementation, {})["judge_model_server"] = {
                "name": model_name
            }
    if not payload and judge_model_name is None:
        raise GymBindingError("runtime binding overlay requires datasets or a judge model")
    return payload


def materialize_runtime_binding(request: Mapping[str, Any]) -> GymRuntimeBinding:
    """Resolve Gym topology and atomically write its generated overlay and evidence."""
    config_paths = request.get("config_paths")
    if (
        not isinstance(config_paths, list)
        or not config_paths
        or not all(isinstance(path, str) and path for path in config_paths)
    ):
        raise GymBindingError("config_paths must be a non-empty sequence of paths")
    agent_name = _identity(request.get("agent_name"), "agent_name")
    resolved, discovery_overrides = _resolved_gym_config(request)
    binding = binding_from_resolved_config(
        resolved,
        agent_name=agent_name,
    )
    if request.get("require_resources_server") and not binding.resources_server_instance:
        raise GymBindingError(f"Gym agent {binding.agent_name!r} has no resources-server route")

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - runtime image contract
        raise GymBindingError("runtime Gym binding resolution requires PyYAML") from exc

    overlay = _merge_overlay(
        discovery_overrides,
        _overlay(
            binding,
            datasets=request.get("datasets"),
            judge_model_name=request.get("judge_model_name"),
        ),
    )
    _atomic_text(
        _identity(request.get("output_config"), "output_config"),
        yaml.safe_dump(overlay, sort_keys=False),
    )
    _atomic_text(
        _identity(request.get("output_binding"), "output_binding"),
        json.dumps(asdict(binding), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    output_env = request.get("output_env")
    if output_env is not None:
        values = {
            "NVFLOW_GYM_RESOURCES_SERVER_INSTANCE": binding.resources_server_instance or "",
        }
        _atomic_text(
            _identity(output_env, "output_env"),
            "".join(f"{key}={shlex.quote(value)}\n" for key, value in values.items()),
        )
    return binding


def render_runtime_binding_command(
    *,
    config_paths: Sequence[str | os.PathLike[str]],
    agent_name: str,
    output_config: str | os.PathLike[str],
    output_binding: str | os.PathLike[str],
    datasets: Sequence[Mapping[str, Any]] = (),
    judge_model_name: str | None = None,
    require_resources_server: bool = False,
    output_env: str | os.PathLike[str] | None = None,
    config_overrides: Mapping[str, Any] | None = None,
) -> str:
    """Render the worker-only resolver without importing Gym on the launcher."""
    # Workers run this module inside Gym's venv, which lacks NVFlow's dependencies.
    from nvflow.integrations.nemo_skills.filesystem import CONTAINER_CODE_DIR

    request: dict[str, Any] = {
        "config_paths": [os.fspath(path) for path in config_paths],
        "agent_name": agent_name,
        "output_config": os.fspath(output_config),
        "output_binding": os.fspath(output_binding),
        "datasets": [dict(dataset) for dataset in datasets],
    }
    if judge_model_name is not None:
        request["judge_model_name"] = judge_model_name
    if require_resources_server:
        request["require_resources_server"] = True
    if output_env is not None:
        request["output_env"] = os.fspath(output_env)
    if config_overrides:
        request["config_overrides"] = dict(config_overrides)
    encoded = base64.urlsafe_b64encode(
        json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode()
    ).decode()
    return (
        f'PYTHONPATH="{CONTAINER_CODE_DIR}${{PYTHONPATH:+:$PYTHONPATH}}" '
        f"python3 -m {_RUNTIME_MODULE} --request {shlex.quote(encoded)}"
    )


def _render_code_root(value: Any, code_root: str) -> Any:
    if isinstance(value, str):
        return value.replace(_CODE_ROOT_TOKEN, code_root)
    if isinstance(value, list):
        return [_render_code_root(item, code_root) for item in value]
    if isinstance(value, dict):
        return {key: _render_code_root(item, code_root) for key, item in value.items()}
    return value


def _parse_request(value: str) -> Mapping[str, Any]:
    try:
        decoded = base64.urlsafe_b64decode(value.encode())
        request = json.loads(decoded)
    except (ValueError, binascii.Error) as exc:
        raise GymBindingError("request must be URL-safe base64 encoded JSON") from exc
    code_root = os.environ.get("NVFLOW_CODE_ROOT", "/nemo_run/code")
    return _mapping(_render_code_root(request, code_root), "request")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    try:
        materialize_runtime_binding(_parse_request(args.request))
    except GymBindingError as exc:
        parser.error(str(exc))


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = [
    "GymBindingError",
    "GymRuntimeBinding",
    "binding_from_resolved_config",
    "materialize_runtime_binding",
    "render_runtime_binding_command",
]
