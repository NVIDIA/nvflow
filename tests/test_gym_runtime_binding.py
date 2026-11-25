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
"""Runtime binding for provider-owned Gym configurations."""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

import pytest
import yaml

from nvflow.integrations.nemo_gym import runtime_binding
from nvflow.integrations.nemo_gym.runtime_binding import (
    GymBindingError,
    binding_from_resolved_config,
    materialize_runtime_binding,
    render_runtime_binding_command,
)
from nvflow.integrations.nemo_skills.filesystem import CONTAINER_CODE_DIR


def _resolved_config() -> dict:
    return {
        "finance_sec_search_resources_server": {
            "resources_servers": {
                "finance_sec_search": {
                    "entrypoint": "app.py",
                    "judge_model_server": {
                        "type": "responses_api_models",
                        "name": "search_judge_model",
                    },
                }
            }
        },
        "finance_agent": {
            "responses_api_agents": {
                "finance_agent": {
                    "entrypoint": "app.py",
                    "resources_server": {
                        "type": "resources_servers",
                        "name": "finance_sec_search_resources_server",
                    },
                }
            }
        },
    }


def test_binding_follows_agent_route_without_filename_conventions() -> None:
    resolved = {
        **_resolved_config(),
        "unrelated": {"resources_servers": {"other": {"entrypoint": "app.py"}}},
    }

    binding = binding_from_resolved_config(
        resolved,
        agent_name="finance_agent",
    )

    assert binding.agent_implementation == "finance_agent"
    assert binding.resources_server_instance == "finance_sec_search_resources_server"
    assert binding.resources_server_implementation == "finance_sec_search"
    assert binding.has_judge_model_server


def test_discovery_placeholders_are_inert_and_url_shaped_when_required() -> None:
    assert runtime_binding._discovery_placeholder("tavily_api_key") == ""
    assert (
        runtime_binding._discovery_placeholder("search_judge_model_base_url")
        == "http://127.0.0.1:1/v1"
    )


def test_binding_does_not_guess_an_unreferenced_resource_route() -> None:
    resolved = _resolved_config()
    resolved["finance_agent"]["responses_api_agents"]["finance_agent"].pop("resources_server")
    resolved["other"] = {"resources_servers": {"other": {"entrypoint": "app.py"}}}

    binding = binding_from_resolved_config(resolved, agent_name="finance_agent")

    assert binding.resources_server_instance is None
    assert binding.resources_server_implementation is None


def test_judge_binding_requires_an_explicit_resource_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved = _resolved_config()
    resolved["finance_agent"]["responses_api_agents"]["finance_agent"].pop("resources_server")
    monkeypatch.setattr(
        "nvflow.integrations.nemo_gym.runtime_binding._resolved_gym_config",
        lambda _request: (resolved, {}),
    )

    with pytest.raises(GymBindingError, match="has no resources-server route"):
        materialize_runtime_binding(
            {
                "config_paths": ["resources_servers/finance/configs/finance.yaml"],
                "agent_name": "finance_agent",
                "output_config": str(tmp_path / "overlay.yaml"),
                "output_binding": str(tmp_path / "binding.json"),
                "judge_model_name": "judge_model",
                "require_resources_server": True,
            }
        )


def test_binding_reports_available_agents() -> None:
    with pytest.raises(
        GymBindingError,
        match="unknown Gym agent instance 'missing'.*available: finance_agent",
    ):
        binding_from_resolved_config(_resolved_config(), agent_name="missing")


def test_materialization_writes_leaf_overlay_and_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "nvflow.integrations.nemo_gym.runtime_binding._resolved_gym_config",
        lambda _request: (
            _resolved_config(),
            {"search_judge_model_base_url": "http://127.0.0.1:1/v1"},
        ),
    )
    overlay = tmp_path / "generated" / "overlay.yaml"
    binding_path = tmp_path / "generated" / "binding.json"
    environment = tmp_path / "generated" / "binding.env"
    request = {
        "config_paths": ["resources_servers/finance/configs/finance.yaml"],
        "agent_name": "finance_agent",
        "output_config": str(overlay),
        "output_binding": str(binding_path),
        "output_env": str(environment),
        "datasets": [
            {
                "name": "nvflow",
                "type": "train",
                "license": "TBD",
                "jsonl_fpath": "/data/train.jsonl",
            }
        ],
        "judge_model_name": "judge_model",
    }

    materialize_runtime_binding(request)

    assert yaml.safe_load(overlay.read_text()) == {
        "search_judge_model_base_url": "http://127.0.0.1:1/v1",
        "finance_agent": {
            "responses_api_agents": {
                "finance_agent": {
                    "datasets": [
                        {
                            "name": "nvflow",
                            "type": "train",
                            "license": "TBD",
                            "jsonl_fpath": "/data/train.jsonl",
                        }
                    ]
                }
            }
        },
        "finance_sec_search_resources_server": {
            "resources_servers": {
                "finance_sec_search": {
                    "judge_model_server": {
                        "name": "judge_model",
                    }
                }
            }
        },
    }
    assert json.loads(binding_path.read_text())["agent_implementation"] == "finance_agent"
    assert (
        "NVFLOW_GYM_RESOURCES_SERVER_INSTANCE=finance_sec_search_resources_server"
        in environment.read_text()
    )


def test_judge_binding_is_noop_for_resource_without_judge_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved = _resolved_config()
    resolved["finance_sec_search_resources_server"]["resources_servers"]["finance_sec_search"].pop(
        "judge_model_server"
    )
    monkeypatch.setattr(
        "nvflow.integrations.nemo_gym.runtime_binding._resolved_gym_config",
        lambda _request: (resolved, {}),
    )
    overlay = tmp_path / "overlay.yaml"

    binding = materialize_runtime_binding(
        {
            "config_paths": ["resources_servers/finance/configs/finance.yaml"],
            "agent_name": "finance_agent",
            "output_config": str(overlay),
            "output_binding": str(tmp_path / "binding.json"),
            "judge_model_name": "policy_model",
        }
    )

    assert not binding.has_judge_model_server
    assert yaml.safe_load(overlay.read_text()) == {}


def test_rendered_resolver_is_worker_only_and_shell_safe(tmp_path: Path) -> None:
    command = render_runtime_binding_command(
        config_paths=("resources_servers/example/configs/example.yaml",),
        agent_name="example_agent",
        output_config=tmp_path / "overlay with spaces.yaml",
        output_binding=tmp_path / "binding.json",
        datasets=(
            {
                "name": "train",
                "type": "train",
                "license": "TBD",
                "jsonl_fpath": "/data/train.jsonl",
            },
        ),
        config_overrides={"service": {"api_key": "configured"}},
    )

    assert "nvflow.integrations.nemo_gym.runtime_binding" in command
    assert f'PYTHONPATH="{CONTAINER_CODE_DIR}${{PYTHONPATH:+:$PYTHONPATH}}"' in command
    assert "overlay with spaces.yaml" not in command
    request = runtime_binding._parse_request(shlex.split(command)[-1])
    assert request["config_overrides"] == {"service": {"api_key": "configured"}}


def test_worker_resolves_code_root_paths_hidden_by_the_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = render_runtime_binding_command(
        config_paths=("{{NVFLOW_CODE_ROOT}}/configs/example.yaml",),
        agent_name="example_agent",
        output_config="/out/overlay.yaml",
        output_binding="/out/binding.json",
    )
    received: list = []
    monkeypatch.setattr(runtime_binding, "materialize_runtime_binding", received.append)
    monkeypatch.setattr(sys, "argv", ["runtime_binding", *shlex.split(command)[-2:]])
    monkeypatch.setenv("NVFLOW_CODE_ROOT", "/ray/working_dir")

    runtime_binding.main()

    assert received[0]["config_paths"] == ["/ray/working_dir/configs/example.yaml"]
