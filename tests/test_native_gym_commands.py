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
"""Tests for the minimal native Gym command renderer."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import pytest

from nvflow.integrations.nemo_gym.native_commands import (
    render_gym_command,
    shell_variable,
)


def test_render_gym_command_composes_native_config_and_overrides() -> None:
    rendered = render_gym_command(
        "gym",
        "eval",
        "run",
        config_paths=("agent.yaml", "sandbox provider.yaml"),
        args=("--input", "/data/input.jsonl", "--output", "/data/rollouts.jsonl"),
        overrides={"num_repeats_add_seed": True, "route_failures_to_sidecar": True},
    )

    assert shlex.split(rendered) == [
        "gym",
        "eval",
        "run",
        "--config",
        "agent.yaml",
        "--config",
        "sandbox provider.yaml",
        "--input",
        "/data/input.jsonl",
        "--output",
        "/data/rollouts.jsonl",
        "+num_repeats_add_seed=true",
        "+route_failures_to_sidecar=true",
    ]


def test_mapping_overrides_render_deterministically() -> None:
    first = render_gym_command("gym", "eval", "run", overrides={"z": 1, "a": 2})
    second = render_gym_command("gym", "eval", "run", overrides={"a": 2, "z": 1})

    assert first == second == "gym eval run +a=2 +z=1"


@pytest.mark.requires_nemo
def test_string_overrides_remain_strings_and_accept_hydra_metacharacters() -> None:
    from hydra.core.override_parser.overrides_parser import OverridesParser

    message = 'Compare the assistant\'s answer: [[rating]], {"strict": true}'
    rendered = render_gym_command(
        "gym",
        "eval",
        "run",
        overrides={
            "judge.message": message,
            "judge.generation_config": '{"temperature":0.0}',
        },
    )
    arguments = shlex.split(rendered)
    parsed = {
        override.key_or_group: override.value()
        for override in OverridesParser.create().parse_overrides(arguments[3:])
    }

    assert parsed == {
        "judge.message": message,
        "judge.generation_config": '{"temperature":0.0}',
    }


def test_shell_variable_is_explicit_and_narrow() -> None:
    rendered = render_gym_command(
        "gym",
        "eval",
        "run",
        args=("--model-url", shell_variable("NVFLOW_POLICY_URL")),
        overrides={"judge_model.base_url": shell_variable("NVFLOW_JUDGE_URL")},
    )
    completed = subprocess.run(
        ["sh", "-c", f"gym() {{ printf '%s\\n' \"$@\"; }}\n{rendered}"],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "NVFLOW_POLICY_URL": "https://policy.example/v1?x=1&y=2",
            "NVFLOW_JUDGE_URL": "https://judge.example/v1",
        },
    )

    assert completed.stdout.splitlines() == [
        "eval",
        "run",
        "--model-url",
        "https://policy.example/v1?x=1&y=2",
        "+judge_model.base_url=https://judge.example/v1",
    ]
    with pytest.raises(ValueError, match="invalid shell variable"):
        shell_variable("URL:-fallback")


def test_literal_values_cannot_inject_shell(tmp_path: Path) -> None:
    marker = tmp_path / "injected"
    attack = f"$(touch {marker}); touch {marker}"
    rendered = render_gym_command(
        "gym",
        "env",
        "validate",
        args=("--manifest", attack),
        overrides={"policy_base_url": attack},
    )

    subprocess.run(["sh", "-c", f"gym() {{ :; }}\n{rendered}"], check=True)
    assert not marker.exists()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"command": ()},
        {"command": ("python",)},
        {"command": ("gym", "")},
        {"command": ("gym",), "config_paths": ("same.yaml", "same.yaml")},
        {"command": ("gym",), "overrides": (("same", 1), ("same", 2))},
        {"command": ("gym",), "overrides": {"unsafe;key": 1}},
    ],
)
def test_renderer_rejects_unsafe_structure(kwargs) -> None:
    command = kwargs.pop("command")
    with pytest.raises(ValueError):
        render_gym_command(*command, **kwargs)
