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
"""Tests for the lightweight nvflow project handoff."""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from nvflow.cli import bootstrap

PROJECT_ROOT = Path(__file__).parents[1]


def test_console_script_is_named_nvflow_only() -> None:
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert project["project"]["scripts"] == {"nvflow": "nvflow.cli.bootstrap:main"}


@pytest.mark.parametrize(
    "arguments",
    [
        ["version"],
        ["list-projects"],
        ["list-workflows", "--project", "finance"],
        ["list-stages", "--workflow", "starter:generate"],
        ["list-environments", "--workflow", "starter:generate"],
        ["show-config", "--workflow", "starter:generate"],
        ["validate", "--recipe", "finance:quickstart"],
    ],
)
def test_structural_commands_skip_bootstrap_handoff(
    arguments: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "nvflow.cli.project_environment.handoff_to_project_environment",
        lambda *_args, **_kwargs: pytest.fail("structural command attempted handoff"),
    )

    bootstrap._handoff(arguments)


def test_workflow_command_hands_off_before_full_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        "nvflow.cli.project_environment.handoff_to_project_environment",
        lambda context, **kwargs: calls.append(
            (context.identity, kwargs["argv"], kwargs["prepare"])
        ),
    )
    arguments = ["validate", "--workflow", "starter:generate"]

    bootstrap._handoff(arguments)

    assert calls == [("starter", arguments, False)]


def test_bootstrap_defers_invalid_selection_to_full_cli() -> None:
    bootstrap._handoff(["plan", "--workflow", "missing:workflow"])


def test_version_path_does_not_import_full_cli() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from nvflow.cli.bootstrap import main; "
                "sys.argv=['nvflow', 'version']; "
                "main(); "
                "assert 'nvflow.cli.main' not in sys.modules"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.startswith("NVFlow version ")


def test_full_cli_does_not_eagerly_import_environment_or_plan_renderer() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import nvflow.cli.main; "
                "assert 'nvflow.project.environment' not in sys.modules; "
                "assert 'nvflow.cli.plan_output' not in sys.modules; "
                "assert 'nvflow.cli.workflow_inspection' not in sys.modules"
            ),
        ],
        check=True,
    )


@pytest.mark.parametrize(
    ("workflow", "expected_exit"),
    [
        ("shared-gym-fixture:profile", 0),
        ("shared-gym-fixture:profile-missing-container", 1),
    ],
)
def test_gym_validation_cli_does_not_import_execution_stacks(
    workflow: str,
    expected_exit: int,
) -> None:
    arguments = [
        "nvflow",
        "validate",
        "collect_rollouts",
        "--workflow",
        workflow,
        "-e",
        "example",
    ]
    script = f"""
import sys
from nvflow.cli.bootstrap import main

sys.argv = {arguments!r}
try:
    main()
except SystemExit as exc:
    exit_code = int(exc.code or 0)
else:
    exit_code = 0
assert exit_code == {expected_exit}, exit_code

forbidden_modules = {{
    "nvflow.integrations.nemo_gym.job_scripts",
    "nvflow.integrations.nemo_gym.native_rewards",
    "nvflow.integrations.nemo_gym.native_rollout",
}}
forbidden_prefixes = (
    "kubeflow",
    "leptonai",
    "nemo_run",
    "nemo_skills",
    "torch",
)
loaded = sorted(
    name
    for name in sys.modules
    if name in forbidden_modules
    or name.startswith(tuple(f"{{prefix}}." for prefix in forbidden_prefixes))
    or name in forbidden_prefixes
)
assert not loaded, loaded
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT / "tests/fixtures/projects/shared_gym_project",
        env={**os.environ, "NVFLOW_DISABLE_PROJECT_ENVIRONMENT": "1"},
        check=True,
        capture_output=True,
        text=True,
    )
