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
"""Contracts for dependency-light NeMo-Skills command adapters."""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import pytest

from nvflow.integrations.nemo_skills import commands

PROJECT_ROOT = Path(__file__).parents[1]


def test_command_adapter_imports_no_nemo_skills_modules() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import nvflow.integrations.nemo_skills.commands; "
                "assert not any(name.startswith('nemo_skills') for name in sys.modules)"
            ),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stderr == ""


def test_wrap_arguments_matches_programmatic_command_context() -> None:
    context = commands.wrap_arguments("  --one value   --two=2 ")

    assert context.args == ["--one", "value", "--two=2"]
    assert context.obj is None


def test_command_adapter_loads_only_requested_leaf(monkeypatch) -> None:
    calls: list[tuple[str, tuple, dict]] = []
    monkeypatch.delitem(sys.modules, "nemo_skills.pipeline.cli", raising=False)
    generate_module = types.ModuleType("nemo_skills.pipeline.generate")
    generate_module.__dict__["generate"] = lambda *args, **kwargs: calls.append(
        ("generate", args, kwargs)
    )
    run_cmd_module = types.ModuleType("nemo_skills.pipeline.run_cmd")
    run_cmd_module.__dict__["run_cmd"] = lambda *args, **kwargs: calls.append(
        ("run_cmd", args, kwargs)
    )
    monkeypatch.setitem(sys.modules, "nemo_skills.pipeline.generate", generate_module)
    monkeypatch.setitem(sys.modules, "nemo_skills.pipeline.run_cmd", run_cmd_module)

    commands.generate("ctx", cluster="test")
    assert "nemo_skills.pipeline.cli" not in sys.modules
    commands.run_cmd("ctx", cluster="test")

    assert calls == [
        ("generate", ("ctx",), {"cluster": "test"}),
        ("run_cmd", ("ctx",), {"cluster": "test"}),
    ]


def test_submission_capture_normalizes_jobs_and_noops(monkeypatch) -> None:
    job = types.SimpleNamespace(handle="slurm://job/1")
    group = types.SimpleNamespace(handles=["slurm://job/2", "slurm://job/3"])
    experiment = types.SimpleNamespace(jobs=[job, group])
    generate_module = types.ModuleType("nemo_skills.pipeline.generate")
    generate_module.__dict__["generate"] = lambda **_kwargs: experiment
    run_cmd_module = types.ModuleType("nemo_skills.pipeline.run_cmd")
    run_cmd_module.__dict__["run_cmd"] = lambda **_kwargs: None
    monkeypatch.setitem(sys.modules, "nemo_skills.pipeline.generate", generate_module)
    monkeypatch.setitem(sys.modules, "nemo_skills.pipeline.run_cmd", run_cmd_module)

    with commands.capture_submissions() as captured:
        commands.generate(expname="generate")
        commands.run_cmd(expname="noop")

    assert captured.handles_by_experiment() == {
        "generate": ("slurm://job/1", "slurm://job/2", "slurm://job/3"),
        "noop": (),
    }


@pytest.mark.requires_nemo
def test_generation_leaf_avoids_unrelated_command_modules() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-W",
            "always",
            "-c",
            (
                "import sys; import nemo_skills.pipeline.generate; "
                "forbidden=('nemo_skills.pipeline.cli', "
                "'nemo_skills.pipeline.start_server', "
                "'nemo_skills.evaluation.metrics.arena_metrics'); "
                "assert not any(name in sys.modules for name in forbidden)"
            ),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "invalid escape sequence" not in completed.stderr
