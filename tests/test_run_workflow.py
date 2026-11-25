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
"""Parity tests for the Python workflow debugging entry point."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from nvflow.cli.run_workflow import main

PROJECT_ROOT = Path(__file__).parents[1]


def test_python_entry_invokes_workflow_runner_without_cli_handoff(monkeypatch) -> None:
    calls = []

    class FakeRunner:
        def __init__(self, config, **kwargs):
            calls.append(("init", config, kwargs))
            self.project_context = SimpleNamespace(
                launcher_python_root=PROJECT_ROOT / "project-source"
            )

        def run(self, **kwargs):
            assert str(self.project_context.launcher_python_root) in sys.path
            calls.append(("run", kwargs))

    monkeypatch.setattr("nvflow.execution.WorkflowRunner", FakeRunner)
    config = PROJECT_ROOT / "projects" / "starter" / "workflows" / "generate.yaml"

    result = main(
        [
            "generate_answer",
            "--config",
            str(config),
            "--cluster",
            "test-cluster",
            "-e",
            "first",
            "second",
        ]
    )

    assert result == 0
    assert calls == [
        (
            "init",
            str(config),
            {
                "project_dir": None,
                "cluster": "test-cluster",
                "project_search_paths": (),
            },
        ),
        (
            "run",
            {
                "stages": ["generate_answer"],
                "environment": ["first", "second"],
            },
        ),
    ]
    assert str(PROJECT_ROOT / "project-source") not in sys.path


def test_script_is_only_a_compatibility_shim() -> None:
    script = (PROJECT_ROOT / "scripts" / "run_flow.py").read_text(encoding="utf-8")

    assert "from nvflow.cli.run_workflow import main" in script
    assert "WorkflowRunner" not in script
