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
"""CLI stream, color, and literal-rendering contracts."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import types
import warnings
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from nvflow.cli import main as cli
from nvflow.cli import output
from nvflow.core import console as orchestration_console
from nvflow.core.console import SUCCESS_STYLE, create_console, message_text

PROJECT_ROOT = Path(__file__).parents[1]


@pytest.fixture
def target_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    name = "test-cluster"
    (tmp_path / f"{name}.yaml").write_text(
        f"executor: slurm\njob_dir: {tmp_path / 'jobs'}\napi_key: do-not-render\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("NVFLOW_CONFIG_DIR", str(tmp_path))
    return name


def test_shared_console_is_tty_aware_and_treats_values_literally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    tty_buffer = StringIO()
    tty_console = create_console(file=tty_buffer, force_terminal=True)
    tty_console.print(message_text("[red]literal[/red]", style=SUCCESS_STYLE))

    plain_buffer = StringIO()
    plain_console = create_console(file=plain_buffer, force_terminal=False)
    plain_console.print(message_text("[red]literal[/red]", style=SUCCESS_STYLE))

    assert "\x1b[" in tty_buffer.getvalue()
    assert "[red]literal[/red]" in tty_buffer.getvalue()
    assert plain_buffer.getvalue() == "[red]literal[/red]\n"


def test_plan_terminal_styling_preserves_plain_redirected_text(monkeypatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    rendered = "\n".join(
        [
            "NVFlow plan: example.workflow",
            "Mode: LIVE",
            "Filesystem inspection: ENABLED — stages may discover inputs or omit reusable work",
            "Source eligibility: READY",
            "Plan coverage: COMPLETE",
            "● collect  1  1 GPU task",
            "Value: [red]literal[/red]",
        ]
    )

    tty_buffer = StringIO()
    monkeypatch.setattr(
        output,
        "stdout_console",
        create_console(file=tty_buffer, force_terminal=True),
    )
    output.emit_plan_text(rendered)
    assert "\x1b[" in tty_buffer.getvalue()
    assert "[red]literal[/red]" in tty_buffer.getvalue()

    plain_buffer = StringIO()
    monkeypatch.setattr(
        output,
        "stdout_console",
        create_console(file=plain_buffer, force_terminal=False),
    )
    monkeypatch.setattr("sys.stdout", plain_buffer)
    output.emit_plan_text(rendered)
    assert plain_buffer.getvalue() == rendered + "\n"
    assert "\x1b[" not in plain_buffer.getvalue()


def test_yaml_terminal_styling_preserves_plain_redirected_text(monkeypatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    long_value = "x" * 120
    rendered = (
        f"root:\n  child: value\n  literal: '[red]literal[/red]'\n  long_value: {long_value}\n"
    )

    tty_buffer = StringIO()
    monkeypatch.setattr(
        output,
        "stdout_console",
        create_console(file=tty_buffer, force_terminal=True),
    )
    output.emit_yaml(rendered)
    assert "\x1b[" in tty_buffer.getvalue()
    assert "[red]literal[/red]" in tty_buffer.getvalue()
    assert tty_buffer.getvalue().count("x") == len(long_value)

    plain_buffer = StringIO()
    monkeypatch.setattr(
        output,
        "stdout_console",
        create_console(file=plain_buffer, force_terminal=False),
    )
    monkeypatch.setattr("sys.stdout", plain_buffer)
    output.emit_yaml(rendered)
    assert plain_buffer.getvalue() == rendered
    assert "\x1b[" not in plain_buffer.getvalue()


def test_yaml_terminal_uses_standard_multicolor_syntax_theme(monkeypatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    rendered = (
        "root:\n"
        "  string: value\n"
        "  number: 42\n"
        "  enabled: true\n"
        "  nested_json: '{\"key\":false}'\n"
        "  # comment\n"
    )
    tty_buffer = StringIO()
    monkeypatch.setattr(
        output,
        "stdout_console",
        create_console(file=tty_buffer, force_terminal=True),
    )

    output.emit_yaml(rendered)

    color_codes = set(re.findall(r"\x1b\[[0-9;]*m", tty_buffer.getvalue()))
    assert output._YAML_SYNTAX_THEME == "ansi_dark"
    assert len(color_codes) >= 3


def test_action_output_routing_includes_native_and_subprocess_stdout() -> None:
    script = """
import os
import subprocess
import sys
from nvflow.cli.output import route_stdout_to_stderr

with route_stdout_to_stderr():
    print("python output", flush=True)
    os.write(1, b"native output\\n")
    subprocess.run(
        [sys.executable, "-c", "print('subprocess output')"],
        check=True,
    )
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stdout == ""
    assert "python output" in completed.stderr
    assert "native output" in completed.stderr
    assert "subprocess output" in completed.stderr


def test_typer_errors_stay_plain_on_github_actions() -> None:
    environment = {
        name: value for name, value in os.environ.items() if name not in {"NO_COLOR", "TERM"}
    }
    environment["GITHUB_ACTIONS"] = "true"
    completed = subprocess.run(
        [sys.executable, "-m", "nvflow.cli.main", "plan", "--workflow", "starter:generate"],
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 2
    assert "Missing option '--cluster'" in completed.stderr
    assert "\x1b[" not in completed.stderr


def test_show_config_stdout_is_parseable_yaml(
    target_config: str,
) -> None:
    result = CliRunner().invoke(
        cli.app,
        ["show-config", "--workflow", "starter:generate"],
    )

    assert result.exit_code == 0, result.output
    assert yaml.safe_load(result.stdout)["project"] == "starter"
    assert result.stderr == ""
    assert "\x1b[" not in result.stdout


def test_plan_json_keeps_payload_on_stdout_and_live_status_on_stderr(
    target_config: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    filesystem = types.SimpleNamespace(batch_exists=lambda paths: dict.fromkeys(paths, False))
    monkeypatch.setattr(cli, "target_filesystem", lambda _cluster: filesystem)
    monkeypatch.setattr(
        "nvflow.execution.inspection.inspect_declared_outputs",
        lambda *_args: pytest.fail("JSON plan performed an informational output scan"),
    )

    result = CliRunner().invoke(
        cli.app,
        [
            "plan",
            "--workflow",
            "starter:generate",
            "--cluster",
            target_config,
            "--live",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["planning_mode"] == "live"
    assert "Inspecting workflow inputs for live task selection" in result.stderr
    assert "do-not-render" not in result.output


def test_plan_text_error_uses_stderr_and_json_error_uses_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NVFLOW_CONFIG_DIR", str(tmp_path))
    base = [
        "plan",
        "--workflow",
        "starter:generate",
        "--cluster",
        "missing",
    ]

    text_result = CliRunner().invoke(cli.app, base)
    json_result = CliRunner().invoke(cli.app, [*base, "--format", "json"])

    assert text_result.exit_code == 1
    assert text_result.stdout == ""
    assert "Target configuration 'missing'" in text_result.stderr
    assert json_result.exit_code == 1
    assert json.loads(json_result.stdout)["error"] == "TargetResolutionError"
    assert json_result.stderr == ""


def test_plan_text_failure_redacts_resolved_secrets(
    target_config: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "resolved-secret-value"
    definition = types.SimpleNamespace(
        cluster=target_config,
        project_context=types.SimpleNamespace(
            manifest=types.SimpleNamespace(entrypoint=None),
        ),
        redact_runtime_values=lambda value: str(value).replace(secret, "<redacted>"),
    )

    class FailingPlanner:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def execution_graph(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError(f"backend rejected token {secret}")

    monkeypatch.setattr(cli, "load_workflow", lambda *_args, **_kwargs: definition)
    monkeypatch.setattr(cli, "WorkflowPlanner", FailingPlanner)

    result = CliRunner().invoke(
        cli.app,
        [
            "plan",
            "--config",
            "workflow.yaml",
            "--cluster",
            target_config,
        ],
    )

    assert result.exit_code == 1
    assert result.stdout == ""
    assert secret not in result.stderr
    assert "<redacted>" in result.stderr


@pytest.mark.parametrize("selection", [["generate_answer"], ["--all"]])
def test_run_commands_keep_stdout_empty(
    selection: list[str],
    target_config: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRunner:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            print("Third-party constructor output")

        def run(self, **_kwargs: Any) -> None:
            print("Third-party launcher output")
            orchestration_console.status("Submitting test job")

    monkeypatch.setattr(cli, "WorkflowRunner", FakeRunner)
    arguments = ["run", *selection]
    arguments.extend(
        [
            "--workflow",
            "starter:generate",
            "--cluster",
            target_config,
        ]
    )

    result = CliRunner().invoke(cli.app, arguments)

    assert result.exit_code == 0, result.output
    assert result.stdout == ""
    assert "Third-party constructor output" in result.stderr
    assert "Third-party launcher output" in result.stderr
    assert "Submitting test job" in result.stderr


def test_run_error_uses_stderr(
    target_config: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingRunner:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def run(self, **_kwargs: Any) -> None:
            raise RuntimeError("submission failed")

    monkeypatch.setattr(cli, "WorkflowRunner", FailingRunner)
    result = CliRunner().invoke(
        cli.app,
        [
            "run",
            "--all",
            "--workflow",
            "starter:generate",
            "--cluster",
            target_config,
        ],
    )

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "submission failed" in result.stderr


def test_run_filters_known_dependency_warnings(
    target_config: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Definition:
        @staticmethod
        def redact_runtime_values(value):
            return value

    class FakeRunner:
        definition = Definition()

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def run(self, **_kwargs: Any) -> None:
            warnings.warn(
                "starlette.middleware.wsgi is deprecated",
                UserWarning,
                stacklevel=2,
            )
            warnings.warn(
                "unclosed file <_io.FileIO name=7 mode='wb' closefd=True>",
                ResourceWarning,
                stacklevel=2,
            )
            warnings.warn("project warning remains visible", UserWarning, stacklevel=2)

    monkeypatch.setattr(cli, "WorkflowRunner", FakeRunner)
    result = CliRunner().invoke(
        cli.app,
        [
            "run",
            "--all",
            "--workflow",
            "starter:generate",
            "--cluster",
            target_config,
        ],
    )

    assert result.exit_code == 0, result.output
    assert "starlette.middleware.wsgi is deprecated" not in result.stderr
    assert "unclosed file" not in result.stderr
    assert "project warning remains visible" in result.stderr
    assert "1 runtime warning(s)" in result.stderr
