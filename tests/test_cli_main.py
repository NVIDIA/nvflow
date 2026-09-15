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
"""Tests for the ``nflow`` CLI (``nvflow.cli.main``).

Exercises the Typer app end-to-end via ``CliRunner`` against the real,
auto-discovered stage registry (the lightweight ``example`` recipe requires
none of the heavy ``nemo-skills``/``torch`` stack, so it registers even in
the CI-lightweight test environment -- see ``tests/requirements-ci.txt``).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from nvflow.cli.main import app

pytest.importorskip("typer")

runner = CliRunner()

REPO_ROOT = Path(__file__).parent.parent
EXAMPLE_CONFIG = REPO_ROOT / "nvflow/recipes/example/workflows/sdg_simple.yaml"


class TestVersion:
    def test_version_prints_version_string(self) -> None:
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "NVFlow version" in result.stdout


class TestListStages:
    def test_no_args_lists_all_recipes(self) -> None:
        result = runner.invoke(app, ["list-stages"])
        assert result.exit_code == 0
        assert "example:" in result.stdout
        assert "generate_answer" in result.stdout

    def test_filter_by_known_recipe(self) -> None:
        result = runner.invoke(app, ["list-stages", "--recipe", "example"])
        assert result.exit_code == 0
        assert "example" in result.stdout
        assert "generate_answer" in result.stdout

    def test_filter_by_unknown_recipe_errors(self) -> None:
        result = runner.invoke(app, ["list-stages", "--recipe", "nonexistent"])
        assert result.exit_code == 1
        assert "not found" in result.stdout

    def test_filter_by_unknown_workflow_reports_no_stages(self) -> None:
        result = runner.invoke(app, ["list-stages", "--workflow", "nonexistent_workflow"])
        assert result.exit_code == 0
        assert "No stages found" in result.stdout

    def test_from_config_file(self) -> None:
        result = runner.invoke(app, ["list-stages", "--config", str(EXAMPLE_CONFIG)])
        assert result.exit_code == 0
        assert "generate_answer" in result.stdout
        assert "Total: 1 stages" in result.stdout

    def test_from_nonexistent_config_file_errors(self) -> None:
        result = runner.invoke(app, ["list-stages", "--config", "does_not_exist.yaml"])
        assert result.exit_code == 1
        assert "Error loading config" in result.stdout


class TestStageInfo:
    def test_full_path(self) -> None:
        result = runner.invoke(app, ["stage-info", "example.sdg_simple.generate_answer"])
        assert result.exit_code == 0
        assert "GenerateAnswerStage" in result.stdout

    def test_short_name_with_recipe_and_workflow(self) -> None:
        result = runner.invoke(
            app,
            ["stage-info", "generate_answer", "--recipe", "example", "--workflow", "sdg_simple"],
        )
        assert result.exit_code == 0
        assert "GenerateAnswerStage" in result.stdout

    def test_short_name_without_recipe_and_workflow_errors(self) -> None:
        result = runner.invoke(app, ["stage-info", "generate_answer"])
        assert result.exit_code == 1
        assert "must provide --recipe and --workflow" in result.stdout

    def test_invalid_path_format_errors(self) -> None:
        result = runner.invoke(app, ["stage-info", "a.b.c.d"])
        assert result.exit_code == 1
        assert "Invalid stage path" in result.stdout

    def test_unknown_stage_errors(self) -> None:
        result = runner.invoke(app, ["stage-info", "example.sdg_simple.nonexistent_stage"])
        assert result.exit_code == 1
        assert "not found" in result.stdout


class TestValidate:
    def test_valid_config(self) -> None:
        result = runner.invoke(app, ["validate", "--config", str(EXAMPLE_CONFIG)])
        assert result.exit_code == 0
        assert "Configuration is valid" in result.stdout
        assert "Recipe: example" in result.stdout

    def test_missing_config_file_errors(self) -> None:
        result = runner.invoke(app, ["validate", "--config", "does_not_exist.yaml"])
        assert result.exit_code == 1
        assert "Configuration is invalid" in result.stdout

    def test_requires_config_option(self) -> None:
        result = runner.invoke(app, ["validate"])
        assert result.exit_code != 0


class TestRun:
    def test_missing_config_file_errors(self) -> None:
        result = runner.invoke(app, ["run", "generate_answer", "--config", "does_not_exist.yaml"])
        assert result.exit_code == 1
        assert "Error:" in result.stdout

    def test_requires_config_option(self) -> None:
        result = runner.invoke(app, ["run", "generate_answer"])
        assert result.exit_code != 0


class TestRunAll:
    def test_missing_config_file_errors(self) -> None:
        result = runner.invoke(app, ["run-all", "--config", "does_not_exist.yaml"])
        assert result.exit_code == 1
        assert "Error:" in result.stdout
