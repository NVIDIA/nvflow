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
"""Consistent workflow navigation contracts across CLI commands."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from nvflow.cli import main as cli
from nvflow.execution.engines import ActiveJob, JobListing
from nvflow.stages.catalog import list_stage_types

PROJECT_ROOT = Path(__file__).parents[1]
STARTER_ROOT = PROJECT_ROOT / "projects" / "starter"
STARTER_CONFIG = STARTER_ROOT / "workflows" / "generate.yaml"


@pytest.fixture(autouse=True)
def _target_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "test-cluster.yaml").write_text("executor: slurm\n")
    monkeypatch.setenv("NVFLOW_CONFIG_DIR", str(tmp_path))


@pytest.fixture(autouse=True)
def _execution_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    # CI lanes without NeMo-Skills must not hand status off to a real environment.
    monkeypatch.setattr(cli, "_execution_stack_installed", lambda: True)


def _normalized_output(result) -> str:
    return " ".join(result.output.split())


def test_status_prints_the_engines_active_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    listed: list[str] = []

    class Engine:
        def active_jobs(self, target):
            listed.append(target.binding.cluster)
            return JobListing(
                (
                    ActiveJob("19352776", "answer_generation", "RUNNING", "4:03", "batch"),
                    ActiveJob("19352777", "prepare_data", "PENDING", "0:00", "cpu"),
                )
            )

    monkeypatch.setattr(cli, "engine_for", lambda _target: Engine())

    result = CliRunner().invoke(cli.app, ["status", "--cluster", "test-cluster"])

    assert result.exit_code == 0, result.output
    assert listed == ["test-cluster"]
    rows = [line.split() for line in result.stdout.splitlines()]
    assert rows[0] == [
        "Job",
        "Workflow",
        "Stage",
        "Experiment",
        "Name",
        "State",
        "Reason",
        "Time",
        "Where",
    ]
    assert ["19352776", "-", "-", "-", "answer_generation", "RUNNING", "-", "4:03", "batch"] in rows
    assert ["19352777", "-", "-", "-", "prepare_data", "PENDING", "-", "0:00", "cpu"] in rows


def test_status_lists_unreachable_locations_below_the_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Engine:
        def active_jobs(self, _target):
            return JobListing(
                (ActiveJob("nvflow-run-controller-a1", "controller", "RUNNING", "0:41", "skills"),),
                {"gym": "RAY_GYM_JOBS_URL is not set"},
            )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "engine_for", lambda _target: Engine())

    result = CliRunner().invoke(cli.app, ["status", "--cluster", "test-cluster"])

    assert result.exit_code == 0, result.output
    rows = [line.split() for line in result.stdout.splitlines()]
    assert [
        "nvflow-run-controller-a1",
        "-",
        "-",
        "-",
        "controller",
        "RUNNING",
        "-",
        "0:41",
        "skills",
    ] in rows
    assert "gym" not in result.stdout
    assert "gym not reachable: RAY_GYM_JOBS_URL is not set" in " ".join(result.stderr.split())


def test_status_says_when_nothing_is_queued(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Engine:
        def active_jobs(self, _target):
            return JobListing()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "engine_for", lambda _target: Engine())

    result = CliRunner().invoke(cli.app, ["status", "--cluster", "test-cluster"])

    assert result.exit_code == 0, result.output
    assert "No queued or running jobs on test-cluster." in _normalized_output(result)


def test_status_continues_in_the_working_projects_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handed_off: list[str] = []

    class Engine:
        def active_jobs(self, _target):
            return JobListing()

    monkeypatch.chdir(STARTER_ROOT / "workflows")
    monkeypatch.setattr(
        cli, "_handoff_project", lambda context: handed_off.append(context.identity)
    )
    monkeypatch.setattr(cli, "engine_for", lambda _target: Engine())

    result = CliRunner().invoke(cli.app, ["status", "--cluster", "test-cluster"])

    assert result.exit_code == 0, result.output
    assert handed_off == ["starter"]


@pytest.mark.parametrize(("stack_installed", "expected"), [(False, ["starter"]), (True, [])])
def test_status_outside_a_project_borrows_the_starter_environment_only_without_the_stack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stack_installed: bool,
    expected: list[str],
) -> None:
    handed_off: list[str] = []

    class Engine:
        def active_jobs(self, _target):
            return JobListing()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_execution_stack_installed", lambda: stack_installed)
    monkeypatch.setattr(
        cli, "_handoff_project", lambda context: handed_off.append(context.identity)
    )
    monkeypatch.setattr(cli, "engine_for", lambda _target: Engine())

    result = CliRunner().invoke(cli.app, ["status", "--cluster", "test-cluster"])

    assert result.exit_code == 0, result.output
    assert handed_off == expected


_LABELED = ActiveJob(
    "101",
    "answer-0",
    "RUNNING",
    "4:03",
    "batch",
    workflow="starter:generate",
    stage="generate_answer",
    experiment="starter-generate-generate_answer",
    reason="node01",
    nodes=1,
    gpus_per_node=8,
    cpus=16,
    start="2026-10-15T16:00:00",
    account="acct",
    priority=7,
)
_OTHER_WORKFLOW = ActiveJob(
    "102",
    "train-0",
    "PENDING",
    "0:00",
    "batch",
    workflow="other:train",
    stage="train",
    experiment="other-train",
    reason="Priority",
)
_UNLABELED = ActiveJob("103", "interactive", "RUNNING", "1:00", "batch")


def _status(
    monkeypatch: pytest.MonkeyPatch,
    *arguments: str,
    jobs: tuple[ActiveJob, ...] = (_LABELED, _OTHER_WORKFLOW, _UNLABELED),
):
    class Engine:
        def active_jobs(self, _target):
            return JobListing(jobs)

    monkeypatch.setattr(cli, "engine_for", lambda _target: Engine())
    monkeypatch.setattr(cli, "_handoff_project", lambda _context: None)
    return CliRunner().invoke(cli.app, ["status", "--cluster", "test-cluster", *arguments])


def _job_ids(result) -> list[str]:
    return [line.split()[0] for line in result.stdout.splitlines()[1:]]


def test_status_shows_each_jobs_workflow_stage_and_experiment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    result = _status(monkeypatch)

    assert result.exit_code == 0, result.output
    rows = [line.split() for line in result.stdout.splitlines()]
    assert [
        "101",
        "starter:generate",
        "generate_answer",
        "starter-generate-generate_answer",
        "answer-0",
        "RUNNING",
        "node01",
        "4:03",
        "batch",
    ] in rows
    assert ["103", "-", "-", "-", "interactive", "RUNNING", "-", "1:00", "batch"] in rows


def test_status_verbose_adds_the_resource_and_scheduling_columns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    result = _status(monkeypatch, "-v")

    assert result.exit_code == 0, result.output
    rows = [line.split() for line in result.stdout.splitlines()]
    assert rows[0][-6:] == ["Nodes", "GPUs/Node", "CPUs", "Start", "Account", "Priority"]
    assert rows[1][-6:] == ["1", "8", "16", "2026-10-15T16:00:00", "acct", "7"]
    assert rows[2][-6:] == ["-", "-", "-", "-", "-", "-"]


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (["-w", "starter:generate"], ["101"]),
        (["-w", "starter:generate", "generate_answer"], ["101"]),
        (["--config", str(STARTER_CONFIG)], ["101"]),
    ],
)
def test_status_selects_one_workflows_jobs(
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
    expected: list[str],
) -> None:
    result = _status(monkeypatch, *arguments)

    assert result.exit_code == 0, result.output
    assert _job_ids(result) == expected


def test_status_says_when_the_selected_workflow_has_no_jobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _status(monkeypatch, "-w", "starter:generate", jobs=(_OTHER_WORKFLOW,))

    assert result.exit_code == 0, result.output
    assert "No queued or running jobs for starter:generate on test-cluster." in (
        _normalized_output(result)
    )


def test_status_rejects_a_stage_the_workflow_does_not_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _status(monkeypatch, "-w", "starter:generate", "missing_stage")

    assert result.exit_code == 1
    assert "Stage 'missing_stage' not found" in _normalized_output(result)


def test_status_lists_a_recipes_jobs_in_recipe_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[str] = []

    def resolve_recipe(recipe, _resolver, *, project_dir=None):
        received.append(recipe)
        return SimpleNamespace(
            context=SimpleNamespace(identity="demo"),
            selector="demo:pipeline",
            workflows=(
                SimpleNamespace(selector="other:train"),
                SimpleNamespace(selector="starter:generate"),
            ),
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "resolve_recipe", resolve_recipe)

    result = _status(monkeypatch, "-r", "demo:pipeline")

    assert result.exit_code == 0, result.output
    assert received == ["demo:pipeline"]
    assert _job_ids(result) == ["102", "101"]


@pytest.mark.parametrize(
    ("arguments", "fragments"),
    [
        (
            ["-r", "demo:pipeline", "-w", "starter:generate"],
            ("Use --recipe or a workflow", "not both"),
        ),
        (["generate_answer"], ("Stage selections apply only to a workflow",)),
    ],
)
def test_status_rejects_selections_other_commands_reject(
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
    fragments: tuple[str, ...],
) -> None:
    result = _status(monkeypatch, *arguments)

    assert result.exit_code == 2
    for fragment in fragments:
        assert fragment in _normalized_output(result)


def test_status_shows_bracketed_scheduler_values_as_written(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    result = _status(
        monkeypatch,
        jobs=(
            ActiveJob(
                "104",
                "train[large]",
                "PENDING",
                "0:00",
                "batch",
                experiment="sweep[lr=1e-5]",
                reason="[bold]waiting",
            ),
        ),
    )

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines()[1].split() == [
        "104",
        "-",
        "-",
        "sweep[lr=1e-5]",
        "train[large]",
        "PENDING",
        "[bold]waiting",
        "0:00",
        "batch",
    ]


def test_status_prints_wide_rows_whole_instead_of_wrapping_them(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    long_name = "job-0-collect-rollouts-chunk-0-attempt-0-policy-0-" * 3
    monkeypatch.chdir(tmp_path)

    result = _status(
        monkeypatch,
        jobs=(ActiveJob("20302316+0", long_name, "PENDING", "0:00", "batch"),),
    )

    assert result.exit_code == 0, result.output
    assert result.stdout.endswith("\n")
    lines = result.stdout.splitlines()
    assert len(lines) == 2
    assert lines[1].split()[0] == "20302316+0"
    assert long_name in lines[1]


def test_cli_hides_only_starlettes_wsgi_deprecation() -> None:
    code = (
        "import warnings\n"
        "import nvflow.cli.main\n"
        "warnings.warn('starlette.middleware.wsgi is deprecated and will be removed', "
        "DeprecationWarning)\n"
        "warnings.warn('another deprecation', DeprecationWarning)\n"
    )

    result = subprocess.run(
        [sys.executable, "-W", "default", "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "starlette.middleware.wsgi" not in result.stderr
    assert "another deprecation" in result.stderr


def test_top_level_help_follows_workflow_lifecycle_order() -> None:
    expected = [
        "list-projects",
        "list-recipes",
        "list-workflows",
        "sync-project",
        "list-stages",
        "list-environments",
        "list-stage-types",
        "stage-type-info",
        "stage-info",
        "show-config",
        "validate",
        "plan-recipe",
        "plan",
        "run",
        "status",
        "version",
    ]

    result = CliRunner().invoke(cli.app, ["--help"])

    assert result.exit_code == 0, result.output
    rows = re.findall(r"^\s*│\s+([a-z][a-z-]+)\s{2,}", result.stdout, flags=re.MULTILINE)
    assert [name for name in rows if name in expected] == expected
    assert "run-all" not in rows


def test_shared_stage_types_are_discoverable_from_platform_catalog() -> None:
    runner = CliRunner()
    list_help = runner.invoke(cli.app, ["list-stage-types", "--help"])
    info_help = runner.invoke(cli.app, ["stage-type-info", "--help"])
    listed = runner.invoke(cli.app, ["list-stage-types"])
    detailed = runner.invoke(
        cli.app,
        ["stage-type-info", "nemo_gym.collect"],
    )

    assert list_help.exit_code == 0
    assert "--namespace" in list_help.output
    assert info_help.exit_code == 0
    assert "STAGE_TYPE" in info_help.output
    assert listed.exit_code == 0, listed.output
    assert "nemo_gym.collect" in listed.output
    assert "nemo_gym.rejudge" in listed.output
    assert "data.train_validation_split" in listed.output
    assert "rubric.build" in listed.output
    assert "data.dedup" in listed.output
    assert "data.decontaminate" in listed.output
    assert "data.filter" in listed.output
    assert "nemo_gym.render" in listed.output
    assert f"Total: {len(list_stage_types())} shared stage types" in listed.output
    assert detailed.exit_code == 0, detailed.output
    assert "CollectRolloutsStage" in detailed.output
    scoped = runner.invoke(
        cli.app,
        ["list-stage-types", "--namespace", "nemo_gym"],
    )
    assert scoped.exit_code == 0
    assert "nemo_gym.collect" in scoped.output
    training = runner.invoke(cli.app, ["list-stage-types", "--namespace", "nemo_rl"])
    assert training.exit_code == 0
    assert "nemo_rl.sft" in training.output
    assert "nemo_rl.grpo" in training.output
    assert "Total: 2 shared stage types" in training.output
    missing_namespace = runner.invoke(
        cli.app,
        ["list-stage-types", "--namespace", "unknown"],
    )
    assert missing_namespace.exit_code == 0
    assert "No shared stage types found." in missing_namespace.output
    unknown = runner.invoke(
        cli.app,
        ["stage-type-info", "unknown.stage"],
    )
    assert unknown.exit_code == 1
    assert "Could not inspect shared stage type" in unknown.output
    assert "nvflow list-stage-types" in unknown.output


def test_cli_selection_options_are_consistent() -> None:
    runner = CliRunner()
    project_aware = (
        "list-projects",
        "list-recipes",
        "list-workflows",
        "sync-project",
        "list-stages",
        "list-environments",
        "stage-info",
        "show-config",
        "validate",
        "plan-recipe",
        "plan",
        "run",
    )
    for command in project_aware:
        result = runner.invoke(cli.app, [command, "--help"])
        assert result.exit_code == 0, result.output
        assert "--project-dir" in result.output
        assert "--project-path" in result.output

    for command in (
        "list-stages",
        "list-environments",
        "show-config",
        "validate",
        "plan",
        "run",
    ):
        result = runner.invoke(cli.app, [command, "--help"])
        assert "--workflow" in result.output
        assert "--config" in result.output

    assert "--project" in runner.invoke(cli.app, ["list-recipes", "--help"]).output
    assert "--recipe" in runner.invoke(cli.app, ["list-workflows", "--help"]).output
    assert "--recipe" in runner.invoke(cli.app, ["validate", "--help"]).output
    assert "RECIPE" in runner.invoke(cli.app, ["plan-recipe", "--help"]).output


def test_common_short_options_are_available_consistently() -> None:
    runner = CliRunner()
    for arguments in (
        ["-h"],
        ["list-workflows", "-p", "starter"],
        ["list-workflows", "-r", "finance:quickstart"],
        ["list-workflows", "-p", "finance", "-a"],
        ["list-stages", "-w", "starter:generate"],
        ["show-config", "-w", "starter:generate"],
        ["validate", "-w", "starter:generate"],
        ["plan-recipe", "finance:quickstart", "-f", "json"],
    ):
        result = runner.invoke(cli.app, arguments)
        assert result.exit_code == 0, f"{arguments}\n{result.output}"

    for command in cli._COMMAND_HELP_ORDER:
        result = runner.invoke(cli.app, [command, "-h"])
        assert result.exit_code == 0, f"{command}\n{result.output}"

    option_help = {
        "sync-project": ("-p", "-a", "-u"),
        "plan": ("-w", "-c", "-e", "-f", "-l", "-v", "-d"),
        "run": ("-w", "-c", "-e", "-a"),
        "show-config": ("-w", "-c", "-V"),
        "validate": ("-w", "-c", "-r"),
    }
    for command, aliases in option_help.items():
        output = runner.invoke(cli.app, [command, "-h"]).output
        assert all(alias in output for alias in aliases)


def test_sync_project_prewarms_selected_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    class FakeManager:
        def __init__(self, *, status):
            calls.append(("status", status))

        def ensure(self, context, *, upgrade=False):
            calls.append(("ensure", context.identity, upgrade))
            return type("Environment", (), {"root": tmp_path / context.identity})()

    monkeypatch.setattr(
        cli,
        "_project_environment_manager",
        lambda *, status: FakeManager(status=status),
    )

    result = CliRunner().invoke(cli.app, ["sync-project", "--project", "starter"])

    assert result.exit_code == 0, result.output
    assert ("ensure", "starter", False) in calls
    assert "Full project execution environment ready: starter" in result.output


def test_sync_project_skips_config_only_projects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "config-only"
    project.mkdir()
    (project / "project.yaml").write_text(
        "manifest_version: 1\nproject: config-only\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("NVFLOW_DISABLE_PROJECT_ENVIRONMENT", raising=False)
    monkeypatch.setattr(
        cli,
        "_project_environment_manager",
        lambda **_kwargs: pytest.fail("config-only project created an environment"),
    )
    monkeypatch.setattr(
        "nvflow.cli.project_environment.ProjectEnvironmentManager",
        lambda **_kwargs: pytest.fail("config-only project created an environment"),
    )

    result = CliRunner().invoke(
        cli.app,
        ["sync-project", "--project-dir", str(project)],
    )

    assert result.exit_code == 0, result.output
    assert "No project environment required: config-only" in result.output


def test_sync_project_prepares_yaml_only_project_with_pyproject(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "shared-only"
    project.mkdir()
    (project / "project.yaml").write_text(
        "manifest_version: 1\nproject: shared-only\n",
        encoding="utf-8",
    )
    (project / "pyproject.toml").write_text("[project]\nname='shared-only'\nversion='0'\n")
    calls = []

    class FakeManager:
        def __init__(self, *, status):
            del status

        def ensure(self, context, *, upgrade=False):
            calls.append((context.identity, upgrade))
            return type("Environment", (), {"root": tmp_path / "environment"})()

    monkeypatch.setattr(
        cli,
        "_project_environment_manager",
        lambda *, status: FakeManager(status=status),
    )
    result = CliRunner().invoke(
        cli.app,
        ["sync-project", "--project-dir", str(project)],
    )

    assert result.exit_code == 0, result.output
    assert calls == [("shared-only", False)]


def test_list_stages_requires_explicit_workflow() -> None:
    runner = CliRunner()

    unscoped = runner.invoke(cli.app, ["list-stages"])

    assert unscoped.exit_code == 2
    assert "--workflow/-w" in _normalized_output(unscoped)
    assert "--config/-c" in _normalized_output(unscoped)


def test_list_environments_reports_selectors_and_handles_workflows_without_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CliRunner()

    listed = runner.invoke(
        cli.app,
        ["list-environments", "--workflow", "finance:grpo/qwen3_4b"],
    )
    none = runner.invoke(
        cli.app,
        ["list-environments", "--workflow", "starter:generate"],
    )
    monkeypatch.delenv("SEC_IDENTITY_EMAIL", raising=False)
    monkeypatch.delenv("SEC_IDENTITY_COMPANY", raising=False)
    unresolved = runner.invoke(
        cli.app,
        ["list-environments", "--workflow", "finance:sdg/template-based-sdg"],
    )

    assert listed.exit_code == 0, listed.output
    assert "Values accepted by --environment/-e" in listed.output
    assert "equivalence_llm_judge" in listed.output
    assert "finance_sec_search" in listed.output
    assert "mcqa" in listed.output
    assert "Total: 3 configured, 2 enabled" in listed.output
    assert none.exit_code == 0, none.output
    assert "No environment selectors configured." in none.output
    assert "-e is not applicable" in none.output
    assert unresolved.exit_code == 0, unresolved.output
    assert "No environment selectors configured." in unresolved.output


def test_dynamic_model_stage_expansion_is_silent() -> None:
    result = CliRunner().invoke(
        cli.app,
        ["list-stages", "--workflow", "finance:eval/demo"],
    )

    assert result.exit_code == 0, result.output
    assert "qwen3-4b" in result.output
    assert "Expanded" not in result.output


def test_sync_project_check_and_upgrade_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    class FakeManager:
        def check(self, context):
            calls.append(("check", context.identity))
            return None

        def ensure(self, context, *, upgrade=False):
            calls.append(("ensure", context.identity, upgrade))
            return type("Environment", (), {"root": tmp_path / context.identity})()

    manager = FakeManager()
    monkeypatch.setattr(cli, "_project_environment_manager", lambda **_kwargs: manager)

    checked = CliRunner().invoke(
        cli.app,
        ["sync-project", "--project", "starter", "--check"],
    )
    upgraded = CliRunner().invoke(
        cli.app,
        ["sync-project", "--project", "starter", "--upgrade"],
    )

    assert checked.exit_code == 1
    assert "Full project execution environment is not ready: starter" in checked.output
    assert upgraded.exit_code == 0, upgraded.output
    assert calls == [("check", "starter"), ("ensure", "starter", True)]


@pytest.mark.parametrize(
    "arguments",
    [
        ["list-projects"],
        ["list-recipes", "--project", "finance"],
        ["list-workflows", "--project", "starter"],
        ["list-stages", "--workflow", "starter:generate"],
        ["show-config", "--workflow", "starter:generate"],
        ["plan-recipe", "finance:quickstart"],
    ],
)
def test_structural_commands_do_not_prepare_project_environments(
    arguments: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NVFLOW_DISABLE_PROJECT_ENVIRONMENT", raising=False)
    monkeypatch.setattr(
        cli,
        "_project_environment_manager",
        lambda **_kwargs: pytest.fail("structural command prepared a project environment"),
    )
    monkeypatch.setattr(
        cli,
        "_handoff_project",
        lambda *_args, **_kwargs: pytest.fail(
            "structural command handed off to a project environment"
        ),
    )

    result = CliRunner().invoke(cli.app, arguments)

    assert result.exit_code == 0, result.output


def test_semantic_validation_hands_off_selected_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contexts = []
    monkeypatch.delenv("NVFLOW_DISABLE_PROJECT_ENVIRONMENT", raising=False)
    monkeypatch.setattr(
        cli,
        "_handoff_project",
        lambda context: contexts.append(context),
    )

    result = CliRunner().invoke(
        cli.app,
        ["validate", "--workflow", "starter:generate"],
    )

    assert result.exit_code == 0, result.output
    assert [context.identity for context in contexts] == ["starter"]


def test_validation_does_not_compute_submission_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "nvflow.project.context.ProjectContext.source_provenance",
        lambda *_args, **_kwargs: pytest.fail("validation computed full Git provenance"),
    )

    result = CliRunner().invoke(
        cli.app,
        ["validate", "--workflow", "starter:generate"],
    )

    assert result.exit_code == 0, result.output


def test_validation_accepts_environment_selection() -> None:
    result = CliRunner().invoke(
        cli.app,
        [
            "validate",
            "--workflow",
            "starter:generate",
            "-e",
            "dataset-a",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Environment selection: dataset-a" in result.output


@pytest.mark.parametrize(
    "arguments",
    [
        ["plan", "--cluster", "test-cluster"],
        ["validate"],
        ["show-config"],
        ["run", "--all", "--cluster", "test-cluster"],
        ["run", "generate_answer", "--cluster", "test-cluster"],
    ],
)
def test_workflow_commands_require_config_or_selector(arguments: list[str]) -> None:
    result = CliRunner().invoke(cli.app, arguments)

    assert result.exit_code == 2
    output = _normalized_output(result)
    assert "One of --config or --workflow" in output
    assert "required" in output


@pytest.mark.parametrize(
    "arguments",
    [
        ["plan", "--cluster", "test-cluster"],
        ["validate"],
        ["show-config"],
        ["run", "--all", "--cluster", "test-cluster"],
        ["run", "generate_answer", "--cluster", "test-cluster"],
        ["list-stages"],
    ],
)
def test_workflow_commands_reject_config_and_selector_together(
    arguments: list[str],
) -> None:
    result = CliRunner().invoke(
        cli.app,
        [
            *arguments,
            "--config",
            str(STARTER_CONFIG),
            "--workflow",
            "starter:generate",
        ],
    )

    assert result.exit_code == 2
    output = _normalized_output(result)
    assert "Use either --config or --workflow" in output
    assert "not both" in output


def test_selector_drives_validation_config_and_stage_discovery() -> None:
    runner = CliRunner()

    validate_result = runner.invoke(
        cli.app,
        ["validate", "--workflow", "starter:generate"],
    )
    config_result = runner.invoke(
        cli.app,
        [
            "show-config",
            "--workflow",
            "starter:generate",
            "--view",
            "source",
        ],
    )
    stages_result = runner.invoke(
        cli.app,
        ["list-stages", "--workflow", "starter:generate"],
    )
    stage_info_result = runner.invoke(
        cli.app,
        [
            "stage-info",
            "generate_answer",
            "--workflow",
            "starter:generate",
        ],
    )

    assert validate_result.exit_code == 0, validate_result.output
    assert "Workflow: answer_generation" in validate_result.output
    assert config_result.exit_code == 0, config_result.output
    assert "project: starter" not in config_result.output
    assert stages_result.exit_code == 0, stages_result.output
    assert "Stages in starter:generate" in stages_result.output
    assert stage_info_result.exit_code == 0, stage_info_result.output
    assert "Stage: starter.answer_generation.generate_answer" in stage_info_result.output


def test_structural_stage_and_config_views_do_not_require_runtime_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SEC_IDENTITY_EMAIL", raising=False)
    monkeypatch.delenv("SEC_IDENTITY_COMPANY", raising=False)
    monkeypatch.setattr(
        cli,
        "WorkflowRunner",
        lambda *_args, **_kwargs: pytest.fail("stage listing constructed WorkflowRunner"),
    )
    monkeypatch.setattr(
        cli,
        "load_workflow",
        lambda *_args, **_kwargs: pytest.fail("unresolved view loaded resolved workflow"),
    )
    runner = CliRunner()

    stages = runner.invoke(
        cli.app,
        ["list-stages", "-w", "finance:download/demo"],
    )
    source = runner.invoke(
        cli.app,
        ["show-config", "-w", "finance:download/base", "-V", "source"],
    )
    merged = runner.invoke(
        cli.app,
        ["show-config", "-w", "finance:download/demo", "-V", "merged"],
    )
    diff = runner.invoke(
        cli.app,
        ["show-config", "-w", "finance:download/demo", "-V", "diff"],
    )

    assert stages.exit_code == 0, stages.output
    assert "Stages in finance:download/demo" in stages.output
    assert source.exit_code == 0, source.output
    assert "${oc.env:SEC_IDENTITY_EMAIL}" in source.output
    assert merged.exit_code == 0, merged.output
    assert "${oc.env:SEC_IDENTITY_EMAIL}" in merged.output
    assert diff.exit_code == 0, diff.output
    assert "scope: demo" in diff.output


def test_positional_stage_scopes_validation_and_config() -> None:
    runner = CliRunner()

    validated = runner.invoke(
        cli.app,
        ["validate", "generate_answer", "--workflow", "starter:generate"],
    )
    configured = runner.invoke(
        cli.app,
        [
            "show-config",
            "generate_answer",
            "--workflow",
            "starter:generate",
        ],
    )
    unknown = runner.invoke(
        cli.app,
        ["validate", "missing", "--workflow", "starter:generate"],
    )

    assert validated.exit_code == 0, validated.output
    assert "Selected stage configuration is valid" in validated.output
    assert "Stages: 1 selected (generate_answer)" in validated.output
    assert configured.exit_code == 0, configured.output
    assert list(yaml.safe_load(configured.output)["stages"]) == ["generate_answer"]
    assert unknown.exit_code == 1
    assert "Stage 'missing' not found" in unknown.output


def test_run_all_option_rejects_positional_stage_names() -> None:
    result = CliRunner().invoke(
        cli.app,
        [
            "run",
            "generate_answer",
            "--all",
            "--workflow",
            "starter:generate",
            "--cluster",
            "test-cluster",
        ],
    )

    assert result.exit_code == 2
    assert "stage name(s) or --all, but not both" in _normalized_output(result).lower()


def test_run_requires_stage_names_or_explicit_all() -> None:
    result = CliRunner().invoke(
        cli.app,
        [
            "run",
            "--workflow",
            "starter:generate",
            "--cluster",
            "test-cluster",
        ],
    )

    assert result.exit_code == 2
    assert "provide stage name(s) or --all" in _normalized_output(result).lower()


def test_run_all_command_is_removed() -> None:
    result = CliRunner().invoke(cli.app, ["run-all"])

    assert result.exit_code == 2
    assert "No such command" in result.output


def test_run_commands_accept_selector_without_submitting(monkeypatch) -> None:
    calls: list[tuple[Any, ...]] = []

    class FakeRunner:
        def __init__(self, config: str | Path, **kwargs: Any) -> None:
            calls.append(("init", Path(config), kwargs.get("project_dir")))

        def run(self, **kwargs: Any) -> None:
            calls.append(("run", kwargs))

    monkeypatch.setattr(cli, "WorkflowRunner", FakeRunner)
    runner = CliRunner()

    selected = runner.invoke(
        cli.app,
        [
            "run",
            "generate_answer",
            "--workflow",
            "starter:generate",
            "--cluster",
            "test-cluster",
        ],
    )
    all_stages = runner.invoke(
        cli.app,
        [
            "run",
            "--all",
            "--workflow",
            "starter:generate",
            "--cluster",
            "test-cluster",
        ],
    )

    assert selected.exit_code == 0, selected.output
    assert all_stages.exit_code == 0, all_stages.output
    assert [call[0] for call in calls] == ["init", "run", "init", "run"]
    assert all(call[1] == STARTER_CONFIG.resolve() for call in calls if call[0] == "init")


def test_list_recipes_defaults_to_all_projects() -> None:
    result = CliRunner().invoke(cli.app, ["list-recipes"])

    assert result.exit_code == 0, result.output
    assert "finance:quickstart" in result.output
    assert "finance:template-based-sft" in result.output
    assert "finance:document-grounded-local-search-grpo" in result.output
    assert "finance:vals-no-web" in result.output
    assert "finance:vals11" in result.output
    assert "hopchain:quickstart" in result.output
    assert "hopchain:vision-reasoning" in result.output
    assert "starter:quickstart" not in result.output
    assert "Total: 7 recipes" in result.output


def test_list_workflows_uses_compact_tables_for_maintained_projects() -> None:
    runner = CliRunner()

    finance = runner.invoke(cli.app, ["list-workflows", "--project", "finance"])
    finance_with_templates = runner.invoke(
        cli.app,
        ["list-workflows", "--project", "finance", "--all"],
    )
    starter = runner.invoke(cli.app, ["list-workflows", "--project", "starter"])
    starter_with_templates = runner.invoke(
        cli.app,
        [
            "list-workflows",
            "--project",
            "starter",
            "--all",
        ],
    )
    selected_starter = runner.invoke(
        cli.app,
        [
            "show-config",
            "--workflow",
            "starter:generate",
            "--view",
            "source",
        ],
    )

    assert finance.exit_code == 0, finance.output
    assert "…" not in finance.output
    assert "finance:sdg/document-grounded-sdg-qwen-demo" in finance.output
    assert all(column in finance.output for column in ("Selector", "Type", "Stages"))
    assert "Total: 51 workflow definitions" in finance.output
    assert len(finance.output.splitlines()) < 58
    assert finance.output.index("finance:download/demo") < finance.output.index(
        "finance:eval/baselines"
    )
    for selector in ("finance:download/demo", "finance:download/smoke", "finance:download/sp500"):
        assert selector in finance.output
    assert finance_with_templates.exit_code == 0, finance_with_templates.output
    assert "Status" in finance_with_templates.output
    assert "template" in finance_with_templates.output
    assert starter.exit_code == 0, starter.output
    assert "starter:generate" in starter.output
    assert "starter — Minimal runnable NVFlow project starter" in starter.output
    assert all(column in starter.output for column in ("Selector", "Type", "Stages"))
    assert "Total: 1 workflow definition" in starter.output
    assert "Status" not in starter.output
    assert starter_with_templates.exit_code == 0, starter_with_templates.output
    assert "Status" in starter_with_templates.output
    assert "runnable" in starter_with_templates.output
    assert selected_starter.exit_code == 0, selected_starter.output
    assert "project: starter" not in selected_starter.output


def test_list_workflows_recipe_uses_same_table_in_dependency_order() -> None:
    result = CliRunner().invoke(
        cli.app,
        ["list-workflows", "--recipe", "finance:quickstart"],
    )

    assert result.exit_code == 0, result.output
    assert all(column in result.output for column in ("Selector", "Type", "Stages", "Optional"))
    assert result.output.index("finance:eval/demo") < result.output.index("finance:download/demo")
    assert "Total: 5 workflow definitions" in result.output
    assert "Order: recipe dependency order" in result.output
    assert "Dependencies: nvflow plan-recipe finance:quickstart" in result.output
    assert "Depends on:" not in result.output


def test_stage_discovery_uses_exact_selectors() -> None:
    runner = CliRunner()

    exact = runner.invoke(
        cli.app,
        ["list-stages", "--workflow", "starter:generate"],
    )
    invalid_short_selector = runner.invoke(
        cli.app,
        [
            "list-stages",
            "--workflow",
            "answer_generation",
        ],
    )
    missing_workflow = runner.invoke(
        cli.app,
        [
            "stage-info",
            "generate_answer",
        ],
    )
    exact_stage_info = runner.invoke(
        cli.app,
        [
            "stage-info",
            "generate_answer",
            "--workflow",
            "starter:generate",
        ],
    )
    assert exact.exit_code == 0, exact.output
    assert "generate_answer" in exact.output
    assert invalid_short_selector.exit_code == 2
    assert "project:relative/workflow" in _normalized_output(invalid_short_selector)
    assert missing_workflow.exit_code == 2
    assert "--workflow" in missing_workflow.output
    assert exact_stage_info.exit_code == 0, exact_stage_info.output


def test_stage_info_accepts_dotted_stage_name_with_exact_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DottedStage(cli.BaseStage):
        def execute(self, config, cluster, expname, run_after=None):
            del config, cluster, expname, run_after

    definition = type(
        "Definition",
        (),
        {
            "pipeline_stages": ["prepare.v2"],
            "config": {"stages": {"prepare.v2": {}}},
            "project": "acme.ml",
            "workflow_name": "training",
            "project_context": object(),
            "resolve_stage_class": lambda self, _stage: DottedStage,
        },
    )()
    monkeypatch.setattr(cli, "load_workflow", lambda *_args, **_kwargs: definition)

    result = CliRunner().invoke(
        cli.app,
        ["stage-info", "prepare.v2", "--workflow", "starter:generate"],
    )

    assert result.exit_code == 0, result.output
    assert "Stage: acme.ml.training.prepare.v2" in result.output
