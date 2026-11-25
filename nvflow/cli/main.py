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
"""Main CLI entry point for NVFlow.

Usage:
    nvflow list-projects
    nvflow list-recipes [--project PROJECT]
    nvflow list-workflows [--project PROJECT | --recipe PROJECT:RECIPE]
    nvflow sync-project [--project PROJECT | --project-dir PROJECT_DIR | --all]
    nvflow list-stages (--workflow PROJECT:WORKFLOW | --config CONFIG)
    nvflow list-environments (--workflow PROJECT:WORKFLOW | --config CONFIG)
    nvflow stage-info STAGE --workflow PROJECT:WORKFLOW
    nvflow run STAGE [STAGE...] (--workflow WORKFLOW | --config CONFIG) --cluster CLUSTER
    nvflow run --all (--workflow WORKFLOW | --config CONFIG) --cluster CLUSTER
    nvflow plan [STAGE...] (--workflow WORKFLOW | --config CONFIG) --cluster CLUSTER
    nvflow show-config [STAGE...] (--workflow WORKFLOW | --config CONFIG) [-e ENVIRONMENT]
    nvflow validate [STAGE...] (--workflow WORKFLOW | --config CONFIG) [-e ENVIRONMENT]
    nvflow status [STAGE...] [--workflow WORKFLOW | --config CONFIG | --recipe RECIPE] --cluster CLUSTER
    nvflow version

Examples:
    # Discover workflows and list active stages
    nvflow list-recipes --project finance
    nvflow list-workflows --project finance
    nvflow list-stages --workflow finance:grpo/qwen3_4b

    # Get stage info from one workflow definition
    nvflow stage-info training --workflow finance:grpo/qwen3_4b

    # Inspect one stage without affecting sibling stages
    nvflow show-config training --workflow finance:grpo/qwen3_4b
    nvflow validate training --workflow finance:grpo/qwen3_4b
    nvflow plan training --workflow finance:grpo/qwen3_4b --cluster my_cluster

    # Run workflow
    nvflow run --all --workflow finance:sft/qwen3_4b --cluster my_cluster

    # GRPO: run a single stage
    nvflow run collect_rollouts --workflow finance:grpo/qwen3_4b --cluster my_cluster

    # GRPO: run a single stage for one environment
    nvflow run collect_rollouts -w finance:grpo/qwen3_4b --cluster my_cluster -e equivalence_llm_judge

    # GRPO: run a stage for multiple environments
    nvflow run training -w finance:grpo/qwen3_4b --cluster my_cluster -e mcqa -e equivalence_llm_judge

    # GRPO: run all stages
    nvflow run --all --workflow finance:grpo/qwen3_4b --cluster my_cluster
"""

import json  # noqa: E402
import os  # noqa: E402

os.environ.setdefault("LITELLM_LOG", "ERROR")

# Limit BLAS/OpenMP thread pools to 1 on the login node. The nvflow CLI only
# orchestrates Slurm jobs -- it never does BLAS compute. Without this cap,
# importing scipy/numpy spawns nproc threads (~96-188 on shared login nodes),
# causing a kernel futex storm that hangs the process for 2-30 minutes.
# Uses setdefault so users can override (e.g., OMP_NUM_THREADS=4 nvflow ...).
# Slurm jobs are unaffected -- they run inside containers with their own env.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import contextlib  # noqa: E402
import importlib.util  # noqa: E402
import io  # noqa: E402
import textwrap  # noqa: E402
import warnings  # noqa: E402
from collections import defaultdict  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import TYPE_CHECKING, Annotated  # noqa: E402

import click  # noqa: E402
import typer  # noqa: E402
import yaml  # noqa: E402
from rich.table import Table  # noqa: E402
from rich.text import Text  # noqa: E402
from typer.core import TyperGroup  # noqa: E402

from nvflow import __version__  # noqa: E402
from nvflow.cli.output import (  # noqa: E402
    diagnostic,
    emit,
    emit_plan_text,
    emit_wide,
    emit_yaml,
    error,
    heading,
    line,
    route_stdout_to_stderr,
    status,
    success,
    warning,
    write_stderr,
    write_stdout,
)
from nvflow.cli.workflow_selection import (  # noqa: E402
    SelectedWorkflow,
    discover_workflows,
    require_workflow,
)
from nvflow.core import BaseStage, load_workflow  # noqa: E402
from nvflow.core.workflow_structure import description_of  # noqa: E402
from nvflow.execution import (  # noqa: E402
    LocalTargetFilesystem,
    TargetBinding,
    TargetResolutionError,
    WorkflowPlanner,
    WorkflowRunner,
    preflight_target_config,
    resolve_target,
)
from nvflow.execution.engines import ActiveJob, engine_for  # noqa: E402
from nvflow.execution.job_labels import label_value, workflow_label  # noqa: E402
from nvflow.integrations.nemo_skills.filesystem import target_filesystem  # noqa: E402
from nvflow.integrations.nemo_skills.target import (  # noqa: E402
    bind_target_config,
)
from nvflow.project import (  # noqa: E402
    ProjectContext,
    ProjectManifestError,
    ProjectResolutionError,
    ProjectResolver,
    resolve_project_directory,
)
from nvflow.recipe import RecipeDefinitionError, RecipeSelector  # noqa: E402
from nvflow.recipe.planning import plan_recipe  # noqa: E402
from nvflow.recipe.resolver import resolve_recipe  # noqa: E402

if TYPE_CHECKING:
    from nvflow.core.workflow_definition import WorkflowDefinition
    from nvflow.recipe.planning import RecipeGraphPlan

# Importing NeMo-Run loads its Lepton executor, whose SDK imports Starlette's
# deprecated WSGI middleware; NVFlow never serves a WSGI app.
warnings.filterwarnings("ignore", message=r"starlette\.middleware\.wsgi is deprecated")


_COMMAND_HELP_ORDER = (
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
)
_COMMAND_HELP_RANK = {name: index for index, name in enumerate(_COMMAND_HELP_ORDER)}


class _WorkflowHelpGroup(TyperGroup):
    """List commands in workflow lifecycle order."""

    def list_commands(self, ctx: click.Context) -> list[str]:
        commands = super().list_commands(ctx)
        return sorted(
            commands,
            key=lambda name: (_COMMAND_HELP_RANK.get(name, len(_COMMAND_HELP_RANK)), name),
        )


app = typer.Typer(
    name="nvflow",
    help=(
        "NVFlow - the workflow orchestration platform for NeMo-based data "
        "generation, training, and evaluation"
    ),
    add_completion=False,
    cls=_WorkflowHelpGroup,
    context_settings={"help_option_names": ["-h", "--help"]},
)

_CONFIG_OPTION_HELP = "Path to workflow configuration file"
_WORKFLOW_OPTION_HELP = "Workflow selector (for example finance:grpo/qwen3_4b)"
_CLUSTER_OPTION_HELP = "Target configuration name (required for planning and submission)"
_PROJECT_DIR_OPTION_HELP = "Explicit external project root"
_PROJECT_PATH_OPTION_HELP = "Ordered project search path (repeatable)"


def _handoff_project(context: ProjectContext) -> None:
    """Continue this CLI command inside the selected project environment."""
    from nvflow.cli.project_environment import handoff_to_project_environment

    handoff_to_project_environment(context, status=status)


def _working_project() -> ProjectContext | None:
    """Return the project containing the working directory, as uv finds its project."""
    for directory in (Path.cwd(), *Path.cwd().parents):
        try:
            return resolve_project_directory(ProjectResolver(), directory)
        except ProjectResolutionError:
            continue
    return None


def _status_project() -> ProjectContext | None:
    """Return the project whose environment ``status`` continues in, if any."""
    project = _working_project()
    if project is None and not _execution_stack_installed():
        # A plain `uv sync` of the checkout omits the execution stack; every
        # project environment carries it, and the Starter's is the smallest.
        with contextlib.suppress(ProjectManifestError):
            project = ProjectResolver().resolve_reference("starter")
    return project


def _execution_stack_installed() -> bool:
    """Return whether this environment has the NeMo-Skills execution stack."""
    return importlib.util.find_spec("nemo_skills") is not None


def _project_environment_manager(*, status):
    from nvflow.project.environment import ProjectEnvironmentManager

    return ProjectEnvironmentManager(status=status)


def _resolve_project_contexts(
    *,
    project: str | None,
    project_dir: str | None,
    project_paths: tuple[str, ...],
) -> tuple[ProjectContext, ...]:
    """Resolve selected or discoverable projects without importing stage code."""
    resolver = ProjectResolver(search_paths=project_paths)
    try:
        if project_dir is not None:
            return (
                resolve_project_directory(
                    resolver,
                    project_dir,
                    expected_project=project,
                ),
            )
        names = (project,) if project else tuple(resolver.project_names())
        return tuple(resolver.resolve_named(name) for name in names)
    except ProjectManifestError as exc:
        raise typer.BadParameter(
            str(exc),
            param_hint="--project/--project-dir",
        ) from exc


def _wrapped_detail(label: str, value: str) -> None:
    """Write one readable, indented recipe-plan detail."""
    prefix = f"   {label}: "
    line(
        textwrap.fill(
            value,
            width=100,
            initial_indent=prefix,
            subsequent_indent=" " * len(prefix),
        )
    )


def _workflow_table(*, show_optional: bool = False, show_status: bool = False) -> Table:
    """Build the shared compact table used by every workflow listing."""
    table = Table(
        show_header=True,
        header_style="bold cyan",
        box=None,
        pad_edge=False,
        collapse_padding=True,
    )
    table.add_column("#", style="dim", justify="right", no_wrap=True)
    table.add_column("Selector", style="green", no_wrap=True, overflow="ignore")
    table.add_column("Type", no_wrap=True)
    table.add_column("Stages", justify="right", no_wrap=True)
    if show_optional:
        table.add_column("Optional", no_wrap=True)
    if show_status:
        table.add_column("Status", no_wrap=True)
    table.add_column("Description")
    return table


def _emit_recipe_graph_workflows(graph_plan: "RecipeGraphPlan") -> None:
    """Render fully resolved graph-plan nodes in dependency order."""
    for index, node in enumerate(graph_plan.workflows, 1):
        optional = " [optional]" if node.optional else ""
        line(f"{index}. {node.selector}{optional}", style="bold cyan")
        _wrapped_detail("Type", node.workflow_type)
        _wrapped_detail("Depends on", ", ".join(node.depends_on) or "—")
        line()


@app.command(name="list-projects")
def list_projects(
    project_dir: str | None = typer.Option(
        None,
        "--project-dir",
        help=_PROJECT_DIR_OPTION_HELP,
    ),
    project_path: Annotated[
        list[str] | None,
        typer.Option("--project-path", help=_PROJECT_PATH_OPTION_HELP),
    ] = None,
):
    """List discoverable NVFlow projects."""
    contexts = _resolve_project_contexts(
        project=None,
        project_dir=project_dir,
        project_paths=tuple(project_path or ()),
    )
    if not contexts:
        line("No projects found.")
        return

    table = Table(show_header=True, header_style="bold cyan", box=None)
    table.add_column("Project", style="green", no_wrap=True)
    table.add_column("Description")
    table.add_column("Recipes", justify="right")
    for context in contexts:
        table.add_row(
            context.identity,
            context.manifest.description or "",
            str(len(context.manifest.recipes)),
        )
    emit(table)
    project_label = "project" if len(contexts) == 1 else "projects"
    line(f"Total: {len(contexts)} {project_label}")


@app.command(name="list-recipes")
def list_recipes(
    project: str | None = typer.Option(
        None,
        "--project",
        "-p",
        help="Project containing the recipes",
    ),
    project_dir: str | None = typer.Option(
        None,
        "--project-dir",
        help=_PROJECT_DIR_OPTION_HELP,
    ),
    project_path: Annotated[
        list[str] | None,
        typer.Option("--project-path", help=_PROJECT_PATH_OPTION_HELP),
    ] = None,
):
    """List recipe DAGs, optionally filtered to one project."""
    contexts = _resolve_project_contexts(
        project=project,
        project_dir=project_dir,
        project_paths=tuple(project_path or ()),
    )
    contexts_with_recipes = [context for context in contexts if context.manifest.recipes]
    if not contexts_with_recipes:
        if len(contexts) == 1:
            context = contexts[0]
            line(f"Project {context.identity!r} defines no recipes.")
            line(f"Next: nvflow list-workflows -p {context.identity}")
        else:
            line("No recipes found.")
        return

    recipe_count = 0
    for index, context in enumerate(contexts_with_recipes):
        if index:
            line()
        heading(
            f"{context.identity} — {context.manifest.description}"
            if context.manifest.description
            else context.identity
        )
        table = Table(show_header=True, header_style="bold cyan", box=None)
        table.add_column("Selector", style="green", no_wrap=True)
        table.add_column("Description")
        table.add_column("Workflows", justify="right")
        for recipe in context.manifest.recipes:
            table.add_row(
                str(RecipeSelector(context.identity, recipe.name)),
                recipe.description or "",
                str(len(recipe.workflows)),
            )
        emit(table)
        recipe_count += len(context.manifest.recipes)
    recipe_label = "recipe" if recipe_count == 1 else "recipes"
    line(f"Total: {recipe_count} {recipe_label}")


@app.command(name="list-workflows")
def list_workflows(
    project: str | None = typer.Option(
        None,
        "--project",
        "-p",
        help="Show reusable workflow definitions in one project",
    ),
    recipe: str | None = typer.Option(
        None,
        "--recipe",
        "-r",
        help="Show workflow nodes in one recipe (project:recipe)",
    ),
    project_dir: str | None = typer.Option(
        None,
        "--project-dir",
        help=_PROJECT_DIR_OPTION_HELP,
    ),
    project_path: Annotated[
        list[str] | None,
        typer.Option("--project-path", help=_PROJECT_PATH_OPTION_HELP),
    ] = None,
    include_all: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Include workflow files explicitly marked as non-runnable templates",
    ),
):
    """List reusable workflow definitions or one recipe's workflow DAG."""
    if project and recipe:
        raise typer.BadParameter(
            "Use either --project or --recipe, not both",
            param_hint="--project/--recipe",
        )
    if recipe and include_all:
        raise typer.BadParameter(
            "--all applies only to reusable workflow discovery",
            param_hint="--all",
        )
    project_paths = tuple(project_path or ())
    if recipe:
        try:
            resolved = resolve_recipe(
                recipe,
                ProjectResolver(search_paths=project_paths),
                project_dir=project_dir,
            )
        except (ProjectManifestError, RecipeDefinitionError) as exc:
            raise typer.BadParameter(str(exc), param_hint="--recipe") from exc

        heading(
            f"{resolved.selector} — {resolved.definition.description}"
            if resolved.definition.description
            else str(resolved.selector)
        )
        line()
        table = _workflow_table(show_optional=True)
        for index, selected in enumerate(resolved.topological_workflows(), 1):
            table.add_row(
                str(index),
                str(selected.selector),
                selected.workflow_type,
                str(selected.stage_count),
                "yes" if selected.node.optional else "no",
                selected.description or "",
            )
        emit_wide(table)
        definition_label = (
            "workflow definition" if len(resolved.workflows) == 1 else "workflow definitions"
        )
        line(f"Total: {len(resolved.workflows)} {definition_label}")
        line("Order: recipe dependency order")
        line(f"Dependencies: nvflow plan-recipe {resolved.selector}")
        return

    listings = discover_workflows(
        project=project,
        project_dir=project_dir,
        project_paths=project_paths,
        include_all=include_all,
    )
    if not listings:
        line("No workflow definitions found.")
        return

    by_project = defaultdict(list)
    for listing in listings:
        by_project[listing.selector.project].append(listing)

    for project_name, project_listings in by_project.items():
        line()
        description = project_listings[0].project_description
        heading(f"{project_name} — {description}" if description else project_name)
        line()
        table = _workflow_table(show_status=include_all)
        for index, listing in enumerate(
            sorted(project_listings, key=lambda item: str(item.selector)),
            1,
        ):
            row = [
                str(index),
                str(listing.selector),
                listing.workflow_type,
                str(listing.stage_count),
            ]
            if include_all:
                row.append("template" if listing.template else "runnable")
            row.append(listing.description or "")
            table.add_row(*row)
        emit_wide(table)
    line()
    definition_label = "workflow definition" if len(listings) == 1 else "workflow definitions"
    project_suffix = f" across {len(by_project)} projects" if len(by_project) > 1 else ""
    line(f"Total: {len(listings)} {definition_label}{project_suffix}")


@app.command(name="sync-project")
def sync_project(
    project: str | None = typer.Option(
        None,
        "--project",
        "-p",
        help="Prepare one project's full execution environment",
    ),
    project_dir: str | None = typer.Option(
        None,
        "--project-dir",
        help=_PROJECT_DIR_OPTION_HELP,
    ),
    project_path: Annotated[
        list[str] | None,
        typer.Option("--project-path", help=_PROJECT_PATH_OPTION_HELP),
    ] = None,
    all_projects: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Prepare every discoverable project",
    ),
    check: bool = typer.Option(
        False,
        "--check",
        help="Check readiness without creating or updating environments",
    ),
    upgrade: bool = typer.Option(
        False,
        "--upgrade",
        "-u",
        help="Re-resolve project dependencies and refresh the cached lock",
    ),
):
    """Prewarm full isolated project execution environments for CI or offline use."""
    if all_projects and (project or project_dir):
        raise typer.BadParameter(
            "--all cannot be combined with --project or --project-dir",
            param_hint="--all",
        )
    if check and upgrade:
        raise typer.BadParameter(
            "--check cannot be combined with --upgrade",
            param_hint="--check/--upgrade",
        )
    resolver = ProjectResolver(search_paths=tuple(project_path or ()))
    try:
        if all_projects:
            contexts = tuple(resolver.resolve_named(name) for name in resolver.project_names())
        elif project_dir:
            contexts = (
                resolve_project_directory(
                    resolver,
                    project_dir,
                    expected_project=project,
                ),
            )
        elif project:
            contexts = (resolver.resolve_named(project),)
        else:
            contexts = (resolver.resolve_current(),)
        if not contexts:
            raise ProjectManifestError("No projects found to prepare")

        manager = None
        missing = []
        for context in contexts:
            if not (context.root / "pyproject.toml").is_file():
                success(f"No project environment required: {context.identity}")
                continue
            if manager is None:
                manager = _project_environment_manager(status=status)
            if check:
                environment = manager.check(context)
                if environment is None:
                    missing.append(context.identity)
                    warning(f"Full project execution environment is not ready: {context.identity}")
                    continue
            else:
                environment = manager.ensure(context, upgrade=upgrade)
            success(f"Full project execution environment ready: {context.identity}")
            diagnostic(f"Cache: {environment.root}")
        if missing:
            raise ProjectManifestError("Project environments are not ready: " + ", ".join(missing))
    except Exception as exc:
        error(f"Could not prepare project environment: {exc}")
        raise typer.Exit(1) from exc


@app.command(name="list-stages")
def list_stages(
    config: str | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Show active stages from a workflow configuration file",
    ),
    workflow: str | None = typer.Option(
        None,
        "--workflow",
        "-w",
        help="Exact workflow selector (for example finance:grpo/qwen3_4b)",
    ),
    project_dir: str | None = typer.Option(
        None, "--project-dir", help="Explicit external project root"
    ),
    project_path: Annotated[
        list[str] | None,
        typer.Option("--project-path", help="Ordered project search path (repeatable)"),
    ] = None,
):
    """List the configured stages and implementations in one workflow."""

    if not (config or workflow):
        raise typer.BadParameter(
            "Use --workflow/-w or --config/-c",
            param_hint="--workflow/--config",
        )

    selection = require_workflow(
        config=config,
        workflow=workflow,
        project_dir=project_dir,
        project_paths=tuple(project_path or ()),
    )
    try:
        from nvflow.cli.workflow_inspection import inspect_workflow_structure

        structure = inspect_workflow_structure(
            selection.config_path,
            project_dir=selection.project_dir,
            project_search_paths=tuple(project_path or ()),
        )

        line()
        heading(f"Stages in {selection.display_name}:")
        line(f"Project: {structure.project}")
        line(f"Workflow: {structure.workflow_name} ({structure.workflow_type})")
        line()

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("#", style="dim", width=4)
        table.add_column("Stage", style="green")
        table.add_column("Description")
        table.add_column("Uses", style="cyan")
        table.add_column("Dependencies", style="yellow")
        table.add_column("Full Path", style="dim")

        for idx, stage in enumerate(structure.stage_entries, 1):
            deps = ", ".join(stage.dependencies) or "None"
            full_path = f"{structure.project}.{structure.workflow_name}.{stage.name}"
            table.add_row(
                str(idx),
                stage.name,
                stage.description or "",
                stage.uses or "missing",
                deps,
                full_path,
            )

        emit_wide(table)
        line()
        if structure.runtime_expanded_stages:
            line(
                "Runtime-expanded stage template(s): "
                + ", ".join(structure.runtime_expanded_stages)
            )
            line()
        line(f"Total: {len(structure.stage_entries)} stages")

    except typer.BadParameter:
        raise
    except Exception as e:
        error(f"Could not load configuration: {e}")
        raise typer.Exit(1) from e


@app.command(name="list-environments")
def list_environments(
    config: str | None = typer.Option(
        None,
        "--config",
        "-c",
        help="List environment selectors from a workflow configuration file",
    ),
    workflow: str | None = typer.Option(
        None,
        "--workflow",
        "-w",
        help="Exact workflow selector (for example finance:grpo/qwen3_4b)",
    ),
    project_dir: str | None = typer.Option(
        None, "--project-dir", help="Explicit external project root"
    ),
    project_path: Annotated[
        list[str] | None,
        typer.Option("--project-path", help="Ordered project search path (repeatable)"),
    ] = None,
):
    """List the workflow values accepted by --environment/-e."""
    selection = require_workflow(
        config=config,
        workflow=workflow,
        project_dir=project_dir,
        project_paths=tuple(project_path or ()),
    )
    try:
        from nvflow.cli.workflow_inspection import inspect_environment_selectors

        environments = inspect_environment_selectors(
            selection.config_path,
            project_dir=selection.project_dir,
            project_search_paths=tuple(project_path or ()),
        )
        heading(f"Environment selectors for {selection.display_name}")
        line("Values accepted by --environment/-e:")
        line()
        if not isinstance(environments, dict) or not environments:
            line("No environment selectors configured.")
            line("This workflow runs each selected stage once; -e is not applicable.")
            return

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Name", style="green")
        table.add_column("Enabled", style="cyan")
        enabled_count = 0
        for name, enabled in environments.items():
            enabled_count += int(enabled)
            table.add_row(str(name), "yes" if enabled else "no")
        emit(table)
        line()
        line(f"Total: {len(environments)} configured, {enabled_count} enabled")
    except Exception as exc:
        error(f"Could not list environment selectors: {exc}")
        raise typer.Exit(1) from exc


@app.command(name="list-stage-types")
def list_stage_types(
    namespace: str | None = typer.Option(
        None,
        "--namespace",
        "-n",
        help="Filter by namespace, e.g. nemo_gym",
    ),
) -> None:
    """List curated platform-owned stage types usable through workflow ``uses``."""
    from nvflow.stages.catalog import list_stage_types as catalog

    stage_types = catalog()
    if namespace:
        stage_types = tuple(
            stage_type for stage_type in stage_types if stage_type.name.startswith(f"{namespace}.")
        )
    if not stage_types:
        line("No shared stage types found.")
        return
    table = Table(show_header=True, header_style="bold cyan", box=None)
    table.add_column("Type", style="green", no_wrap=True)
    table.add_column("Description")
    for stage_type in stage_types:
        table.add_row(stage_type.name, stage_type.description)
    emit(table)
    type_label = "stage type" if len(stage_types) == 1 else "stage types"
    line(f"Total: {len(stage_types)} shared {type_label}")


@app.command(name="stage-type-info")
def stage_type_info(
    stage_type: str = typer.Argument(..., help="Shared stage type, e.g. nemo_gym.collect"),
) -> None:
    """Show details for one curated shared stage type."""
    from nvflow.stages.catalog import get_stage_type, resolve_stage_type

    try:
        spec = get_stage_type(stage_type)
        stage_class = resolve_stage_type(stage_type)
        heading(f"Shared stage type: {spec.name}")
        line(f"Class: {spec.module}.{spec.class_name}")
        line(f"Contract versions: {', '.join(map(str, spec.contract_versions))}")
        line(f"Documentation: {spec.documentation}")
        line(f"Description: {spec.description}")
        if stage_class.__doc__:
            line()
            line(stage_class.__doc__)
    except Exception as exc:
        error(f"Could not inspect shared stage type: {exc}")
        diagnostic("Use 'nvflow list-stage-types' to inspect available shared stage types")
        raise typer.Exit(1) from exc


def _execute_selected_workflow(
    selection: SelectedWorkflow,
    *,
    cluster: str,
    project_paths: tuple[str, ...],
    stages: list[str] | None,
    environment: list[str] | None,
) -> None:
    """Run a CLI-selected workflow while preserving the action stream contract."""
    runner = None
    caught_warnings: list[warnings.WarningMessage] = []
    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            if selection.project_context is not None:
                _handoff_project(selection.project_context)
            target_config = preflight_target_config(cluster)
            with route_stdout_to_stderr():
                runner = WorkflowRunner(
                    selection.config_path,
                    project_dir=selection.project_dir,
                    cluster=cluster,
                    project_search_paths=project_paths,
                    target_config_path=target_config,
                )
                if selection.project_context is None:
                    _handoff_project(runner.project_context)
                runner.run(stages=stages, environment=environment)
        warning_messages = _format_plan_warnings(caught_warnings)
        if warning_messages:
            _emit_plan_warnings(
                runner.definition,
                warning_messages,
                activity="runtime",
            )
    except Exception as exc:
        if runner is not None:
            warning_messages = _format_plan_warnings(caught_warnings)
            if warning_messages:
                _emit_plan_warnings(
                    runner.definition,
                    warning_messages,
                    activity="runtime",
                )
        error(exc)
        raise typer.Exit(1) from exc


@app.command()
def run(
    cluster: Annotated[
        str,
        typer.Option("--cluster", help=_CLUSTER_OPTION_HELP),
    ],
    stages: Annotated[
        list[str] | None,
        typer.Argument(help="Stage name(s) to run, or use --all"),
    ] = None,
    config: Annotated[
        str | None,
        typer.Option("--config", "-c", help=_CONFIG_OPTION_HELP),
    ] = None,
    workflow: Annotated[
        str | None,
        typer.Option(
            "--workflow",
            "-w",
            help=_WORKFLOW_OPTION_HELP,
        ),
    ] = None,
    environment: Annotated[
        list[str] | None,
        typer.Option("--environment", "-e", help="Run for specific environment(s) only"),
    ] = None,
    all_stages: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Run the complete configured workflow",
    ),
    project_dir: Annotated[
        str | None,
        typer.Option("--project-dir", help="Explicit external project root"),
    ] = None,
    project_path: Annotated[
        list[str] | None,
        typer.Option("--project-path", help="Ordered project search path (repeatable)"),
    ] = None,
):
    """Run selected stages or explicitly run the complete workflow."""
    selected_stages = list(stages or ())
    if all_stages == bool(selected_stages):
        raise typer.BadParameter(
            "Provide stage name(s) or --all, but not both",
            param_hint="STAGES/--all",
        )

    selection = require_workflow(
        config=config,
        workflow=workflow,
        project_dir=project_dir,
        project_paths=tuple(project_path or ()),
    )
    _execute_selected_workflow(
        selection,
        cluster=cluster,
        project_paths=tuple(project_path or ()),
        stages=None if all_stages else selected_stages,
        environment=environment,
    )


_STATUS_COLUMNS = (
    ("Job", "job_id"),
    ("Workflow", "workflow"),
    ("Stage", "stage"),
    ("Experiment", "experiment"),
    ("Name", "name"),
    ("State", "state"),
    ("Reason", "reason"),
    ("Time", "elapsed"),
    ("Where", "location"),
)
_STATUS_VERBOSE_COLUMNS = (
    ("Nodes", "nodes"),
    ("GPUs/Node", "gpus_per_node"),
    ("CPUs", "cpus"),
    ("Start", "start"),
    ("Account", "account"),
    ("Priority", "priority"),
)
_STATUS_RIGHT_ALIGNED = frozenset({"elapsed", "nodes", "gpus_per_node", "cpus", "priority"})


@dataclass(frozen=True)
class _StatusSelection:
    """The workflows, and optionally their stages, whose jobs ``status`` shows."""

    workflows: tuple[str, ...]
    stages: tuple[str, ...]
    description: str

    def select(self, jobs: tuple[ActiveJob, ...]) -> tuple[ActiveJob, ...]:
        selected = [
            job
            for job in jobs
            if job.workflow in self.workflows and (not self.stages or job.stage in self.stages)
        ]
        return tuple(sorted(selected, key=lambda job: self.workflows.index(str(job.workflow))))


@app.command(name="status")
def status_command(
    cluster: Annotated[
        str,
        typer.Option("--cluster", help=_CLUSTER_OPTION_HELP),
    ],
    stages: Annotated[
        list[str] | None,
        typer.Argument(help="Optional stage name(s) to show (requires a workflow selection)"),
    ] = None,
    config: str | None = typer.Option(
        None,
        "--config",
        "-c",
        help=_CONFIG_OPTION_HELP,
    ),
    workflow: str | None = typer.Option(
        None,
        "--workflow",
        "-w",
        help=_WORKFLOW_OPTION_HELP,
    ),
    recipe: str | None = typer.Option(
        None,
        "--recipe",
        "-r",
        help="Show the jobs of every workflow in a recipe (project:recipe)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Also show nodes, GPUs per node, CPUs, start time, account, and priority",
    ),
    project_dir: str | None = typer.Option(None, "--project-dir", help=_PROJECT_DIR_OPTION_HELP),
    project_path: Annotated[
        list[str] | None,
        typer.Option("--project-path", help=_PROJECT_PATH_OPTION_HELP),
    ] = None,
):
    """Show your queued and running jobs on a Slurm or Ray target.

    Select a workflow, some of its stages, or a recipe to show only their jobs.
    """
    if recipe and (config or workflow):
        raise typer.BadParameter(
            "Use --recipe or a workflow selection, not both",
            param_hint="--recipe/--workflow/--config",
        )
    if stages and not (config or workflow):
        raise typer.BadParameter(
            "Stage selections apply only to a workflow selection",
            param_hint="STAGES",
        )
    try:
        selection = _status_selection(
            config=config,
            workflow=workflow,
            recipe=recipe,
            stages=stages,
            project_dir=project_dir,
            project_paths=tuple(project_path or ()),
        )
        target = resolve_target(
            TargetBinding(cluster=cluster, source="override"),
            preflight_target_config(cluster),
        )
        listing = engine_for(target).active_jobs(target)
    except click.ClickException:
        raise
    except Exception as exc:
        error(exc)
        raise typer.Exit(1) from exc
    jobs = listing.jobs if selection is None else selection.select(listing.jobs)
    if jobs:
        emit_wide(_status_table(jobs, verbose=verbose))
    elif selection is None:
        line(f"No queued or running jobs on {cluster}.")
    else:
        line(f"No queued or running jobs for {selection.description} on {cluster}.")
    for location, reason in listing.unreachable.items():
        warning(f"{location} not reachable: {reason}")


def _status_selection(
    *,
    config: str | None,
    workflow: str | None,
    recipe: str | None,
    stages: list[str] | None,
    project_dir: str | None,
    project_paths: tuple[str, ...],
) -> _StatusSelection | None:
    """Resolve whose jobs ``status`` shows, continuing in the selected project's environment."""
    if recipe:
        resolved = resolve_recipe(
            recipe,
            ProjectResolver(search_paths=project_paths),
            project_dir=project_dir,
        )
        _handoff_project(resolved.context)
        return _StatusSelection(
            workflows=tuple(label_value(str(node.selector)) for node in resolved.workflows),
            stages=(),
            description=str(resolved.selector),
        )
    if not (config or workflow):
        project = _status_project()
        if project is not None:
            # A project's own launcher environment omits the execution stack.
            _handoff_project(project)
        return None
    selection = require_workflow(
        config=config,
        workflow=workflow,
        project_dir=project_dir,
        project_paths=project_paths,
    )
    if selection.project_context is not None:
        _handoff_project(selection.project_context)
    definition = load_workflow(
        selection.config_path,
        project_dir=selection.project_dir,
        project_search_paths=project_paths,
    )
    if selection.project_context is None:
        _handoff_project(definition.project_context)
    selected = definition.select_active_stages(stages) if stages else []
    label = workflow_label(definition.project_context, definition.config_path)
    return _StatusSelection(
        workflows=(label_value(label),),
        stages=tuple(label_value(stage) for stage in selected),
        description=label + (f" ({', '.join(selected)})" if selected else ""),
    )


def _status_table(jobs: tuple[ActiveJob, ...], *, verbose: bool) -> Table:
    columns = (*_STATUS_COLUMNS, *(_STATUS_VERBOSE_COLUMNS if verbose else ()))
    table = Table(show_header=True, header_style="bold cyan", box=None)
    for title, attribute in columns:
        table.add_column(
            title,
            style="green" if attribute == "name" else None,
            justify="right" if attribute in _STATUS_RIGHT_ALIGNED else "left",
            no_wrap=True,
        )
    for job in jobs:
        # Text cells keep scheduler values such as "train[large]" from being read as markup.
        table.add_row(*(Text(_status_cell(getattr(job, attribute))) for _, attribute in columns))
    return table


def _status_cell(value: object) -> str:
    return "-" if value is None or value == "" else str(value)


def _format_plan_warnings(caught: list[warnings.WarningMessage]) -> list[str]:
    """Return stable, actionable warnings without import or cleanup noise."""
    messages = []
    seen = set()
    for caught_warning in caught:
        message = str(caught_warning.message).strip()
        if not message or "starlette.middleware.wsgi is deprecated" in message:
            continue
        if issubclass(caught_warning.category, ResourceWarning):
            continue
        warning_path = Path(caught_warning.filename)
        if "site-packages" in warning_path.parts and issubclass(
            caught_warning.category,
            DeprecationWarning | FutureWarning,
        ):
            continue
        if (
            caught_warning.category is SyntaxWarning
            and "site-packages" in warning_path.parts
            and "nemo_skills" in warning_path.parts
            and message.startswith("invalid escape sequence")
        ):
            continue
        source = warning_path.stem
        category = (
            ""
            if caught_warning.category is UserWarning
            else f"{caught_warning.category.__name__}: "
        )
        rendered = f"{source}: {category}{message}"
        if rendered not in seen:
            seen.add(rendered)
            messages.append(rendered)
    return messages


def _emit_plan_warnings(
    definition: "WorkflowDefinition",
    messages: list[str],
    *,
    activity: str = "planning",
) -> None:
    if not messages:
        return
    warning(f"{len(messages)} {activity} warning(s)")
    for message in messages:
        diagnostic(
            textwrap.fill(
                definition.redact_runtime_values(message),
                width=100,
                initial_indent="  - ",
                subsequent_indent="    ",
            )
        )


def _emit_plan_diagnostics(
    definition: "WorkflowDefinition | None",
    diagnostic_output: str,
    *,
    debug: bool,
) -> None:
    if not debug or not diagnostic_output:
        return
    if definition is None:
        diagnostic("Planning diagnostics unavailable before configuration loaded.")
        return
    redacted = definition.redact_runtime_values(diagnostic_output).rstrip()
    if redacted:
        diagnostic("Planning diagnostics:")
        write_stderr(redacted)


def _safe_plan_error_message(
    definition: "WorkflowDefinition | None",
    exc: Exception,
) -> str:
    """Return a planning error that cannot expose unresolved configuration secrets."""
    if definition is not None:
        return definition.redact_runtime_values(str(exc))
    if isinstance(exc, TargetResolutionError):
        return str(exc)
    return "Workflow planning failed before configuration could be loaded"


@app.command(name="plan-recipe")
def plan_recipe_command(
    recipe: str = typer.Argument(..., help="Recipe selector (project:recipe)"),
    output_format: str = typer.Option(
        "text",
        "--format",
        "-f",
        help="Output format: text or json",
    ),
    project_dir: str | None = typer.Option(None, "--project-dir", help=_PROJECT_DIR_OPTION_HELP),
    project_path: Annotated[
        list[str] | None,
        typer.Option("--project-path", help=_PROJECT_PATH_OPTION_HELP),
    ] = None,
):
    """Render a target-independent recipe dependency graph."""
    if output_format not in {"text", "json"}:
        raise typer.BadParameter("--format must be either 'text' or 'json'")
    try:
        graph_plan = plan_recipe(
            recipe,
            ProjectResolver(search_paths=tuple(project_path or ())),
            project_dir=project_dir,
        )
        if output_format == "json":
            write_stdout(json.dumps(graph_plan.to_dict(), indent=2, sort_keys=True))
            return

        heading(f"NVFlow recipe plan: {graph_plan.project}:{graph_plan.recipe}")
        if graph_plan.description:
            line(graph_plan.description)
        line()
        _emit_recipe_graph_workflows(graph_plan)
        provenance = graph_plan.provenance
        line(f"Project commit: {provenance.commit or 'unavailable'}")
        line(f"Project content: {provenance.content_digest or 'unavailable'}")
        line(
            "Project source: "
            + ("clean and submittable" if provenance.submittable else "not submittable")
        )
        line(f"Graph digest: {graph_plan.digest}")
        line("Target: not bound; no cluster contacted")
        line("Execution: recipe DAG submission is not implemented")
    except Exception as exc:
        if output_format == "json":
            write_stdout(
                json.dumps(
                    {
                        "complete": False,
                        "error": type(exc).__name__,
                        "message": str(exc),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            error(f"Recipe planning failed: {exc}")
        raise typer.Exit(1) from exc


@app.command()
def plan(
    cluster: Annotated[
        str,
        typer.Option("--cluster", help=_CLUSTER_OPTION_HELP),
    ],
    stages: Annotated[
        list[str] | None,
        typer.Argument(help="Optional stage name(s) to plan (default: complete workflow)"),
    ] = None,
    config: str | None = typer.Option(
        None,
        "--config",
        "-c",
        help=_CONFIG_OPTION_HELP,
    ),
    workflow: str | None = typer.Option(
        None,
        "--workflow",
        "-w",
        help=_WORKFLOW_OPTION_HELP,
    ),
    environment: Annotated[
        list[str] | None,
        typer.Option("--environment", "-e", help="Plan specific environment(s) only"),
    ] = None,
    output_format: str = typer.Option(
        "text",
        "--format",
        "-f",
        help="Output format: text or json",
    ),
    live: bool = typer.Option(
        False,
        "--live",
        "-l",
        help=(
            "Enable stage-specific filesystem discovery/resume checks; "
            "does not query Slurm or submit jobs"
        ),
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Include every task, command, dependency, resource, and output",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        "-d",
        help="Show captured stage-builder diagnostics on stderr",
    ),
    project_dir: str | None = typer.Option(
        None, "--project-dir", help="Explicit external project root"
    ),
    project_path: Annotated[
        list[str] | None,
        typer.Option("--project-path", help="Ordered project search path (repeatable)"),
    ] = None,
):
    """Plan the complete workflow or selected stages without submission."""
    if output_format not in {"text", "json"}:
        raise typer.BadParameter("--format must be either 'text' or 'json'")
    if output_format == "json" and verbose:
        raise typer.BadParameter("--verbose applies to text output; JSON is already detailed")
    selection = require_workflow(
        config=config,
        workflow=workflow,
        project_dir=project_dir,
        project_paths=tuple(project_path or ()),
    )
    definition = None
    diagnostics = io.StringIO()
    try:
        if selection.project_context is not None:
            _handoff_project(selection.project_context)
        target_config = preflight_target_config(cluster)
        with contextlib.redirect_stdout(diagnostics), contextlib.redirect_stderr(diagnostics):
            definition = load_workflow(
                selection.config_path,
                project_dir=selection.project_dir,
                project_search_paths=tuple(project_path or ()),
            )
            target = TargetBinding.resolve(definition.cluster, cluster)
            resolved_target = resolve_target(target, target_config)
        target_scope = (
            bind_target_config(target_config)
            if resolved_target.engine == "slurm"
            else contextlib.nullcontext()
        )
        with (
            target_scope,
            warnings.catch_warnings(record=True) as caught_warnings,
        ):
            warnings.simplefilter("always")
            if selection.project_context is None:
                _handoff_project(definition.project_context)
            if live:
                status(
                    "Inspecting workflow inputs and declared outputs"
                    if output_format == "text"
                    else "Inspecting workflow inputs for live task selection"
                )
            with contextlib.redirect_stdout(diagnostics), contextlib.redirect_stderr(diagnostics):
                planning_filesystem = (
                    (
                        target_filesystem(target.cluster)
                        if resolved_target.engine == "slurm"
                        else LocalTargetFilesystem()
                    )
                    if live
                    else None
                )
                from nvflow.execution.inspection import inspect_declared_outputs

                execution_graph = WorkflowPlanner(
                    definition,
                    target,
                    filesystem=planning_filesystem,
                ).execution_graph(
                    resolved_target,
                    stages=stages,
                    environment=environment,
                    live=live,
                )
                if resolved_target.engine == "ray":
                    from nvflow.integrations.ray_jobs.lowering import (
                        lower_execution_graph,
                    )

                    lowered = lower_execution_graph(execution_graph)
                    assert resolved_target.ray is not None
                    for task in lowered.tasks:
                        for role in {command.require_runtime_role() for command in task.commands}:
                            resolved_target.ray.cluster_for_role(role)
                launch_plan = execution_graph.to_plan(planning_mode="live" if live else "static")
                output_inspections = (
                    inspect_declared_outputs(
                        launch_plan,
                        planning_filesystem,
                        visible=(
                            planning_filesystem.mounted
                            if isinstance(planning_filesystem, LocalTargetFilesystem)
                            else None
                        ),
                    )
                    if live and output_format == "text"
                    else ()
                )
        warning_messages = _format_plan_warnings(caught_warnings)
        if output_format == "json":
            payload = definition.redact_runtime_values(launch_plan.to_dict())
            write_stdout(json.dumps(payload, indent=2, sort_keys=True))
        else:
            from nvflow.cli.plan_output import render_plan_text

            emit_plan_text(
                definition.redact_runtime_values(
                    render_plan_text(
                        launch_plan,
                        verbose=verbose,
                        output_inspections=output_inspections,
                        jobs_per_stage=engine_for(resolved_target).jobs_per_stage(execution_graph),
                    )
                )
            )
        _emit_plan_warnings(definition, warning_messages)
        _emit_plan_diagnostics(definition, diagnostics.getvalue(), debug=debug)
        if not launch_plan.complete:
            raise typer.Exit(2)
    except typer.Exit:
        raise
    except Exception as exc:
        _emit_plan_diagnostics(definition, diagnostics.getvalue(), debug=debug)
        message = _safe_plan_error_message(definition, exc)
        if output_format == "json":
            write_stdout(
                json.dumps(
                    {
                        "complete": False,
                        "error": type(exc).__name__,
                        "message": message,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            error(message)
        raise typer.Exit(1) from exc


@app.command(name="show-config")
def show_config(
    stages: Annotated[
        list[str] | None,
        typer.Argument(help="Optional stage name(s) to show (resolved view only)"),
    ] = None,
    config: str | None = typer.Option(
        None,
        "--config",
        "-c",
        help=_CONFIG_OPTION_HELP,
    ),
    workflow: str | None = typer.Option(
        None,
        "--workflow",
        "-w",
        help=_WORKFLOW_OPTION_HELP,
    ),
    view: str = typer.Option(
        "resolved",
        "--view",
        "-V",
        help="Configuration view: source, merged, resolved, or diff",
    ),
    environment: Annotated[
        list[str] | None,
        typer.Option(
            "--environment",
            "-e",
            help="Show specific environment/fan-out value(s) only",
        ),
    ] = None,
    project_dir: str | None = typer.Option(
        None, "--project-dir", help="Explicit external project root"
    ),
    project_path: Annotated[
        list[str] | None,
        typer.Option("--project-path", help="Ordered project search path (repeatable)"),
    ] = None,
):
    """Print a safe workflow or resolved stage configuration view."""
    valid_views = {"source", "merged", "resolved", "diff"}
    if view not in valid_views:
        raise typer.BadParameter(
            f"--view must be one of {', '.join(sorted(valid_views))}, got {view!r}"
        )
    if stages and view != "resolved":
        raise typer.BadParameter("Positional stage names require --view resolved")
    if environment and not stages:
        raise typer.BadParameter("--environment requires positional stage name(s)")
    selection = require_workflow(
        config=config,
        workflow=workflow,
        project_dir=project_dir,
        project_paths=tuple(project_path or ()),
    )
    try:
        if view in {"source", "merged", "diff"}:
            from nvflow.cli.workflow_inspection import unresolved_config_view

            value = unresolved_config_view(
                selection.config_path,
                view=view,
                project_dir=selection.project_dir,
                project_search_paths=tuple(project_path or ()),
            )
        else:
            definition = load_workflow(
                selection.config_path,
                project_dir=selection.project_dir,
                project_search_paths=tuple(project_path or ()),
            )
            value = definition.config_view(
                view,
                stages=stages,
                environment=environment,
            )
        rendered = yaml.safe_dump(value, sort_keys=False)
        emit_yaml(rendered)
    except Exception as exc:
        error(exc)
        raise typer.Exit(1) from exc


@app.command()
def validate(
    stages: Annotated[
        list[str] | None,
        typer.Argument(help="Optional stage name(s) to validate (default: complete workflow)"),
    ] = None,
    config: str | None = typer.Option(
        None,
        "--config",
        "-c",
        help=_CONFIG_OPTION_HELP,
    ),
    workflow: str | None = typer.Option(
        None,
        "--workflow",
        "-w",
        help=_WORKFLOW_OPTION_HELP,
    ),
    recipe: str | None = typer.Option(
        None,
        "--recipe",
        "-r",
        help="Recipe DAG selector (project:recipe)",
    ),
    environment: Annotated[
        list[str] | None,
        typer.Option(
            "--environment",
            "-e",
            help="Validate specific environment/fan-out value(s) only",
        ),
    ] = None,
    project_dir: str | None = typer.Option(None, "--project-dir", help=_PROJECT_DIR_OPTION_HELP),
    project_path: Annotated[
        list[str] | None,
        typer.Option("--project-path", help=_PROJECT_PATH_OPTION_HELP),
    ] = None,
):
    """Validate one workflow, selected stages/environments, or one recipe DAG."""
    if recipe and (config or workflow):
        raise typer.BadParameter(
            "Use --recipe or a workflow selection, not both",
            param_hint="--recipe/--workflow/--config",
        )
    if recipe and (stages or environment):
        raise typer.BadParameter(
            "Stage and environment selections apply only to workflow validation",
            param_hint="STAGES/--environment",
        )
    project_paths = tuple(project_path or ())
    if recipe:
        try:
            resolved = resolve_recipe(
                recipe,
                ProjectResolver(search_paths=project_paths),
                project_dir=project_dir,
                validate_resources=True,
            )
            line()
            success("Recipe definition is valid")
            line()
            line(f"Project: {resolved.selector.project}")
            line(f"Recipe: {resolved.selector.recipe}")
            line(f"Workflows: {len(resolved.workflows)}")
            line()
            line(
                "Checked: project manifest, recipe dependency graph, workflow references, "
                "complete workflow definitions, and local project resources"
            )
            line(
                "Not checked: cluster access, worker images, models, mounts, cluster-only "
                "data, or multi-workflow submission"
            )
            return
        except Exception as exc:
            error(f"Recipe definition is invalid: {exc}")
            raise typer.Exit(1) from exc

    selection = require_workflow(
        config=config,
        workflow=workflow,
        project_dir=project_dir,
        project_paths=project_paths,
    )
    try:
        if selection.project_context is not None:
            _handoff_project(selection.project_context)
        runner = WorkflowRunner(
            selection.config_path,
            project_dir=selection.project_dir,
            project_search_paths=project_paths,
        )
        if selection.project_context is None:
            _handoff_project(runner.project_context)
        runner.validate_config(stages=stages, environment=environment)
        selected_stages = runner.definition.select_active_stages(stages)
        line()
        if stages:
            success("Selected stage configuration is valid")
        else:
            success("Workflow definition is valid")
        line()
        line(f"Config: {runner.config_path}")
        line(f"Project: {runner.project}")
        line(f"Workflow: {runner.workflow_name}")
        line(f"Type: {runner.workflow_type}")
        line(f"Cluster: {runner.cluster} (declared; not contacted)")
        if environment:
            line(f"Environment selection: {', '.join(environment)}")
        if stages:
            line(f"Stages: {len(selected_stages)} selected ({', '.join(selected_stages)})")
            line()
            line(
                "Checked: manifest, local project resources, selected stage configuration, "
                "and transitive dependency graph"
            )
        else:
            line(f"Stages: {len(selected_stages)}")
            line()
            line(
                "Checked: manifest, local project resources, stage configuration, "
                "and dependency graph"
            )
        line("Not checked: cluster access, worker images, models, mounts, or cluster-only data")
    except Exception as exc:
        error(f"Workflow definition is invalid: {exc}")
        raise typer.Exit(1) from exc


@app.command(name="stage-info")
def stage_info(
    stage: str = typer.Argument(
        ...,
        help="Configured stage name",
    ),
    workflow: str = typer.Option(
        ...,
        "--workflow",
        "-w",
        help="Exact workflow selector",
    ),
    project_dir: str | None = typer.Option(
        None, "--project-dir", help="Explicit external project root"
    ),
    project_path: Annotated[
        list[str] | None,
        typer.Option("--project-path", help="Ordered project search path (repeatable)"),
    ] = None,
):
    """Show detailed information about a specific stage.

    Usage:
        nvflow stage-info training --workflow finance:grpo/qwen3_4b
    """
    try:
        selection = require_workflow(
            config=None,
            workflow=workflow,
            project_dir=project_dir,
            project_paths=tuple(project_path or ()),
        )
        if selection.project_context is not None:
            _handoff_project(selection.project_context)
        definition = load_workflow(
            selection.config_path,
            project_dir=selection.project_dir,
            project_search_paths=tuple(project_path or ()),
        )
        if stage not in definition.pipeline_stages:
            available = ", ".join(definition.pipeline_stages)
            raise KeyError(
                f"Stage {stage!r} is not active in {selection.display_name}. Available: {available}"
            )
        project_name = definition.project
        workflow_name = definition.workflow_name
        stage_name = stage

        stage_class = definition.resolve_stage_class(stage_name)

        # Create instance to get methods
        stage_instance = stage_class()

        # Display information
        full_path = f"{project_name}.{workflow_name}.{stage_name}"
        line()
        heading(f"Stage: {full_path}")
        line()

        line(f"Project: {project_name}")
        line(f"Workflow: {workflow_name}")
        line(f"Stage: {stage_name}")
        stage_config = definition.config["stages"].get(stage_name) or {}
        if description := description_of(stage_config):
            line(f"Description: {description}")
        if isinstance(substages := stage_config.get("substages"), dict):
            line(f"Substages: {', '.join(substages)}")
        line(f"Class Name: {stage_class.__name__}")

        # Show docstring
        if stage_class.__doc__:
            line()
            heading("Documentation:")
            line(stage_class.__doc__)

        # Show module path
        line()
        heading("Source File:")
        import inspect

        source_file = inspect.getfile(stage_class)
        line(f"  {source_file}")

        # Show available methods
        line()
        heading("Methods:")
        line("  • execution() - Exact backend-neutral execution contract")
        if (
            hasattr(stage_instance, "validate_config")
            and stage_instance.validate_config.__func__ != BaseStage.validate_config
        ):
            line("  • validate_config() - Custom validation implemented")

    except (KeyError, ValueError, ImportError, TypeError) as e:
        error(e)
        diagnostic(
            f"Use 'nvflow list-stages --workflow {workflow}' for configured aliases or "
            "'nvflow list-stage-types' for shared implementations"
        )
        raise typer.Exit(1) from e


@app.command()
def version():
    """Show NVFlow version."""
    write_stdout(f"NVFlow version {__version__}")


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
