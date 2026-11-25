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
"""Contracts for project workflows and optional selector-keyed recipe DAGs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from typer.testing import CliRunner

from nvflow import load_workflow
from nvflow.cli import main as cli
from nvflow.core.workflow_definition import load_config_with_inheritance
from nvflow.project import ProjectManifest, ProjectManifestError, ProjectResolver, WorkflowSelector
from nvflow.recipe import RecipeDefinitionError, RecipeSelector
from nvflow.recipe.planning import plan_recipe
from nvflow.recipe.resolver import resolve_recipe

runner = CliRunner()


def _project(
    tmp_path: Path,
    *,
    recipes: str = "",
) -> Path:
    root = tmp_path / "demo-project"
    (root / "workflows").mkdir(parents=True)
    (root / "project.yaml").write_text(
        f'manifest_version: 1\nproject: demo\ndescription: "Demo project"\n{recipes}',
        encoding="utf-8",
    )
    (root / "workflows" / "generate.yaml").write_text(
        "workflow:\n"
        "  name: generate\n"
        "  type: sdg\n"
        "cluster: test\n"
        "pipeline_stages: [prepare, generate]\n"
        "stages:\n"
        "  prepare:\n"
        "    dependencies: []\n"
        "  generate:\n"
        "    dependencies: [prepare]\n",
        encoding="utf-8",
    )
    return root


def test_minimal_project_has_no_inline_recipes(tmp_path: Path) -> None:
    root = _project(tmp_path)

    manifest = ProjectManifest.load(root / "project.yaml")

    assert manifest.project == "demo"
    assert manifest.recipes == ()


def test_inline_recipe_preserves_dag_not_yaml_execution_order(tmp_path: Path) -> None:
    root = _project(
        tmp_path,
        recipes=(
            "recipes:\n"
            "  quickstart:\n"
            '    description: "Generate demo data"\n'
            "    workflows:\n"
            "      generate:\n"
            "        depends_on: [prepare-data]\n"
            "      prepare-data:\n"
            "        optional: true\n"
        ),
    )

    recipe = ProjectManifest.load(root / "project.yaml").recipe_map["quickstart"]

    assert [node.selector for node in recipe.workflows] == ["generate", "prepare-data"]
    assert [node.selector for node in recipe.topological_workflows()] == [
        "prepare-data",
        "generate",
    ]


def test_recipe_may_reference_one_workflow(tmp_path: Path) -> None:
    root = _project(
        tmp_path,
        recipes=(
            "recipes:\n"
            "  quickstart:\n"
            '    description: "Generate demo data"\n'
            "    workflows:\n"
            "      generate: {}\n"
        ),
    )

    recipe = ProjectManifest.load(root / "project.yaml").recipe_map["quickstart"]
    resolved = resolve_recipe("demo:quickstart", ProjectResolver(), project_dir=root)

    assert recipe.description == "Generate demo data"
    assert [node.selector for node in recipe.workflows] == ["generate"]
    assert resolved.workflows[0].selector == WorkflowSelector.parse("demo:generate")


@pytest.mark.parametrize(
    "recipes, message",
    [
        (
            "recipes:\n  broken:\n    workflows:\n      first:\n        depends_on: [missing]\n",
            "unknown workflow selector",
        ),
        (
            "recipes:\n  broken:\n    workflows:\n      first:\n        depends_on: [first]\n",
            "depends on itself",
        ),
        (
            "recipes:\n"
            "  broken:\n"
            "    workflows:\n"
            "      first:\n"
            "        depends_on: [second]\n"
            "      second:\n"
            "        depends_on: [first]\n",
            "dependency cycle",
        ),
        (
            "recipes:\n  broken:\n    workflows:\n      generate:\n        environments: [train]\n",
            "unknown field.*environments",
        ),
        (
            "recipes:\n  broken:\n    workflow: generate\n    workflows:\n      generate: {}\n",
            "unknown field.*workflow",
        ),
        (
            "recipes:\n"
            "  broken:\n"
            "    workflows:\n"
            "      generate: {}\n"
            "      generate: {optional: true}\n",
            "duplicate key.*generate",
        ),
        (
            "recipes:\n  broken:\n    workflows:\n      eval//demo: {}\n",
            "extension-free path beneath workflows",
        ),
        (
            "recipes:\n  broken:\n    workflows:\n      1: {}\n",
            "relative POSIX workflow selector path",
        ),
    ],
)
def test_project_manifest_rejects_invalid_recipe_graphs(
    tmp_path: Path,
    recipes: str,
    message: str,
) -> None:
    root = _project(tmp_path, recipes=recipes)

    with pytest.raises(ProjectManifestError, match=message):
        ProjectManifest.load(root / "project.yaml")


@pytest.mark.parametrize(
    "selector",
    ["demo", "demo:", ":quickstart", "demo:bad/name", " demo:quickstart"],
)
def test_recipe_selector_rejects_malformed_values(selector: str) -> None:
    with pytest.raises(RecipeDefinitionError):
        RecipeSelector.parse(selector)


def test_recipe_nodes_reject_stage_subsets(
    tmp_path: Path,
) -> None:
    root = _project(
        tmp_path,
        recipes=(
            "recipes:\n"
            "  broken-stage:\n"
            "    workflows:\n"
            "      generate:\n"
            "        stages: [missing]\n"
        ),
    )

    with pytest.raises(ProjectManifestError, match="unknown field.*stages"):
        ProjectManifest.load(root / "project.yaml")


def test_recipe_resolution_reports_malformed_workflow_yaml(tmp_path: Path) -> None:
    root = _project(
        tmp_path,
        recipes=("recipes:\n  quickstart:\n    workflows:\n      generate: {}\n"),
    )
    (root / "workflows" / "generate.yaml").write_text(
        "workflow: [\n",
        encoding="utf-8",
    )

    with pytest.raises(ProjectManifestError, match="Unable to inspect workflow"):
        resolve_recipe("demo:quickstart", ProjectResolver(), project_dir=root)


def test_recipe_resolution_accepts_config_local_pipeline_reference(tmp_path: Path) -> None:
    root = _project(
        tmp_path,
        recipes=("recipes:\n  quickstart:\n    workflows:\n      generate: {}\n"),
    )
    workflow = root / "workflows" / "generate.yaml"
    workflow.write_text(
        workflow.read_text(encoding="utf-8").replace(
            "pipeline_stages: [prepare, generate]\n",
            "stage_order: [prepare, generate]\npipeline_stages: ${stage_order}\n",
        ),
        encoding="utf-8",
    )

    resolved = resolve_recipe("demo:quickstart", ProjectResolver(), project_dir=root)

    assert resolved.workflows[0].selector == WorkflowSelector.parse("demo:generate")


def test_recipe_resolution_requires_workflow_cluster(tmp_path: Path) -> None:
    root = _project(
        tmp_path,
        recipes=("recipes:\n  quickstart:\n    workflows:\n      generate: {}\n"),
    )
    workflow = root / "workflows" / "generate.yaml"
    workflow.write_text(
        workflow.read_text(encoding="utf-8").replace("cluster: test\n", ""),
        encoding="utf-8",
    )

    with pytest.raises(ProjectManifestError, match="must declare cluster"):
        resolve_recipe("demo:quickstart", ProjectResolver(), project_dir=root)


def test_recipe_resolves_complete_checkpoint_workflow(
    tmp_path: Path,
) -> None:
    root = _project(
        tmp_path,
        recipes=("recipes:\n  evaluation:\n    workflows:\n      generate: {}\n"),
    )
    workflow_path = root / "workflows" / "generate.yaml"
    workflow_path.write_text(
        "workflow:\n"
        "  name: evaluation\n"
        "  type: evaluation\n"
        "cluster: test\n"
        "pipeline_stages: [checkpoint]\n"
        "stages: {}\n"
        "checkpoints:\n"
        "  checkpoint:\n"
        "    eval_steps: [10, final]\n",
        encoding="utf-8",
    )

    resolved = resolve_recipe(
        "demo:evaluation",
        ProjectResolver(),
        project_dir=root,
    )
    workflow = load_workflow(workflow_path, project_dir=root)

    assert resolved.workflows[0].selector == WorkflowSelector.parse("demo:generate")
    assert resolved.workflows[0].stage_count == 2
    assert tuple(workflow.pipeline_stages) == ("checkpoint-10", "checkpoint-final")


def test_workflow_and_stage_descriptions_show_in_every_listing(tmp_path: Path) -> None:
    root = _project(tmp_path)
    workflow_path = root / "workflows" / "generate.yaml"
    workflow_path.write_text(
        workflow_path.read_text(encoding="utf-8")
        .replace("  type: sdg\n", "  type: sdg\n  description: Generate demo answers\n")
        .replace(
            "  prepare:\n",
            "  prepare:\n    uses: data.filter\n    description: Write the demo prompts\n",
        ),
        encoding="utf-8",
    )
    workflow = "demo:generate"
    project = ["--project-dir", str(root)]

    workflows = runner.invoke(cli.app, ["list-workflows", *project])
    stages = runner.invoke(cli.app, ["list-stages", "-w", workflow, *project])
    info = runner.invoke(cli.app, ["stage-info", "prepare", "-w", workflow, *project])

    assert workflows.exit_code == 0, workflows.output
    assert "Generate demo answers" in workflows.output
    assert stages.exit_code == 0, stages.output
    assert "Write the demo prompts" in stages.output
    assert info.exit_code == 0, info.output
    assert "Description: Write the demo prompts" in info.output
    assert "description" not in load_workflow(workflow_path).stage_runtime_config("prepare")


def test_project_recipe_workflow_cli_drill_down(tmp_path: Path) -> None:
    root = _project(
        tmp_path,
        recipes=(
            "recipes:\n"
            "  quickstart:\n"
            '    description: "Generate demo data"\n'
            "    workflows:\n"
            "      generate: {}\n"
        ),
    )

    projects = runner.invoke(cli.app, ["list-projects", "--project-dir", str(root)])
    recipes = runner.invoke(cli.app, ["list-recipes", "--project-dir", str(root)])
    workflows = runner.invoke(
        cli.app,
        [
            "list-workflows",
            "--recipe",
            "demo:quickstart",
            "--project-dir",
            str(root),
        ],
    )

    assert projects.exit_code == 0, projects.output
    assert "demo" in projects.output
    assert "1" in projects.output
    assert recipes.exit_code == 0, recipes.output
    assert "demo:quickstart" in recipes.output
    assert "Generate demo data" in recipes.output
    assert workflows.exit_code == 0, workflows.output
    assert "demo:generate" in workflows.output
    assert all(column in workflows.output for column in ("Selector", "Type", "Stages", "Optional"))
    assert "Total: 1 workflow definition" in workflows.output
    assert "Order: recipe dependency order" in workflows.output
    assert "Dependencies: nvflow plan-recipe demo:quickstart" in workflows.output
    assert "Environments:" not in workflows.output
    assert "…" not in workflows.output


def test_project_with_no_recipes_is_reported_without_error(tmp_path: Path) -> None:
    root = _project(tmp_path)

    result = runner.invoke(cli.app, ["list-recipes", "--project-dir", str(root)])

    assert result.exit_code == 0, result.output
    assert "defines no recipes" in result.output


def test_recipe_validation_and_graph_plan_are_target_independent(
    tmp_path: Path,
) -> None:
    root = _project(
        tmp_path,
        recipes=("recipes:\n  quickstart:\n    workflows:\n      generate: {}\n"),
    )
    resolver = ProjectResolver()

    first = plan_recipe("demo:quickstart", resolver, project_dir=str(root))
    second = plan_recipe("demo:quickstart", resolver, project_dir=str(root))
    validation = runner.invoke(
        cli.app,
        ["validate", "--recipe", "demo:quickstart", "--project-dir", str(root)],
    )
    planned = runner.invoke(
        cli.app,
        [
            "plan-recipe",
            "demo:quickstart",
            "--project-dir",
            str(root),
            "--format",
            "json",
        ],
    )

    assert first.digest == second.digest
    assert first.workflows[0].selector == "demo:generate"
    assert validation.exit_code == 0, validation.output
    assert "cluster access" in validation.output.lower()
    assert "complete workflow definitions" in validation.output
    assert "stage selections" not in validation.output
    assert planned.exit_code == 0, planned.output
    payload = json.loads(planned.stdout)
    assert payload["digest"] == first.digest
    assert payload["plan_version"] == 2
    assert payload["kind"] == "recipe"
    assert set(payload["workflows"][0]) == {
        "selector",
        "workflow_type",
        "depends_on",
        "optional",
    }
    assert "manifest_version" not in payload


def test_finance_quickstart_recipe_connects_compatible_data_paths() -> None:
    resolved = resolve_recipe(
        "finance:quickstart",
        ProjectResolver(),
    )
    workflows = resolved.workflow_map

    download = load_config_with_inheritance(workflows["download/demo"].config_path)
    sdg = load_config_with_inheritance(workflows["sdg/template-based-sdg-demo"].config_path)
    sft = load_config_with_inheritance(workflows["sft/qwen3_4b"].config_path)
    grpo = load_config_with_inheritance(workflows["grpo/qwen3_4b"].config_path)

    download_output = OmegaConf.select(download, "stages.download.output_dir")
    sdg_input = OmegaConf.select(sdg, "filings_dir")
    sdg_output = OmegaConf.select(sdg, "directories.step-5-filter-answers")
    assert download_output == sdg_input
    assert OmegaConf.select(sft, "directories.raw_train_data") == sdg_output
    assert OmegaConf.select(grpo, "environments.equivalence_llm_judge.raw_train_data") == sdg_output
    assert OmegaConf.select(grpo, "stages.validate_questions.source_data") == sdg_output


def test_finance_template_based_sft_recipe_connects_single_turn_models() -> None:
    resolved = resolve_recipe(
        "finance:template-based-sft",
        ProjectResolver(),
    )
    workflows = resolved.workflow_map

    assert set(workflows) == {
        "eval/baselines",
        "download/sp500",
        "sdg/template-based-sdg",
        "sft/qwen3_14b",
        "sft/nemotron-3-nano",
    }
    download = load_config_with_inheritance(workflows["download/sp500"].config_path)
    template_sdg = load_config_with_inheritance(workflows["sdg/template-based-sdg"].config_path)
    download_output = OmegaConf.select(download, "stages.download.output_dir")
    assert OmegaConf.select(template_sdg, "filings_dir") == download_output

    template_output = OmegaConf.select(template_sdg, "directories.step-5-filter-answers")
    for selector in ("sft/qwen3_14b", "sft/nemotron-3-nano"):
        sft = load_config_with_inheritance(workflows[selector].config_path)
        assert OmegaConf.select(sft, "directories.raw_train_data") == template_output


def test_finance_production_grpo_recipe_connects_document_grounded_data() -> None:
    resolved = resolve_recipe(
        "finance:document-grounded-local-search-grpo",
        ProjectResolver(),
    )
    workflows = resolved.workflow_map

    assert set(workflows) == {
        "eval/baselines",
        "download/sp500",
        "sdg/document-grounded-sdg-sp500",
        "grpo/nemotron_nano_30b_a3b",
    }
    download = load_config_with_inheritance(workflows["download/sp500"].config_path)
    document_sdg = load_config_with_inheritance(
        workflows["sdg/document-grounded-sdg-sp500"].config_path
    )
    grpo = load_config_with_inheritance(workflows["grpo/nemotron_nano_30b_a3b"].config_path)
    assert OmegaConf.select(document_sdg, "filings_dir") == OmegaConf.select(
        download,
        "stages.download.output_dir",
    )
    # GRPO consumers append final_result.jsonl, which dgsdg_post_process writes
    # into its own step directory rather than the SDG workflow root.
    assert OmegaConf.select(grpo, "_dg_sdg_data") == OmegaConf.select(
        document_sdg,
        "stages.dgsdg_post_process.output_dir",
    )
    assert OmegaConf.select(grpo, "stages.validate_questions.source_data") == OmegaConf.select(
        grpo,
        "_dg_sdg_data",
    )


@pytest.mark.parametrize(
    ("recipe_name", "image_filter_selector", "sdg_selector"),
    [
        (
            "quickstart",
            "image_filter/demo",
            "sdg/demo",
        ),
        (
            "vision-reasoning",
            "image_filter/full",
            "sdg/full",
        ),
    ],
)
def test_hopchain_recipes_connect_image_filter_to_sdg(
    recipe_name: str,
    image_filter_selector: str,
    sdg_selector: str,
) -> None:
    resolved = resolve_recipe(
        f"hopchain:{recipe_name}",
        ProjectResolver(),
    )
    workflows = resolved.workflow_map
    image_filter = load_config_with_inheritance(workflows[image_filter_selector].config_path)
    sdg = load_config_with_inheritance(workflows[sdg_selector].config_path)

    expected_input = (
        f"{OmegaConf.select(image_filter, 'stages.image_filter.output_dir')}/kept_images.jsonl"
    )
    assert (
        OmegaConf.select(sdg, "stages.prepare_filtered_image_inputs.input_file") == expected_input
    )
