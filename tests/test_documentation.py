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
"""Correctness checks for platform- and project-owned Markdown documentation."""

from __future__ import annotations

import ast
import os
import re
import shlex
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from typer.testing import CliRunner

from nvflow import load_workflow
from nvflow.cli.main import app

PROJECT_ROOT = Path(__file__).parents[1]
STARTER_ROOT = PROJECT_ROOT / "projects" / "starter"
_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
_PYTHON_FENCE_RE = re.compile(r"```python[^\n]*\n(.*?)```", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:bash|sh|shell)[^\n]*\n(.*?)```", re.DOTALL)
_PROJECT_CONFIG_RE = re.compile(r"--config(?:=|\s+)project://")
_YAML_EXCERPT_MARKER = "doc-check: yaml-excerpt"
_STALE_PATHS = (
    "nvflow/recipes/",
    "nvflow.projects.",
    "docs/projects/finance",
    "docs/projects/multimodal",
    "docs/development/nemo-rl-gym.md",
    "getting-started/creating-a-recipe.md",
    "recipe.yaml",
    "recipe://",
    "--recipe-dir",
    "--recipe-path",
    "NVFLOW_RECIPE_PATH",
    'recipe="finance"',
    "--workflow-name",
    "--workflow-type",
    "run-all",
    "workflow_order",
    "finance:download_sec_filings",
    "download_sec_filings.yaml",
    "`download_sec_filings`",
    "sap-500",
    "sap500",
    "SAP-500",
    "nvflow_recipe_finance",
    "nvflow_recipe_multimodal",
    "/nemo_run/code/recipes",
    "recipes/finance",
    "recipes/multimodal",
    "projects/multimodal",
    "nvmultimodal",
    "multimodal:quickstart",
    "multimodal:vision",
    "multimodal:image_filter/",
    "multimodal:sdg/",
    "recipes/example",
    "examples/external-recipe",
)

_WORKFLOW_DOCS = (
    ("projects/starter/README.md", "projects/starter/workflows/generate.yaml"),
    (
        "projects/finance/docs/workflows/01-download-sec.md",
        "projects/finance/workflows/download/demo.yaml",
    ),
    (
        "projects/finance/docs/workflows/02-template-based-sdg.md",
        "projects/finance/workflows/sdg/template-based-sdg-demo.yaml",
    ),
    (
        "projects/finance/docs/workflows/03-document-grounded-sdg.md",
        "projects/finance/workflows/sdg/document-grounded-sdg-demo.yaml",
    ),
    (
        "projects/finance/docs/workflows/04-sft.md",
        "projects/finance/workflows/sft/qwen3_4b.yaml",
    ),
    (
        "projects/finance/docs/workflows/05-eval.md",
        "projects/finance/workflows/eval/demo.yaml",
    ),
    (
        "projects/finance/docs/workflows/06-grpo.md",
        "projects/finance/workflows/grpo/qwen3_4b.yaml",
    ),
    (
        "projects/hopchain/README.md",
        "projects/hopchain/workflows/image_filter/demo.yaml",
    ),
    (
        "projects/hopchain/README.md",
        "projects/hopchain/workflows/sdg/demo.yaml",
    ),
)


def _markdown_files() -> list[Path]:
    files = {
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "INSTALL.md",
        PROJECT_ROOT / "CONTRIBUTING.md",
        PROJECT_ROOT / "SECURITY.md",
    }
    for root in (
        PROJECT_ROOT / "docs",
        PROJECT_ROOT / "projects",
        PROJECT_ROOT / "dockerfiles",
        PROJECT_ROOT / "scripts",
    ):
        files.update(root.rglob("*.md"))
        files.update(root.rglob("*.mmd"))
    return sorted(path for path in files if path.is_file())


def _skip_target(target: str) -> bool:
    return (
        not target
        or target.startswith(("#", "http://", "https://", "mailto:"))
        or any(marker in target for marker in ("<", ">", "{", "}", "$"))
    )


def _strip_blockquote_prefix(line: str, depth: int) -> str:
    stripped = line
    for _ in range(depth):
        match = re.match(r"^\s*>\s?", stripped)
        if match is None:
            return stripped
        stripped = stripped[match.end() :]
    return stripped


def _fenced_blocks(
    text: str,
    *,
    languages: set[str],
) -> list[tuple[int, str, str]]:
    lines = text.splitlines()
    blocks = []
    index = 0
    while index < len(lines):
        normalized = lines[index]
        quote_depth = 0
        while re.match(r"^\s*>", normalized):
            normalized = _strip_blockquote_prefix(normalized, 1)
            quote_depth += 1
        opening = re.match(r"^```(?P<language>[\w+-]+)", normalized.strip())
        if opening is None or opening.group("language").lower() not in languages:
            index += 1
            continue

        opening_index = index
        index += 1
        body = []
        while index < len(lines):
            candidate = _strip_blockquote_prefix(lines[index], quote_depth)
            if candidate.strip() == "```":
                break
            body.append(candidate)
            index += 1
        preceding = "\n".join(lines[max(0, opening_index - 3) : opening_index])
        blocks.append((opening_index + 1, "\n".join(body), preceding))
        index += 1
    return blocks


@pytest.mark.parametrize(
    "path", _markdown_files(), ids=lambda path: str(path.relative_to(PROJECT_ROOT))
)
def test_markdown_local_links_resolve(path: Path) -> None:
    failures = []
    for raw_target in _LINK_RE.findall(path.read_text(encoding="utf-8")):
        target = raw_target.split(maxsplit=1)[0].strip("<>")
        target = target.split("#", 1)[0]
        if _skip_target(target):
            continue
        resolved = (path.parent / target).resolve()
        if not resolved.exists():
            failures.append(f"{raw_target} -> {resolved}")
    assert failures == []


@pytest.mark.parametrize(
    "path", _markdown_files(), ids=lambda path: str(path.relative_to(PROJECT_ROOT))
)
def test_python_fences_parse(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    failures = []
    for match in _PYTHON_FENCE_RE.finditer(text):
        preceding = text[max(0, match.start() - 160) : match.start()]
        if "doc-check: skip" in preceding:
            continue
        try:
            ast.parse(match.group(1))
        except SyntaxError as exc:
            failures.append(f"line {text.count(chr(10), 0, match.start()) + 1}: {exc.msg}")
    assert failures == []


@pytest.mark.parametrize(
    "path", _markdown_files(), ids=lambda path: str(path.relative_to(PROJECT_ROOT))
)
def test_yaml_fences_are_standalone_or_marked_as_excerpts(path: Path) -> None:
    failures = []
    text = path.read_text(encoding="utf-8")
    for line_number, block, preceding in _fenced_blocks(text, languages={"yaml", "yml"}):
        if _YAML_EXCERPT_MARKER in preceding:
            continue
        try:
            list(yaml.safe_load_all(block))
        except yaml.YAMLError as exc:
            failures.append(f"line {line_number}: {exc}")
    assert failures == []


def _documented_commands() -> list[tuple[Path, str, bool]]:
    commands = []
    for path in _markdown_files():
        for block in _FENCE_RE.findall(path.read_text(encoding="utf-8")):
            execute = "doc-check: execute" in block
            logical = ""
            for line in block.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                logical = f"{logical} {stripped}".strip()
                if logical.endswith("\\"):
                    logical = logical[:-1].rstrip()
                    continue
                if logical.startswith("uv run nvflow "):
                    commands.append((path, logical.removeprefix("uv run "), execute))
                elif logical.startswith("nvflow "):
                    commands.append((path, logical, execute))
                logical = ""
    return commands


def _yaml_comment_commands() -> list[tuple[Path, str]]:
    commands = []
    for path in sorted((PROJECT_ROOT / "projects").rglob("*.yaml")):
        commands.extend(_yaml_comment_commands_from(path))
    return commands


def _yaml_comment_commands_from(path: Path) -> list[tuple[Path, str]]:
    commands = []
    pending = ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.lstrip()
        if not stripped.startswith("#"):
            pending = ""
            continue
        content = stripped.removeprefix("#").strip()
        if not content or (not pending and not content.startswith("uv run nvflow ")):
            continue
        pending = f"{pending} {content}".strip()
        if pending.endswith("\\"):
            pending = pending[:-1].rstrip()
            continue
        commands.append((path, pending.removeprefix("uv run ")))
        pending = ""
    return commands


def test_documented_nvflow_commands_use_supported_options() -> None:
    runner = CliRunner()
    help_options: dict[str, set[str]] = {}
    failures = []
    for path, command, _execute in _documented_commands():
        if "..." in command or "<" in command or ">" in command:
            continue
        tokens = shlex.split(command)
        if len(tokens) < 2:
            continue
        verb = tokens[1]
        if verb in {"plan", "run"} and not any(
            token == "--cluster" or token.startswith("--cluster=") for token in tokens[2:]
        ):
            failures.append(f"{path.relative_to(PROJECT_ROOT)}: {verb} requires explicit --cluster")
        if verb not in help_options:
            result = runner.invoke(app, [verb, "--help"])
            if result.exit_code != 0:
                failures.append(f"{path.relative_to(PROJECT_ROOT)}: unknown command {verb}")
                help_options[verb] = set()
                continue
            help_options[verb] = set(re.findall(r"(?<!\w)--?[a-zA-Z][\w-]*", result.output))
        for token in tokens[2:]:
            option = token.split("=", 1)[0]
            if option.startswith("-") and not option.startswith("++"):
                if option not in help_options[verb]:
                    failures.append(
                        f"{path.relative_to(PROJECT_ROOT)}: {verb} does not support {option}"
                    )
    assert failures == []


def test_project_yaml_command_examples_select_an_explicit_cluster() -> None:
    failures = []
    for path, command in _yaml_comment_commands():
        tokens = shlex.split(command)
        if len(tokens) < 2 or tokens[1] not in {"plan", "run"}:
            continue
        if not any(token == "--cluster" or token.startswith("--cluster=") for token in tokens[2:]):
            failures.append(f"{path.relative_to(PROJECT_ROOT)}: {command}")

    assert failures == []


def test_project_starter_commands_work_from_project_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    external_project_root = tmp_path / "copied-project"
    shutil.copytree(STARTER_ROOT, external_project_root)
    config = external_project_root / "workflows" / "generate.yaml"
    target_dir = tmp_path / "team-targets"
    target_dir.mkdir()
    (target_dir / "my_cluster.yaml").write_text("executor: slurm\n")
    monkeypatch.setenv("NVFLOW_CONFIG_DIR", str(target_dir))
    monkeypatch.chdir(external_project_root)

    guide = PROJECT_ROOT / "docs" / "getting-started" / "creating-a-project.md"
    commands = [
        command for path, command, execute in _documented_commands() if path == guide and execute
    ]
    assert commands

    environment = os.environ.copy()
    for command in commands:
        tokens = [
            token.replace("your-project-name:", "starter:").replace("your-project-name", "starter")
            for token in shlex.split(command)
        ]
        if "--config" in tokens:
            config_index = tokens.index("--config") + 1
            tokens[config_index] = str(config)
        if "--project-dir" in tokens:
            project_dir_index = tokens.index("--project-dir") + 1
            tokens[project_dir_index] = str(external_project_root)
        cli_arguments = tokens[3:] if tokens[:3] == ["uv", "run", "nvflow"] else tokens[1:]
        result = subprocess.run(
            [sys.executable, "-c", "from nvflow.cli.main import app; app()", *cli_arguments],
            cwd=external_project_root,
            env=environment,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{command}\n{result.stdout}\n{result.stderr}"


def test_copied_starter_accepts_a_new_project_identity(tmp_path: Path) -> None:
    project_root = tmp_path / "renamed-project"
    shutil.copytree(STARTER_ROOT, project_root)
    manifest = project_root / "project.yaml"
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace("project: starter", "project: renamed"),
        encoding="utf-8",
    )
    config = project_root / "workflows" / "generate.yaml"

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from nvflow import load_workflow; "
                f"workflow=load_workflow({str(config)!r}, project_dir={str(project_root)!r}); "
                "assert workflow.project_context.identity == 'renamed'; "
                "assert workflow.resolve_stage_class('generate_answer').__name__ "
                "== 'GenerateAnswerStage'"
            ),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_project_starter_environment_contract_is_consistent() -> None:
    project = tomllib.loads((STARTER_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    manifest = yaml.safe_load((STARTER_ROOT / "project.yaml").read_text(encoding="utf-8"))
    nvflow = next(
        Requirement(dependency)
        for dependency in project["project"]["dependencies"]
        if Requirement(dependency).name == "nvflow"
    )

    assert nvflow.specifier == SpecifierSet(manifest["requires_nvflow"])
    assert nvflow.specifier.contains("1.1")
    assert not nvflow.specifier.contains("2")
    assert manifest["description"] == "Minimal runnable NVFlow project starter"
    assert "workflow_order" not in manifest
    assert "recipes" not in manifest
    assert set(project["tool"]["uv"]) == {"sources"}
    assert set(project["tool"]["uv"]["sources"]) == {"nvflow"}
    assert not (STARTER_ROOT / "uv.lock").exists()

    guide = (PROJECT_ROOT / "docs" / "getting-started" / "creating-a-project.md").read_text(
        encoding="utf-8"
    )
    assert "uv sync" in guide
    assert "lightweight `nvflow` bootstrap" in guide
    assert "uv run nvflow sync-project" in guide
    assert "Worker dependencies remain in worker images" in guide
    assert "## Run the starter before customizing it" in guide
    assert guide.index("## Run the starter before customizing it") < guide.index(
        "## Understand project identity"
    )
    assert "[Run your first workflow](first-workflow.md)" in guide
    assert "NVFLOW_CONFIG_DIR" in guide
    assert "worker" in guide and "image" in guide
    assert guide.index("NVFLOW_CONFIG_DIR") < guide.index("nvflow plan \\")
    assert 'NEW_PROJECT_ROOT="/path/to/your-chosen-project-directory"' in guide
    assert "RECIPE_PARENT=" not in guide
    assert "RECIPE_NAME=" not in guide
    assert "$HOME/my-recipe" not in guide

    example = (STARTER_ROOT / "README.md").read_text(encoding="utf-8")
    for section in (
        "What it demonstrates",
        "What it runs",
        "Repository contents",
        "Use it as a starter",
    ):
        assert section in example
    assert "/workspace/outputs/starter/generate-answer/" in example
    assert "uv sync" in example
    assert "list-recipes --project starter" not in example
    assert "docs/getting-started/creating-a-project.md" in example
    assert "cp -R" not in example

    docs_index = (PROJECT_ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    assert "Create an NVFlow project from the starter" in docs_index
    assert "cp -R" not in docs_index


def test_project_manifests_include_discovery_metadata() -> None:
    manifests = sorted((PROJECT_ROOT / "projects").glob("*/project.yaml"))
    assert {path.parent.name for path in manifests} == {"finance", "hopchain", "starter"}

    for path in manifests:
        manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(manifest["description"], str)
        assert manifest["description"].strip() == manifest["description"]
        assert "\n" not in manifest["description"]
        assert "workflow_order" not in manifest
        assert manifest["project"]
        assert isinstance(manifest.get("recipes", {}), dict)
        if manifest["project"] == "starter":
            assert "recipes" not in manifest
        for recipe in manifest.get("recipes", {}).values():
            workflows = recipe["workflows"]
            assert workflows
            for selector, node in workflows.items():
                assert isinstance(selector, str) and not selector.endswith((".yaml", ".yml"))
                assert set(node) <= {"depends_on", "optional"}
                assert set(node.get("depends_on", [])) <= set(workflows)


def test_authored_workflows_derive_project_identity_from_manifest() -> None:
    workflow_files = sorted((PROJECT_ROOT / "projects").glob("*/workflows/**/*.yaml"))
    assert workflow_files
    for path in workflow_files:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            assert "project" not in payload, path.relative_to(PROJECT_ROOT)

    finance_workflows = PROJECT_ROOT / "projects" / "finance" / "workflows"
    assert not (finance_workflows / "download_sec_filings.yaml").exists()
    assert {path.stem for path in (finance_workflows / "download").glob("*.yaml")} == {
        "base",
        "smoke",
        "demo",
        "sp500",
    }


def test_scripts_follow_platform_project_and_test_ownership() -> None:
    finance_scripts = {
        "convert_checkpoint_to_hf.sh",
        "eval_metrics.py",
        "wandb_consolidate.py",
    }
    retired_fixture_tools = {"_dump_rollout_fixtures.py", "_dump_verify_fixtures.py"}

    assert finance_scripts <= {
        path.name for path in (PROJECT_ROOT / "projects" / "finance" / "scripts").iterdir()
    }
    assert not retired_fixture_tools & {
        path.name for path in (PROJECT_ROOT / "tests").rglob("*.py")
    }
    assert not finance_scripts & {path.name for path in (PROJECT_ROOT / "scripts").iterdir()}
    converter = (PROJECT_ROOT / "projects/finance/scripts/convert_checkpoint_to_hf.sh").read_text(
        encoding="utf-8"
    )
    assert "export PYTHONPATH=/nemo_run/code:" in converter
    assert "from nvflow.integrations.nemo_rl.checkpoint_converter import" in converter
    assert "build_dcp_conversion_script" in converter
    assert 'RUN_ROOT="$(dirname "$CHECKPOINT_DIR")"' in converter
    assert '"${RUN_ROOT}" "${STEP_NUMBER}" "${OUTPUT_DIR}"' in converter
    assert "convert_dcp_to_hf.py" not in converter
    assert "DTensorPolicyWorker/bin/python" not in converter
    assert "/opt/NeMo-RL" not in converter
    assert "/nemo_run/code/projects/finance/src" not in converter
    assert "nvflow_project_finance" not in converter
    assert "/nemo_run/code/recipes/" not in converter
    converter_module = (
        PROJECT_ROOT / "nvflow/integrations/nemo_rl/checkpoint_converter.py"
    ).read_text(encoding="utf-8")
    assert "automodel_policy_worker.AutomodelPolicyWorker" in converter_module
    assert "dtensor_policy_worker" not in converter_module
    assert "shard-*.safetensors" in converter_module
    assert "examples/converters/convert_dcp_to_hf.py" in converter_module
    assert (
        converter_module.count('ray_venv_python_preamble(AUTOMODEL_VENV_PYTHON, "automodel")') == 2
    )
    wandb_tool = (PROJECT_ROOT / "projects/finance/scripts/wandb_consolidate.py").read_text(
        encoding="utf-8"
    )
    assert "Path(__file__).resolve().parents[3]" in wandb_tool


def test_docs_do_not_pass_project_uris_as_cli_config_paths() -> None:
    failures = []
    for path in _markdown_files():
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if _PROJECT_CONFIG_RE.search(line):
                failures.append(f"{path.relative_to(PROJECT_ROOT)}:{line_number}")

    yaml_files: set[Path] = set()
    yaml_files.update((PROJECT_ROOT / "projects").rglob("*.yaml"))
    yaml_files.update((PROJECT_ROOT / "projects").rglob("*.yml"))
    for path in sorted(yaml_files):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#") and _PROJECT_CONFIG_RE.search(line):
                failures.append(f"{path.relative_to(PROJECT_ROOT)}:{line_number}")

    assert failures == []


@pytest.mark.parametrize(("doc_path", "workflow_path"), _WORKFLOW_DOCS)
def test_documented_workflow_stage_flow_matches_config(
    doc_path: str,
    workflow_path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SEC_IDENTITY_EMAIL", "nvflow-tests@example.com")
    monkeypatch.setenv("SEC_IDENTITY_COMPANY", "NVFlow Tests")
    text = (PROJECT_ROOT / doc_path).read_text(encoding="utf-8")
    workflow = load_workflow(PROJECT_ROOT / workflow_path)

    missing = [stage for stage in workflow.pipeline_stages if stage not in text]
    assert missing == []


def test_normative_platform_docs_do_not_own_concrete_projects() -> None:
    forbidden = (
        "nvfinance",
        "nvhopchain",
        "/hf_models/Qwen/Qwen3.5",
        "SEC-QUE score",
    )
    failures = []
    for path in (PROJECT_ROOT / "docs" / "platform").rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        for value in forbidden:
            if value in text:
                failures.append(f"{path.relative_to(PROJECT_ROOT)}: {value}")
    assert failures == []


def test_stale_project_and_doc_paths_are_absent() -> None:
    failures = []
    for path in _markdown_files():
        # RFCs are dated decision records and keep the paths they describe.
        if path.is_relative_to(PROJECT_ROOT / "docs" / "rfcs"):
            continue
        text = path.read_text(encoding="utf-8")
        for value in _STALE_PATHS:
            if value in text:
                failures.append(f"{path.relative_to(PROJECT_ROOT)}: {value}")
    assert failures == []


def test_runtime_sources_do_not_use_removed_project_contracts() -> None:
    failures = []
    forbidden = (
        "/nemo_run/code/recipes",
        "nvflow_recipe_finance",
        "nvflow_recipe_multimodal",
        "workflow_order",
        "finance:download_sec_filings",
        "download_sec_filings.yaml",
        "sap-500",
        "sap500",
        "SAP-500",
        "nemo_skills.pipeline.cli",
    )
    for root in (PROJECT_ROOT / "nvflow", PROJECT_ROOT / "projects", PROJECT_ROOT / "scripts"):
        for path in root.rglob("*"):
            if path.suffix not in {".py", ".sh", ".yaml", ".yml", ".md", ".txt"}:
                continue
            text = path.read_text(encoding="utf-8")
            for value in forbidden:
                if value in text:
                    failures.append(f"{path.relative_to(PROJECT_ROOT)}: {value}")

    assert failures == []
