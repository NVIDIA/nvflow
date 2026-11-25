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
"""Characterize and enforce NVFlow package dependency boundaries."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from nvflow.project import ProjectManifest

PROJECT_ROOT = Path(__file__).parents[1]


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    class RuntimeImportVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.imports: set[str] = set()

        def visit_If(self, node: ast.If) -> None:  # noqa: N802 - ast visitor protocol
            if isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
                return
            self.generic_visit(node)

        def visit_Import(self, node: ast.Import) -> None:  # noqa: N802 - ast visitor protocol
            self.imports.update(alias.name for alias in node.names)

        def visit_ImportFrom(  # noqa: N802 - ast visitor protocol
            self,
            node: ast.ImportFrom,
        ) -> None:
            if node.module:
                self.imports.add(node.module)

    visitor = RuntimeImportVisitor()
    visitor.visit(tree)
    return visitor.imports


def test_core_cross_layer_imports_are_explicitly_bounded() -> None:
    core = PROJECT_ROOT / "nvflow" / "core"
    forbidden_prefixes = ("nvflow.execution", "nvflow.integrations")
    actual = {
        path.name: sorted(
            module for module in _imports(path) if module.startswith(forbidden_prefixes)
        )
        for path in core.glob("*.py")
    }
    actual = {path: imports for path, imports in actual.items() if imports}

    assert actual == {"__init__.py": ["nvflow.execution.runner"]}


def test_project_backend_imports_are_explicitly_bounded() -> None:
    allowed = {
        "nvflow.integrations.nemo_gym.environments",
        "nvflow.integrations.nemo_gym.native_contract",
        "nvflow.integrations.nemo_gym.native_rollout",
        "nvflow.integrations.nemo_gym.native_sharding",
        "nvflow.integrations.nemo_rl.checkpoints",
        "nvflow.integrations.nemo_skills",
        "nvflow.integrations.nemo_skills.commands",
    }
    project_roots: list[Path] = []
    for manifest_path in sorted((PROJECT_ROOT / "projects").glob("*/project.yaml")):
        manifest = ProjectManifest.load(manifest_path)
        source_root = manifest_path.parent / manifest.python_source_root
        project_roots.extend(
            path for path in source_root.iterdir() if (path / "__init__.py").is_file()
        )
    assert project_roots
    actual = {
        module
        for root in project_roots
        for path in root.rglob("*.py")
        for module in _imports(path)
        if module.startswith("nvflow.integrations")
    }

    assert actual == allowed


def test_project_and_recipe_contract_layers_do_not_import_integrations() -> None:
    offenders = {
        str(path.relative_to(PROJECT_ROOT)): sorted(
            module for module in _imports(path) if module.startswith("nvflow.integrations")
        )
        for root in (PROJECT_ROOT / "nvflow" / "project", PROJECT_ROOT / "nvflow" / "recipe")
        for path in root.rglob("*.py")
    }

    assert {path: imports for path, imports in offenders.items() if imports} == {}


def test_platform_code_uses_canonical_target_filesystem_import() -> None:
    compatibility_module = "nvflow.integrations.nemo_gym.launcher_fs"
    offenders = {
        str(path.relative_to(PROJECT_ROOT))
        for root in (PROJECT_ROOT / "nvflow", PROJECT_ROOT / "projects")
        for path in root.rglob("*.py")
        if path.name != "launcher_fs.py" and compatibility_module in _imports(path)
    }

    assert offenders == set()


def test_execution_does_not_import_reusable_stage_implementations() -> None:
    execution = PROJECT_ROOT / "nvflow" / "execution"
    offenders = {
        str(path.relative_to(PROJECT_ROOT)): sorted(
            module for module in _imports(path) if module.startswith("nvflow.stages")
        )
        for path in execution.rglob("*.py")
    }

    assert {path: imports for path, imports in offenders.items() if imports} == {}


def test_execution_concrete_integrations_are_limited_to_composition_points() -> None:
    execution = PROJECT_ROOT / "nvflow" / "execution"
    actual = {
        path.name: sorted(
            module for module in _imports(path) if module.startswith("nvflow.integrations")
        )
        for path in execution.glob("*.py")
    }
    actual = {path: imports for path, imports in actual.items() if imports}

    assert actual == {
        "capture.py": [
            "nvflow.integrations.nemo_gym.generation",
            "nvflow.integrations.nemo_skills.commands",
        ],
        "runner.py": [
            "nvflow.integrations.nemo_skills.filesystem",
            "nvflow.integrations.nemo_skills.target",
        ],
    }


def test_nemo_gym_modules_and_workers_are_split_at_explicit_boundaries() -> None:
    package = PROJECT_ROOT / "nvflow" / "integrations" / "nemo_gym"

    expected_modules = {
        "compare.py",
        "contracts.py",
        "environments.py",
        "generation.py",
        "job_scripts.py",
        "launcher_fs.py",
        "native_commands.py",
        "native_contract.py",
        "native_rewards.py",
        "native_rollout.py",
        "native_sharding.py",
        "operations.py",
        "operation_config.py",
        "postprocess.py",
        "reward_profiles.py",
        "shell_snippets.py",
        "vllm_judge.py",
        "workers/__init__.py",
        "workers/build_training_dataset.py",
        "workers/shuffle_jsonl.py",
        "workers/validate_dataset_selection.py",
        "workers/verify.py",
    }
    assert all((package / relative_path).is_file() for relative_path in expected_modules)
    removed_modules = {
        "create_overlay.py",
        "helpers.py",
        "merge_script.py",
        "resume_filter.py",
        "rollout.py",
        "rollout_script.py",
        "select_profiled_data.py",
        "verify.py",
        "verify_worker.py",
        "workers/create_overlay.py",
        "workers/select_profiled_data.py",
        "workers/validate_profile_selection.py",
        "workers/resume_filter.py",
    }
    assert all(not (package / relative_path).exists() for relative_path in removed_modules)

    for init_path in (package / "__init__.py", package / "workers" / "__init__.py"):
        init_tree = ast.parse(
            init_path.read_text(encoding="utf-8"),
            filename=str(init_path),
        )
        assert not any(
            isinstance(node, ast.Import | ast.ImportFrom) for node in ast.walk(init_tree)
        )


def test_superseded_finance_rollout_modules_and_overlays_are_removed() -> None:
    finance = PROJECT_ROOT / "projects" / "finance"
    removed_paths = {
        finance / "nvfinance/stages/evaluation/compare.py",
        finance / "nvfinance/stages/evaluation/prepare_gym_data.py",
        finance / "nvfinance/stages/rl/collect_rollouts.py",
        finance / "nvfinance/stages/rl/compute_rewards.py",
        finance / "nvfinance/stages/rl/prepare_data.py",
        finance / "nvfinance/stages/rl/rejudge.py",
        finance / "nvfinance/utils/rl/aggregate_seeds.py",
        finance / "nvfinance/utils/rl/analyze_rollouts.py",
        finance / "nvfinance/utils/rl/enrich_rollouts.py",
        finance / "nvfinance/utils/rl/export_difficulty.py",
        finance / "nvfinance/utils/rl/filter_training_data.py",
        finance / "nvfinance/utils/rl/shuffle_jsonl.py",
    }
    assert all(not path.exists() for path in removed_paths)
    assert not list((finance / "workflows/eval/overlays").glob("*.yaml"))
    assert not list((finance / "workflows/grpo/overlays").glob("*.yaml"))


def test_nemo_gym_module_dependencies_are_acyclic() -> None:
    package = PROJECT_ROOT / "nvflow" / "integrations" / "nemo_gym"
    modules = {path.stem: path for path in package.glob("*.py") if path.stem != "__init__"}
    graph: dict[str, set[str]] = {name: set() for name in modules}
    prefix = "nvflow.integrations.nemo_gym."

    for name, path in modules.items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            if node.level == 1:
                target = node.module.split(".", 1)[0]
            elif node.module.startswith(prefix):
                target = node.module.removeprefix(prefix).split(".", 1)[0]
            else:
                continue
            if target in modules:
                graph[name].add(target)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(module: str) -> None:
        if module in visiting:
            raise AssertionError(f"NeMo-Gym module import cycle at {module}")
        if module in visited:
            return
        visiting.add(module)
        for dependency in graph[module]:
            visit(dependency)
        visiting.remove(module)
        visited.add(module)

    for module in graph:
        visit(module)


def test_gym_semantic_validation_does_not_import_execution_dependencies() -> None:
    script = """
import sys
from nvflow.integrations.nemo_gym.operations import (
    normalize_reward_operation,
    normalize_rollout_operation,
)

normalize_rollout_operation(
    {
        "output_dir": "/outputs/rollouts",
        "container": "nemo-gym",
        "environments": {
            "example": {
                "config_paths": ["environment.yaml"],
                "agent_name": "example_agent",
            }
        },
        "rollout": {
            "input_data": "/data/input.jsonl",
            "num_repeats": 1,
            "models": {
                "policy": {
                    "deployment": "external",
                    "model": "policy",
                    "base_url": "https://policy.invalid/v1",
                }
            },
        },
    }
)
normalize_reward_operation(
    {
        "output_dir": "/outputs/rejudge",
        "rollouts_dir": "/outputs/rollouts",
        "container": "nemo-gym",
        "environments": {
            "example": {
                "config_paths": ["environment.yaml"],
                "agent_name": "example_agent",
            }
        },
        "rejudge": {
            "mode": "native",
            "models": {
                "judge": {
                    "deployment": "external",
                    "model_type": "openai_model",
                    "model": "judge",
                    "base_url": "https://judge.invalid/v1",
                }
            },
        },
    }
)

forbidden = {
    "nvflow.integrations.nemo_gym.job_scripts",
    "nvflow.integrations.nemo_gym.native_rewards",
    "nvflow.integrations.nemo_gym.native_rollout",
    "torch",
}
loaded = sorted(
    name
    for name in sys.modules
    if name in forbidden
    or name.startswith("nemo_skills")
    or name.startswith("torch.")
)
assert not loaded, loaded
"""
    subprocess.run([sys.executable, "-c", script], cwd=PROJECT_ROOT, check=True)


def test_generic_split_imports_no_gym_or_nemo_rl_modules() -> None:
    script = """
import sys

import nvflow.stages.data.dataset_splitter
import nvflow.stages.data.train_validation_split

families = ("nemo_gym", "nemo_rl")
prefixes = tuple(f"nvflow.{area}.{family}" for area in ("integrations", "stages") for family in families)
loaded = sorted(
    name for name in sys.modules if name.split(".", 1)[0] in families or name.startswith(prefixes)
)
assert not loaded, loaded
"""
    subprocess.run([sys.executable, "-c", script], cwd=PROJECT_ROOT, check=True)


def test_training_container_workers_import_without_launcher_dependencies() -> None:
    script = """
import importlib.abc
import sys

blocked = {"nemo_run", "nemo_skills", "omegaconf", "orjson", "rich", "typer", "yaml"}

class BlockLauncherDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in blocked:
            raise ImportError(f"worker-boundary import blocked: {fullname}")
        return None

sys.meta_path.insert(0, BlockLauncherDependencies())
import nvflow.integrations.nemo_rl.checkpoint_converter
import nvflow.stages.data.dataset_splitter
import nvflow.stages.data.split_contract
"""
    subprocess.run([sys.executable, "-c", script], cwd=PROJECT_ROOT, check=True)


def test_data_stage_workers_import_without_launcher_or_model_dependencies() -> None:
    # The decontamination merge step runs in whichever container embedded the rows.
    script = """
import importlib.abc
import sys

blocked = {
    "datasketch", "nemo_run", "nemo_skills", "omegaconf", "orjson", "rich",
    "sentence_transformers", "torch", "typer", "vllm", "yaml",
}

class BlockHeavyDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in blocked:
            raise ImportError(f"worker-boundary import blocked: {fullname}")
        return None

sys.meta_path.insert(0, BlockHeavyDependencies())
import nvflow.integrations.embeddings.settings
import nvflow.integrations.nemo_gym.workers.split_by_reward
import nvflow.stages.data.decontaminate_worker
import nvflow.stages.data.dedup_worker
import nvflow.stages.data.filter_rows_worker
"""
    subprocess.run([sys.executable, "-c", script], cwd=PROJECT_ROOT, check=True)


def test_shared_packages_do_not_encode_finance_policy() -> None:
    roots = (
        PROJECT_ROOT / "nvflow" / "integrations" / "nemo_gym",
        PROJECT_ROOT / "nvflow" / "integrations" / "nemo_rl",
        PROJECT_ROOT / "nvflow" / "stages" / "data",
        PROJECT_ROOT / "nvflow" / "stages" / "nemo_gym",
        PROJECT_ROOT / "nvflow" / "stages" / "nemo_rl",
    )
    forbidden = ("nvfinance", "finance_sec_search")
    offenders = {
        str(path.relative_to(PROJECT_ROOT)): token
        for root in roots
        for path in root.rglob("*.py")
        for token in forbidden
        if token in path.read_text(encoding="utf-8")
    }
    assert offenders == {}
