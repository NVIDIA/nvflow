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

"""Check syntax and import boundaries required by current Python 3.13 workers."""

from __future__ import annotations

import ast
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
SYNTAX_FEATURE_VERSION = (3, min(sys.version_info.minor, 13))
PROJECT_PACKAGES = (
    ("finance", PROJECT_ROOT / "projects" / "finance", "nvfinance"),
    ("hopchain", PROJECT_ROOT / "projects" / "hopchain", "nvhopchain"),
    ("starter", PROJECT_ROOT / "projects" / "starter", "my_project"),
)
PROJECT_IMPORTS = tuple(
    (root, module, root / module / "__init__.py") for _name, root, module in PROJECT_PACKAGES
)
WORKER_ENTRYPOINTS = (
    *(
        PROJECT_ROOT / "nvflow" / "integrations" / "nemo_gym" / "workers" / name
        for name in (
            "build_training_dataset.py",
            "shuffle_jsonl.py",
            "validate_dataset_selection.py",
            "verify.py",
        )
    ),
    PROJECT_ROOT / "nvflow" / "integrations" / "nemo_gym" / "native_sharding.py",
    PROJECT_ROOT / "nvflow" / "integrations" / "nemo_gym" / "reward_profiles.py",
    PROJECT_ROOT / "nvflow" / "integrations" / "model_servers" / "serve.py",
    PROJECT_ROOT / "nvflow" / "integrations" / "nemo_rl" / "checkpoint_converter.py",
    PROJECT_ROOT / "nvflow" / "stages" / "data" / "dataset_splitter.py",
    PROJECT_ROOT / "nvflow" / "stages" / "data" / "split_contract.py",
    *(
        PROJECT_ROOT / "nvflow" / "integrations" / "ray_jobs" / name
        for name in (
            "bundle.py",
            "client.py",
            "controller.py",
            "strict_site/sitecustomize.py",
            "worker.py",
        )
    ),
    *(
        PROJECT_ROOT / "nvflow" / "execution" / name
        for name in (
            "codec.py",
            "contracts.py",
            "run_records.py",
            "target.py",
        )
    ),
)


def check_project_imports(interpreters: Iterable[str]) -> None:
    import_script = """
import importlib.abc
import sys

class BlockNVFlow(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "nvflow" or fullname.startswith("nvflow."):
            raise ImportError(f"worker-boundary import blocked: {{fullname}}")
        return None

sys.meta_path.insert(0, BlockNVFlow())
sys.path.insert(0, {source_root!r})
__import__({module_name!r})
"""
    for source_root, module_name, source_path in PROJECT_IMPORTS:
        source = source_path.read_text(encoding="utf-8")
        ast.parse(
            source,
            filename=str(source_root),
            feature_version=SYNTAX_FEATURE_VERSION,
        )
        for interpreter in interpreters:
            subprocess.run(
                [
                    interpreter,
                    "-I",
                    "-c",
                    import_script.format(
                        source_root=str(source_root),
                        module_name=module_name,
                    ),
                ],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )


def check_worker_syntax() -> None:
    for path in WORKER_ENTRYPOINTS:
        ast.parse(
            path.read_text(encoding="utf-8"),
            filename=str(path),
            feature_version=SYNTAX_FEATURE_VERSION,
        )


def check_ray_worker_import_boundary(interpreter: str) -> None:
    """Import the Ray worker while launcher-only dependencies are unavailable."""
    script = """
import importlib.abc
import sys

blocked = {
    "omegaconf",
    "rich",
    "yaml",
    "packaging",
    "torch",
    "nemo_run",
    "nemo_skills",
    "ray",
}

class BlockLauncherDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in blocked:
            raise ImportError(f"worker-boundary import blocked: {fullname}")
        return None

sys.meta_path.insert(0, BlockLauncherDependencies())
sys.path.insert(0, __SOURCE_ROOT__)
__import__("nvflow.integrations.ray_jobs.worker")
"""
    subprocess.run(
        [
            interpreter,
            "-I",
            "-c",
            script.replace("__SOURCE_ROOT__", repr(str(PROJECT_ROOT))),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def main() -> None:
    if not ((3, 13, 14) <= sys.version_info[:3] < (3, 14, 0)):
        raise SystemExit(
            "check_worker_python313.py requires Python >=3.13.14,<3.14, got "
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        )
    check_project_imports((sys.executable,))
    check_worker_syntax()
    check_ray_worker_import_boundary(sys.executable)


if __name__ == "__main__":
    main()
