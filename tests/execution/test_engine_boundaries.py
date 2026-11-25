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
"""Static and runtime boundaries between direct Slurm and supplied Ray."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

from nvflow import load_workflow
from nvflow.execution.engines import engine_for
from nvflow.execution.planner import WorkflowPlanner
from nvflow.execution.target import (
    RayTarget,
    RegisteredCluster,
    ResolvedTarget,
    TargetBinding,
)

PROJECT_ROOT = Path(__file__).parents[2]


def test_slurm_engine_has_no_ray_imports() -> None:
    imports = _imports(PROJECT_ROOT / "nvflow/execution/engines/slurm.py")

    assert not any(name == "ray" or name.startswith("ray.") for name in imports)
    assert not any("integrations.ray_jobs" in name for name in imports)


def test_ray_engine_and_integration_have_no_slurm_imports_or_commands() -> None:
    paths = [
        PROJECT_ROOT / "nvflow/execution/engines/ray.py",
        *(PROJECT_ROOT / "nvflow/integrations/ray_jobs").glob("*.py"),
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        imports = _imports(path)
        assert not any("nemo_skills" in name or name.endswith(".slurm") for name in imports), path
        for forbidden in ("sbatch", "scancel", "SLURM_JOB_ID", "SLURM_NODEID"):
            assert forbidden not in text, f"{path}: {forbidden}"


def test_engine_dispatch_loads_only_selected_engine(tmp_path: Path) -> None:
    ray_module = "nvflow.execution.engines.ray"
    original = sys.modules.pop(ray_module, None)
    target = ResolvedTarget(
        binding=TargetBinding.resolve("cluster"),
        engine="slurm",
        fingerprint="sha256:target",
        source_path=tmp_path / "cluster.yaml",
    )

    try:
        selected = engine_for(target)

        assert type(selected).__name__ == "DirectSlurmEngine"
        assert ray_module not in sys.modules
    finally:
        if original is not None:
            sys.modules[ray_module] = original


def test_ray_dispatch_does_not_require_slurm_target_state(tmp_path: Path) -> None:
    cluster = RegisteredCluster(
        name="shared",
        jobs_url="http://127.0.0.1:8265",
    )
    target = ResolvedTarget(
        binding=TargetBinding.resolve("ray"),
        engine="ray",
        fingerprint="sha256:target",
        source_path=tmp_path / "ray.yaml",
        ray=RayTarget(
            control_cluster="shared",
            artifact_root="/shared",
            clusters={"shared": cluster},
            routes={"project": "shared"},
        ),
    )

    assert type(engine_for(target)).__name__ == "SuppliedRayJobsEngine"


def test_same_project_graph_is_shared_by_slurm_and_ray_targets(
    tmp_path: Path,
) -> None:
    definition = load_workflow(
        PROJECT_ROOT / "projects" / "starter" / "workflows" / "generate.yaml"
    )
    binding = TargetBinding.resolve(definition.cluster, "portable")
    planner = WorkflowPlanner(definition, binding)
    slurm = ResolvedTarget(
        binding=binding,
        engine="slurm",
        fingerprint="sha256:slurm",
        source_path=tmp_path / "slurm.yaml",
    )
    shared = RegisteredCluster(
        name="shared",
        jobs_url="http://127.0.0.1:8265",
    )
    ray = ResolvedTarget(
        binding=binding,
        engine="ray",
        fingerprint="sha256:ray",
        source_path=tmp_path / "ray.yaml",
        ray=RayTarget(
            control_cluster="shared",
            artifact_root=str(tmp_path),
            clusters={"shared": shared},
            routes={"nemo-skills": "shared", "vllm": "shared"},
        ),
    )

    slurm_graph = planner.execution_graph(slurm, live=False)
    ray_graph = planner.execution_graph(ray, live=False)

    assert slurm_graph.stages == ray_graph.stages
    assert slurm_graph.engine == "slurm"
    assert ray_graph.engine == "ray"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names
