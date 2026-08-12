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
"""Core infrastructure for workflow orchestration."""

from typing import TYPE_CHECKING, Any

from nvflow.core import console
from nvflow.core.base_stage import BaseStage
from nvflow.core.stage_registry import StageRegistry

if TYPE_CHECKING:
    from nvflow.core.ray_workflow_runner import RayWorkflowRunner, create_workflow_runner
    from nvflow.core.workflow_runner import WorkflowRunner

__all__ = [
    "BaseStage",
    "StageRegistry",
    "WorkflowRunner",
    "RayWorkflowRunner",
    "create_workflow_runner",
    "console",
]


def __getattr__(name: str) -> Any:
    # WorkflowRunner pulls in omegaconf, which is absent from minimal worker
    # containers (e.g. the SAM localization image). Those workers import only
    # leaf helper modules under nvflow.recipes, and recipe auto-discovery
    # touches this package -- so importing WorkflowRunner eagerly here would
    # crash them with ModuleNotFoundError. Resolve it lazily instead. The Ray
    # runner symbols subclass/build on WorkflowRunner, so they stay lazy too.
    if name == "WorkflowRunner":
        from nvflow.core.workflow_runner import WorkflowRunner

        return WorkflowRunner
    if name in ("RayWorkflowRunner", "create_workflow_runner"):
        from nvflow.core import ray_workflow_runner

        return getattr(ray_workflow_runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Note: nemo-skills functions are imported directly in stage files when needed:
# from nemo_skills.pipeline.cli import generate, run_cmd, wrap_arguments
