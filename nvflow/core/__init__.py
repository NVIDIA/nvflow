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

if TYPE_CHECKING:
    from nvflow.core.workflow_definition import WorkflowDefinition, load_workflow
    from nvflow.execution.runner import WorkflowRunner

__all__ = [
    "BaseStage",
    "WorkflowDefinition",
    "WorkflowRunner",
    "console",
    "load_workflow",
]


def __getattr__(name: str) -> Any:
    # WorkflowDefinition and WorkflowRunner pull in omegaconf, which is absent
    # from minimal worker containers. Resolve them lazily so workers can import
    # dependency-light leaf modules without loading the launcher stack.
    if name in {"WorkflowDefinition", "load_workflow"}:
        from nvflow.core.workflow_definition import WorkflowDefinition, load_workflow

        return {
            "WorkflowDefinition": WorkflowDefinition,
            "load_workflow": load_workflow,
        }[name]
    if name == "WorkflowRunner":
        from nvflow.execution.runner import WorkflowRunner

        return WorkflowRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Note: nemo-skills functions are imported directly in stage files when needed:
# from nvflow.integrations.nemo_skills.commands import generate, run_cmd, wrap_arguments
