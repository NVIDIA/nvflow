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
"""Execution planning and target contracts."""

from typing import TYPE_CHECKING, Any

from nvflow.execution.contracts import (
    ArtifactDeclaration,
    CommandPlan,
    DependencyPlan,
    ExecutionGraph,
    LaunchPlan,
    ResourceRequest,
    ServicePlan,
    StageExecution,
    StagePlan,
    TaskPlan,
    internalize_dependencies,
    rekey_tasks,
)
from nvflow.execution.filesystem import (
    LocalTargetFilesystem,
    MutableTargetFilesystem,
    TargetFilesystem,
)
from nvflow.execution.inspection import OutputPresence, StageOutputInspection
from nvflow.execution.operations import (
    declare_task_outputs,
    opaque_payload_marker,
    plan_generate,
    plan_run_cmd,
)
from nvflow.execution.planner import PlanningContext, WorkflowPlanner, plan_stage
from nvflow.execution.receipts import (
    SubmissionReceipt,
    SubmittedExperiment,
    SubmittedTask,
    discover_submission_receipt,
)
from nvflow.execution.run_records import RunEvent, RunJournal
from nvflow.execution.stages import CapturedExecutionStage, PlannedExecutionStage
from nvflow.execution.target import (
    RayTarget,
    RegisteredCluster,
    ResolvedTarget,
    TargetBinding,
    TargetResolutionError,
    load_target_document,
    preflight_target_config,
    resolve_target,
)

if TYPE_CHECKING:
    from nvflow.execution.runner import WorkflowRunner

__all__ = [
    "ArtifactDeclaration",
    "CommandPlan",
    "CapturedExecutionStage",
    "DependencyPlan",
    "ExecutionGraph",
    "LaunchPlan",
    "LocalTargetFilesystem",
    "MutableTargetFilesystem",
    "OutputPresence",
    "PlanningContext",
    "PlannedExecutionStage",
    "RayTarget",
    "RegisteredCluster",
    "ResolvedTarget",
    "ResourceRequest",
    "RunEvent",
    "RunJournal",
    "ServicePlan",
    "StageExecution",
    "StagePlan",
    "StageOutputInspection",
    "SubmissionReceipt",
    "SubmittedExperiment",
    "SubmittedTask",
    "TargetBinding",
    "TargetResolutionError",
    "TargetFilesystem",
    "TaskPlan",
    "WorkflowPlanner",
    "WorkflowRunner",
    "declare_task_outputs",
    "discover_submission_receipt",
    "internalize_dependencies",
    "load_target_document",
    "opaque_payload_marker",
    "plan_generate",
    "plan_run_cmd",
    "plan_stage",
    "preflight_target_config",
    "rekey_tasks",
    "resolve_target",
]


def __getattr__(name: str) -> Any:
    if name == "WorkflowRunner":
        from nvflow.execution.runner import WorkflowRunner

        return WorkflowRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
