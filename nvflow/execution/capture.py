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
"""Capture supported NeMo-Skills invocations from existing stage builders."""

from __future__ import annotations

import contextlib
import io
import sys
from contextvars import ContextVar
from dataclasses import replace
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

from nvflow.execution.contracts import StageExecution, TaskPlan, rekey_tasks
from nvflow.execution.operations import (
    declared_task_outputs,
    plan_generate,
    task_from_invocation,
)

if TYPE_CHECKING:
    from nvflow.execution.filesystem import TargetFilesystem
    from nvflow.execution.planner import PlanningContext
    from nvflow.execution.stages import CapturedExecutionStage


_SLURM_CLEANUP_EXPERIMENTS: ContextVar[list[str] | None] = ContextVar(
    "nvflow_slurm_cleanup_experiments",
    default=None,
)
_LIVE_FILESYSTEM: ContextVar[TargetFilesystem | None] = ContextVar(
    "nvflow_live_filesystem",
    default=None,
)


def reusable_output(path: str) -> bool:
    """Return whether a live plan finds *path* as a non-empty file on the target.

    Static plans inspect nothing, so the stage plans the work.
    """
    filesystem = _LIVE_FILESYSTEM.get()
    return filesystem is not None and filesystem.is_nonempty_file(path)


def declare_slurm_experiment_cleanup(expnames: list[str]) -> None:
    """Declare stale NeMo-Run experiment directories for Slurm lowering."""
    active = _SLURM_CLEANUP_EXPERIMENTS.get()
    if active is None:
        raise RuntimeError("Slurm experiment cleanup declarations require execution capture")
    active.extend(str(expname) for expname in expnames)


def capture_first_party_execution(
    stage: CapturedExecutionStage,
    context: PlanningContext,
) -> StageExecution:
    """Run a trusted stage builder while replacing supported submit calls."""
    stage_module_name = stage.__class__.__module__
    strategy = stage.capture_strategy

    import nvflow.integrations.nemo_skills.commands as pipeline_cli

    tasks: list[TaskPlan] = []
    no_op_reasons: list[str] = []
    cleanup_experiments: list[str] = []

    def capture_run_cmd(*args: Any, **kwargs: Any) -> None:
        tasks.append(
            task_from_invocation(
                context,
                index=len(tasks),
                operation="nemo_skills.run_cmd",
                positional=args,
                keyword=kwargs,
            )
        )

    def capture_generate(*args: Any, **kwargs: Any) -> None:
        if args:
            raise ValueError("Captured generate invocation must use keyword arguments")
        ctx = kwargs.pop("ctx", None)
        arguments = " ".join(getattr(ctx, "args", []))
        child_context = replace(
            context,
            expname=str(kwargs.get("expname") or context.expname),
            run_after=tuple(kwargs.get("run_after") or context.run_after),
        )
        contribution = plan_generate(
            child_context,
            arguments,
            **kwargs,
        )
        if contribution.disposition == "noop" and contribution.reason:
            no_op_reasons.append(contribution.reason)
        declared = declared_task_outputs()
        contribution_tasks = contribution.tasks
        if declared is not None and contribution_tasks:
            contribution_tasks = (
                *contribution_tasks[:-1],
                replace(contribution_tasks[-1], outputs=declared),
            )
        prefix = f"{context.stage}.generate{len(tasks)}"
        tasks.extend(
            replace(task, parameters={**task.parameters, "generation_invocation": prefix})
            for task in rekey_tasks(contribution_tasks, stage=context.stage, prefix=prefix)
        )

    def capture_gym_generation(**kwargs: Any) -> None:
        from nvflow.integrations.nemo_gym.generation import plan_gym_generation

        contribution = plan_gym_generation(
            context,
            task_index=len(tasks),
            **kwargs,
        )
        tasks.extend(contribution.tasks)
        if contribution.disposition == "noop" and contribution.reason:
            no_op_reasons.append(contribution.reason)

    cleanup_token = _SLURM_CLEANUP_EXPERIMENTS.set(cleanup_experiments)
    filesystem_token = _LIVE_FILESYSTEM.set(context.filesystem if context.live else None)
    try:
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(pipeline_cli, "run_cmd", capture_run_cmd))
            stack.enter_context(patch.object(pipeline_cli, "generate", capture_generate))
            if strategy == "gym_generation":
                stage_module = sys.modules[stage_module_name]
                stack.enter_context(
                    patch.object(
                        stage_module,
                        "submit_gym_generation",
                        capture_gym_generation,
                    )
                )
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            stage.build_tasks(
                config=dict(context.config),
                cluster=context.target.cluster,
                expname=context.expname,
                run_after=list(context.run_after) or None,
            )
    finally:
        _LIVE_FILESYSTEM.reset(filesystem_token)
        _SLURM_CLEANUP_EXPERIMENTS.reset(cleanup_token)

    if cleanup_experiments and tasks:
        tasks[0] = replace(
            tasks[0],
            parameters={
                **tasks[0].parameters,
                "slurm_cleanup_experiments": tuple(dict.fromkeys(cleanup_experiments)),
            },
        )

    if not tasks:
        if no_op_reasons:
            return StageExecution.noop(
                context.stage,
                "; ".join(dict.fromkeys(no_op_reasons)),
            )
        if stage.capture_empty_reason is not None:
            return StageExecution.noop(
                context.stage,
                stage.capture_empty_reason,
            )
        raise ValueError(
            f"Audited capture for {stage.__class__.__module__}.{stage.__class__.__name__} "
            "produced no tasks; use StagePlan.noop() for an intentional no-op"
        )
    return StageExecution(stage=context.stage, tasks=tuple(tasks))
