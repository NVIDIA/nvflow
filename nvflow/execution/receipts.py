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
"""Non-durable scheduler receipts for submitted launch plans."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from nvflow.execution.contracts import LaunchPlan
from nvflow.project.provenance import ProjectProvenance


@dataclass(frozen=True)
class SubmittedTask:
    """Planned task and the experiment used for best-effort discovery."""

    task_id: str
    name: str
    experiment: str


@dataclass(frozen=True)
class SubmittedExperiment:
    """Best-effort live handles discovered at experiment granularity."""

    name: str
    scheduler_handles: tuple[str, ...] = ()
    discovery_complete: bool = False


@dataclass(frozen=True)
class SubmissionReceipt:
    """Non-durable result of submitting one launch plan."""

    plan_digest: str
    target: str
    plan_complete: bool
    opaque_stages: tuple[str, ...]
    tasks: tuple[SubmittedTask, ...]
    experiments: tuple[SubmittedExperiment, ...]
    provenance: ProjectProvenance | None = None
    engine: Literal["slurm", "ray"] = "slurm"
    execution_digest: str | None = None
    target_fingerprint: str | None = None
    controller_handle: str | None = None


def discover_submission_receipt(
    plan: LaunchPlan,
    *,
    captured_handles: Mapping[str, Sequence[str]] | None = None,
    engine: Literal["slurm", "ray"] = "slurm",
    execution_digest: str | None = None,
    target_fingerprint: str | None = None,
    controller_handle: str | None = None,
) -> SubmissionReceipt:
    """Discover live NeMo-Run handles for a plan after legacy submission."""
    experiment_names: list[str] = []
    submitted_tasks = []
    for task in plan.tasks:
        experiment = str(
            task.parameters.get("experiment_name") or task.parameters.get("expname") or task.name
        )
        if experiment not in experiment_names:
            experiment_names.append(experiment)
        submitted_tasks.append(
            SubmittedTask(
                task_id=task.task_id,
                name=task.name,
                experiment=experiment,
            )
        )
    submitted_experiments = []
    get_exp_handles: Any = None
    fallback_loaded = False
    for experiment in experiment_names:
        handles: tuple[str, ...] = ()
        discovery_complete = False
        if captured_handles is not None and experiment in captured_handles:
            handles = tuple(str(handle) for handle in captured_handles[experiment])
            discovery_complete = True
        else:
            if engine == "slurm" and not fallback_loaded:
                fallback_loaded = True
                try:
                    from nemo_skills.pipeline.utils.exp import get_exp_handles as fallback
                except ImportError:
                    fallback = None
                get_exp_handles = fallback
        if not discovery_complete and get_exp_handles is not None:
            try:
                handles = tuple(
                    str(handle)
                    for handle in get_exp_handles(
                        experiment,
                        ignore_finished=False,
                        ignore_exp_not_exists=True,
                    )
                )
                discovery_complete = True
            except (OSError, RuntimeError, ValueError):
                pass
        submitted_experiments.append(
            SubmittedExperiment(
                name=experiment,
                scheduler_handles=handles,
                discovery_complete=discovery_complete,
            )
        )
    return SubmissionReceipt(
        plan_digest=plan.digest,
        target=plan.target,
        plan_complete=plan.complete,
        opaque_stages=tuple(stage.stage for stage in plan.stages if stage.disposition == "opaque"),
        tasks=tuple(submitted_tasks),
        experiments=tuple(submitted_experiments),
        provenance=plan.provenance,
        engine=engine,
        execution_digest=execution_digest,
        target_fingerprint=target_fingerprint,
        controller_handle=controller_handle,
    )
