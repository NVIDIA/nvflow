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
"""Stable human-readable rendering for workflow launch plans."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

from nvflow.execution.contracts import LaunchPlan, ResourceRequest, TaskPlan
from nvflow.execution.inspection import StageOutputInspection


@dataclass(frozen=True)
class _TaskResources:
    """Normalized scheduler resources for one task."""

    kind: str
    gpus: int


def render_plan_text(
    plan: LaunchPlan,
    *,
    verbose: bool = False,
    output_inspections: Sequence[StageOutputInspection] = (),
    jobs_per_stage: Mapping[str, int] | None = None,
) -> str:
    """Render a compact plan summary, with task details on request.

    *jobs_per_stage* adds the scheduler jobs that submitting each stage starts.
    """
    lines = [
        f"NVFlow plan: {plan.project}.{plan.workflow}",
        f"Mode: {plan.planning_mode.upper()}",
        f"Target config: {plan.target} loaded",
        _filesystem_inspection_line(plan.planning_mode),
        _tasks_scope_line(plan.planning_mode),
    ]
    if plan.environments:
        label = "Environment" if len(plan.environments) == 1 else "Environments"
        lines.append(f"{label}: {', '.join(plan.environments)}")
    lines.extend(
        [
            _source_line(plan),
            _managed_model_paths_line(plan),
            _coverage_line(plan),
        ]
    )
    if plan.planning_mode == "live":
        lines.append(_inspection_result_line(plan, output_inspections))
    if verbose:
        lines.append(f"Plan digest: {plan.digest}")
        if plan.provenance is not None:
            provenance = plan.provenance
            revision = provenance.commit[:12] if provenance.commit else "unavailable"
            lines.append(
                "Project provenance: "
                f"{provenance.source_subpath}@{revision} · "
                f"{provenance.content_digest or 'no content digest'}"
            )

    all_tasks = list(plan.tasks)
    resources = [_task_resources(task) for task in all_tasks]
    lines.append("")
    totals = [_count(len(plan.stages), "stage"), _count(len(all_tasks), "task")]
    if jobs_per_stage is not None:
        totals.append(_count(sum(jobs_per_stage.values()), "job"))
    lines.append(" · ".join(totals))

    task_mix = _format_task_mix(resources, separator=" · ")
    if task_mix != "none":
        lines.append(task_mix)
    declared_resources = [resource for task in all_tasks for resource in task.resources]
    if largest := _format_largest_allocation(declared_resources):
        lines.append(f"Largest resource group: {largest}")
    stage_gpus = {stage.stage: _gpus_at_once(stage.tasks) for stage in plan.stages}
    busiest = max(stage_gpus, key=stage_gpus.__getitem__, default=None)
    if busiest is not None and stage_gpus[busiest]:
        lines.append(f"Busiest stage: {busiest} ({stage_gpus[busiest]:,} GPUs at once)")
    cumulative_gpus = sum(resource.gpus for resource in resources if resource.kind == "GPU")
    if cumulative_gpus:
        lines.append(
            f"Cumulative requested GPUs: {cumulative_gpus:,} (sum across tasks; not concurrent)"
        )

    lines.extend(
        [
            "",
            *_format_stage_table(
                plan, output_inspections=output_inspections, jobs_per_stage=jobs_per_stage
            ),
        ]
    )
    if any(inspection.presence == "exists" for inspection in output_inspections):
        lines.append(
            "● an output of the stage's final tasks exists; "
            "may indicate prior activity, not proof of completion."
        )
    unmounted = sorted(
        {
            f"/{PurePosixPath(path).parts[1]}"
            for inspection in output_inspections
            for path in inspection.hidden_paths
        }
    )
    if unmounted:
        verb, pronoun = ("is", "it") if len(unmounted) == 1 else ("are", "them")
        lines.append(
            f"unavailable: {', '.join(unmounted)} {verb} not mounted on this machine, "
            f"so outputs under {pronoun} cannot be checked from here."
        )
    if verbose:
        lines.append("")
        lines.append("Task details:")
        for stage in plan.stages:
            for task in stage.tasks:
                lines.extend(_format_task_details(task))
    elif all_tasks:
        lines.extend(
            [
                "",
                "Use --verbose for task details, output paths, and full provenance; "
                "--format json for the complete plan.",
            ]
        )
    return "\n".join(lines)


def _source_line(plan: LaunchPlan) -> str:
    provenance = plan.provenance
    if provenance is None:
        return "Source eligibility: UNKNOWN — project provenance unavailable"
    if not provenance.submittable:
        reason = provenance.reason or "project source is not submittable"
        return f"Source eligibility: BLOCKED — {reason}"
    return "Source eligibility: READY"


def _coverage_line(plan: LaunchPlan) -> str:
    if plan.complete:
        return "Plan coverage: COMPLETE"
    opaque = sum(not stage.complete for stage in plan.stages)
    return f"Plan coverage: PARTIAL — {_count(opaque, 'opaque stage')}"


def _managed_model_paths_line(plan: LaunchPlan) -> str:
    paths = tuple(
        dict.fromkeys(
            path
            for stage in plan.stages
            for task in stage.tasks
            for path in task.parameters.get("managed_model_paths", ())
        )
    )
    if not paths:
        return "Managed model paths: NOT APPLICABLE"
    return "Managed model paths: NOT CHECKED — " + ", ".join(paths)


def _count(value: int, noun: str) -> str:
    return f"{value} {noun if value == 1 else noun + 's'}"


def _filesystem_inspection_line(mode: str) -> str:
    if mode == "live":
        return "Filesystem inspection: ENABLED — stages may discover inputs or omit reusable work"
    return "Filesystem inspection: OFF — workflow inputs and outputs are not inspected"


def _tasks_scope_line(mode: str) -> str:
    if mode == "live":
        return "Tasks shown: stage-specific selection after filesystem inspection"
    return "Tasks shown: all configured work"


def _task_resources(task: TaskPlan) -> _TaskResources:
    if not task.resources:
        return _TaskResources("undeclared", 0)
    known_gpu_counts = [
        resource.nodes * resource.gpus_per_node
        for resource in task.resources
        if resource.gpus_per_node is not None
    ]
    gpus = sum(known_gpu_counts)
    if gpus > 0:
        kind = "GPU"
    elif len(known_gpu_counts) == len(task.resources):
        kind = "CPU"
    else:
        kind = "undeclared"
    return _TaskResources(kind, gpus)


def _format_task_mix(
    resources: list[_TaskResources],
    *,
    separator: str = " / ",
) -> str:
    counts = {
        "GPU": sum(resource.kind == "GPU" for resource in resources),
        "CPU": sum(resource.kind == "CPU" for resource in resources),
        "undeclared": sum(resource.kind == "undeclared" for resource in resources),
    }
    parts = [
        _count(counts["GPU"], "GPU task") if counts["GPU"] else "",
        _count(counts["CPU"], "CPU task") if counts["CPU"] else "",
        (
            f"{_count(counts['undeclared'], 'task')} with "
            f"{'resource' if counts['undeclared'] == 1 else 'resources'} undeclared"
            if counts["undeclared"]
            else ""
        ),
    ]
    return separator.join(part for part in parts if part) or "none"


def _format_largest_allocation(resources: list[ResourceRequest]) -> str:
    gpu_resources = [
        resource
        for resource in resources
        if resource.gpus_per_node is not None and resource.gpus_per_node > 0
    ]
    if gpu_resources:
        largest = max(
            gpu_resources,
            key=lambda resource: (
                resource.nodes * (resource.gpus_per_node or 0),
                resource.nodes,
            ),
        )
        gpus = largest.nodes * (largest.gpus_per_node or 0)
        return f"{_count(largest.nodes, 'node')} / {_count(gpus, 'GPU')}"
    cpu_resources = [resource for resource in resources if resource.gpus_per_node == 0]
    if cpu_resources:
        largest = max(cpu_resources, key=lambda resource: resource.nodes)
        return _count(largest.nodes, "CPU node")
    return ""


def _inspection_result_line(
    plan: LaunchPlan,
    output_inspections: Sequence[StageOutputInspection],
) -> str:
    selected = _count(len(plan.tasks), "task")
    counts = {
        presence: sum(inspection.presence == presence for inspection in output_inspections)
        for presence in ("exists", "missing", "not_declared", "unavailable")
    }
    descriptions = {
        "exists": "with existing output",
        "missing": "with missing output",
        "not_declared": "without declared output",
        "unavailable": "with unavailable inspection",
    }
    summary = [
        f"{_count(count, 'stage')} {descriptions[presence]}"
        for presence, count in counts.items()
        if count
    ]
    return f"Output presence: {' · '.join(summary) or 'unavailable'}; {selected} selected"


def _format_stage_table(
    plan: LaunchPlan,
    *,
    output_inspections: Sequence[StageOutputInspection] = (),
    jobs_per_stage: Mapping[str, int] | None = None,
) -> list[str]:
    presence_by_stage = {inspection.stage: inspection.presence for inspection in output_inspections}
    show_presence = plan.planning_mode == "live"
    rows = []
    for stage in plan.stages:
        task_resources = [_task_resources(task) for task in stage.tasks]
        declared_resources = [resource for task in stage.tasks for resource in task.resources]
        task_count = str(len(stage.tasks))
        if stage.disposition == "planned":
            work = _format_task_mix(task_resources)
        elif stage.disposition == "noop":
            work = "no work"
            if stage.reason:
                work = f"{work} — {stage.reason}"
        else:
            work = stage.disposition
            if stage.reason:
                work = f"{work} — {stage.reason}"
        presence = presence_by_stage.get(stage.stage, "unavailable")
        gpus_at_once = _gpus_at_once(stage.tasks)
        total_gpus = sum(resource.gpus for resource in task_resources)
        row = (
            f"● {stage.stage}" if presence == "exists" else stage.stage,
            task_count,
            *(() if jobs_per_stage is None else (str(jobs_per_stage.get(stage.stage, 0)),)),
            work,
            _format_largest_allocation(declared_resources) or "—",
            f"{gpus_at_once:,}" if gpus_at_once else "—",
            f"{total_gpus:,}" if total_gpus else "—",
        )
        if show_presence:
            row = (*row, presence.replace("_", " "))
        rows.append(row)

    headers = (
        "Stage",
        "Tasks",
        *(() if jobs_per_stage is None else ("Jobs",)),
        "Task mix",
        "Largest resource group",
        "GPUs at once",
        "Total GPUs",
    )
    if show_presence:
        headers = (*headers, "Output")
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def render_row(row: tuple[str, ...]) -> str:
        return "  ".join(value.ljust(widths[index]) for index, value in enumerate(row)).rstrip()

    return [
        render_row(headers),
        render_row(tuple("-" * width for width in widths)),
        *(render_row(row) for row in rows),
    ]


def _gpus_at_once(tasks: Sequence[TaskPlan]) -> int:
    """Estimate the most GPUs that *tasks* hold at the same time.

    Tasks are grouped into waves: a task that waits for another to finish joins
    a later wave, and one that waits only for another to start, such as a judge
    service and its training job, joins the same wave.
    """
    members = {task.task_id: task for task in tasks}
    waves: dict[str, int] = {}

    def wave(task_id: str) -> int:
        if task_id not in waves:
            waves[task_id] = max(
                (
                    wave(dependency.upstream)
                    + (0 if dependency.condition in {"after_start", "after_ready"} else 1)
                    for dependency in members[task_id].dependencies
                    if not dependency.external and dependency.upstream in members
                ),
                default=0,
            )
        return waves[task_id]

    totals: dict[int, int] = {}
    for task in tasks:
        index = wave(task.task_id)
        totals[index] = totals.get(index, 0) + _task_resources(task).gpus
    return max(totals.values(), default=0)


def _format_task_details(task: TaskPlan) -> list[str]:
    resources = ", ".join(_format_resource(resource) for resource in task.resources)
    lines = [
        f"    - {task.name} [{task.operation}]",
        f"      resources: {resources or 'not declared'}",
    ]
    if task.dependencies:
        dependencies = ", ".join(
            f"{dependency.upstream} ({dependency.condition}"
            f"{', external' if dependency.external else ''})"
            for dependency in task.dependencies
        )
        lines.append(f"      dependencies: {dependencies}")
    for command in task.commands:
        template = ", runtime template" if command.runtime_template else ""
        lines.append(f"      command ({command.role}, {command.container}{template}):")
        lines.extend(f"        {line}" for line in command.command.splitlines())
    if task.outputs:
        outputs = ", ".join(
            f"{output.name}={output.path}{' (optional)' if output.optional else ''}"
            for output in task.outputs
        )
        lines.append(f"      outputs: {outputs}")
    return lines


def _format_resource(resource: ResourceRequest) -> str:
    nodes = _count(resource.nodes, "node")
    if resource.gpus_per_node is None:
        accelerator = "GPUs unspecified"
    elif resource.gpus_per_node == 0:
        accelerator = "CPU"
    else:
        accelerator = f"{_count(resource.gpus_per_node, 'GPU')}/node"
    qualifiers = [
        value
        for value in (
            resource.partition,
            resource.account,
            "exclusive" if resource.exclusive else None,
        )
        if value
    ]
    suffix = f", {', '.join(qualifiers)}" if qualifiers else ""
    return f"{resource.role}={nodes}, {accelerator}{suffix}"
