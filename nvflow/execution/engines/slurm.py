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
"""Direct Slurm execution lowering.

This module is the only execution engine allowed to import the NeMo-Skills /
NeMo-Run Slurm integration.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
from collections import Counter
from contextlib import AbstractContextManager
from dataclasses import replace
from pathlib import Path
from typing import Any

from nvflow.core.code_root import render_code_root
from nvflow.execution.contracts import (
    SLURM_GROUP_JOBS,
    DependencyPlan,
    ExecutionGraph,
    ResourceRequest,
    TaskPlan,
    completion_task_ids,
)
from nvflow.execution.engines.base import ActiveJob, EngineContext, ExecutionEngine, JobListing
from nvflow.execution.job_labels import JobLabel, task_label
from nvflow.execution.receipts import SubmissionReceipt, discover_submission_receipt
from nvflow.execution.target import ResolvedTarget

# The name and comment come last so a name containing "|" stays whole.
_SQUEUE_FORMAT = "|".join(
    ("%i", "%T", "%M", "%P", "%D", "%C", "%b", "%a", "%Q", "%S", "%R", "%j", "%k")
)


class DirectSlurmEngine(ExecutionEngine):
    """Lower exact tasks through the existing direct Slurm command adapters."""

    def submit(
        self,
        graph: ExecutionGraph,
        context: EngineContext,
    ) -> SubmissionReceipt:
        if graph.engine != "slurm" or context.target.engine != "slurm":
            raise ValueError("DirectSlurmEngine requires a Slurm execution graph and target")

        from nvflow.integrations.nemo_skills.commands import capture_submissions
        from nvflow.integrations.nemo_skills.filesystem import submission_tunnel_scope
        from nvflow.integrations.nemo_skills.slurm import (
            apply_sbatch_args_autopatch,
            job_comment,
            jobs_excluded_from_dependencies,
        )
        from nvflow.integrations.nemo_skills.target import bind_target_config
        from nvflow.project.packaging import project_submission

        def labeled(task: TaskPlan) -> AbstractContextManager[None]:
            return job_comment(task_label(context.workflow, task).to_comment())

        apply_sbatch_args_autopatch()
        groups = _job_groups(_topological_tasks(graph))
        tasks = tuple(_grouped_task(group) for group in groups)
        task_experiments = {task.task_id: _experiment_name(task) for task in tasks}
        pipeline_groups = _pipeline_groups(tasks)
        submitted_pipelines: set[str] = set()
        nemo_rl_groups = _nemo_rl_groups(tasks)
        submitted_nemo_rl: set[str] = set()
        generation_groups = _generation_groups(tasks)
        submitted_generations: set[str] = set()
        with (
            project_submission(
                context.project_context,
                graph.provenance,
                context.platform_source,
            ),
            bind_target_config(context.target.source_path),
            submission_tunnel_scope(),
            capture_submissions() as captured,
            jobs_excluded_from_dependencies() as excluded_jobs,
        ):
            _clean_stale_experiments(
                tasks,
                cluster=context.target.binding.cluster,
            )
            for task in tasks:
                invocation = _generation_invocation(task)
                if invocation is not None:
                    if invocation not in submitted_generations:
                        with labeled(task):
                            _submit_generation(
                                generation_groups[invocation],
                                cluster=context.target.binding.cluster,
                                task_experiments=task_experiments,
                            )
                        submitted_generations.add(invocation)
                    continue
                if _is_nemo_rl_task(task):
                    experiment = _experiment_name(task)
                    if experiment not in submitted_nemo_rl:
                        with labeled(task):
                            _submit_nemo_rl(
                                nemo_rl_groups[experiment],
                                cluster=context.target.binding.cluster,
                                task_experiments=task_experiments,
                                excluded_jobs=excluded_jobs,
                            )
                        submitted_nemo_rl.add(experiment)
                    continue
                if _requires_pipeline(task):
                    experiment = _experiment_name(task)
                    if experiment not in submitted_pipelines:
                        with labeled(task):
                            _submit_pipeline(
                                pipeline_groups[experiment],
                                cluster=context.target.binding.cluster,
                                task_experiments=task_experiments,
                                excluded_jobs=excluded_jobs,
                            )
                        submitted_pipelines.add(experiment)
                    continue
                with labeled(task):
                    _submit_task(
                        task,
                        cluster=context.target.binding.cluster,
                        dependency_names=tuple(
                            dict.fromkeys(
                                task_experiments[dependency.upstream]
                                for dependency in task.dependencies
                                if not dependency.external
                            )
                        ),
                    )

        handles = _receipt_handles(captured.handles_by_experiment(), tasks)
        return discover_submission_receipt(
            graph.to_plan(planning_mode="live"),
            captured_handles=_grouped_handles(handles, groups),
            engine="slurm",
            execution_digest=graph.digest,
            target_fingerprint=context.target.fingerprint,
        )

    def jobs_per_stage(self, graph: ExecutionGraph) -> dict[str, int]:
        return dict(Counter(group[0].stage for group in _job_groups(_topological_tasks(graph))))

    def active_jobs(self, target: ResolvedTarget) -> JobListing:
        from nvflow.integrations.nemo_skills import launcher_is_remote, load_target_config
        from nvflow.integrations.nemo_skills.target import bind_target_config

        command = ["squeue", "--me", "--noheader", f"--format={_SQUEUE_FORMAT}"]
        with bind_target_config(target.source_path):
            cluster_config = load_target_config(target.binding.cluster)
            if launcher_is_remote(cluster_config):
                from nemo_skills.pipeline.utils.cluster import get_tunnel

                output = get_tunnel(cluster_config).run(shlex.join(command), hide=True).stdout
            else:
                output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
        rows = []
        for row in output.splitlines():
            fields = row.split("|", 11)
            if len(fields) != 12:
                continue
            name, _, comment = fields[11].rpartition("|")
            rows.append((fields[:11], name, JobLabel.from_comment(comment)))
        # A heterogeneous job's components share one label.
        labels = {
            fields[0].partition("+")[0]: label for fields, _, label in rows if label is not None
        }
        jobs = []
        for fields, name, label in rows:
            (
                job_id,
                state,
                elapsed,
                partition,
                nodes,
                cpus,
                tres,
                account,
                priority,
                start,
                reason,
            ) = fields
            label = label or labels.get(job_id.partition("+")[0]) or JobLabel()
            jobs.append(
                ActiveJob(
                    job_id,
                    name,
                    state,
                    elapsed,
                    partition,
                    workflow=label.workflow or None,
                    stage=label.stage or None,
                    experiment=label.experiment or None,
                    reason=_reported(reason.removeprefix("(").removesuffix(")")),
                    nodes=_count(nodes),
                    gpus_per_node=_gpus_per_node(tres),
                    cpus=_count(cpus),
                    start=_reported(start),
                    account=_reported(account),
                    priority=_count(priority),
                )
            )
        return JobListing(tuple(sorted(jobs, key=lambda job: _submission_order(job.job_id))))


def _submission_order(job_id: str) -> tuple[tuple[int, ...], str]:
    """Order job IDs as Slurm assigns them: by job, then heterogeneous part or array task."""
    return tuple(int(number) for number in re.findall(r"\d+", job_id)), job_id


def _reported(value: str) -> str | None:
    """Return a squeue field, or None where squeue reports nothing."""
    value = value.strip()
    return None if value in {"", "N/A", "(null)"} else value


def _count(value: str) -> int | None:
    value = value.strip()
    return int(value) if value.isdigit() else None


def _gpus_per_node(tres: str) -> int | None:
    """Return the GPU count in a squeue TRES-per-node field such as ``gres/gpu:h100:4``."""
    for entry in tres.split(","):
        parts = re.split(r"[:=]", entry.partition("(")[0])
        if any(part.endswith("gpu") for part in parts[:-1]) and parts[-1].isdigit():
            return int(parts[-1])
    return None


def _clean_stale_experiments(
    tasks: tuple[TaskPlan, ...],
    *,
    cluster: str,
) -> None:
    """Remove explicitly declared stale NeMo-Run experiment directories."""
    names = tuple(
        dict.fromkeys(
            str(name)
            for task in tasks
            for name in task.parameters.get("slurm_cleanup_experiments", ())
        )
    )
    if not names:
        return
    if any(not name or Path(name).name != name or name in {".", ".."} for name in names):
        raise ValueError("Slurm cleanup experiment names must be safe basenames")

    from nvflow.integrations.nemo_skills import launcher_is_remote, load_target_config

    cluster_config = load_target_config(cluster)
    tunnel_config = cluster_config.get("ssh_tunnel") or {}
    job_dir = cluster_config.get("job_dir") or tunnel_config.get("job_dir")
    if not job_dir:
        return
    root = Path(str(job_dir)) / "experiments"
    targets = tuple(root / name for name in names)
    if launcher_is_remote(cluster_config):
        from nemo_skills.pipeline.utils.cluster import get_tunnel

        command = "rm -rf -- " + " ".join(shlex.quote(str(target)) for target in targets)
        get_tunnel(cluster_config).run(command, hide=True)
        return
    for target in targets:
        shutil.rmtree(target, ignore_errors=True)


def _submit_task(
    task: TaskPlan,
    *,
    cluster: str,
    dependency_names: tuple[str, ...],
) -> None:
    _submit_command(task, cluster=cluster, dependency_names=dependency_names)


def _submit_pipeline(
    tasks: tuple[TaskPlan, ...],
    *,
    cluster: str,
    task_experiments: dict[str, str],
    excluded_jobs: set[str],
) -> None:
    from nemo_skills.pipeline.utils.declarative import (
        Command,
        CommandGroup,
        HardwareConfig,
        Pipeline,
    )

    from nvflow.integrations.nemo_gym.job_scripts import BashScript
    from nvflow.integrations.nemo_skills import load_target_config

    job_by_task: dict[str, dict[str, Any]] = {}
    jobs = []
    external_dependencies: list[str] = []
    for task in tasks:
        resources = {resource.role: resource for resource in task.resources}
        commands_by_group: dict[str, list[Any]] = {}
        for command in task.commands:
            resource = resources.get(command.group)
            commands_by_group.setdefault(command.group, []).append(
                Command(
                    script=BashScript(
                        cmd=_slurm_command(command.command),
                        span_group_nodes=(
                            command.kind == "service"
                            and resource is not None
                            and resource.nodes > 1
                        ),
                    ),
                    container=command.container,
                    name=command.role,
                )
            )
        log_dirs = task.parameters.get("group_log_dirs") or {}
        groups = []
        for group_name, commands in commands_by_group.items():
            resource = resources.get(group_name)
            if resource is None:
                raise ValueError(
                    f"Slurm task {task.task_id!r} has no resources for group {group_name!r}"
                )
            sbatch_kwargs = {}
            if resource.exclusive is not None:
                sbatch_kwargs["exclusive"] = resource.exclusive
            condition = _one_dependency_condition(task.dependencies)
            if condition != "after_success":
                sbatch_kwargs["dependency_type"] = {
                    "after_start": "after",
                    "after_ready": "after",
                    "after_any": "afterany",
                }[condition]
            groups.append(
                CommandGroup(
                    commands=commands,
                    hardware=HardwareConfig(
                        partition=resource.partition,
                        account=resource.account,
                        num_gpus=resource.gpus_per_node,
                        num_nodes=resource.nodes,
                        num_tasks=resource.tasks_per_node,
                        sbatch_kwargs=sbatch_kwargs or None,
                    ),
                    name=group_name,
                    log_dir=log_dirs.get(group_name),
                )
            )
        dependencies = []
        for dependency in task.dependencies:
            if dependency.external:
                external_dependencies.append(dependency.upstream)
            elif dependency.upstream not in job_by_task:
                external_dependencies.append(task_experiments[dependency.upstream])
            else:
                dependencies.append(job_by_task[dependency.upstream])
        key = "group" if len(groups) == 1 else "groups"
        job = {
            "name": task.name,
            key: groups[0] if len(groups) == 1 else groups,
            "dependencies": dependencies or None,
        }
        jobs.append(job)
        job_by_task[task.task_id] = job

    experiment = Pipeline(
        name=_experiment_name(tasks[0]),
        cluster_config=load_target_config(cluster),
        jobs=jobs,
        run_after=list(dict.fromkeys(external_dependencies)) or None,
    ).run()
    completion = set(completion_task_ids(tasks))
    superseded = [index for index, task in enumerate(tasks) if task.task_id not in completion]
    if not superseded:
        return
    # NeMo-Skills names every job of a pipeline alike and adds them in job order.
    launched = [_job_handle(job) for job in getattr(experiment, "jobs", None) or ()]
    if len(launched) != len(tasks):
        from nvflow.core import console

        console.warning(
            f"Could not match the jobs of {_experiment_name(tasks[0])!r} to its tasks; "
            "later stages wait on every one of its jobs, retry attempts included."
        )
        return
    excluded_jobs.update(launched[index] for index in superseded if launched[index])


def _submit_nemo_rl(
    tasks: tuple[TaskPlan, ...],
    *,
    cluster: str,
    task_experiments: dict[str, str],
    excluded_jobs: set[str],
) -> None:
    from nemo_skills.pipeline.nemo_rl.grpo import parse_kwargs
    from nemo_skills.pipeline.utils.cluster import get_slurm_timeout_str, get_timeout_str
    from nemo_skills.pipeline.utils.exp import add_task, get_exp, run_exp

    from nvflow.execution.runtime import CHECKPOINT_SAVE_BY_ENV
    from nvflow.integrations.nemo_skills import load_target_config
    from nvflow.integrations.nemo_skills.slurm import dependency_sbatch_kwargs

    cluster_config = load_target_config(cluster)
    handles: dict[str, Any] = {}
    cleanup_tasks = [task for task in tasks if task.operation == "cleanup"]
    experiment = _experiment_name(tasks[0])
    with get_exp(experiment, cluster_config) as exp:
        for task in tasks:
            if task.operation == "cleanup":
                continue
            if len(task.commands) != 1 or len(task.resources) != 1:
                raise ValueError(
                    f"Direct Slurm NeMo-RL task {task.task_id!r} requires one "
                    "command and one resource group"
                )
            command = task.commands[0]
            resource = task.resources[0]
            internal_dependencies = [
                handles[dependency.upstream]
                for dependency in task.dependencies
                if not dependency.external and dependency.upstream in handles
            ]
            run_after = [
                task_experiments[dependency.upstream]
                for dependency in task.dependencies
                if not dependency.external and dependency.upstream not in handles
            ]
            run_after.extend(
                dependency.upstream for dependency in task.dependencies if dependency.external
            )
            raw_sbatch = (task.parameters.get("slurm") or {}).get(
                "sbatch_kwargs",
                "",
            )
            sbatch_kwargs = parse_kwargs(raw_sbatch)
            condition = _one_dependency_condition(task.dependencies)
            sbatch_kwargs = dict(
                dependency_sbatch_kwargs(
                    sbatch_kwargs,
                    continue_after_failure=condition == "after_any",
                )
                or {}
            )
            if condition == "after_start":
                sbatch_kwargs["dependency_type"] = "after"
            environment = {}
            if task.operation in {"nemo_rl.sft", "nemo_rl.grpo"}:
                environment[CHECKPOINT_SAVE_BY_ENV] = get_timeout_str(
                    cluster_config,
                    resource.partition,
                )
            elif task.operation == "service":
                sbatch_kwargs.setdefault(
                    "time",
                    get_slurm_timeout_str(
                        cluster_config,
                        resource.partition,
                        with_save_delay=False,
                    ),
                )
            with_ray = bool(task.parameters.get("ray_application"))
            handles[task.task_id] = add_task(
                exp,
                cmd=_slurm_command(command.command, environment),
                task_name=task.name,
                log_dir=str(task.parameters["slurm"]["log_dir"]),
                container=cluster_config["containers"].get(
                    command.container,
                    command.container,
                ),
                num_gpus=resource.gpus_per_node,
                num_nodes=resource.nodes,
                cluster_config=cluster_config,
                with_ray=with_ray,
                ray_template=(cluster_config.get("ray_template") if with_ray else None),
                sbatch_kwargs=sbatch_kwargs,
                installation_command=(
                    (task.parameters.get("slurm") or {}).get("installation_command")
                    or task.parameters.get("installation_command")
                ),
                partition=resource.partition,
                run_after=list(dict.fromkeys(run_after)) or None,
                task_dependencies=internal_dependencies or None,
            )
        run_exp(exp, cluster_config, sequential=False)
        launched = _launched_job_handles(exp)
        completion = set(completion_task_ids(tasks))
        for task_id, job_id in handles.items():
            if task_id not in completion and launched.get(str(job_id)):
                excluded_jobs.add(launched[str(job_id)])
        for cleanup in cleanup_tasks:
            _submit_host_cleanup(
                cleanup,
                {task_id: launched.get(str(job_id), "") for task_id, job_id in handles.items()},
                cluster_config,
            )


def _job_handle(job: Any) -> str:
    return str(getattr(job, "handle", "") or next(iter(getattr(job, "handles", None) or ()), ""))


def _launched_job_handles(exp: Any) -> dict[str, str]:
    return {str(job.id): _job_handle(job) for job in getattr(exp, "jobs", None) or ()}


def _submit_host_cleanup(
    task: TaskPlan,
    handles: dict[str, str],
    cluster_config: dict[str, Any],
) -> None:
    """Best-effort cancellation of a service after its consumer ends."""
    from nvflow.core import console

    dependencies = [dependency for dependency in task.dependencies if not dependency.external]
    if len(dependencies) != 1 or dependencies[0].condition != "after_any":
        raise ValueError(f"Host cleanup task {task.task_id!r} requires one after_any dependency")
    handle = handles.get(dependencies[0].upstream, "")
    if not handle:
        return
    slurm_job_id = _slurm_job_id_from_handle(handle)
    if slurm_job_id is None:
        console.warning(
            f"Could not extract Slurm job ID from handle '{handle}', skipping {task.name}"
        )
        return
    target_task_id = str(task.parameters.get("slurm_cleanup_target_task", ""))
    target_handle = handles.get(target_task_id, "") if target_task_id else ""
    target_job_id = _slurm_job_id_from_handle(target_handle) if target_handle else None
    if target_task_id and target_job_id is None:
        console.warning(
            f"Could not extract cleanup target Slurm job ID from handle "
            f"'{target_handle}', skipping {task.name}"
        )
        return
    try:
        _sbatch_host_cleanup(
            task,
            slurm_job_id,
            cluster_config,
            target_job_id=target_job_id,
        )
    except Exception as exc:
        console.warning(f"Could not submit {task.name}: {exc}")


def _slurm_job_id_from_handle(handle: str) -> str | None:
    try:
        path = handle.partition("://")[2]
        slurm_job_id = path.split("/")[1]
        int(slurm_job_id)
    except (ValueError, IndexError):
        return None
    return slurm_job_id


def _sbatch_host_cleanup(
    task: TaskPlan,
    slurm_job_id: str,
    cluster_config: dict[str, Any],
    *,
    target_job_id: str | None = None,
) -> None:
    from nemo_skills.pipeline.utils import get_unmounted_path

    from nvflow.integrations.nemo_skills import launcher_is_remote
    from nvflow.integrations.nemo_skills.slurm import active_job_comment

    target = str(task.parameters["slurm_cleanup_target"])
    prefix = str(cluster_config.get("job_name_prefix", ""))
    account = str(cluster_config.get("account", ""))
    partition = str(cluster_config.get("cpu_partition") or cluster_config.get("partition", "batch"))
    log_dir = get_unmounted_path(cluster_config, str(task.parameters["slurm"]["log_dir"]))
    directives = [
        "#!/bin/bash",
        f"#SBATCH --job-name={prefix}{task.name}",
        f"#SBATCH --partition={partition}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --gpus-per-node=0",
        "#SBATCH --time=00:05:00",
        f"#SBATCH --output={log_dir}/judge-cleanup-%j.log",
        f"#SBATCH --error={log_dir}/judge-cleanup-%j.log",
        f"#SBATCH --dependency=afterany:{slurm_job_id}",
    ]
    if account:
        directives.insert(2, f"#SBATCH --account={account}")
    if comment := active_job_comment():
        directives.append(f"#SBATCH --comment={shlex.quote(comment)}")
    cancel_command = (
        f"scancel {target_job_id} 2>/dev/null || true"
        if target_job_id is not None
        else f"scancel --name={prefix}{target} --user=$USER 2>/dev/null || true"
    )
    script = "\n".join(
        [
            *directives,
            cancel_command,
            "",
        ]
    )
    if launcher_is_remote(cluster_config):
        from nemo_skills.pipeline.utils.cluster import get_tunnel

        result = get_tunnel(cluster_config).run(
            "sbatch <<'NVFLOW_SBATCH_EOF'\n" + script + "NVFLOW_SBATCH_EOF\n",
            hide=True,
            warn=True,
        )
        return_code = getattr(
            result,
            "exited",
            getattr(result, "return_code", 1),
        )
        stderr = str(getattr(result, "stderr", ""))
    else:
        result = subprocess.run(
            ["sbatch"],
            input=script,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return_code = result.returncode
        stderr = result.stderr
    if return_code != 0:
        raise RuntimeError(f"Failed to submit cleanup task {task.task_id!r}: {stderr.strip()}")


def _submit_command(
    task: TaskPlan,
    *,
    cluster: str,
    dependency_names: tuple[str, ...],
) -> None:
    from nvflow.integrations.nemo_skills.commands import run_cmd, wrap_arguments

    if any(command.runtime_template for command in task.commands):
        raise ValueError(f"Slurm task {task.task_id!r} still contains runtime templates")
    containers = {command.container for command in task.commands}
    if len(containers) != 1:
        raise ValueError(
            f"Slurm task {task.task_id!r} requires multiple runtime containers: "
            + ", ".join(sorted(containers))
        )
    command_text = _slurm_command("\n".join(command.command for command in task.commands))
    parameters = _parameters(task, cluster=cluster, dependency_names=dependency_names)
    parameters.pop("arguments", None)
    parameters.pop("command", None)
    parameters.setdefault("command", command_text)
    parameters.setdefault("container", next(iter(containers)))
    _apply_resources(parameters, task.resources)
    run_cmd(ctx=wrap_arguments(""), **parameters)


def _parameters(
    task: TaskPlan,
    *,
    cluster: str,
    dependency_names: tuple[str, ...],
) -> dict[str, Any]:
    parameters = _render_value(dict(task.parameters), "/nemo_run/code")
    parameters.pop("positional", None)
    parameters.pop("slurm_cleanup_experiments", None)
    parameters.pop("generation_invocation", None)
    stage_kwargs = parameters.pop("stage_kwargs", None)
    if isinstance(stage_kwargs, dict):
        parameters.update(stage_kwargs)
    parameters["cluster"] = cluster
    parameters.pop("experiment_name", None)
    parameters["expname"] = _experiment_name(task)
    external = [dependency.upstream for dependency in task.dependencies if dependency.external]
    run_after = [*dependency_names, *external]
    parameters["run_after"] = run_after or None
    condition = _one_dependency_condition(task.dependencies)
    if condition != "after_success":
        sbatch_kwargs = dict(parameters.get("sbatch_kwargs") or {})
        sbatch_kwargs["dependency_type"] = {
            "after_start": "after",
            "after_ready": "after",
            "after_any": "afterany",
        }[condition]
        parameters["sbatch_kwargs"] = sbatch_kwargs
    return parameters


def _apply_resources(
    parameters: dict[str, Any],
    resources: tuple[ResourceRequest, ...],
) -> None:
    if not resources:
        return
    if len(resources) != 1:
        raise ValueError("Direct run_cmd lowering currently requires one resource group per task")
    resource = resources[0]
    parameters.setdefault("num_nodes", resource.nodes)
    if resource.gpus_per_node is not None and parameters.get("num_gpus") is None:
        parameters["num_gpus"] = resource.gpus_per_node
    parameters.setdefault("num_tasks", resource.tasks_per_node)
    if resource.partition is not None:
        parameters.setdefault("partition", resource.partition)
    if resource.account is not None:
        parameters.setdefault("account", resource.account)
    if resource.exclusive is not None:
        sbatch_kwargs = dict(parameters.get("sbatch_kwargs") or {})
        sbatch_kwargs.setdefault("exclusive", resource.exclusive)
        parameters["sbatch_kwargs"] = sbatch_kwargs


def _one_dependency_condition(
    dependencies: tuple[DependencyPlan, ...],
) -> str:
    conditions = {dependency.condition for dependency in dependencies}
    if len(conditions) > 1:
        raise ValueError(
            "Direct Slurm task dependencies require one condition, got "
            + ", ".join(sorted(conditions))
        )
    return next(iter(conditions), "after_success")


def _pipeline_groups(tasks: tuple[TaskPlan, ...]) -> dict[str, tuple[TaskPlan, ...]]:
    grouped: dict[str, list[TaskPlan]] = {}
    for task in tasks:
        if _requires_pipeline(task):
            grouped.setdefault(_experiment_name(task), []).append(task)
    return {name: tuple(group) for name, group in grouped.items()}


def _nemo_rl_groups(tasks: tuple[TaskPlan, ...]) -> dict[str, tuple[TaskPlan, ...]]:
    grouped: dict[str, list[TaskPlan]] = {}
    for task in tasks:
        if _is_nemo_rl_task(task):
            grouped.setdefault(_experiment_name(task), []).append(task)
    return {name: tuple(group) for name, group in grouped.items()}


def _generation_invocation(task: TaskPlan) -> str | None:
    """Return the NeMo-Skills ``generate`` call that submits *task*.

    A chunked call's planned merge task is submitted on its own after the
    chunks: NeMo-Skills merges inside the chunk jobs and exits 0 when a chunk
    is missing, so only the merge task's checks fail such a step.
    """
    if task.operation != "nemo_skills.generate":
        return None
    invocation = task.parameters.get("generation_invocation")
    return None if invocation is None else str(invocation)


def _generation_groups(tasks: tuple[TaskPlan, ...]) -> dict[str, tuple[TaskPlan, ...]]:
    grouped: dict[str, list[TaskPlan]] = {}
    for task in tasks:
        invocation = _generation_invocation(task)
        if invocation is not None:
            grouped.setdefault(invocation, []).append(task)
    return {name: tuple(group) for name, group in grouped.items()}


_GENERATION_PLAN_KEYS = frozenset(
    {"arguments", "experiment_name", "generation_invocation", "group_log_dirs", "slurm"}
)


def _submit_generation(
    tasks: tuple[TaskPlan, ...],
    *,
    cluster: str,
    task_experiments: dict[str, str],
) -> None:
    """Submit one captured generation invocation through NeMo-Skills ``generate``."""
    from nvflow.integrations.nemo_skills.commands import generate, wrap_arguments

    source = tasks[0]
    members = {task.task_id for task in tasks}
    parameters = _render_value(dict(source.parameters), "/nemo_run/code")
    arguments = str(parameters.get("arguments") or "")
    scheduler = parameters.get("slurm") or {}
    for key in _GENERATION_PLAN_KEYS:
        parameters.pop(key, None)
    if scheduler.get("installation_command"):
        parameters["installation_command"] = scheduler["installation_command"]
    run_after = [
        dependency.upstream if dependency.external else task_experiments[dependency.upstream]
        for task in tasks
        for dependency in task.dependencies
        if dependency.external or dependency.upstream not in members
    ]
    parameters["cluster"] = cluster
    parameters["run_after"] = list(dict.fromkeys(run_after)) or None
    generate(ctx=wrap_arguments(arguments), **parameters)


def _experiment_name(task: TaskPlan) -> str:
    if _is_generation_merge(task):
        # NeMo-Run ids are the title plus the second, so the merge job cannot reuse
        # the title of the chunk jobs it follows.
        return task.name
    return str(
        task.parameters.get("experiment_name") or task.parameters.get("expname") or task.name
    )


def _is_generation_merge(task: TaskPlan) -> bool:
    return task.operation == "nemo_skills.run_cmd" and "generation_invocation" in task.parameters


def _receipt_handles(
    handles: dict[str, tuple[str, ...]],
    tasks: tuple[TaskPlan, ...],
) -> dict[str, tuple[str, ...]]:
    """File each merge job's handles under the planned experiment of its task."""
    for task in tasks:
        if _is_generation_merge(task) and task.name in handles:
            planned = str(task.parameters["experiment_name"])
            handles[planned] = (*handles.get(planned, ()), *handles.pop(task.name))
    return handles


def _requires_pipeline(task: TaskPlan) -> bool:
    return (
        task.operation == "nemo_skills.pipeline"
        or len(task.resources) > 1
        or len({command.container for command in task.commands}) > 1
    )


def _is_nemo_rl_task(task: TaskPlan) -> bool:
    return task.operation in {"nemo_rl.sft", "nemo_rl.grpo"} or bool(
        task.parameters.get("nemo_rl_group")
    )


def _slurm_command(command: str, environment: dict[str, str] | None = None) -> str:
    exports = "".join(
        f"export {name}={shlex.quote(value)}\n" for name, value in (environment or {}).items()
    )
    return (
        'export NVFLOW_CODE_ROOT="${NVFLOW_CODE_ROOT:-/nemo_run/code}"\n'
        + exports
        + render_code_root(command, "/nemo_run/code")
    )


def _render_value(value: Any, code_root: str) -> Any:
    if isinstance(value, str):
        return render_code_root(value, code_root)
    if isinstance(value, dict):
        return {key: _render_value(nested, code_root) for key, nested in value.items()}
    if isinstance(value, list):
        return [_render_value(nested, code_root) for nested in value]
    if isinstance(value, tuple):
        return tuple(_render_value(nested, code_root) for nested in value)
    return value


def _topological_tasks(graph: ExecutionGraph) -> tuple[TaskPlan, ...]:
    tasks = {task.task_id: task for task in graph.tasks}
    pending = dict(tasks)
    ordered: list[TaskPlan] = []
    while pending:
        ready = [
            task
            for task in pending.values()
            if all(
                dependency.external or dependency.upstream not in pending
                for dependency in task.dependencies
            )
        ]
        if not ready:
            raise ValueError("Execution graph contains an unresolved dependency cycle")
        ready.sort(key=lambda task: tuple(tasks).index(task.task_id))
        for task in ready:
            ordered.append(task)
            pending.pop(task.task_id)
    return tuple(ordered)


_GROUPABLE_PARAMETERS = frozenset(
    {
        SLURM_GROUP_JOBS,
        "arguments",
        "cluster",
        "container",
        "experiment_name",
        "expname",
        "group_log_dirs",
        "job_index",
        "log_dir",
        "logical_role",
        "managed_model_paths",
        "output_dir",
        "run_after",
        "sbatch_kwargs",
        "slurm_cleanup_experiments",
    }
)


def _groupable(task: TaskPlan) -> bool:
    return (
        task.parameters.get(SLURM_GROUP_JOBS) is True
        and task.operation in {"nemo_skills.run_cmd", "nemo_skills.pipeline"}
        and len(task.commands) == 1
        and task.commands[0].kind == "command"
        and not task.commands[0].runtime_template
        and not task.services
        and len(task.resources) == 1
        and not task.parameters.get("sbatch_kwargs")
        and set(task.parameters) <= _GROUPABLE_PARAMETERS
        and all(dependency.condition == "after_success" for dependency in task.dependencies)
    )


def _hardware(task: TaskPlan) -> tuple[object, ...]:
    resource = task.resources[0]
    return (
        resource.nodes,
        resource.gpus_per_node or 0,
        resource.tasks_per_node,
        resource.partition,
        resource.account,
        resource.exclusive,
    )


def _shares_job(previous: TaskPlan, task: TaskPlan) -> bool:
    return (
        _groupable(previous)
        and _groupable(task)
        and previous.stage == task.stage
        and previous.commands[0].container == task.commands[0].container
        and _hardware(previous) == _hardware(task)
    )


def _job_groups(tasks: tuple[TaskPlan, ...]) -> list[list[TaskPlan]]:
    """Split ordered tasks into the chains that each run as one Slurm job.

    A substage task joins its upstream's job when both belong to the same stage,
    it is that task's only dependent and depends on nothing else, and both run
    one command in the same container on the same hardware.
    """
    dependents = Counter(dependency.upstream for task in tasks for dependency in task.dependencies)
    chain_of: dict[str, list[TaskPlan]] = {}
    groups: list[list[TaskPlan]] = []
    for task in tasks:
        upstream = task.dependencies[0] if len(task.dependencies) == 1 else None
        chain = (
            chain_of.get(upstream.upstream)
            if upstream is not None and not upstream.external and dependents[upstream.upstream] == 1
            else None
        )
        if chain is not None and _shares_job(chain[-1], task):
            chain.append(task)
        else:
            chain = [task]
            groups.append(chain)
        chain_of[task.task_id] = chain
    return groups


def _grouped_task(group: list[TaskPlan]) -> TaskPlan:
    """Return one task running *group*'s commands in order.

    Each command runs in its own subshell, so its ``exit 0`` or exports cannot
    reach the next one; the first failure stops the job. The subshell's status
    is checked on the next line: as the left side of ``||``, bash would ignore
    the command's own ``set -e``.
    """
    if len(group) == 1:
        [task] = group
        parameters = {
            key: value for key, value in task.parameters.items() if key != SLURM_GROUP_JOBS
        }
        return replace(task, parameters=parameters)
    first, last = group[0], group[-1]
    script = "\n".join(
        f"echo {shlex.quote(f'NVFlow stage {task.stage}: {task.task_id}')}\n"
        f"(\n{task.commands[0].command}\n)\n"
        'status=$?; [ "$status" -eq 0 ] || exit "$status"'
        for task in group
    )
    return replace(
        last,
        operation="nemo_skills.run_cmd",
        commands=(
            replace(
                first.commands[0],
                command=script,
                role="main",
                group="main",
                display_command=None,
            ),
        ),
        resources=(replace(first.resources[0], role="main"),),
        dependencies=first.dependencies,
        outputs=tuple(output for task in group for output in task.outputs),
        services=(),
        parameters={
            "expname": _experiment_name(last),
            "log_dir": _group_log_dir(group),
            "slurm_cleanup_experiments": tuple(
                dict.fromkeys(
                    name
                    for task in group
                    for name in task.parameters.get("slurm_cleanup_experiments", ())
                )
            ),
        },
    )


def _group_log_dir(group: list[TaskPlan]) -> str | None:
    """Return the ``logs`` folder under the tasks' shared output folder."""
    log_dirs = []
    for task in group:
        log_dir = task.parameters.get("log_dir") or next(
            iter((task.parameters.get("group_log_dirs") or {}).values()), None
        )
        if log_dir:
            log_dirs.append(str(log_dir))
    if not log_dirs:
        return None
    common = os.path.commonpath(log_dirs)
    if common in log_dirs:
        return common
    return log_dirs[0] if common == "/" else f"{common}/logs"


def _grouped_handles(
    handles: dict[str, tuple[str, ...]],
    groups: list[list[TaskPlan]],
) -> dict[str, tuple[str, ...]]:
    """File each grouped job's handles under the experiment of every task it ran."""
    for group in groups:
        for task in group[:-1]:
            handles.setdefault(_experiment_name(task), handles.get(_experiment_name(group[-1]), ()))
    return handles


__all__ = ["DirectSlurmEngine"]
