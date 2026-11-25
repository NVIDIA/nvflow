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
"""Direct-Slurm engine regression tests."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from nvflow.execution import (
    CommandPlan,
    DependencyPlan,
    ResolvedTarget,
    ResourceRequest,
    TargetBinding,
    TaskPlan,
)
from nvflow.execution.contracts import SLURM_GROUP_JOBS, ExecutionGraph, StageExecution
from nvflow.execution.engines.base import ActiveJob
from nvflow.execution.engines.slurm import (
    DirectSlurmEngine,
    _apply_resources,
    _clean_stale_experiments,
    _gpus_per_node,
    _grouped_handles,
    _grouped_task,
    _job_groups,
    _submission_order,
    _submit_task,
)


def _job(
    task_id: str,
    container: str = "nemo-gym",
    *,
    after: tuple[str, ...] = (),
    gpus: int = 0,
    command: str = "true",
    grouped: bool = True,
    stage: str = "checks",
) -> TaskPlan:
    name = task_id.split(".")[0]
    return TaskPlan(
        task_id=task_id,
        stage=stage,
        name=task_id,
        operation="nemo_skills.run_cmd",
        commands=(CommandPlan(command, container, runtime_role=container),),
        resources=(ResourceRequest(gpus_per_node=gpus),),
        dependencies=tuple(DependencyPlan(upstream) for upstream in after),
        parameters={
            "expname": f"run-{name}",
            "log_dir": f"/out/checks/{name}/logs",
            SLURM_GROUP_JOBS: grouped,
        },
    )


def test_a_chain_in_one_container_runs_as_one_job() -> None:
    tasks = (
        _job("render.0"),
        _job("collect.0", after=("render.0",)),
        _job("split.0", after=("collect.0",)),
    )

    [group] = _job_groups(tasks)
    job = _grouped_task(group)

    assert (job.task_id, job.stage, job.dependencies) == ("split.0", "checks", ())
    assert job.parameters == {
        "expname": "run-split",
        "log_dir": "/out/checks/logs",
        "slurm_cleanup_experiments": (),
    }


def test_a_grouped_jobs_handle_is_filed_under_every_stage_it_ran() -> None:
    groups = _job_groups((_job("render.0"), _job("split.0", after=("render.0",))))

    handles = _grouped_handles({"run-split": ("slurm://41",)}, groups)

    assert handles == {"run-split": ("slurm://41",), "run-render": ("slurm://41",)}


def test_jobs_split_at_image_hardware_and_stage_changes_fan_out_and_unmarked_tasks() -> None:
    tasks = (
        _job("clean.0", "nemo-skills"),
        _job("embed.0", "vllm", after=("clean.0",), gpus=1),
        _job("judge.0", after=("embed.0",)),
        _job("left.0", after=("judge.0",)),
        _job("right.0", after=("judge.0",)),
        _job("manual.0", after=("right.0",), grouped=False),
        _job("report.0", after=("left.0",), stage="report"),
    )

    assert [[task.task_id for task in group] for group in _job_groups(tasks)] == [
        ["clean.0"],
        ["embed.0"],
        ["judge.0"],
        ["left.0"],
        ["right.0"],
        ["manual.0"],
        ["report.0"],
    ]


def test_the_engine_counts_the_jobs_each_stage_submits() -> None:
    implementation = {
        "kind": "project",
        "name": "my_project.stage:ChecksStage",
        "contract_version": None,
        "provider": "my_project",
        "platform": {"distribution": "nvflow", "version": "1.2.0", "source": "git:platform"},
    }
    graph = ExecutionGraph(
        execution_version=1,
        project="starter",
        workflow="checks",
        target="cluster",
        engine="slurm",
        target_fingerprint="sha256:target",
        stages=(
            StageExecution(
                stage="checks",
                tasks=(
                    _job("render.0"),
                    _job("split.0", after=("render.0",)),
                    _job("embed.0", "vllm", after=("split.0",), gpus=1),
                ),
                implementation=implementation,
            ),
            StageExecution(
                stage="report",
                tasks=(_job("report.0", after=("embed.0",), stage="report"),),
                implementation=implementation,
            ),
        ),
    )

    assert DirectSlurmEngine().jobs_per_stage(graph) == {"checks": 2, "report": 1}


def test_a_grouped_job_runs_each_command_in_its_own_shell_until_one_fails() -> None:
    commands = (
        "echo one",
        "if true; then echo two; exit 0; fi; echo skipped",
        "export SEEN=1; echo three",
        'echo "four${SEEN:-}"; set -e; false; echo kept-going',
        "echo five",
    )
    tasks = tuple(
        _job(f"step{index}.0", command=command, after=(f"step{index - 1}.0",) if index else ())
        for index, command in enumerate(commands)
    )
    [group] = _job_groups(tasks)

    result = subprocess.run(
        ["bash", "-c", _grouped_task(group).commands[0].command],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    output = [line for line in result.stdout.splitlines() if not line.startswith("NVFlow stage")]
    assert output == ["one", "two", "three", "four"]


def test_planned_gpu_count_replaces_an_explicit_none() -> None:
    parameters: dict[str, object] = {"num_gpus": None}

    _apply_resources(parameters, (ResourceRequest(gpus_per_node=0),))

    assert parameters["num_gpus"] == 0


@pytest.mark.requires_nemo
def test_slurm_status_reads_the_users_queue_through_the_target_tunnel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[str] = []
    label = "nvflow,workflow=demo:eval/small,stage=model-a,experiment=eval-model-a-bench"
    rows = (
        # In squeue's own order: one labeled job, two components of a heterogeneous job of
        # which only the first reports the label, and an unlabeled job whose name contains "|".
        ("20302316+1", "PENDING", "0:00", "batch", "1", "1", "gres/gpu:h100:4", "acct", "76449")
        + ("N/A", "(None)", "job-0-judge-1", "(null)"),
        ("19352777", "PENDING", "0:00", "cpu", "1", "4", "N/A", "acct", "1")
        + ("N/A", "(Dependency)", "prepare|data", "(null)"),
        ("20302316+0", "PENDING", "0:00", "batch", "1", "1", "gres/gpu:h100:4", "acct", "76449")
        + ("N/A", "(None)", "job-0-policy-0", label),
        ("19352776", "RUNNING", "4:03", "batch", "2", "16", "gres/gpu:8", "acct", "76452")
        + ("2026-10-15T16:00:00", "node[01-02]", "answer_generation-rs0-chunk0", label),
    )

    class Tunnel:
        def run(self, command: str, hide: bool):
            commands.append(command)
            return SimpleNamespace(stdout="".join("|".join(row) + "\n" for row in rows))

    monkeypatch.setattr(
        "nvflow.integrations.nemo_skills.load_target_config",
        lambda _name: {"ssh_tunnel": {"host": "login"}},
    )
    monkeypatch.setattr("nvflow.integrations.nemo_skills.launcher_is_remote", lambda _config: True)
    monkeypatch.setattr("nemo_skills.pipeline.utils.cluster.get_tunnel", lambda _config: Tunnel())
    target = ResolvedTarget(
        binding=TargetBinding.resolve("cluster"),
        engine="slurm",
        fingerprint="sha256:target",
        source_path=tmp_path / "cluster.yaml",
    )

    listing = DirectSlurmEngine().active_jobs(target)

    assert commands == ["squeue --me --noheader '--format=%i|%T|%M|%P|%D|%C|%b|%a|%Q|%S|%R|%j|%k'"]
    labeled: dict[str, Any] = {
        "workflow": "demo:eval/small",
        "stage": "model-a",
        "experiment": "eval-model-a-bench",
    }
    pending: dict[str, Any] = {
        "reason": "None",
        "nodes": 1,
        "gpus_per_node": 4,
        "cpus": 1,
        "account": "acct",
        "priority": 76449,
    }
    assert listing.jobs == (
        ActiveJob(
            "19352776",
            "answer_generation-rs0-chunk0",
            "RUNNING",
            "4:03",
            "batch",
            **labeled,
            reason="node[01-02]",
            nodes=2,
            gpus_per_node=8,
            cpus=16,
            start="2026-10-15T16:00:00",
            account="acct",
            priority=76452,
        ),
        ActiveJob(
            "19352777",
            "prepare|data",
            "PENDING",
            "0:00",
            "cpu",
            reason="Dependency",
            nodes=1,
            cpus=4,
            account="acct",
            priority=1,
        ),
        ActiveJob("20302316+0", "job-0-policy-0", "PENDING", "0:00", "batch", **labeled, **pending),
        ActiveJob("20302316+1", "job-0-judge-1", "PENDING", "0:00", "batch", **labeled, **pending),
    )
    assert listing.unreachable == {}


def test_slurm_status_lists_jobs_in_submission_order() -> None:
    job_ids = ["123_[5-10]", "1000+1", "99", "123_4", "1000+0", "1000"]

    assert sorted(job_ids, key=_submission_order) == [
        "99",
        "123_4",
        "123_[5-10]",
        "1000",
        "1000+0",
        "1000+1",
    ]


@pytest.mark.parametrize(
    ("tres", "gpus"),
    [
        ("gres/gpu:4", 4),
        ("gres:gpu:4", 4),
        ("gres/gpu:h100:8", 8),
        ("gres/gpu:3090:2", 2),
        ("gres/gpu=8", 8),
        ("gres/gpu:4(IDX:0-3)", 4),
        ("gres/shard:2,gres/gpu:4", 4),
        ("gres/gpumem:40G", None),
        ("gres/gpu:0", 0),
        ("N/A", None),
        ("(null)", None),
    ],
)
def test_slurm_status_reads_gpus_per_node_from_every_tres_form(tres: str, gpus: int | None) -> None:
    assert _gpus_per_node(tres) == gpus


def _task(*names: str) -> TaskPlan:
    return TaskPlan(
        task_id="stage.main",
        stage="stage",
        name="stage",
        operation="test",
        parameters={"slurm_cleanup_experiments": names},
    )


def test_declared_stale_experiments_are_removed_locally(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "experiments"
    stale = root / "stale"
    retained = root / "retained"
    stale.mkdir(parents=True)
    retained.mkdir()
    (stale / "cached-script.sh").write_text("old", encoding="utf-8")

    monkeypatch.setattr(
        "nvflow.integrations.nemo_skills.load_target_config",
        lambda _cluster: {"job_dir": str(tmp_path)},
    )
    monkeypatch.setattr(
        "nvflow.integrations.nemo_skills.launcher_is_remote",
        lambda _config: False,
    )

    _clean_stale_experiments((_task("stale", "stale"),), cluster="test")

    assert not stale.exists()
    assert retained.is_dir()


def test_cleanup_rejects_path_like_experiment_names() -> None:
    with pytest.raises(ValueError, match="safe basenames"):
        _clean_stale_experiments((_task("../other-run"),), cluster="test")


@pytest.mark.requires_nemo
def test_declared_stale_experiments_use_remote_tunnel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[str] = []

    class Tunnel:
        def run(self, command: str, *, hide: bool) -> None:
            assert hide is True
            commands.append(command)

    monkeypatch.setattr(
        "nvflow.integrations.nemo_skills.load_target_config",
        lambda _cluster: {
            "ssh_tunnel": {
                "job_dir": "/remote/jobs",
            }
        },
    )
    monkeypatch.setattr(
        "nvflow.integrations.nemo_skills.launcher_is_remote",
        lambda _config: True,
    )
    monkeypatch.setattr(
        "nemo_skills.pipeline.utils.cluster.get_tunnel",
        lambda _config: Tunnel(),
    )

    _clean_stale_experiments((_task("stage-one", "stage-two"),), cluster="test")

    assert commands == [
        "rm -rf -- /remote/jobs/experiments/stage-one /remote/jobs/experiments/stage-two"
    ]


def test_a_substage_task_alone_in_its_job_submits_without_the_grouping_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "nvflow.integrations.nemo_skills.commands.run_cmd", lambda **kwargs: calls.append(kwargs)
    )
    monkeypatch.setattr(
        "nvflow.integrations.nemo_skills.commands.wrap_arguments",
        lambda value: value,
    )
    [group] = _job_groups((_job("embed.0", "vllm", gpus=1),))

    _submit_task(_grouped_task(group), cluster="test", dependency_names=())

    [call] = calls
    assert SLURM_GROUP_JOBS not in call
    assert (call["container"], call["num_gpus"]) == ("vllm", 1)


def test_cleanup_declaration_is_not_forwarded_to_run_cmd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def run_cmd(
        *,
        ctx: str,
        command: str,
        cluster: str,
        expname: str,
        run_after: None,
        container: str,
        log_dir: str,
    ) -> None:
        calls.append(
            {
                "ctx": ctx,
                "command": command,
                "cluster": cluster,
                "expname": expname,
                "run_after": run_after,
                "container": container,
                "log_dir": log_dir,
            }
        )

    monkeypatch.setattr("nvflow.integrations.nemo_skills.commands.run_cmd", run_cmd)
    monkeypatch.setattr(
        "nvflow.integrations.nemo_skills.commands.wrap_arguments",
        lambda value: value,
    )
    task = TaskPlan(
        task_id="stage.main",
        stage="stage",
        name="stage",
        operation="nemo_skills.run_cmd",
        commands=(
            CommandPlan(
                "echo ready",
                "nemo-skills",
                runtime_role="nemo-skills",
            ),
        ),
        parameters={
            "log_dir": "/logs",
            "slurm_cleanup_experiments": ("stage",),
        },
    )

    _submit_task(task, cluster="test", dependency_names=())

    assert len(calls) == 1


@pytest.mark.requires_nemo
def test_service_cleanup_job_carries_the_job_comment(monkeypatch: pytest.MonkeyPatch) -> None:
    from nvflow.execution.engines.slurm import _sbatch_host_cleanup
    from nvflow.integrations.nemo_skills.slurm import job_comment

    scripts: list[str] = []

    def run(_argv: list[str], *, input: str, **_kwargs: object) -> SimpleNamespace:
        scripts.append(input)
        return SimpleNamespace(returncode=0, stdout="Submitted batch job 2", stderr="")

    monkeypatch.setattr("nvflow.execution.engines.slurm.subprocess", SimpleNamespace(run=run))
    monkeypatch.setattr("nvflow.integrations.nemo_skills.launcher_is_remote", lambda _config: False)
    task = TaskPlan(
        task_id="training.cleanup.0",
        stage="training",
        name="train-judge-cleanup-0",
        operation="cleanup",
        parameters={"slurm_cleanup_target": "train-judge-0", "slurm": {"log_dir": "/logs"}},
    )
    comment = "nvflow,workflow=demo:train/small,stage=training"
    config = {"executor": "slurm", "account": "acct", "mounts": ["/lustre/logs:/logs"]}

    with job_comment(comment):
        _sbatch_host_cleanup(task, "1000", config)
    _sbatch_host_cleanup(task, "1001", config)

    assert f"#SBATCH --comment={comment}" in scripts[0].splitlines()
    assert "--comment" not in scripts[1]
