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
"""Tests for NVFlow's NeMo-Skills dependency and sbatch policy adapter."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from nvflow.integrations.nemo_skills import slurm as sbatch
from nvflow.integrations.nemo_skills.slurm import (
    dependency_sbatch_kwargs,
    parse_extra_sbatch_args,
)

# ---------------------------------------------------------------------------
# parse_extra_sbatch_args
# ---------------------------------------------------------------------------


def test_parse_value_form():
    cfg = {"extra_sbatch_args": ["--qos=xlarge_qos", "--partition=gb200"]}
    assert parse_extra_sbatch_args(cfg) == {"qos": "xlarge_qos", "partition": "gb200"}


def test_parse_boolean_form():
    cfg = {"extra_sbatch_args": ["--exclusive", "--requeue"]}
    assert parse_extra_sbatch_args(cfg) == {"exclusive": True, "requeue": True}


def test_parse_strips_dashes():
    cfg = {"extra_sbatch_args": ["---qos=xlarge_qos"]}
    assert parse_extra_sbatch_args(cfg) == {"qos": "xlarge_qos"}


def test_parse_skips_garbage_entries():
    cfg = {"extra_sbatch_args": ["", "--qos=xlarge_qos", None, 42, "---"]}  # type: ignore[list-item]
    assert parse_extra_sbatch_args(cfg) == {"qos": "xlarge_qos"}


@pytest.mark.parametrize(
    "cluster_config",
    [None, {}, {"extra_sbatch_args": None}, {"extra_sbatch_args": []}],  # type: ignore[dict-item]
)
def test_parse_empty(cluster_config):
    assert parse_extra_sbatch_args(cluster_config) == {}


def test_dependency_policy_is_afterok_except_for_resume_edges():
    assert dependency_sbatch_kwargs(
        {"qos": "normal"},
        continue_after_failure=False,
    ) == {"qos": "normal"}
    assert dependency_sbatch_kwargs(
        {"qos": "normal", "dependency_type": "afterok"},
        continue_after_failure=True,
    ) == {"qos": "normal", "dependency_type": "afterany"}


_REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent


def test_rollout_uses_shared_parser():
    """Native rollout must use parse_extra_sbatch_args, not an inline loop.

    Reads the file directly so the test does not require ``nemo_skills``
    (which ``rollout`` imports at module load) to be installed.
    """
    src = (_REPO_ROOT / "nvflow" / "integrations" / "nemo_gym" / "native_rollout.py").read_text()
    assert "parse_extra_sbatch_args" in src


def test_autopatch_is_isolated_in_direct_slurm_engine():
    """Autopatch must be in the Slurm engine, not shared runner or CLI.

    Reads the files directly so the test does not require ``typer``
    (a CLI-only dep) to be installed.
    """
    cli_src = (_REPO_ROOT / "nvflow" / "cli" / "main.py").read_text()
    runner_src = (_REPO_ROOT / "nvflow" / "execution" / "runner.py").read_text()
    slurm_src = (_REPO_ROOT / "nvflow" / "execution" / "engines" / "slurm.py").read_text()

    assert "apply_sbatch_args_autopatch" not in cli_src
    assert "apply_sbatch_args_autopatch" not in runner_src
    assert "apply_sbatch_args_autopatch" in slurm_src


# ---------------------------------------------------------------------------
# apply_sbatch_args_autopatch
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_nemo_skills(monkeypatch):
    """Minimal fake nemo_skills so the autopatch can target get_executor."""
    calls: list[dict[str, Any]] = []
    packager_calls: list[tuple[str, ...] | None] = []

    # If another test patched the installed package, restore it before replacing
    # its module entries.  The monkeypatch fixture will put every original entry
    # back after this fixture tears down.
    sbatch._reset_for_tests()

    def _make(name: str, **attrs):
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    def _original_get_executor(cluster_config, *args, **kwargs):
        calls.append({"cluster_config": cluster_config, "args": args, "kwargs": kwargs})
        env_vars = {
            name.strip(): value.strip()
            for entry in cluster_config.get("env_vars", [])
            if isinstance(entry, str) and "=" in entry
            for name, value in (entry.split("=", 1),)
        }
        return types.SimpleNamespace(
            packager=object(),
            env_vars=env_vars,
            srun_args=[f"--container-env={','.join(env_vars)}"],
        )

    def _original_check_mounts(_directories, _cluster_config, exit_on_failure=True):
        del exit_on_failure

    def _original_get_packager(extra_package_dirs=None):
        packager_calls.append(extra_package_dirs)
        return object()

    _make("nemo_skills")
    _make("nemo_skills.pipeline")
    utils_mod = _make(
        "nemo_skills.pipeline.utils",
        get_executor=_original_get_executor,
        get_packager=_original_get_packager,
        check_remote_mount_directories=_original_check_mounts,
    )
    exp_mod = _make(
        "nemo_skills.pipeline.utils.exp",
        get_executor=_original_get_executor,
        get_packager=_original_get_packager,
        check_remote_mount_directories=_original_check_mounts,
    )
    _make(
        "nemo_skills.pipeline.utils.mounts",
        check_remote_mount_directories=_original_check_mounts,
    )
    _make(
        "nemo_skills.pipeline.utils.declarative",
        get_executor=_original_get_executor,
        get_packager=_original_get_packager,
    )
    exp_mod.__dict__["packager_calls"] = packager_calls

    try:
        yield exp_mod, utils_mod, calls
    finally:
        sbatch._reset_for_tests()


def test_autopatch_injects_qos_into_get_executor(fake_nemo_skills):
    exp_mod, _, calls = fake_nemo_skills

    sbatch.apply_sbatch_args_autopatch()

    exp_mod.get_executor(
        {"extra_sbatch_args": ["--qos=xlarge_qos"]},
        "container",
        num_nodes=1,
    )

    assert len(calls) == 1
    assert calls[0]["kwargs"]["sbatch_kwargs"] == {"qos": "xlarge_qos"}


def test_autopatch_existing_kwargs_win(fake_nemo_skills):
    exp_mod, _, calls = fake_nemo_skills

    sbatch.apply_sbatch_args_autopatch()

    exp_mod.get_executor(
        {"extra_sbatch_args": ["--qos=xlarge_qos", "--time=02:00:00"]},
        "container",
        sbatch_kwargs={"qos": "explicit_override"},
    )

    assert calls[0]["kwargs"]["sbatch_kwargs"] == {
        "qos": "explicit_override",
        "time": "02:00:00",
    }


def test_autopatch_noop_when_no_extras(fake_nemo_skills):
    exp_mod, _, calls = fake_nemo_skills

    sbatch.apply_sbatch_args_autopatch()

    exp_mod.get_executor({"extra_sbatch_args": []}, "container")

    assert len(calls) == 1
    assert calls[0]["cluster_config"]["dependency_type"] == "afterok"
    assert calls[0]["kwargs"].get("sbatch_kwargs") is None


def test_downstream_jobs_never_wait_on_excluded_jobs(fake_nemo_skills):
    exp_mod, _, calls = fake_nemo_skills

    sbatch.apply_sbatch_args_autopatch()
    with sbatch.jobs_excluded_from_dependencies() as excluded_jobs:
        excluded_jobs.add("slurm:///1001/master/0")
        exp_mod.get_executor(
            {},
            "container",
            dependencies=["slurm:///1000/master/0", "slurm:///1001/master/0"],
        )
    exp_mod.get_executor({}, "container", dependencies=["slurm:///1001/master/0"])

    assert calls[0]["kwargs"]["dependencies"] == ["slurm:///1000/master/0"]
    assert calls[1]["kwargs"]["dependencies"] == ["slurm:///1001/master/0"]


def test_autopatch_preserves_explicit_dependency_policy(fake_nemo_skills):
    exp_mod, _, calls = fake_nemo_skills

    sbatch.apply_sbatch_args_autopatch()
    exp_mod.get_executor({"dependency_type": "afterany"}, "container")

    assert calls[0]["cluster_config"]["dependency_type"] == "afterany"


def test_jobs_carry_the_job_comment_only_inside_its_block(fake_nemo_skills):
    exp_mod, _, calls = fake_nemo_skills

    sbatch.apply_sbatch_args_autopatch()
    with sbatch.job_comment("nvflow,stage=first"):
        exp_mod.get_executor({}, "container", sbatch_kwargs={"qos": "high"})
        with sbatch.job_comment("nvflow,stage=second"):
            exp_mod.get_executor({}, "container")
        exp_mod.get_executor({}, "container")
    exp_mod.get_executor({}, "container")

    assert [call["kwargs"].get("sbatch_kwargs") for call in calls] == [
        {"qos": "high", "comment": "nvflow,stage=first"},
        {"comment": "nvflow,stage=second"},
        {"comment": "nvflow,stage=first"},
        None,
    ]
    assert sbatch.active_job_comment() is None


@pytest.mark.parametrize(
    ("cluster_config", "sbatch_kwargs"),
    [
        ({"extra_sbatch_args": ["--comment=site-policy"]}, None),
        ({}, {"comment": "site-policy"}),
    ],
)
def test_a_comment_the_cluster_or_task_sets_wins_over_the_job_comment(
    fake_nemo_skills,
    cluster_config,
    sbatch_kwargs,
):
    exp_mod, _, calls = fake_nemo_skills

    sbatch.apply_sbatch_args_autopatch()
    with sbatch.job_comment("nvflow,stage=first"):
        exp_mod.get_executor(cluster_config, "container", sbatch_kwargs=sbatch_kwargs)

    assert calls[0]["kwargs"]["sbatch_kwargs"] == {"comment": "site-policy"}


@pytest.mark.requires_nemo
def test_nemo_skills_executors_receive_the_job_comment(tmp_path):
    import nemo_skills.pipeline.utils.exp as exp_mod

    arguments = {
        "cluster_config": {
            "executor": "slurm",
            "account": "acct",
            "partition": "batch",
            "job_dir": str(tmp_path),
            "containers": {"nemo-skills": "image"},
            "mounts": [f"{tmp_path}:/logs"],
            "timeouts": {"batch": "04:00:00"},
        },
        "container": "image",
        "num_nodes": 1,
        "tasks_per_node": 1,
        "gpus_per_node": 8,
        "job_name": "model-a",
        "log_dir": "/logs",
    }
    was_patched = sbatch._PATCHED
    sbatch.apply_sbatch_args_autopatch()
    try:
        with sbatch.job_comment("nvflow,stage=model-a"):
            labeled = exp_mod.get_executor(**arguments, sbatch_kwargs={"qos": "high"})
        unlabeled = exp_mod.get_executor(**arguments)
    finally:
        if not was_patched:
            sbatch._reset_for_tests()

    assert (labeled.comment, labeled.qos) == ("nvflow,stage=model-a", "high")
    assert unlabeled.comment is None


@pytest.mark.requires_nemo
def test_every_heterogeneous_component_carries_the_job_comment(tmp_path):
    import nemo_run as run
    from nemo_run.core.execution.slurm import SlurmBatchRequest
    from nemo_run.core.tunnel.client import LocalTunnel

    comment = "nvflow,workflow=demo:eval/small,stage=model-a"
    executor = run.SlurmExecutor.merge(
        [
            run.SlurmExecutor(
                account="acct",
                tunnel=LocalTunnel(job_dir=str(tmp_path)),
                heterogeneous=True,
                comment=comment,
            )
            for _ in range(2)
        ],
        num_tasks=2,
    )
    executor.job_dir = str(tmp_path / "experiment")
    script = SlurmBatchRequest(
        launch_cmd=["sbatch", "--parsable"],
        jobs=["policy", "judge"],
        command_groups=[["true"], ["true"]],
        executor=executor,
        max_retries=0,
        extra_env={},
    ).materialize()

    lines = script.splitlines()
    assert f"#SBATCH --comment={comment}" in lines[: lines.index("#SBATCH hetjob")]
    assert lines.count(f"#SBATCH --comment={comment}") == 2


@pytest.mark.requires_nemo
def test_ray_on_slurm_jobs_carry_the_job_comment(tmp_path):
    import nemo_run as run
    from nemo_run.core.tunnel.client import LocalTunnel
    from nemo_run.run.ray.slurm import SlurmRayRequest

    comment = "nvflow,workflow=demo:train/small,stage=training"
    executor = run.SlurmExecutor(
        account="acct",
        tunnel=LocalTunnel(job_dir=str(tmp_path)),
        comment=comment,
    )
    executor.job_dir = str(tmp_path / "experiment")
    script = SlurmRayRequest(
        name="training",
        cluster_dir=str(tmp_path / "cluster"),
        template_name="ray.sub.j2",
        executor=executor,
        command="true",
        launch_cmd=["sbatch", "--parsable"],
    ).materialize()

    assert f"#SBATCH --comment={comment}" in script.splitlines()


@pytest.mark.requires_nemo
def test_every_heterogeneous_component_waits_on_the_dependencies(fake_nemo_skills, tmp_path):
    import nemo_run as run
    from nemo_run.core.execution.slurm import SlurmBatchRequest
    from nemo_run.core.tunnel.client import LocalTunnel

    sbatch.apply_sbatch_args_autopatch()
    executor = run.SlurmExecutor.merge(
        [
            run.SlurmExecutor(
                account="acct",
                tunnel=LocalTunnel(job_dir=str(tmp_path)),
                heterogeneous=True,
            )
            for _ in range(2)
        ],
        num_tasks=2,
    )
    executor.job_dir = str(tmp_path / "experiment")
    executor.dependencies = ["slurm:///1234/master/0"]
    script = SlurmBatchRequest(
        launch_cmd=["sbatch", "--parsable"],
        jobs=["policy", "judge"],
        command_groups=[["true"], ["true"]],
        executor=executor,
        max_retries=0,
        extra_env={},
    ).materialize()

    lines = script.splitlines()
    dependency = "#SBATCH --dependency=afterok:1234"
    assert dependency in lines[: lines.index("#SBATCH hetjob")]
    assert lines.count(dependency) == 2
    assert executor.dependencies == ["slurm:///1234/master/0"]
    assert "dependency" not in (executor.additional_parameters or {})


def test_autopatch_patches_utils_reexport(fake_nemo_skills):
    _, utils_mod, calls = fake_nemo_skills

    sbatch.apply_sbatch_args_autopatch()

    utils_mod.get_executor(
        {"extra_sbatch_args": ["--qos=test"]},
        "container",
    )

    assert len(calls) == 1
    assert calls[0]["kwargs"]["sbatch_kwargs"] == {"qos": "test"}


def test_autopatch_replaces_mount_check_aliases(fake_nemo_skills):
    exp_mod, utils_mod, _ = fake_nemo_skills
    mounts_mod = sys.modules["nemo_skills.pipeline.utils.mounts"]

    sbatch.apply_sbatch_args_autopatch()

    assert exp_mod.check_remote_mount_directories is utils_mod.check_remote_mount_directories
    assert mounts_mod.check_remote_mount_directories is exp_mod.check_remote_mount_directories
    assert hasattr(exp_mod.check_remote_mount_directories, "__wrapped__")


def test_autopatch_reuses_base_packager_within_project_submission(fake_nemo_skills):
    from nvflow.project import ProjectResolver
    from nvflow.project.packaging import project_submission

    exp_mod, _, _ = fake_nemo_skills
    context = ProjectResolver().resolve_reference("starter")
    sbatch.apply_sbatch_args_autopatch()

    with project_submission(context):
        first = exp_mod.get_packager(extra_package_dirs=("extra",))
        second = exp_mod.get_packager(extra_package_dirs=("extra",))

    assert first is second
    assert exp_mod.packager_calls == [("extra",)]


def test_autopatch_replaces_previously_captured_declarative_alias(fake_nemo_skills):
    from nvflow.project import ProjectResolver
    from nvflow.project.packaging import ProjectOverlayPackager, project_submission

    exp_mod, _, calls = fake_nemo_skills
    declarative_mod = sys.modules["nemo_skills.pipeline.utils.declarative"]
    bundle = _REPO_ROOT / "tests" / "fixtures" / "projects" / "python_project"
    config = bundle / "workflows" / "noop.yaml"
    context = ProjectResolver().resolve(config, project_dir=bundle)

    sbatch.apply_sbatch_args_autopatch()
    with project_submission(context):
        executor = declarative_mod.get_executor(
            {"extra_sbatch_args": ["--qos=declarative"]},
            "container",
        )

    assert declarative_mod.get_executor is exp_mod.get_executor
    assert calls[0]["kwargs"]["sbatch_kwargs"] == {"qos": "declarative"}
    assert isinstance(executor.packager, ProjectOverlayPackager)
    assert executor.env_vars["PYTHONPATH"].startswith("/nemo_run/code/projects/fixture-python/src:")
    assert "PYTHONPATH" in executor.srun_args[0]


def test_autopatch_is_idempotent(fake_nemo_skills):
    exp_mod, utils_mod, _ = fake_nemo_skills

    sbatch.apply_sbatch_args_autopatch()
    first = (exp_mod.get_executor, utils_mod.get_executor)

    sbatch.apply_sbatch_args_autopatch()
    second = (exp_mod.get_executor, utils_mod.get_executor)

    assert first == second


def test_autopatch_applies_external_project_packager(fake_nemo_skills):
    from nvflow.project import ProjectResolver
    from nvflow.project.packaging import ProjectOverlayPackager, project_submission

    exp_mod, _, _ = fake_nemo_skills
    bundle = _REPO_ROOT / "tests" / "fixtures" / "projects" / "python_project"
    config = bundle / "workflows" / "noop.yaml"
    context = ProjectResolver().resolve(config, project_dir=bundle)
    sbatch.apply_sbatch_args_autopatch()

    with project_submission(context):
        executor = exp_mod.get_executor({}, "container")

    assert isinstance(executor.packager, ProjectOverlayPackager)


def test_autopatch_skips_when_nemo_skills_absent(monkeypatch):
    """No crash when nemo-skills isn't importable."""
    sbatch._reset_for_tests()
    blocked_modules = (
        "nemo_skills",
        "nemo_skills.pipeline",
        "nemo_skills.pipeline.utils",
        "nemo_skills.pipeline.utils.exp",
    )

    try:
        # Install sentinels unconditionally so this exercises the ImportError
        # path even when the full-deps environment has not imported NeMo-Skills
        # yet.  The context restores pre-existing modules and removes new keys.
        with monkeypatch.context() as module_patch:
            for mod_name in blocked_modules:
                module_patch.setitem(sys.modules, mod_name, None)

            sbatch.apply_sbatch_args_autopatch()  # must not raise
            assert sbatch._PATCHED is False
    finally:
        sbatch._reset_for_tests()
