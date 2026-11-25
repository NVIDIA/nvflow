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
"""Apply NVFlow's dependency policy and global sbatch arguments.

Cluster configs may declare global Slurm flags via ``extra_sbatch_args``
(e.g. ``--qos=xlarge_qos``, ``--exclusive``).  These must reach the
generated sbatch script for every stage — SDG, SFT, eval, and GRPO.

NVFlow defaults omitted ``dependency_type`` to ``afterok`` so ordinary
downstream work is fail-closed. Explicit retry and analyze jobs opt into
``afterany`` through :func:`dependency_sbatch_kwargs`.

NeMo-Skills expands a ``run_after`` experiment into every pending job in it,
including services that end by cancellation and retry attempts that a later
attempt supersedes. Jobs therefore never wait on the jobs recorded in
:func:`jobs_excluded_from_dependencies`.

Jobs created inside :func:`job_comment` carry its comment, which ``nvflow status``
reads to tell the workflow and stage of each job. A comment the cluster's
``extra_sbatch_args`` or the task's ``sbatch_kwargs`` already set wins.

This module patches ``nemo_skills.pipeline.utils.exp.get_executor``, the
single funnel through which all ``SlurmExecutor`` instances are created.
It also replaces aliases captured by already-imported pipeline modules;
modules imported later receive the patched function from ``exp``. It also
writes a heterogeneous job's dependencies into every component of its script.

:func:`parse_extra_sbatch_args` is also used by the NeMo-Gym rollout adapter
which previously inlined the same parsing loop.
"""

from __future__ import annotations

import contextlib
import functools
import sys
from collections.abc import Iterator
from typing import Any

__all__ = [
    "active_job_comment",
    "apply_sbatch_args_autopatch",
    "dependency_sbatch_kwargs",
    "job_comment",
    "jobs_excluded_from_dependencies",
    "parse_extra_sbatch_args",
]

_EXCLUDED_JOBS: set[str] = set()
_JOB_COMMENT: list[str] = []


@contextlib.contextmanager
def jobs_excluded_from_dependencies() -> Iterator[set[str]]:
    """Yield the set of job handles no job in this submission may wait on."""
    _EXCLUDED_JOBS.clear()
    try:
        yield _EXCLUDED_JOBS
    finally:
        _EXCLUDED_JOBS.clear()


@contextlib.contextmanager
def job_comment(comment: str) -> Iterator[None]:
    """Give every Slurm job created inside the block *comment* as its job comment."""
    _JOB_COMMENT.append(comment)
    try:
        yield
    finally:
        _JOB_COMMENT.pop()


def active_job_comment() -> str | None:
    """Return the comment jobs created now receive, if any."""
    return _JOB_COMMENT[-1] if _JOB_COMMENT else None


def parse_extra_sbatch_args(cluster_config: dict | None) -> dict[str, Any]:
    """Parse ``extra_sbatch_args`` list into a kwargs dict.

    Accepts entries like ``--qos=xlarge_qos`` (value) or ``--exclusive``
    (boolean flag).  Leading dashes are stripped.  Empty / non-string
    entries are skipped so malformed YAML doesn't crash submission.
    """
    if not cluster_config:
        return {}
    out: dict[str, Any] = {}
    for arg in cluster_config.get("extra_sbatch_args") or []:
        if not isinstance(arg, str):
            continue
        key, sep, val = arg.lstrip("-").partition("=")
        if not key:
            continue
        out[key] = val if sep else True
    return out


def dependency_sbatch_kwargs(
    sbatch_kwargs: dict[str, Any] | None,
    *,
    continue_after_failure: bool,
) -> dict[str, Any] | None:
    """Keep ``afterok`` by default and opt retry/postprocess edges into ``afterany``."""
    merged = dict(sbatch_kwargs or {})
    if continue_after_failure:
        merged["dependency_type"] = "afterany"
    return merged or None


_PATCHED = False
_PATCHED_ATTRIBUTES: list[tuple[Any, str, Any]] = []


def apply_sbatch_args_autopatch() -> None:
    """Wrap ``get_executor`` so every Slurm submission honours ``extra_sbatch_args``.

    Patches both the home module (``exp``) and the ``utils`` package
    re-export so callers that import via either path see the wrapper.

    Safe to call repeatedly; subsequent calls are no-ops.  Silently does
    nothing if ``nemo-skills`` isn't importable.
    """
    global _PATCHED
    if _PATCHED:
        return

    try:
        import nemo_skills.pipeline.utils.exp as exp_mod
        import nemo_skills.pipeline.utils.mounts as mounts_mod
    except ImportError:
        return

    _PATCHED = True

    _original_get_executor = exp_mod.get_executor
    _original_get_packager = exp_mod.get_packager

    @functools.wraps(_original_get_executor)
    def _patched_get_executor(cluster_config, *args, **kwargs):
        cluster_config = dict(cluster_config)
        cluster_config.setdefault("dependency_type", "afterok")
        if kwargs.get("dependencies") and _EXCLUDED_JOBS:
            kwargs["dependencies"] = [
                handle for handle in kwargs["dependencies"] if handle not in _EXCLUDED_JOBS
            ] or None
        extras = parse_extra_sbatch_args(cluster_config)
        if extras:
            existing = kwargs.get("sbatch_kwargs") or {}
            merged = {**extras, **existing}
            kwargs["sbatch_kwargs"] = merged
        comment = active_job_comment()
        if comment and "comment" not in (kwargs.get("sbatch_kwargs") or {}):
            kwargs["sbatch_kwargs"] = {**(kwargs.get("sbatch_kwargs") or {}), "comment": comment}
        from nvflow.project.packaging import apply_project_environment, apply_project_packager

        cluster_config = apply_project_environment(cluster_config)
        executor = _original_get_executor(cluster_config, *args, **kwargs)
        return apply_project_packager(executor)

    @functools.wraps(_original_get_packager)
    def _patched_get_packager(extra_package_dirs=None):
        from nvflow.project.packaging import get_submission_base_packager

        return get_submission_base_packager(_original_get_packager, extra_package_dirs)

    _original_check_mounts = mounts_mod.check_remote_mount_directories

    @functools.wraps(_original_check_mounts)
    def _patched_check_mounts(directories, cluster_config, exit_on_failure=True):
        from nvflow.integrations.nemo_skills.filesystem import (
            check_remote_mount_directories,
        )

        return check_remote_mount_directories(
            directories,
            cluster_config,
            exit_on_failure=exit_on_failure,
        )

    for module_name, module in tuple(sys.modules.items()):
        if not module_name.startswith("nemo_skills.pipeline") or module is None:
            continue
        for attribute, original, replacement in (
            ("get_executor", _original_get_executor, _patched_get_executor),
            ("get_packager", _original_get_packager, _patched_get_packager),
            (
                "check_remote_mount_directories",
                _original_check_mounts,
                _patched_check_mounts,
            ),
        ):
            if getattr(module, attribute, None) is original:
                setattr(module, attribute, replacement)
                _PATCHED_ATTRIBUTES.append((module, attribute, original))

    try:
        from nemo_run.core.execution.slurm import SlurmBatchRequest
    except ImportError:
        return
    _original_materialize = SlurmBatchRequest.materialize

    @functools.wraps(_original_materialize)
    def _patched_materialize(self):
        """Put a heterogeneous job's dependency in every component of its script.

        NeMo-Run writes ``--dependency`` after the last ``#SBATCH hetjob``, where
        Slurm applies it to the last component only. In an experiment with
        dependencies between its own jobs, NeMo-Run replaces the executor's
        dependencies after rendering, so one on another experiment is left only
        in the script.
        """
        executor = self.executor
        if not executor.heterogeneous or not executor.dependencies:
            return _original_materialize(self)
        dependencies = executor.dependencies
        additional_parameters = executor.additional_parameters
        executor.additional_parameters = {
            **(additional_parameters or {}),
            "dependency": f"{executor.dependency_type}:{':'.join(executor.parse_deps())}",
        }
        executor.dependencies = []
        try:
            return _original_materialize(self)
        finally:
            executor.dependencies = dependencies
            executor.additional_parameters = additional_parameters

    SlurmBatchRequest.materialize = _patched_materialize
    _PATCHED_ATTRIBUTES.append((SlurmBatchRequest, "materialize", _original_materialize))


def _reset_for_tests() -> None:
    """Test-only hook: clear the idempotency flag and undo monkey-patches."""
    global _PATCHED
    if not _PATCHED:
        return
    for module, attribute, original in _PATCHED_ATTRIBUTES:
        setattr(module, attribute, original)
    _PATCHED_ATTRIBUTES.clear()
    _PATCHED = False
