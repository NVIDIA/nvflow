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
"""Executor-detection utilities shared across nvflow.

Provides a single canonical :func:`is_ray_backend` check so that the
Ray/Slurm submission branch (``with_ray=``) does not need to be
re-implemented in every file that submits jobs.
"""

from __future__ import annotations

from typing import Any


def is_ray_backend(cluster_config: dict[str, Any]) -> bool:
    """Return True when the cluster config targets the Ray Jobs API backend.

    Handles all known config shapes:

    * ``backend: {name: ray, dashboard_url: ...}``        — standard single-cluster
    * ``backend: {name: ray, gpu_nemo_rl_dashboard_url: ...}`` — 2-cluster
    * ``backend: {name: ray, kubernetes: {dashboard_url: ...}}`` — K8s
    * ``backend: "ray"``                                  — shorthand string
    * ``execution_backend: ...``                          — legacy alias

    A bare ``backend.name: ray`` with no dashboard URL is NOT treated as the
    Ray Jobs backend — it indicates an incomplete config rather than a real
    Ray cluster, and submission would fail anyway.
    """
    # An explicit Slurm executor is authoritative.  This prevents stale or
    # partially migrated Ray metadata from activating Ray-only retries,
    # command rendering, or Jobs API submission in an ordinary Slurm run.
    executor = str(cluster_config.get("executor") or "").strip().lower()
    if executor == "slurm":
        return False

    backend = cluster_config.get("backend") or cluster_config.get("execution_backend") or {}

    if isinstance(backend, str):
        # shorthand: executor: none + backend: ray (no dashboard)
        return backend.strip().lower() == "ray"

    if not isinstance(backend, dict):
        return False

    name = str(backend.get("name") or "").strip().lower()
    if name != "ray":
        return False

    # Require at least one resolvable dashboard URL so that a partial config
    # (backend.name = ray but no URL yet) doesn't accidentally route to Ray.
    k8s = backend.get("kubernetes") or {}
    dashboard = (
        backend.get("gpu_nemo_rl_dashboard_url")
        or backend.get("cpu_nemo_skills_dashboard_url")
        or backend.get("cpu_nemo_gym_dashboard_url")
        or backend.get("dashboard_url")
        or backend.get("jobs_api_url")
        or k8s.get("dashboard_url")
    )
    return bool(dashboard)


# Slurm container working dir. Ray captures the absolute extracted
# ``backend.working_dir`` path in ``NEMO_RUN_CODE_DIR`` before commands can
# change directory (for example to /opt/Gym).
WORKSPACE_ROOT = "/workspace"
RAY_WORKING_DIR_ROOT = "$NEMO_RUN_CODE_DIR"


def path_roots(cluster_config: dict[str, Any]) -> dict[str, str]:
    """Return the backend-keyed named path roots for recipe interpolation.

    Recipes interpolate ``${repo_root}/...`` and ``${nvflow_root}/...`` so a single
    backend-agnostic recipe works on both Slurm and the Ray Jobs backend:

    * ``repo_root`` — ``/workspace`` on Slurm; ``$NEMO_RUN_CODE_DIR`` on Ray,
      the absolute extracted working-directory archive captured before a job
      command can change cwd. Used for tracked repo references (prompts, scripts).
    * ``nvflow_root`` — where workflow data/outputs live. On Slurm the driver runs
      in-container so this is ``/workspace`` (making ``${nvflow_root}/X`` byte-identical
      to the legacy ``/workspace/X``). On the Ray Jobs backend (``executor: none``) the
      nvflow-client driver has no mutable ``/workspace`` data mount, so driver-touched
      paths need an absolute driver- and worker-visible
      shared path (e.g. ``/lustre/<you>/nvflow``).

    On Ray the absolute root is read from the cluster config ``nvflow_root`` (or
    ``backend.nvflow_root``). Slurm (and any non-Ray / unresolved config) keeps both
    roots at ``/workspace``, so existing recipes resolve unchanged. (The key is named
    ``nvflow_root`` rather than ``data_root`` so it never collides with a recipe that
    already defines its own ``data_root`` key, e.g. grpo.)
    """
    if not is_ray_backend(cluster_config):
        return {"repo_root": WORKSPACE_ROOT, "nvflow_root": WORKSPACE_ROOT}

    backend = cluster_config.get("backend")
    backend = backend if isinstance(backend, dict) else {}
    nvflow_root = cluster_config.get("nvflow_root") or backend.get("nvflow_root")
    if not nvflow_root:
        raise ValueError(
            "Ray backend requires an absolute 'nvflow_root' (a shared path visible "
            "identically in the nvflow-client driver AND worker containers, "
            "e.g. /lustre/<you>/nvflow) in the cluster config."
        )
    return {"repo_root": RAY_WORKING_DIR_ROOT, "nvflow_root": str(nvflow_root)}
