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
        or backend.get("dashboard_url")
        or backend.get("jobs_api_url")
        or k8s.get("dashboard_url")
    )
    return bool(dashboard)
