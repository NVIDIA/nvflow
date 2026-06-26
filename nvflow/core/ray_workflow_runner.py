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
"""Ray-specific workflow runner with 2-cluster GPU/CPU routing.

Extends :class:`WorkflowRunner` with per-stage cluster routing for
pre-provisioned 2-cluster Ray setups (a GPU cluster running the nemo-rl
image and a CPU cluster running the nemo-skills image).  Slurm-specific
logic (sbatch args autopatch, Slurm dependency wiring) is intentionally
absent here — Ray jobs are submitted via the Ray Jobs HTTP API, not sbatch.

Use :func:`create_workflow_runner` to auto-detect whether a config targets
Ray or Slurm and return the right runner without changing the call site.

Typical usage via the factory (transparent to the CLI)::

    from nvflow.core.ray_workflow_runner import create_workflow_runner

    runner = create_workflow_runner("recipes/finance/workflows/sft/qwen3_4b.yaml")
    runner.run()

Or directly when the executor is known::

    from nvflow.core.ray_workflow_runner import RayWorkflowRunner

    runner = RayWorkflowRunner("cluster_configs/my_cluster-2c.yaml")
    runner.run()
"""

from nvflow.core.workflow_runner import WorkflowRunner
from nvflow.lib.executor import is_ray_backend


class RayWorkflowRunner(WorkflowRunner):
    """WorkflowRunner variant for pre-provisioned Ray clusters.

    Adds per-stage CPU/GPU cluster routing for 2-cluster Ray setups:

    - Stages with a positive GPU resource count (``total_gpus``, ``num_gpus``,
      ``gpus``) are routed to ``gpu_nemo_rl_dashboard_url``.
    - CPU stages (no positive GPU count) are routed to
      ``cpu_nemo_skills_dashboard_url``.
    - An explicit ``target_cluster: cpu|gpu`` on the stage config overrides
      the inference (e.g. eval, whose generation server is external).

    For single-cluster Ray (only a plain ``dashboard_url`` defined) or Slurm
    configs the routing is a no-op — the workflow cluster name is returned
    unchanged, so ``RayWorkflowRunner`` degrades gracefully to the same
    behaviour as ``WorkflowRunner``.

    The Slurm sbatch-args autopatch (:meth:`WorkflowRunner._before_run`) is
    overridden to a no-op: Ray jobs are submitted via the Ray Jobs HTTP API,
    not via ``sbatch``.
    """

    # GPU-resource keys a stage may declare (at any nesting depth).  A
    # positive value routes the stage to the GPU cluster, mirroring how Slurm
    # picks ``partition`` vs ``cpu_partition`` from the stage's own request.
    _GPU_RESOURCE_KEYS = ("total_gpus", "num_gpus", "gpus")

    # Keys whose subtree describes OTHER components' resources, not this
    # stage's own compute.  The ``environments`` block is embedded verbatim
    # into every stage's config (``environments: ${environments}``) and
    # carries each environment's server GPU counts (e.g. a judge or policy
    # vLLM's ``num_gpus``).  Walking it would classify a pure-CPU stage (e.g.
    # ``validate_questions``, ``train_validation_split``) as GPU and misroute
    # it to the GPU cluster, so the GPU-need walk skips these subtrees.
    _NON_STAGE_RESOURCE_KEYS = ("environments", "_environment")

    def _before_run(self) -> None:
        """No-op for Ray: jobs are submitted via the Ray Jobs HTTP API, not sbatch."""

    def _resolve_stage_cluster(self, stage_config: dict):
        """Route each stage to its GPU or CPU Ray cluster.

        Reads ``gpu_nemo_rl_dashboard_url`` and
        ``cpu_nemo_skills_dashboard_url`` from the cluster config backend
        and returns a config copy whose generic ``backend.dashboard_url``
        (the field nemo-skills reads) points at the correct cluster.
        Falls back to the workflow cluster name unchanged for single-cluster
        Ray or any non-Ray config.
        """
        try:
            from nemo_skills.pipeline.utils.cluster import get_cluster_config

            cluster_config = get_cluster_config(self.cluster)
        except Exception:
            return self.cluster
        resolved = self._route_cluster(cluster_config, stage_config, self.cluster)
        if resolved is not self.cluster:
            from nvflow.core.console import info

            info("Routing stage to its resolved Ray cluster")
        return resolved

    @staticmethod
    def _route_cluster(cluster_config, stage_config: dict, cluster_name):
        """Pick the GPU vs CPU cluster for a stage.  Pure (no I/O).

        For a two-cluster Ray backend that defines both
        ``gpu_nemo_rl_dashboard_url`` and ``cpu_nemo_skills_dashboard_url``,
        return a cluster-config copy whose generic ``backend.dashboard_url``
        (the field NeMo-Skills reads) is the GPU dashboard for GPU stages and
        the CPU dashboard for CPU stages.  An explicit ``target_cluster:
        cpu|gpu`` overrides the ``num_gpus`` inference.  Returns
        ``cluster_name`` unchanged for Slurm or a single-cluster Ray config.
        """
        backend = cluster_config.get("backend") if isinstance(cluster_config, dict) else None
        if not isinstance(backend, dict):
            return cluster_name
        gpu_url = backend.get("gpu_nemo_rl_dashboard_url")
        cpu_url = backend.get("cpu_nemo_skills_dashboard_url")
        if not gpu_url and not cpu_url:
            return cluster_name
        target = stage_config.get("target_cluster")
        if target not in ("cpu", "gpu"):
            target = "gpu" if RayWorkflowRunner._stage_requests_gpus(stage_config) else "cpu"
        chosen = cpu_url if (target == "cpu" and cpu_url) else (gpu_url or cpu_url)
        return {**cluster_config, "backend": {**backend, "dashboard_url": chosen}}

    @staticmethod
    def _stage_requests_gpus(node) -> bool:
        """True if a positive GPU count appears anywhere in the stage config.

        Walks nested dicts/lists so a GPU request declared under a nested
        server (e.g. ``rollout.policy_vllm.num_gpus``) is detected, not just
        a top-level count.  Only the ``_GPU_RESOURCE_KEYS`` are treated as
        GPU signals, and only when the value is a positive int (so ``0``
        external-endpoint placeholders do not count).  The embedded
        ``environments`` block is skipped (see ``_NON_STAGE_RESOURCE_KEYS``)
        because it describes other components' endpoints, not this stage's
        own compute.  An explicit ``target_cluster`` (checked first in
        ``_route_cluster``) still wins over this inference.
        """
        if isinstance(node, dict):
            for k, v in node.items():
                if k in RayWorkflowRunner._GPU_RESOURCE_KEYS and isinstance(v, int) and v > 0:
                    return True
                if k in RayWorkflowRunner._NON_STAGE_RESOURCE_KEYS:
                    continue
                if RayWorkflowRunner._stage_requests_gpus(v):
                    return True
            return False
        if isinstance(node, list | tuple):
            return any(RayWorkflowRunner._stage_requests_gpus(v) for v in node)
        return False


def create_workflow_runner(config_path: str) -> WorkflowRunner:
    """Factory: return a :class:`RayWorkflowRunner` or :class:`WorkflowRunner`.

    Peeks at the cluster config referenced by the workflow YAML.  If the
    cluster targets the Ray Jobs backend (per :func:`is_ray_backend` — a
    ``name: ray`` backend with a resolvable dashboard URL) the factory returns
    a :class:`RayWorkflowRunner`; otherwise it returns a plain
    :class:`WorkflowRunner` (Slurm / local).

    This keeps the CLI transparent — ``nflow run-all`` works for both
    executors without the call site knowing which runner to instantiate::

        runner = create_workflow_runner(config_path)
        runner.run()
    """
    from pathlib import Path

    from omegaconf import OmegaConf

    raw = OmegaConf.load(Path(config_path))
    cluster_name = raw.get("cluster")
    if cluster_name:
        try:
            from nemo_skills.pipeline.utils.cluster import get_cluster_config

            cluster_cfg = get_cluster_config(cluster_name)
            # Use the single canonical gate so factory routing and per-stage
            # with_ray= submission agree on what counts as a Ray backend.
            if isinstance(cluster_cfg, dict) and is_ray_backend(cluster_cfg):
                return RayWorkflowRunner(config_path)
        except Exception:
            pass
    return WorkflowRunner(config_path)
