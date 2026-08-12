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

    def _get_run_after_names(self, dependencies, environment):
        """Ray backend: drop cross-stage Slurm-style dependency wiring.

        The base :meth:`WorkflowRunner._get_run_after_names` builds
        ``run_after`` experiment names so nemo-run can chain one stage's
        Slurm job to the next via ``--dependency=afterok`` -- an *async*
        submit-all-then-exit model that needs explicit ordering.

        The Ray Jobs backend is not async: nvflow runs stages sequentially
        in-process and each ``stage.execute`` blocks until its Ray jobs reach
        a terminal state (the ``RayBackend`` ready-queue polls every job to
        completion) before the next stage is submitted.  A dependency stage's
        output (e.g. ``prepare_data``'s benchmark datasets on shared storage)
        is therefore already present before any dependent stage starts, so
        cross-stage ordering is guaranteed without a ``run_after`` handle.

        Worse, the Slurm-style names are *unresolvable* on Ray: each stage
        submits under its own ``expname`` (a distinct nemo-run experiment),
        and in a 2-cluster setup ``prepare_data`` runs on the CPU cluster
        while eval runs on the GPU cluster -- two separate Ray job registries.
        The dependent job then declares a handle (``eval-prepare_data``) that
        lives in neither the current experiment nor the current cluster and
        fails with a "dependency not found" error.  Returning ``None`` drops
        the redundant, unresolvable cross-stage handle.

        Intra-stage dependencies (e.g. checkpoint-conversion job -> eval job)
        are wired *inside* a single ``stage.execute`` from the same submission
        via nemo-skills' internal ``job_name_to_handle`` map and are
        unaffected by this override.
        """
        return None

    def _resolve_stage_cluster(self, stage_config: dict):
        """Route each stage to its GPU/CPU/gym Ray cluster.

        Returns a cluster-config copy whose generic ``backend.dashboard_url``
        (the field nemo-skills reads) points at the resolved cluster; falls back
        to the workflow cluster name for single-cluster or non-Ray configs.
        """
        try:
            from nemo_skills.pipeline.utils.cluster import get_cluster_config

            cluster_config = get_cluster_config(self.cluster)
        except Exception as exc:
            # Log the fallback -- a silent skip here surfaced as a cryptic
            # "requires a dashboard URL" deep in the nemo-skills backend.
            from nvflow.core.console import warning

            warning(
                f"Ray per-stage cluster routing skipped: could not load cluster "
                f"config for '{self.cluster}' ({type(exc).__name__}: {exc}). "
                f"Add 'backend.dashboard_url' if this is a multi-cluster Ray backend."
            )
            return self.cluster
        resolved = self._route_cluster(cluster_config, stage_config, self.cluster)
        if resolved is not self.cluster:
            from nvflow.core.console import info

            info("Routing stage to its resolved Ray cluster")
        return resolved

    @staticmethod
    def _default_dashboard_url(backend: dict):
        """Generic ``dashboard_url`` default for a multi-cluster Ray backend.

        nemo-skills reads only the generic ``dashboard_url``; when a config
        defines just the per-role keys, return a CPU-first default (skills ->
        gym -> rl) so backend resolution doesn't raise "requires a dashboard
        URL". Returns ``None`` for single-cluster / non-Ray backends.
        """
        if not isinstance(backend, dict):
            return None
        return (
            backend.get("cpu_nemo_skills_dashboard_url")
            or backend.get("cpu_nemo_gym_dashboard_url")
            or backend.get("gpu_nemo_rl_dashboard_url")
        )

    @staticmethod
    def _route_cluster(cluster_config, stage_config: dict, cluster_name):
        """Pick the GPU/CPU/gym cluster for a stage.  Pure (no I/O).

        Returns a cluster-config copy whose generic ``backend.dashboard_url`` is
        the resolved per-role URL. ``target_cluster: cpu|gpu`` overrides the
        num_gpus inference, but a ``container: nemo-gym`` CPU stage is forced to
        the gym cluster regardless (only that image ships the ``gym`` CLI). For a
        multi-cluster backend with no generic ``dashboard_url``, fills a default
        (see :meth:`_default_dashboard_url`). Unchanged for Slurm/single-cluster.
        """
        backend = cluster_config.get("backend") if isinstance(cluster_config, dict) else None
        if not isinstance(backend, dict):
            return cluster_name
        gpu_url = backend.get("gpu_nemo_rl_dashboard_url")
        cpu_url = backend.get("cpu_nemo_skills_dashboard_url")
        gym_url = backend.get("cpu_nemo_gym_dashboard_url")
        # Any per-role key => multi-cluster; none => single-cluster / non-Ray.
        if not gpu_url and not cpu_url and not gym_url:
            return cluster_name
        target = stage_config.get("target_cluster")
        # A container: nemo-gym stage runs the gym CLI (baked only in the gym
        # image), so force it onto the gym cluster even over an explicit gpu/cpu
        # target -- that target describes server placement, not the client's.
        # Gated on no own-GPU request + a configured gym_url.
        is_cpu_gym_stage = stage_config.get(
            "container"
        ) == "nemo-gym" and not RayWorkflowRunner._stage_requests_gpus(stage_config)
        if is_cpu_gym_stage and gym_url:
            if target in ("cpu", "gpu"):
                from nvflow.core.console import warning

                warning(
                    f"Stage sets target_cluster: {target} but runs the nemo-gym "
                    "CLI (container: nemo-gym); redirecting to the gym cluster "
                    "(cpu_nemo_gym_dashboard_url) — only the nemo-gym image "
                    "ships the `gym` CLI."
                )
            target = "gym"
        elif target not in ("cpu", "gpu", "gym"):
            target = "gpu" if RayWorkflowRunner._stage_requests_gpus(stage_config) else "cpu"
        if target == "gym":
            # Fall back to CPU when no gym cluster is configured (2-cluster).
            chosen = gym_url or cpu_url or gpu_url
        elif target == "cpu":
            chosen = cpu_url or RayWorkflowRunner._default_dashboard_url(backend)
        else:
            chosen = gpu_url or cpu_url or gym_url
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

    The cluster name is resolved honouring ``_base_`` inheritance: a raw
    ``OmegaConf.load`` misses ``cluster:`` inherited by an overlay recipe, which
    would silently fall back to the base (no-routing) ``WorkflowRunner``. So we
    build the base runner first (its ``__init__`` merges ``_base_`` and exposes
    ``base.cluster``), then upgrade to ``RayWorkflowRunner`` iff it's a Ray
    backend; otherwise return the already-built base (no wasted re-parse).
    """
    base = WorkflowRunner(config_path)
    try:
        # WorkflowRunner already loaded the small routing subset directly from
        # YAML while injecting path roots.  Reuse it here: importing
        # nemo_skills.pipeline merely to detect a default Slurm backend adds
        # seconds to `validate`/`list-stages` and repeats the same lookup.
        cluster_cfg = getattr(base, "_cluster_config", {})
        cluster_error = getattr(base, "_cluster_config_error", None)
        if cluster_error is not None:
            raise cluster_error
        # Use the single canonical gate so factory routing and per-stage
        # with_ray= submission agree on what counts as a Ray backend.
        if isinstance(cluster_cfg, dict) and is_ray_backend(cluster_cfg):
            return RayWorkflowRunner(config_path)
    except Exception as exc:
        # Never silently downgrade -- log, then keep the safe base-runner fallback.
        from nvflow.core.console import warning

        warning(
            f"Ray backend detection failed for cluster "
            f"'{getattr(base, 'cluster', '?')}' ({type(exc).__name__}: {exc}); "
            f"falling back to the base WorkflowRunner (per-stage Ray routing disabled)."
        )
    return base
