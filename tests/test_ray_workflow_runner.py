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
"""Regression tests for RayWorkflowRunner per-stage cluster routing.

Locks the mycluster 3-cluster GRPO fix: a multi-cluster Ray backend that defines
the per-role ``*_dashboard_url`` keys but NO generic ``backend.dashboard_url``
must still yield a resolvable generic ``dashboard_url`` (the CPU default) so the
pinned nemo-skills backend resolution does not raise "requires a dashboard URL"
before nvflow's routing runs. Per-stage routing must still override that default.

Also locks Fix A: a failed cluster-config lookup in ``_resolve_stage_cluster``
must LOG A WARNING (not silently swallow) before falling back to the workflow
cluster name.
"""

from __future__ import annotations

import sys
import types

import pytest
from omegaconf import OmegaConf

from nvflow.core import console
from nvflow.core.ray_workflow_runner import RayWorkflowRunner

# A representative mycluster 3-cluster backend: all three per-role dashboard keys
# defined, but NO generic ``dashboard_url`` (the exact shape that broke GRPO).
_MULTICLUSTER_BACKEND = {
    "name": "ray",
    "gpu_nemo_rl_dashboard_url": "http://gpu-head:8265",
    "cpu_nemo_skills_dashboard_url": "http://skills-head:8266",
    "cpu_nemo_gym_dashboard_url": "http://gym-head:8267",
}


def _cluster_config():
    return {"executor": "none", "backend": dict(_MULTICLUSTER_BACKEND)}


def _cluster_config_2c():
    """A 2-cluster Ray backend (gpu + skills, NO gym cluster)."""
    return {
        "executor": "none",
        "backend": {
            k: v for k, v in _MULTICLUSTER_BACKEND.items() if k != "cpu_nemo_gym_dashboard_url"
        },
    }


def _extserve_gym_stage(target_cluster=None):
    """External-serve collect_rollouts-like stage: nemo-gym CLI, num_gpus: 0.

    Mirrors qwen3_4b_smoke_extserve.yaml — the policy vLLM is BYO (base_url
    set), so the stage requests no GPUs of its own.
    """
    stage = {
        "container": "nemo-gym",
        "rollout": {"policy_vllm": {"num_gpus": 0, "base_url": "http://policy-host:5000/v1"}},
    }
    if target_cluster is not None:
        stage["target_cluster"] = target_cluster
    return stage


def _install_fake_get_cluster_config(monkeypatch, resolver):
    """Register a fake ``nemo_skills.pipeline.utils.cluster`` with ``resolver``."""
    ns = types.ModuleType("nemo_skills")
    pipeline = types.ModuleType("nemo_skills.pipeline")
    utils = types.ModuleType("nemo_skills.pipeline.utils")
    cluster_mod = types.ModuleType("nemo_skills.pipeline.utils.cluster")
    cluster_mod.get_cluster_config = resolver
    for name, mod in {
        "nemo_skills": ns,
        "nemo_skills.pipeline": pipeline,
        "nemo_skills.pipeline.utils": utils,
        "nemo_skills.pipeline.utils.cluster": cluster_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)


def _runner():
    """A RayWorkflowRunner without the heavy __init__ (routing is stateless)."""
    runner = RayWorkflowRunner.__new__(RayWorkflowRunner)
    runner.cluster = "mycluster-3c"
    return runner


# ---------------------------------------------------------------------------
# Fix B — generic dashboard_url default is injected for the multi-cluster shape
# ---------------------------------------------------------------------------


def test_multicluster_without_generic_url_gets_cpu_default(monkeypatch):
    """A plain CPU stage on a multi-cluster backend (no generic dashboard_url)
    resolves to the CPU nemo-skills cluster — the seamless default."""
    _install_fake_get_cluster_config(monkeypatch, lambda name: _cluster_config())
    resolved = _runner()._resolve_stage_cluster({})  # no gpu, no target
    assert isinstance(resolved, dict), "must return a routed config dict, not the bare name"
    assert resolved["backend"]["dashboard_url"] == "http://skills-head:8266"


def test_route_cluster_default_is_resolvable_pure():
    """Pure _route_cluster: a no-GPU/no-target stage yields a non-empty
    generic dashboard_url even though the config defines none."""
    cfg = _cluster_config()
    assert "dashboard_url" not in cfg["backend"]  # precondition: none defined
    routed = RayWorkflowRunner._route_cluster(cfg, {}, "mycluster-3c")
    assert routed["backend"]["dashboard_url"]  # truthy / resolvable


def test_multicluster_routing_preserves_opt_in_working_dir():
    """Per-stage URL routing must not drop the install-free code archive."""

    cfg = _cluster_config()
    cfg["backend"]["working_dir"] = "/opt/nvflow-ray-code.zip"

    routed = RayWorkflowRunner._route_cluster(cfg, {"total_gpus": 8}, "mycluster-3c")

    assert routed["backend"]["dashboard_url"] == "http://gpu-head:8265"
    assert routed["backend"]["working_dir"] == "/opt/nvflow-ray-code.zip"


def test_default_dashboard_url_preference_order():
    """CPU-first preference: skills -> gym -> rl."""
    d = RayWorkflowRunner._default_dashboard_url
    assert d(dict(_MULTICLUSTER_BACKEND)) == "http://skills-head:8266"
    no_skills = {
        k: v for k, v in _MULTICLUSTER_BACKEND.items() if k != "cpu_nemo_skills_dashboard_url"
    }
    assert d(no_skills) == "http://gym-head:8267"
    only_gpu = {"name": "ray", "gpu_nemo_rl_dashboard_url": "http://gpu-head:8265"}
    assert d(only_gpu) == "http://gpu-head:8265"
    assert d({"name": "ray"}) is None  # single-cluster / non-multi -> no default


# ---------------------------------------------------------------------------
# Fix B — per-stage routing still OVERRIDES the default
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stage_config, expected",
    [
        ({"total_gpus": 8}, "http://gpu-head:8265"),  # GPU stage -> nemo-rl
        ({"num_gpus": 4}, "http://gpu-head:8265"),  # nested-style GPU signal
        ({"container": "nemo-gym"}, "http://gym-head:8267"),  # CPU gym-client stage
        ({}, "http://skills-head:8266"),  # plain CPU stage -> nemo-skills
        ({"target_cluster": "gpu"}, "http://gpu-head:8265"),  # explicit override
        ({"target_cluster": "gym"}, "http://gym-head:8267"),
        ({"target_cluster": "cpu"}, "http://skills-head:8266"),
    ],
)
def test_per_stage_routing_overrides_default(stage_config, expected):
    routed = RayWorkflowRunner._route_cluster(_cluster_config(), stage_config, "mycluster-3c")
    assert routed["backend"]["dashboard_url"] == expected


def test_gpu_request_beats_gym_container():
    """A nemo-gym stage that ALSO requests GPUs is not a CPU gym-client stage."""
    routed = RayWorkflowRunner._route_cluster(
        _cluster_config(), {"container": "nemo-gym", "total_gpus": 8}, "mycluster-3c"
    )
    assert routed["backend"]["dashboard_url"] == "http://gpu-head:8265"


# ---------------------------------------------------------------------------
# Fix C — a nemo-gym-container CPU stage force-routes to gym, TAKING PRECEDENCE
# over an explicit target_cluster: cpu|gpu (the confirmed collect_rollouts
# mis-route: base.yaml sets target_cluster: gpu + container: nemo-gym).
# ---------------------------------------------------------------------------


def test_gym_container_with_explicit_cpu_target_routes_to_gym():
    """target_cluster: cpu on a nemo-gym stage must NOT land on skills (:8266)."""
    routed = RayWorkflowRunner._route_cluster(
        _cluster_config(), _extserve_gym_stage("cpu"), "mycluster-3c"
    )
    assert routed["backend"]["dashboard_url"] == "http://gym-head:8267"


def test_gym_container_with_explicit_gpu_target_routes_to_gym():
    """target_cluster: gpu (base.yaml default) must NOT land on nemo-rl (:8265)."""
    routed = RayWorkflowRunner._route_cluster(
        _cluster_config(), _extserve_gym_stage("gpu"), "mycluster-3c"
    )
    assert routed["backend"]["dashboard_url"] == "http://gym-head:8267"


def test_gym_container_without_target_routes_to_gym():
    """A nemo-gym CPU stage with no target_cluster still infers gym."""
    routed = RayWorkflowRunner._route_cluster(
        _cluster_config(), _extserve_gym_stage(None), "mycluster-3c"
    )
    assert routed["backend"]["dashboard_url"] == "http://gym-head:8267"


def test_internal_serve_gym_stage_requesting_gpus_is_not_redirected():
    """A gym stage that genuinely requests its OWN GPUs is left on gpu.

    base.yaml's internal-serve collect_rollouts launches an in-gym policy vLLM
    (rollout.policy_vllm.num_gpus > 0) and sets target_cluster: gpu. That case
    can't run on the CPU gym cluster, so the guard (gated on
    ``not _stage_requests_gpus``) must NOT redirect it — it stays on gpu.
    """
    stage = {
        "container": "nemo-gym",
        "target_cluster": "gpu",
        "rollout": {"policy_vllm": {"num_gpus": 2}},
    }
    routed = RayWorkflowRunner._route_cluster(_cluster_config(), stage, "mycluster-3c")
    assert routed["backend"]["dashboard_url"] == "http://gpu-head:8265"


def test_override_emits_warning(monkeypatch):
    """Overriding target_cluster: gpu -> gym logs a warning naming the redirect."""
    calls: list[str] = []
    monkeypatch.setattr(console, "warning", calls.append)

    RayWorkflowRunner._route_cluster(_cluster_config(), _extserve_gym_stage("gpu"), "mycluster-3c")
    assert len(calls) == 1
    assert "gym" in calls[0].lower()


def test_no_warning_when_target_already_gym(monkeypatch):
    """No redirect warning when the stage already targets gym (nothing overridden)."""
    calls: list[str] = []
    monkeypatch.setattr(console, "warning", calls.append)

    RayWorkflowRunner._route_cluster(_cluster_config(), _extserve_gym_stage("gym"), "mycluster-3c")
    assert calls == []


# ---------------------------------------------------------------------------
# 2-cluster (no gym URL): gym routing degrades gracefully to skills.
# ---------------------------------------------------------------------------


def test_two_cluster_gym_target_falls_back_to_skills():
    """With no gym cluster configured, target_cluster: gym falls back to skills.

    2-cluster setups keep their existing behaviour — the stage then fails
    loudly at runtime if the image lacks the CLI (a clearer signal than a
    routing KeyError).
    """
    routed = RayWorkflowRunner._route_cluster(
        _cluster_config_2c(), _extserve_gym_stage("gym"), "mycluster-2c"
    )
    assert routed["backend"]["dashboard_url"] == "http://skills-head:8266"


def test_two_cluster_gym_container_no_target_falls_back_to_skills():
    """A nemo-gym CPU stage in a 2-cluster setup still lands on skills."""
    routed = RayWorkflowRunner._route_cluster(
        _cluster_config_2c(), _extserve_gym_stage(None), "mycluster-2c"
    )
    assert routed["backend"]["dashboard_url"] == "http://skills-head:8266"


# ---------------------------------------------------------------------------
# Single-cluster / non-Ray configs are left untouched (no routing)
# ---------------------------------------------------------------------------


def test_single_cluster_ray_untouched():
    """A single-cluster Ray backend (plain dashboard_url only) is not rerouted."""
    cfg = {"executor": "none", "backend": {"name": "ray", "dashboard_url": "http://head:8265"}}
    assert RayWorkflowRunner._route_cluster(cfg, {}, "single") == "single"


def test_non_ray_backend_untouched():
    cfg = {"executor": "slurm"}
    assert RayWorkflowRunner._route_cluster(cfg, {}, "slurm") == "slurm"


# ---------------------------------------------------------------------------
# Fix A — failed cluster-config lookup logs a warning (no silent swallow)
# ---------------------------------------------------------------------------


def test_lookup_failure_logs_warning_and_falls_back(monkeypatch):
    def _boom(name):
        raise RuntimeError("config dir not mounted")

    _install_fake_get_cluster_config(monkeypatch, _boom)

    warnings: list[str] = []
    monkeypatch.setattr(console, "warning", warnings.append)

    runner = _runner()
    result = runner._resolve_stage_cluster({})

    assert result == "mycluster-3c"  # degrades to the workflow cluster name
    assert len(warnings) == 1, "the failure must NOT be swallowed silently"
    msg = warnings[0]
    assert "mycluster-3c" in msg
    assert "dashboard_url" in msg  # points the user at the real remedy


# ---------------------------------------------------------------------------
# Fix #1 — factory detects Ray for _base_ overlay recipes (cluster inherited)
#
# An overlay recipe (``_base_: base.yaml``) that does NOT repeat ``cluster:``
# inherits it from the base. A raw ``OmegaConf.load`` of the overlay reports
# ``cluster=None`` and the Ray detection was silently skipped, so the factory
# returned the base WorkflowRunner (per-stage routing DISABLED) — the confirmed
# mycluster collect_rollouts mis-route. The factory must resolve ``_base_`` first.
# ---------------------------------------------------------------------------


def _write_base_overlay(tmp_path, base_cluster: str = "my_cluster"):
    """Write a minimal base recipe (declares cluster:) + an overlay that
    inherits it via _base_ and does NOT repeat cluster:.  Returns the overlay."""
    (tmp_path / "base.yaml").write_text(
        f"recipe: finance\nworkflow:\n  name: smoke\n  type: grpo\ncluster: {base_cluster}\n"
    )
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text("_base_: base.yaml\nstages:\n  collect_rollouts:\n    target_cluster: gym\n")
    return overlay


def test_factory_overlay_inherits_cluster_returns_ray_runner(tmp_path, monkeypatch):
    """An overlay whose cluster: comes ONLY from _base_ is still detected as Ray."""
    ray_backend = {
        "executor": "none",
        "nvflow_root": "/lustre/test/nvflow",
        "backend": dict(_MULTICLUSTER_BACKEND),
    }
    config_dir = tmp_path / "cluster_configs"
    config_dir.mkdir()
    (config_dir / "my_cluster.yaml").write_text(OmegaConf.to_yaml(ray_backend))
    monkeypatch.setenv("NEMO_SKILLS_CONFIG_DIR", str(config_dir))
    from nvflow.core.ray_workflow_runner import create_workflow_runner

    runner = create_workflow_runner(str(_write_base_overlay(tmp_path)))
    assert isinstance(runner, RayWorkflowRunner), (
        "factory must resolve _base_ to see the inherited cluster and detect Ray"
    )


def test_factory_non_ray_cluster_returns_base_runner(tmp_path, monkeypatch):
    """A non-Ray cluster still returns the plain WorkflowRunner (not the subclass)."""
    slurm_backend = {"executor": "slurm", "backend": {"name": "slurm"}}
    config_dir = tmp_path / "cluster_configs"
    config_dir.mkdir()
    (config_dir / "my_cluster.yaml").write_text(OmegaConf.to_yaml(slurm_backend))
    monkeypatch.setenv("NEMO_SKILLS_CONFIG_DIR", str(config_dir))
    from nvflow.core.ray_workflow_runner import create_workflow_runner
    from nvflow.core.workflow_runner import WorkflowRunner

    runner = create_workflow_runner(str(_write_base_overlay(tmp_path)))
    assert isinstance(runner, WorkflowRunner)
    assert not isinstance(runner, RayWorkflowRunner)
