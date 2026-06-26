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
"""Regression tests for Ray/Slurm executor routing.

Two invariants that must hold so the Ray work never breaks a pure-Slurm run:

1. NeMo-RL *training* jobs (SFT, GRPO) submit with ``with_ray=True`` regardless
   of backend. On Slurm, ``with_ray=True`` is what makes nemo-skills set
   ``use_with_ray_cluster`` (the in-allocation Ray cluster that ``run_sft.py`` /
   ``run_grpo.py`` attach to). Gating SFT on ``is_ray_backend()`` set it False on
   Slurm and broke SFT-on-Slurm — this test locks it back to True.
2. ``create_workflow_runner`` selects ``RayWorkflowRunner`` using the single
   canonical ``is_ray_backend()`` gate, so factory routing and per-stage
   ``with_ray=`` submission agree (a ``name: ray`` backend with no dashboard URL
   is NOT a Ray backend and must fall back to the plain Slurm runner).
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

from nvflow.recipes.finance.stages.sft.training import SFTStage


def _install_fake_nemo_skills(monkeypatch, *, exp=None, grpo=None, cluster=None):
    """Register a fake ``nemo_skills`` package tree in ``sys.modules``.

    Lets the lazy ``from nemo_skills... import ...`` calls inside the stage
    code resolve without nemo_skills actually being installed.
    """
    ns = types.ModuleType("nemo_skills")
    pipeline = types.ModuleType("nemo_skills.pipeline")
    nemo_rl = types.ModuleType("nemo_skills.pipeline.nemo_rl")
    utils = types.ModuleType("nemo_skills.pipeline.utils")
    ns.pipeline = pipeline
    pipeline.nemo_rl = nemo_rl
    pipeline.utils = utils
    mapping = {
        "nemo_skills": ns,
        "nemo_skills.pipeline": pipeline,
        "nemo_skills.pipeline.nemo_rl": nemo_rl,
        "nemo_skills.pipeline.utils": utils,
    }
    if exp is not None:
        utils.exp = exp
        mapping["nemo_skills.pipeline.utils.exp"] = exp
    if grpo is not None:
        nemo_rl.grpo = grpo
        mapping["nemo_skills.pipeline.nemo_rl.grpo"] = grpo
    if cluster is not None:
        utils.cluster = cluster
        mapping["nemo_skills.pipeline.utils.cluster"] = cluster
    for name, mod in mapping.items():
        monkeypatch.setitem(sys.modules, name, mod)


def _submit_sft_and_capture(monkeypatch, cluster_config):
    """Drive ``SFTStage._submit_sft_job`` with stubs; return add_task kwargs."""
    captured = {}

    def fake_add_task(exp, **kwargs):
        captured.update(kwargs)
        return object()

    class _FakeExp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    exp_mod = types.ModuleType("nemo_skills.pipeline.utils.exp")
    exp_mod.add_task = fake_add_task
    exp_mod.get_exp = lambda *a, **k: _FakeExp()
    exp_mod.run_exp = lambda *a, **k: None

    grpo_mod = types.ModuleType("nemo_skills.pipeline.nemo_rl.grpo")
    grpo_mod.parse_kwargs = lambda _s: {}

    _install_fake_nemo_skills(monkeypatch, exp=exp_mod, grpo=grpo_mod)

    stage = SFTStage.__new__(SFTStage)
    # Stub the two helpers that build the command (exercised elsewhere) so the
    # test isolates the submission path.
    stage._config_shell_snippet = lambda prepared: ("snippet", "/cfg/run_config.yaml")
    stage._build_train_cmd = lambda *a, **k: "TRAIN_CMD"

    prepared = SimpleNamespace(expname="sft-exp", output_dir="/out", num_gpus=8, num_nodes=1)
    config = {"stage_kwargs": {}, "dependent_jobs": 0}

    SFTStage._submit_sft_job(stage, prepared, cluster_config, config)
    return captured


def test_sft_training_submits_with_ray_true_on_slurm(monkeypatch):
    """SFT on pure Slurm MUST submit with_ray=True (the regression guard).

    With_ray=False on Slurm means no use_with_ray_cluster -> no in-allocation
    Ray cluster -> run_sft.py has nothing to attach to -> SFT-on-Slurm fails.
    """
    slurm_cfg = {"executor": "slurm", "containers": {"nemo-rl": "img"}}
    captured = _submit_sft_and_capture(monkeypatch, slurm_cfg)
    assert captured["with_ray"] is True


def test_sft_training_submits_with_ray_true_on_ray_backend(monkeypatch):
    """SFT on a Ray Jobs backend also submits with_ray=True (matches GRPO)."""
    ray_cfg = {
        "executor": "none",
        "backend": {"name": "ray", "gpu_nemo_rl_dashboard_url": "http://h:8265"},
        "containers": {"nemo-rl": "img"},
    }
    captured = _submit_sft_and_capture(monkeypatch, ray_cfg)
    assert captured["with_ray"] is True


def test_factory_routes_by_is_ray_backend(monkeypatch, tmp_path):
    """create_workflow_runner uses is_ray_backend(), not a bare name==ray check."""
    from nvflow.core import ray_workflow_runner as rwr

    # No-op constructors isolate the routing decision from config loading.
    monkeypatch.setattr(rwr.WorkflowRunner, "__init__", lambda self, p: None)
    monkeypatch.setattr(rwr.RayWorkflowRunner, "__init__", lambda self, p: None)

    clusters = {
        "ray2c": {"backend": {"name": "ray", "gpu_nemo_rl_dashboard_url": "http://h:8265"}},
        "raynourl": {"backend": {"name": "ray"}},  # name=ray but NO dashboard URL
        "slurm": {"executor": "slurm"},
    }
    cluster_mod = types.ModuleType("nemo_skills.pipeline.utils.cluster")
    cluster_mod.get_cluster_config = lambda name: clusters[name]
    _install_fake_nemo_skills(monkeypatch, cluster=cluster_mod)

    def make(cluster_name):
        p = tmp_path / f"{cluster_name}.yaml"
        p.write_text(f"cluster: {cluster_name}\n")
        return rwr.create_workflow_runner(str(p))

    # Real Ray 2-cluster config (name=ray + dashboard URL) -> RayWorkflowRunner
    assert isinstance(make("ray2c"), rwr.RayWorkflowRunner)
    # name=ray but no URL is an incomplete config, not a Ray backend -> plain runner
    assert not isinstance(make("raynourl"), rwr.RayWorkflowRunner)
    # Slurm -> plain runner
    assert not isinstance(make("slurm"), rwr.RayWorkflowRunner)
