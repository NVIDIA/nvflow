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
"""Tests for core functionality."""

import pytest

from nvflow.core import BaseStage, StageRegistry
from nvflow.core.ray_workflow_runner import RayWorkflowRunner


def test_stage_registry_hierarchical():
    """Test hierarchical stage registration and retrieval."""

    # Clear registry for testing
    StageRegistry.clear()

    # Register a test stage
    @StageRegistry.register(recipe="test", workflow="example", stage="stage1")
    class TestStage(BaseStage):
        workflow = "example"

        def execute(self, config, cluster, expname, run_after=None):
            pass

    # Check stage is registered
    assert StageRegistry.has("test", "example", "stage1")

    # Retrieve stage
    stage_class = StageRegistry.get("test", "example", "stage1")
    assert stage_class == TestStage

    # List stages
    stages = StageRegistry.list_stages("test", "example")
    assert "stage1" in stages

    # List all stages
    all_stages = StageRegistry.list_all_stages()
    assert ("test", "example", "stage1") in all_stages


def test_stage_registry_multiple_levels():
    """Test registry with multiple recipes and workflows."""

    # Clear registry
    StageRegistry.clear()

    # Register stages in different recipes/workflows
    @StageRegistry.register(recipe="finance", workflow="training", stage="sft")
    class FinanceSFTStage(BaseStage):
        def execute(self, config, cluster, expname, run_after=None):
            pass

    @StageRegistry.register(recipe="finance", workflow="sdg", stage="generate")
    class FinanceSDGStage(BaseStage):
        def execute(self, config, cluster, expname, run_after=None):
            pass

    @StageRegistry.register(recipe="retail", workflow="training", stage="sft")
    class RetailSFTStage(BaseStage):
        def execute(self, config, cluster, expname, run_after=None):
            pass

    # Check recipes
    recipes = StageRegistry.list_recipes()
    assert "finance" in recipes
    assert "retail" in recipes

    # Check workflows
    finance_workflows = StageRegistry.list_workflows("finance")
    assert "training" in finance_workflows
    assert "sdg" in finance_workflows

    # Check stages
    assert StageRegistry.get("finance", "training", "sft") == FinanceSFTStage
    assert StageRegistry.get("finance", "sdg", "generate") == FinanceSDGStage
    assert StageRegistry.get("retail", "training", "sft") == RetailSFTStage

    # Verify same stage name can exist in different workflows
    assert StageRegistry.has("finance", "training", "sft")
    assert StageRegistry.has("retail", "training", "sft")


def test_stage_registry_error():
    """Test stage registry error handling."""

    # Clear registry
    StageRegistry.clear()

    # Try to get non-existent recipe
    with pytest.raises(KeyError):
        StageRegistry.get("nonexistent", "workflow", "stage")

    # Try to get non-existent workflow
    @StageRegistry.register(recipe="test", workflow="example", stage="stage1")
    class TestStage(BaseStage):
        def execute(self, config, cluster, expname, run_after=None):
            pass

    with pytest.raises(KeyError):
        StageRegistry.get("test", "nonexistent", "stage")

    # Try to get non-existent stage
    with pytest.raises(KeyError):
        StageRegistry.get("test", "example", "nonexistent")


def test_stage_registry_duplicate():
    """Test that duplicate registration raises error."""

    # Clear registry
    StageRegistry.clear()

    # Register first stage
    @StageRegistry.register(recipe="test", workflow="example", stage="duplicate")
    class TestStage1(BaseStage):
        def execute(self, config, cluster, expname, run_after=None):
            pass

    # Try to register with same path
    with pytest.raises(ValueError):

        @StageRegistry.register(recipe="test", workflow="example", stage="duplicate")
        class TestStage2(BaseStage):
            def execute(self, config, cluster, expname, run_after=None):
                pass


def test_stage_validation():
    """Test stage config validation."""

    @StageRegistry.register(recipe="test", workflow="example", stage="validation")
    class ValidationStage(BaseStage):
        workflow = "example"

        def execute(self, config, cluster, expname, run_after=None):
            pass

        def validate_config(self, config):
            if "required_field" not in config:
                raise ValueError("required_field is required")

    stage = ValidationStage()

    # Valid config
    stage.validate_config({"required_field": "value"})

    # Invalid config
    with pytest.raises(ValueError):
        stage.validate_config({})


# Two-dashboard Ray config used by the per-stage routing tests below.
_RAY_2URL = {
    "backend": {
        "name": "ray",
        "gpu_nemo_rl_dashboard_url": "http://gpu-head:8265",
        "cpu_nemo_skills_dashboard_url": "http://cpu-head:8265",
    }
}


def test_stage_requests_gpus():
    """A positive top-level GPU count marks a stage as GPU-class."""
    assert RayWorkflowRunner._stage_requests_gpus({"total_gpus": 8})
    assert RayWorkflowRunner._stage_requests_gpus({"num_gpus": 1})
    assert RayWorkflowRunner._stage_requests_gpus({"gpus": 4})
    assert not RayWorkflowRunner._stage_requests_gpus({"num_gpus": 0})
    assert not RayWorkflowRunner._stage_requests_gpus({"dependencies": []})


def test_stage_requests_gpus_nested():
    """A positive GPU count nested under a server is detected; 0 is not."""
    # e.g. rollout.policy_vllm.num_gpus
    assert RayWorkflowRunner._stage_requests_gpus({"rollout": {"policy_vllm": {"num_gpus": 2}}})
    # nested inside a list of dicts
    assert RayWorkflowRunner._stage_requests_gpus({"servers": [{"num_gpus": 0}, {"num_gpus": 1}]})
    # nested zero placeholders (external endpoint) do not count
    assert not RayWorkflowRunner._stage_requests_gpus({"judge_vllm": {"num_gpus": 0}})
    assert not RayWorkflowRunner._stage_requests_gpus({"benchmarks": {"secque": {"seeds": 5}}})


def test_stage_requests_gpus_ignores_embedded_environments():
    """Regression (NV-1): the ``environments`` block embedded in every stage
    (``environments: ${environments}``) carries other environments' server GPU
    counts; walking it would misroute a CPU stage to the GPU cluster.  Only the
    stage's OWN resource request counts."""
    # A pure-CPU stage (e.g. validate_questions / train_validation_split) embeds
    # the full environments dict, which includes a GPU-bearing policy/judge
    # server -- this must NOT classify the stage as GPU.
    cpu_stage = {
        "dependencies": [],
        "environments": {"finance_sec_search": {"policy_vllm": {"num_gpus": 4}}},
        "_environment": "equivalence_llm_judge",
    }
    assert not RayWorkflowRunner._stage_requests_gpus(cpu_stage)
    # The stage's OWN nested server GPU request is still detected even when an
    # environments block is present alongside it.
    assert RayWorkflowRunner._stage_requests_gpus(
        {"environments": {"x": {"num_gpus": 4}}, "rollout": {"policy_vllm": {"num_gpus": 2}}}
    )


def test_route_cluster_gpu_stage_uses_gpu_dashboard():
    """GPU stage -> config copy whose generic dashboard_url is the GPU dashboard."""
    out = RayWorkflowRunner._route_cluster(_RAY_2URL, {"total_gpus": 8}, "ray")
    assert isinstance(out, dict)
    assert out["backend"]["dashboard_url"] == "http://gpu-head:8265"


def test_route_cluster_cpu_stage_swaps_to_cpu_dashboard():
    """CPU stage (no GPU field) -> config copy on the CPU dashboard."""
    out = RayWorkflowRunner._route_cluster(_RAY_2URL, {"dependencies": []}, "ray")
    assert isinstance(out, dict)
    assert out["backend"]["dashboard_url"] == "http://cpu-head:8265"
    assert out["backend"]["name"] == "ray"
    # input config is not mutated
    assert _RAY_2URL["backend"]["gpu_nemo_rl_dashboard_url"] == "http://gpu-head:8265"
    assert "dashboard_url" not in _RAY_2URL["backend"]


def test_route_cluster_explicit_override_wins():
    """target_cluster overrides the num_gpus inference (e.g. eval)."""
    out = RayWorkflowRunner._route_cluster(_RAY_2URL, {"gpus": 4, "target_cluster": "cpu"}, "ray")
    assert isinstance(out, dict)
    assert out["backend"]["dashboard_url"] == "http://cpu-head:8265"
    gpu_out = RayWorkflowRunner._route_cluster(_RAY_2URL, {"target_cluster": "gpu"}, "ray")
    assert gpu_out["backend"]["dashboard_url"] == "http://gpu-head:8265"


def test_route_cluster_override_beats_nested_gpus():
    """eval-style stage: target_cluster=cpu wins over a nested GPU server."""
    eval_like = {"target_cluster": "cpu", "judge_vllm": {"num_gpus": 4}}
    out = RayWorkflowRunner._route_cluster(_RAY_2URL, eval_like, "ray")
    assert isinstance(out, dict)
    assert out["backend"]["dashboard_url"] == "http://cpu-head:8265"


def test_route_cluster_nested_gpus_no_override_routes_gpu():
    """A nested-only num_gpus with no override infers GPU -> GPU dashboard."""
    nested = {"rollout": {"policy_vllm": {"num_gpus": 2}}}
    out = RayWorkflowRunner._route_cluster(_RAY_2URL, nested, "ray")
    assert out["backend"]["dashboard_url"] == "http://gpu-head:8265"


def test_route_cluster_noop_without_cpu_dashboard():
    """Single-URL Ray and Slurm are unchanged (no routing)."""
    single = {"backend": {"name": "ray", "dashboard_url": "http://gpu-head:8265"}}
    assert RayWorkflowRunner._route_cluster(single, {}, "ray") == "ray"
    assert RayWorkflowRunner._route_cluster({"executor": "slurm"}, {}, "draco") == "draco"
