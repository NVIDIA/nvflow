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

import sys
from unittest.mock import MagicMock

import pytest

from nvflow.core import BaseStage, StageRegistry
from nvflow.core.ray_workflow_runner import RayWorkflowRunner
from nvflow.core.workflow_runner import WorkflowRunner


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
    assert RayWorkflowRunner._route_cluster({"executor": "slurm"}, {}, "myslurm") == "myslurm"


def test_ray_runner_drops_cross_stage_run_after():
    """Ray backend suppresses Slurm-style cross-stage ``run_after`` handles.

    On Ray, stages run sequentially and each blocks to completion, so
    ``prepare_data`` finishes before a dependent eval stage submits.  The
    Slurm-style name ``eval-prepare_data`` references a different nemo-run
    experiment / Ray cluster registry and is unresolvable on the Ray Jobs
    backend, so the override must return ``None`` regardless of the deps.
    """
    # Bypass __init__ (needs a real config on disk); we only exercise the
    # override, which is pure and self-independent.
    runner = object.__new__(RayWorkflowRunner)
    assert runner._get_run_after_names(["prepare_data"], None) is None
    assert runner._get_run_after_names(["prepare_data"], ["secque"]) is None
    assert runner._get_run_after_names([], None) is None


@pytest.mark.parametrize(
    ("executor", "expected_header", "expected_verb", "expects_slurm_note"),
    [
        (None, "✅ Workflow Submitted", "Submitted", True),
        ("none", "✅ Workflow Complete!", "Completed", False),
        ("local", "✅ Workflow Complete!", "Completed", False),
        ("slurm", "✅ Workflow Submitted", "Submitted", True),
    ],
)
def test_workflow_footer_reflects_executor_semantics(
    monkeypatch, executor, expected_header, expected_verb, expects_slurm_note
):
    """Synchronous Ray success must not be described as a Slurm submission."""
    runner = object.__new__(WorkflowRunner)
    runner.cluster = "test-cluster"
    monkeypatch.setattr(runner, "_executor_of", lambda cluster: executor)

    calls = []
    module = sys.modules[WorkflowRunner.__module__]
    monkeypatch.setattr(module, "header", lambda message: calls.append(("header", message)))
    monkeypatch.setattr(module, "success", lambda message: calls.append(("success", message)))
    monkeypatch.setattr(module, "detail", lambda label, message: calls.append((label, message)))

    runner._report_workflow_completion(["hello_world"])

    assert ("header", expected_header) in calls
    assert ("success", f"{expected_verb} 1 stage(s): hello_world") in calls
    assert (("Note", "Stages run as Slurm jobs -- track them with squeue") in calls) is (
        expects_slurm_note
    )


# ---------------------------------------------------------------------------
# Stage task-failure propagation (WorkflowRunner._verify_stage_tasks)
#
# Regression guard: a stage whose underlying nemo-run task crashes
# (ModuleNotFoundError, non-zero Hydra job, ...) reaches a terminal FAILED
# state, but nemo-run's exp.run(detach=False) does NOT raise -- so without an
# explicit status check the runner printed "Stage completed" / "Workflow
# Complete!" on a run that actually failed.  These tests pin the check.
# ---------------------------------------------------------------------------


class _State:
    """Minimal stand-in for a torchx AppState (has an upper-case ``.name``)."""

    def __init__(self, name: str):
        self.name = name


def _runner_with_status(monkeypatch, status_dict, executor="none"):
    """Build a bare WorkflowRunner whose experiment reports ``status_dict``."""
    runner = object.__new__(WorkflowRunner)
    monkeypatch.setattr(runner, "_executor_of", lambda cluster: executor)

    exp = MagicMock()
    exp.status.return_value = status_dict
    ctx = MagicMock()
    ctx.__enter__.return_value = exp
    fake_run = MagicMock()
    fake_run.Experiment.from_title.return_value = ctx
    monkeypatch.setitem(sys.modules, "nemo_run", fake_run)
    return runner


def test_verify_stage_tasks_raises_on_failed(monkeypatch):
    """A FAILED task makes the stage fail (the false-success bug)."""
    runner = _runner_with_status(monkeypatch, {"create_seed_data": {"status": _State("FAILED")}})
    with pytest.raises(RuntimeError, match="did not complete successfully"):
        runner._verify_stage_tasks("create_seed_data", "sdg-create_seed_data", {"executor": "none"})


def test_verify_stage_tasks_raises_on_cancelled(monkeypatch):
    """A CANCELLED task is also treated as a failure."""
    runner = _runner_with_status(monkeypatch, {"gen": {"status": _State("CANCELLED")}})
    with pytest.raises(RuntimeError, match="gen"):
        runner._verify_stage_tasks("filter_answers", "sdg-filter_answers", {"executor": "none"})


def test_verify_stage_tasks_ok_on_succeeded(monkeypatch):
    """A SUCCEEDED task does not raise."""
    runner = _runner_with_status(monkeypatch, {"gen": {"status": _State("SUCCEEDED")}})
    runner._verify_stage_tasks("filter_answers", "sdg-filter_answers", {"executor": "none"})


def test_verify_stage_tasks_ok_on_skip_no_tasks(monkeypatch):
    """A legitimate skip (skip_filled / no data -> no tasks) is not a failure.

    nemo-skills returns early when there is nothing to generate, so the
    experiment has no failed task; the empty/all-succeeded status must be
    treated as success, not misreported as a false failure.
    """
    runner = _runner_with_status(monkeypatch, {})
    runner._verify_stage_tasks("filter_answers", "sdg-filter_answers", {"executor": "none"})


def test_verify_stage_tasks_skips_slurm(monkeypatch):
    """Slurm submits async: tasks are not terminal here, so the check is a no-op.

    Even if a status lookup *would* show FAILED, the executor gate must skip it
    so the Slurm submit-all-then-exit path stays byte-identical (failure
    propagation on Slurm is handled by ``--dependency=afterok``).
    """
    runner = _runner_with_status(
        monkeypatch, {"gen": {"status": _State("FAILED")}}, executor="slurm"
    )
    # Must NOT raise (gated out before any status lookup).
    runner._verify_stage_tasks("sft", "training_sft-sft", "myslurm")
    # And it must not even attempt to reconstruct the experiment on Slurm.
    assert sys.modules["nemo_run"].Experiment.from_title.call_count == 0


def test_verify_stage_tasks_best_effort_on_missing_exp(monkeypatch):
    """If the experiment cannot be located, do not raise (never a false fail)."""
    runner = object.__new__(WorkflowRunner)
    monkeypatch.setattr(runner, "_executor_of", lambda cluster: "none")
    fake_run = MagicMock()
    fake_run.Experiment.from_title.side_effect = FileNotFoundError("no exp")
    monkeypatch.setitem(sys.modules, "nemo_run", fake_run)
    runner._verify_stage_tasks("gen", "sdg-gen", {"executor": "none"})


def test_executor_of_reads_dict_and_defaults_none():
    """Executor resolves from a config dict; unknown shapes return None."""
    runner = object.__new__(WorkflowRunner)
    assert runner._executor_of({"executor": "none"}) == "none"
    assert runner._executor_of({}) is None


def test_task_state_name_handles_enum_and_string():
    """State name normalises both an AppState-like enum and a plain string."""
    assert WorkflowRunner._task_state_name({"status": _State("FAILED")}) == "FAILED"
    assert WorkflowRunner._task_state_name({"status": "failed"}) == "FAILED"
