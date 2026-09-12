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
"""Tests that ``run_after`` names match the experiments dependency stages submit.

When a ``run_after`` name does not identify a submitted experiment, nemo-skills
logs a warning and submits the dependent job without that dependency.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest
import yaml

import nvflow.recipes  # noqa: F401  # registers recipe stages
from nvflow.core import BaseStage, StageRegistry
from nvflow.core.workflow_runner import WorkflowRunner

GRPO_WORKFLOW = (
    Path(__file__).resolve().parent.parent
    / "nvflow/recipes/finance/workflows/grpo/qwen3_30b_a3b.yaml"
)

# Snapshot at collection time: tests/test_core.py calls StageRegistry.clear().
_DISCOVERED_STAGES = {
    recipe: {workflow: dict(stages) for workflow, stages in workflows.items()}
    for recipe, workflows in StageRegistry._stages.items()
}


@pytest.fixture
def registry(monkeypatch):
    """Restore the stages registered by recipe discovery for one test."""
    stages = {
        recipe: {workflow: dict(entries) for workflow, entries in workflows.items()}
        for recipe, workflows in _DISCOVERED_STAGES.items()
    }
    monkeypatch.setattr(StageRegistry, "_stages", stages)
    return StageRegistry


def _stub_modules(monkeypatch, modules: dict[str, dict[str, Any]]) -> None:
    """Install fake modules, and their parent packages, for one test."""
    created: dict[str, types.ModuleType] = {}
    for dotted, attributes in modules.items():
        parts = dotted.split(".")
        for depth in range(1, len(parts) + 1):
            name = ".".join(parts[:depth])
            if name not in created:
                created[name] = types.ModuleType(name)
                monkeypatch.setitem(sys.modules, name, created[name])
        for attribute, value in attributes.items():
            setattr(created[dotted], attribute, value)


class _DefaultNamingStage(BaseStage):
    def execute(self, config, cluster, expname, run_after=None):
        pass


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({}, ["wf-stage"]),
        ({"environments": {}}, ["wf-stage"]),
        ({"environments": {"a": {}, "b": {}}}, ["wf-stage-a", "wf-stage-b"]),
        ({"environments": {"a": {}, "b": {}}, "_environment": []}, ["wf-stage-a", "wf-stage-b"]),
        ({"environments": {"a": {}, "b": {}}, "_environment": ["b"]}, ["wf-stage-b"]),
        (
            {"environments": {"a": {}, "b": {}}, "_environment": ["b", "a"]},
            ["wf-stage-b", "wf-stage-a"],
        ),
        ({"environments": {"a": {}}, "_environment": ["a", "other"]}, ["wf-stage-a"]),
        ({"environments": {"a": {}, "b": {}}, "_environment": "b"}, ["wf-stage-b"]),
    ],
)
def test_default_submitted_expnames(config, expected):
    assert _DefaultNamingStage.submitted_expnames(config, "wf-stage") == expected


def test_run_after_uses_names_reported_by_the_dependency(tmp_path, monkeypatch):
    import nvflow.lib.sbatch as sbatch

    calls: dict[str, Any] = {"producer_instances": 0}

    class Producer(BaseStage):
        def __init__(self):
            calls["producer_instances"] += 1

        def execute(self, config, cluster, expname, run_after=None):
            calls["execute"] = (config, expname)

        @classmethod
        def submitted_expnames(cls, config, expname):
            calls["submitted_expnames"] = (config, expname)
            return [f"{expname}-last"]

    class Consumer(BaseStage):
        def execute(self, config, cluster, expname, run_after=None):
            calls["run_after"] = run_after

    # run() patches nemo-skills' executor factory, which is not under test here.
    monkeypatch.setattr(sbatch, "apply_sbatch_args_autopatch", lambda: None)
    monkeypatch.setattr(
        StageRegistry, "_stages", {"test": {"wf": {"producer": Producer, "consumer": Consumer}}}
    )
    workflow = tmp_path / "workflow.yaml"
    workflow.write_text(
        yaml.safe_dump(
            {
                "recipe": "test",
                "workflow": {"name": "wf"},
                "cluster": "local",
                "environments": {"a": {}, "b": {}},
                "pipeline_stages": ["producer", "consumer"],
                "stages": {
                    "producer": {"environments": {"a": {}, "b": {}}},
                    "consumer": {"dependencies": ["producer"]},
                },
            }
        )
    )

    WorkflowRunner(str(workflow)).run(environment=["b"])

    assert calls["run_after"] == ["wf-producer-last"]
    assert calls["submitted_expnames"] == calls["execute"]
    # Only _run_stage instantiates the stage; dependency naming is class-level.
    assert calls["producer_instances"] == 1


@pytest.mark.parametrize(
    "environment",
    [None, ["equivalence_llm_judge"], ["equivalence_llm_judge", "mcqa"]],
    ids=["all-environments", "one-environment", "two-environments"],
)
def test_grpo_eval_waits_on_the_training_experiment(registry, monkeypatch, environment):
    """Training submits one experiment for all selected environments."""
    training_cls = registry.get("finance", "grpo", "training")
    eval_cls = registry.get("finance", "grpo", "eval")
    submitted: list[str] = []
    received: dict[str, Any] = {}

    _stub_modules(
        monkeypatch,
        {"nemo_skills.pipeline.utils.cluster": {"get_cluster_config": lambda cluster: {}}},
    )
    # Record the experiment name that _submit_grpo_job passes to get_exp().
    monkeypatch.setattr(
        training_cls,
        "_prepare_grpo_config",
        lambda self, config, cluster, expname, **kwargs: types.SimpleNamespace(expname=expname),
    )
    monkeypatch.setattr(training_cls, "_display_grpo_summary", lambda self, *args: None)
    monkeypatch.setattr(
        training_cls,
        "_submit_grpo_job",
        lambda self, prepared, *args: submitted.append(prepared.expname),
    )
    monkeypatch.setattr(eval_cls, "validate_config", lambda self, config: None)
    monkeypatch.setattr(
        eval_cls,
        "execute",
        lambda self, config, cluster, expname, run_after=None: received.update(run_after=run_after),
    )

    runner = WorkflowRunner(str(GRPO_WORKFLOW))
    stages = ["training", "eval"]
    runner._run_stage("training", environment=environment, stages_to_run=stages)
    runner._run_stage("eval", environment=environment, stages_to_run=stages)

    assert len(submitted) == 1
    assert received["run_after"] == submitted


@pytest.mark.parametrize(
    "environment",
    [None, ["equivalence_llm_judge"]],
    ids=["all-environments", "one-environment"],
)
def test_grpo_data_transformation_waits_on_validate_questions_final_jobs(
    registry, monkeypatch, environment
):
    """validate_questions chains phase 1 -> phase 2 per environment."""
    transform_cls = registry.get("finance", "grpo", "data_transformation")
    submissions: list[tuple[str, list[str]]] = []
    received: dict[str, Any] = {}

    def record(**kwargs):
        submissions.append((kwargs["expname"], list(kwargs.get("run_after") or [])))

    _stub_modules(
        monkeypatch,
        {
            "nemo_skills.pipeline.cli": {
                "generate": record,
                "run_cmd": record,
                "wrap_arguments": lambda text: text,
            }
        },
    )
    monkeypatch.setattr(transform_cls, "validate_config", lambda self, config: None)
    monkeypatch.setattr(
        transform_cls,
        "execute",
        lambda self, config, cluster, expname, run_after=None: received.update(run_after=run_after),
    )

    runner = WorkflowRunner(str(GRPO_WORKFLOW))
    stages = ["validate_questions", "data_transformation"]
    runner._run_stage("validate_questions", environment=environment, stages_to_run=stages)
    runner._run_stage("data_transformation", environment=environment, stages_to_run=stages)

    submitted = {expname for expname, _ in submissions}
    chained = {name for _, run_after in submissions for name in run_after}
    assert sorted(received["run_after"]) == sorted(submitted - chained)
