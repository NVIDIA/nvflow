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
"""``WorkflowRunner.run(stages=...)`` must submit stages in dependency order.

``run_after`` names only resolve to Slurm job handles for experiments that
were already submitted, so a stage submitted before its dependency loses
the dependency (nemo-skills logs "Experiment ... not found!" and submits
the job with no Slurm dependency).
"""

import pytest

from nvflow.core import BaseStage, StageRegistry, WorkflowRunner

RECIPE = "_test_stage_order"
WORKFLOW = "wf"

CONFIG = f"""
recipe: {RECIPE}
workflow:
  name: {WORKFLOW}
cluster: local
pipeline_stages: [prep, train, eval]
stages:
  prep: {{}}
  train:
    dependencies: [prep]
  eval:
    dependencies: [train]
"""


@pytest.fixture
def runner(tmp_path):
    submitted: list[tuple[str, list[str] | None]] = []

    def make(name):
        class _Stage(BaseStage):
            def execute(self, config, cluster, expname, run_after=None):
                submitted.append((expname, run_after))

        _Stage.__name__ = f"{name}_stage"
        return _Stage

    for name in ("prep", "train", "eval"):
        StageRegistry.register(recipe=RECIPE, workflow=WORKFLOW, stage=name)(make(name))
    cfg = tmp_path / "wf.yaml"
    cfg.write_text(CONFIG)
    try:
        yield WorkflowRunner(str(cfg)), submitted
    finally:
        StageRegistry._stages.pop(RECIPE, None)


def test_run_submits_dependency_before_dependent_regardless_of_cli_order(runner):
    wf, submitted = runner
    wf.run(stages=["train", "prep"])
    assert submitted == [("wf-prep", None), ("wf-train", ["wf-prep"])]


def test_run_matches_run_all_order_for_full_reversed_list(runner):
    wf, submitted = runner
    wf.run(stages=["eval", "train", "prep"])
    assert [name for name, _ in submitted] == ["wf-prep", "wf-train", "wf-eval"]


def test_run_submits_repeated_stage_once(runner):
    wf, submitted = runner
    wf.run(stages=["prep", "prep"])
    assert submitted == [("wf-prep", None)]


def test_run_in_pipeline_order_is_unchanged(runner):
    wf, submitted = runner
    wf.run(stages=["prep", "train"])
    assert submitted == [("wf-prep", None), ("wf-train", ["wf-prep"])]
