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
"""Labels that tie scheduler jobs to their workflow, stage, and experiment."""

from __future__ import annotations

import shlex
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from nvflow.execution import ResourceRequest, TaskPlan
from nvflow.execution.job_labels import (
    JobLabel,
    label_value,
    planned_resources,
    task_label,
    workflow_label,
)


def test_a_label_round_trips_through_a_slurm_comment_and_ray_metadata() -> None:
    label = JobLabel(workflow="demo:eval/small", stage="model-a", experiment="eval-model-a-bench")

    assert label.to_comment() == (
        "nvflow,workflow=demo:eval/small,stage=model-a,experiment=eval-model-a-bench"
    )
    assert JobLabel.from_comment(label.to_comment()) == label
    assert label.to_metadata() == {
        "nvflow_workflow": "demo:eval/small",
        "nvflow_stage": "model-a",
        "nvflow_experiment": "eval-model-a-bench",
    }
    assert JobLabel.from_metadata(label.to_metadata()) == label


def test_a_comment_needs_no_shell_quoting_whatever_the_names_contain() -> None:
    label = JobLabel(workflow="demo:my eval/run=1", stage="a,b", experiment="x'y\"z")

    comment = label.to_comment()

    assert shlex.quote(comment) == comment
    assert JobLabel.from_comment(comment) == label
    assert label.workflow == label_value("demo:my eval/run=1") == "demo:my_eval/run_1"


@pytest.mark.parametrize(
    "comment",
    ["", "(null)", "site-policy", '{"reaper": {"exempt": true}}', "nvflowish,stage=a"],
)
def test_other_comments_carry_no_label(comment: str) -> None:
    assert JobLabel.from_comment(comment) is None


def test_metadata_without_a_label_carries_none() -> None:
    assert JobLabel.from_metadata({"nvflow_kind": "controller", "nvflow_user": "alice"}) is None


def test_a_task_label_names_the_tasks_planned_experiment() -> None:
    task = TaskPlan(
        task_id="generate.merge",
        stage="generate",
        name="demo-generate-merge",
        operation="nemo_skills.run_cmd",
        parameters={"experiment_name": "demo-generate"},
    )

    assert task_label("demo:generate", task) == JobLabel(
        "demo:generate", "generate", "demo-generate"
    )
    assert task_label(None, replace(task, parameters={})) == JobLabel(
        "", "generate", "demo-generate-merge"
    )


@pytest.mark.parametrize(
    ("resources", "expected"),
    [
        ((), (None, None)),
        ((ResourceRequest(nodes=2, gpus_per_node=8),), (2, 8)),
        (
            (
                ResourceRequest(role="policy", gpus_per_node=4),
                ResourceRequest(role="judge", gpus_per_node=4),
            ),
            (2, 4),
        ),
        (
            (
                ResourceRequest(role="policy", gpus_per_node=8),
                ResourceRequest(role="judge", gpus_per_node=4),
            ),
            (2, None),
        ),
    ],
)
def test_planned_resources_count_every_resource_group(
    resources: tuple[ResourceRequest, ...],
    expected: tuple[int | None, int | None],
) -> None:
    task = TaskPlan(task_id="t.0", stage="t", name="t", operation="shell", resources=resources)

    assert planned_resources(task) == expected


def test_a_workflow_is_labeled_by_its_selector_or_by_its_path_outside_the_project(
    tmp_path: Path,
) -> None:
    workflows = tmp_path / "workflows"
    (workflows / "eval").mkdir(parents=True)
    inside = workflows / "eval" / "small.yaml"
    outside = tmp_path / "elsewhere.yaml"
    for path in (inside, outside):
        path.write_text("", encoding="utf-8")
    context = cast(Any, SimpleNamespace(identity="demo", workflows_root=workflows))

    assert workflow_label(context, inside) == "demo:eval/small"
    assert workflow_label(context, outside) == outside.resolve().as_posix()
