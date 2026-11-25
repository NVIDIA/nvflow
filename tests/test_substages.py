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
"""A stage with substages runs ordinary stages in order, in its container."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from nvflow.execution import (
    PlannedExecutionStage,
    PlanningContext,
    StagePlan,
    TargetBinding,
    plan_run_cmd,
)
from nvflow.execution.contracts import SLURM_GROUP_JOBS
from nvflow.stages.substages import SubstagesStage


class _Step(PlannedExecutionStage):
    """Plans one command, or nothing once its config marks it done."""

    def plan(self, context: PlanningContext) -> StagePlan:
        if context.config.get("done"):
            return StagePlan.noop(context.stage, "done")
        plan = plan_run_cmd(
            context,
            f"echo {context.stage}",
            cluster=context.target.cluster,
            container=context.config["container"],
            installation_command=context.config.get("installation_command"),
            expname=context.expname,
            run_after=list(context.run_after) or None,
            sbatch_kwargs=None,
        )
        if display := context.config.get("display"):
            tasks = tuple(
                replace(
                    task,
                    commands=tuple(replace(c, display_command=display) for c in task.commands),
                )
                for task in plan.tasks
            )
            plan = replace(plan, tasks=tasks)
        return plan


@pytest.fixture(autouse=True)
def _steps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "nvflow.core.workflow_definition.resolve_stage_uses",
        lambda _uses, _project: _Step,
    )


def _context(substages: dict[str, Any], **stage: Any) -> PlanningContext:
    config = {"container": "nemo-gym", "substages": substages, **stage}
    return PlanningContext(
        stage="checks",
        config=config,
        launcher_config=config,
        target=TargetBinding.resolve("test"),
        expname="run-checks",
        run_after=("run-prepare",),
    )


def _upstream(task: Any) -> list[tuple[str, bool]]:
    return [(dependency.upstream, dependency.external) for dependency in task.dependencies]


def test_substages_run_in_order_in_the_stages_container() -> None:
    setup = "export PATH=/opt/gym/bin:$PATH"
    substages = {"render": {"uses": "step"}, "collect": {"uses": "step"}}

    render, collect = SubstagesStage().plan(_context(substages, installation_command=setup)).tasks

    assert (render.task_id, collect.task_id) == ("checks.render.0", "checks.collect.0")
    assert _upstream(render) == [("run-prepare", True)]
    assert _upstream(collect) == [("checks.render.0", False)]
    for task in (render, collect):
        assert task.stage == "checks"
        assert task.commands[0].container == "nemo-gym"
        assert task.commands[0].command.startswith(f"set -e\n{setup}\n")
        assert task.parameters[SLURM_GROUP_JOBS] is True


def test_a_substage_runs_its_command_and_keeps_its_display_text() -> None:
    [task] = SubstagesStage().plan(_context({"train": {"uses": "step", "display": "train"}})).tasks

    assert task.commands[0].command.endswith("echo train")
    assert task.commands[0].display_command == "train"


def test_a_finished_substage_is_skipped() -> None:
    finished = {"uses": "step", "done": True}

    [collect] = (
        SubstagesStage().plan(_context({"render": finished, "collect": {"uses": "step"}})).tasks
    )

    assert _upstream(collect) == [("run-prepare", True)]
    assert SubstagesStage().plan(_context({"render": finished})).disposition == "noop"


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"substages": {"render": {"uses": "step"}}}, "'container'"),
        ({"container": "nemo-gym", "substages": {}}, "'substages' must map"),
        ({"container": "nemo-gym", "substages": {"render": {}}}, "must declare 'uses'"),
        (
            {
                "container": "nemo-gym",
                "substages": {"render": {"uses": "step", "dependencies": []}},
            },
            "drop 'dependencies'",
        ),
        (
            {
                "container": "nemo-gym",
                "substages": {"render": {"uses": "step", "container": "vllm"}},
            },
            "stage's container",
        ),
    ],
)
def test_stage_config_fails_closed(config: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SubstagesStage().validate_config(config)
