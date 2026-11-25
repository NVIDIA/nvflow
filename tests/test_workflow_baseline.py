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
"""Characterize workflow loading and submission before the planning refactor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from my_project.stage import GenerateAnswerStage

from nvflow.execution import PlanningContext, TargetBinding, WorkflowRunner


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def test_runner_resolves_inheritance_and_expands_dynamic_stages(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_root = tmp_path / "project"
    workflows = project_root / "workflows"
    workflows.mkdir(parents=True)
    _write_yaml(
        project_root / "project.yaml",
        {
            "manifest_version": 1,
            "project": "baseline-fixture",
        },
    )
    _write_yaml(
        workflows / "base.yaml",
        {
            "workflow": {"name": "baseline", "type": "evaluation"},
            "cluster": "baseline-cluster",
            "api_key": "${oc.env:NVFLOW_BASELINE_SECRET,null}",
            "pipeline_stages": ["model-a", "checkpoint-a"],
            "models": {
                "model-a": {
                    "model": "/models/a",
                    "dependencies": [],
                }
            },
            "checkpoints": {
                "checkpoint-a": {
                    "checkpoint": "/checkpoints/a",
                    "eval_steps": [10, 20],
                    "dependencies": ["model-a"],
                }
            },
            "stages": {},
        },
    )
    _write_yaml(
        workflows / "child.yaml",
        {
            "_base_": "base.yaml",
            "workflow": {"name": "baseline-child"},
        },
    )
    monkeypatch.setenv("NVFLOW_BASELINE_SECRET", "baseline-secret-value")

    runner = WorkflowRunner(str(workflows / "child.yaml"))

    assert runner.project == "baseline-fixture"
    assert runner.workflow_name == "baseline-child"
    assert runner.workflow_type == "evaluation"
    assert runner.cluster == "baseline-cluster"
    assert runner.config["api_key"] == "baseline-secret-value"
    assert runner.config["pipeline_stages"] == [
        "model-a",
        "checkpoint-a-10",
        "checkpoint-a-20",
    ]
    assert runner.config["stages"]["model-a"] == {
        "api_key": "baseline-secret-value",
        "model": "/models/a",
        "dependencies": [],
        "_source_section": "models",
        "_source_name": "model-a",
    }
    assert runner.config["stages"]["checkpoint-a-10"]["_step"] == 10
    assert runner.config["stages"]["checkpoint-a-20"]["_step"] == 20


def test_example_stage_submission_arguments_are_stable(
    tmp_path: Path,
) -> None:
    input_file = tmp_path / "input.jsonl"
    input_file.write_text('{"question":"What is NVFlow?"}\n', encoding="utf-8")
    config = {
        "input_file": str(input_file),
        "output_dir": "/workspace/output",
        "prompt_config": "/workspace/prompt.yaml",
        "inline_args": "++temperature=0.2",
        "stage_kwargs": {
            "model": "/hf/model",
            "num_chunks": 2,
            "server_gpus": 1,
        },
    }
    execution = GenerateAnswerStage().execution(
        PlanningContext(
            stage="generate",
            config=config,
            launcher_config=config,
            target=TargetBinding.resolve("baseline-cluster"),
            expname="baseline-generate",
            run_after=("baseline-download",),
        )
    )

    assert len(execution.tasks) == 3
    client_commands = [
        command.command
        for task in execution.tasks
        for command in task.commands
        if command.role == "client"
    ]
    assert all("++prompt_config=/workspace/prompt.yaml" in command for command in client_commands)
    assert all("++temperature=0.2" in command for command in client_commands)
    assert execution.tasks[0].dependencies[0].upstream == "baseline-download"
    assert execution.tasks[0].name == "baseline-generate-chunk0"
    assert execution.tasks[-1].name == "baseline-generate-merge"
