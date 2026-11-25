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

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from nvfinance.stages.rl.prepare_grpo_data import (
    PrepareRubricGRPODataStage,
    build_grpo_data_command,
    prepare_grpo_data,
)

from nvflow.execution import PlanningContext, TargetBinding
from nvflow.stages.data.split_contract import validate_split_marker


def _rubric_row(record_id: str, question: str, reference: str) -> dict[str, object]:
    return {
        "id": record_id,
        "question": question,
        "reference_answer": reference,
        "rubric_method": "simple",
        "rubric": [
            {
                "id": "R1",
                "type": "fact",
                "criterion": f"States {reference}",
                "severity": 3,
                "must_pass": True,
                "evidence_summary": "",
            }
        ],
    }


def test_grpo_handoff_joins_each_rubric_to_its_own_question(tmp_path: Path) -> None:
    """A template contributes shape only; every answer must come from its own row."""
    rubric_file = tmp_path / "rubrics.jsonl"
    rubric_file.write_text(
        "\n".join(
            json.dumps(_rubric_row(f"q{index}", f"Question {index}?", f"Answer {index}"))
            for index in range(4)
        )
        + "\n"
    )
    template_file = tmp_path / "template.jsonl"
    template_file.write_text(
        json.dumps(
            {
                "uuid": "template",
                "problem": "Template question?",
                "question": "Template question?",
                "prompt": "Instructions\nQuestion:\nTemplate question?\n",
                "expected_answer": "Template answer",
                "responses_create_params": {"input": []},
                "metadata": {"reference_answer": "Template answer", "source_split": "train"},
            }
        )
        + "\n"
    )

    stats = prepare_grpo_data(
        rubric_file=rubric_file,
        output_dir=tmp_path / "grpo",
        template_rl_file=template_file,
        validation_size=1,
    )
    assert stats == {"rubrics": 4, "records": 4, "train": 3, "val": 1}

    split = tmp_path / "grpo" / "finance_sec_search"
    marker = cast(dict[str, Any], validate_split_marker(split))
    assert marker["train"]["rows"] == 3
    assert marker["validation"]["rows"] == 1
    rows = [
        json.loads(line)
        for name in ("train.jsonl", "val.jsonl")
        for line in (split / name).read_text().splitlines()
    ]
    assert {row["uuid"] for row in rows} == {"q0", "q1", "q2", "q3"}
    for row in rows:
        index = row["uuid"].removeprefix("q")
        spec = json.loads(row["expected_answer"])
        assert spec["reference_answer"] == f"Answer {index}"
        assert row["metadata"]["reference_answer"] == f"Answer {index}"
        assert row["problem"] == f"Question {index}?"
        assert row["metadata"]["source_split"] == "train"


def test_grpo_handoff_stage_validation_command_and_plan() -> None:
    config = {
        "rubric_file": "/out/rubrics.jsonl",
        "base_rl_file": "/data/train.jsonl",
        "output_dir": "/out/step-3-grpo-data",
        "env_name": "finance_sec_search",
        "allow_missing_rubrics": True,
    }
    stage = PrepareRubricGRPODataStage()
    stage.validate_config(config)
    command = build_grpo_data_command(config)
    assert "nvfinance.stages.rl.prepare_grpo_data" in command
    assert "--env_name finance_sec_search" in command
    assert command.endswith("--allow-missing-rubrics")

    context = PlanningContext(
        stage="prepare_grpo",
        config=config,
        launcher_config=config,
        target=TargetBinding.resolve("test-cluster"),
        expname="rubric-smoke-prepare-grpo",
        run_after=(),
    )
    execution = stage.execution(context)
    assert execution.disposition == "executable"
    assert execution.tasks
    assert all(task.commands for task in execution.tasks)

    with pytest.raises(ValueError, match="base_rl_file or template_rl_file"):
        stage.validate_config({"rubric_file": "/out/rubrics.jsonl", "output_dir": "/out"})
