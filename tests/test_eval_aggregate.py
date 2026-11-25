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
"""Tests for finance reporting over canonical Gym repeats."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from nvfinance.utils.rl.eval_aggregate import (
    INVALID_TOOL,
    _ordered,
    _parse_row,
    aggregate,
)


def _rollout(reward=1.0, tools=(), calls=(), tokens=10):
    return {
        "reward": reward,
        "response": {
            "tools": [{"name": tool} for tool in tools],
            "output": [{"type": "function_call", "name": call} for call in calls],
            "usage": {"output_tokens": tokens, "input_tokens": 100},
        },
    }


def _write_canonical(
    directory: Path,
    repeats: list[list[dict]],
) -> tuple[Path, Path]:
    rollouts_file = directory / "rollouts.jsonl"
    materialized_file = directory / "rollouts_materialized_inputs.jsonl"
    rollout_rows = []
    materialized_rows = []
    for repeat_index, rows in enumerate(repeats):
        for task_index, row in enumerate(rows):
            identity = {
                "_ng_task_index": task_index,
                "_ng_rollout_index": repeat_index,
            }
            rollout_rows.append({**row["rollout"], **identity})
            materialized_rows.append(
                {
                    **identity,
                    "uuid": row.get("uuid", f"q{task_index}"),
                    "question_type": row.get("question_type", "Trends"),
                }
            )
    rollouts_file.write_text("".join(json.dumps(row) + "\n" for row in rollout_rows))
    materialized_file.write_text("".join(json.dumps(row) + "\n" for row in materialized_rows))
    return rollouts_file, materialized_file


def test_parse_row_uses_materialized_metadata_and_counts_only_offered_tools() -> None:
    parsed = _parse_row(
        _rollout(
            tools=("edgar_search", "web_search"),
            calls=("edgar_search", "invented_tool"),
        ),
        {"uuid": "q1", "question_type": "Trends"},
    )

    assert parsed["uuid"] == "q1"
    assert parsed["question_type"] == "Trends"
    assert parsed["tool_calls"]["edgar_search"] == 1
    assert parsed["tool_calls"][INVALID_TOOL] == 1
    assert "invented_tool" not in parsed["tool_calls"]


def test_known_tools_keep_display_order() -> None:
    assert _ordered(
        ["calculator", "zzz_custom", "web_search"],
        ["web_search", "calculator"],
    ) == ["web_search", "calculator", "zzz_custom"]


def test_qa_report_has_no_tool_rows_and_uses_repeat_metrics(tmp_path: Path) -> None:
    rollouts, inputs = _write_canonical(
        tmp_path,
        [
            [
                {"uuid": "q1", "rollout": _rollout(1.0)},
                {"uuid": "q2", "rollout": _rollout(0.0)},
            ],
            [
                {"uuid": "q1", "rollout": _rollout(1.0)},
                {"uuid": "q2", "rollout": _rollout(1.0)},
            ],
        ],
    )
    output = tmp_path / "report"

    aggregate(str(rollouts), str(inputs), str(output), expected_repeats=2)

    summary = (output / "summary.txt").read_text()
    metrics = json.loads((output / "metrics.json").read_text())
    assert "2 repeats" in summary
    assert "Avg. Num turns / question" not in summary
    assert metrics["finance"]["num_repeats"] == 2
    assert metrics["finance"]["overall_accuracy"] == {
        "mean": 0.75,
        "std": pytest.approx(0.3535533906),
    }


def test_agentic_report_lists_offered_tools_even_when_unused(tmp_path: Path) -> None:
    rollouts, inputs = _write_canonical(
        tmp_path,
        [
            [
                {
                    "uuid": "q1",
                    "rollout": _rollout(
                        tools=("edgar_search", "web_search"),
                        calls=("edgar_search",),
                    ),
                },
                {
                    "uuid": "q2",
                    "rollout": _rollout(tools=("edgar_search", "web_search")),
                },
            ]
        ],
    )
    output = tmp_path / "report"

    aggregate(str(rollouts), str(inputs), str(output))

    tools = json.loads((output / "metrics.json").read_text())["finance"]["tool_calls_per_question"]
    assert tools["edgar_search"]["mean"] == 0.5
    assert tools["web_search"]["mean"] == 0.0


def test_missing_repeat_or_mismatched_artifacts_fail_loudly(tmp_path: Path) -> None:
    rollouts, inputs = _write_canonical(
        tmp_path,
        [[{"uuid": "q1", "rollout": _rollout()}]],
    )

    with pytest.raises(RuntimeError, match="Expected 3 repeats"):
        aggregate(str(rollouts), str(inputs), str(tmp_path / "report"), expected_repeats=3)

    inputs.write_text("")
    with pytest.raises(RuntimeError, match="identical non-empty keys"):
        aggregate(str(rollouts), str(inputs), str(tmp_path / "report"))
