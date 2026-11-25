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
"""Repeated canonical rollout enrichment contracts for DG-SDG."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nvflow.stages.sdg.document_grounded.enrich_rollouts import enrich
from nvflow.stages.sdg.document_grounded.preprocess import (
    construct_question_verify_input,
)


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_question_expansion_drops_prior_native_gym_identity(tmp_path: Path) -> None:
    generated = tmp_path / "generated"
    _write(
        generated / "output.jsonl",
        [
            {
                "uuid": "source-0",
                "_ng_task_index": 21,
                "_ng_rollout_index": 0,
                "generation": json.dumps({"Numerical Reasoning": ["question one", "question two"]}),
            }
        ],
    )
    verify_input = tmp_path / "verify_input.jsonl"

    construct_question_verify_input(generated, verify_input)

    rows = [json.loads(line) for line in verify_input.read_text().splitlines()]
    assert [row["problem"] for row in rows] == ["question one", "question two"]
    assert all("_ng_task_index" not in row for row in rows)
    assert all("_ng_rollout_index" not in row for row in rows)


def test_question_expansion_uses_final_json_after_reasoning(tmp_path: Path) -> None:
    generated = tmp_path / "generated"
    _write(
        generated / "output.jsonl",
        [
            {
                "generation": (
                    'An example would be {"Beat_or_Miss": ["example"]}.\n'
                    "</think>\n"
                    '{"Beat_or_Miss": ["supported question"]}'
                )
            }
        ],
    )
    verify_input = tmp_path / "verify_input.jsonl"

    construct_question_verify_input(generated, verify_input)

    rows = [json.loads(line) for line in verify_input.read_text().splitlines()]
    assert [row["problem"] for row in rows] == ["supported question"]
    assert [row["question_type"] for row in rows] == ["Beat_or_Miss"]


def test_enrichment_preserves_native_repeat_identity(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs.jsonl"
    rollouts = tmp_path / "rollouts.jsonl"
    source_rows = [
        {
            "uuid": f"task-{task}",
            "context": f"context-{task}",
            "responses_create_params": {"input": [{"role": "user", "content": f"question-{task}"}]},
        }
        for task in range(2)
    ]
    _write(inputs, source_rows)
    rollout_rows = []
    for task, source in enumerate(source_rows):
        for repeat in range(2):
            rollout_rows.append(
                {
                    "responses_create_params": source["responses_create_params"],
                    "response": {
                        "output": [
                            {
                                "role": "assistant",
                                "content": f"answer-{task}-{repeat}",
                            }
                        ]
                    },
                    "_ng_task_index": task,
                    "_ng_rollout_index": repeat,
                    "reward": 1.0,
                }
            )
    _write(rollouts, rollout_rows)

    enrich(str(inputs), str(rollouts), expected_repeats=2)

    enriched = [json.loads(line) for line in rollouts.read_text().splitlines()]
    assert len(enriched) == 4
    assert {(row["_ng_task_index"], row["_ng_rollout_index"]) for row in enriched} == {
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    }
    assert {row["uuid"] for row in enriched} == {"task-0", "task-1"}
    assert enriched[0]["generation"] == "answer-0-0"


def test_wrong_repeat_count_fails_without_mutating_rollouts(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs.jsonl"
    rollouts = tmp_path / "rollouts.jsonl"
    request = {"responses_create_params": {"input": [{"role": "user", "content": "q"}]}}
    _write(inputs, [request])
    _write(
        rollouts,
        [
            {
                **request,
                "response": {"output": [{"role": "assistant", "content": "a"}]},
                "_ng_task_index": 0,
                "_ng_rollout_index": repeat,
            }
            for repeat in range(2)
        ],
    )
    original = rollouts.read_bytes()

    with pytest.raises(RuntimeError, match="1 inputs × 1 repeats"):
        enrich(str(inputs), str(rollouts), expected_repeats=1)

    assert rollouts.read_bytes() == original
