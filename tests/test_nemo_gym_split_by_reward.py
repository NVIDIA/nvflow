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
"""The reward-split report extension: repeats, replies, unscored and partial collections."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from nvflow.integrations.nemo_gym.responses import to_responses_api
from nvflow.integrations.nemo_gym.workers.split_by_reward import main


def _request(uuid: str, problem: str | None = None) -> dict[str, Any]:
    row = to_responses_api({"uuid": uuid, "problem": problem or f"question {uuid}"}, "check")
    row["verifier"] = {"type": "string_match", "expected_markers": ["Verdict: PASS"]}
    row["agent_ref"] = {"name": "check_agent"}
    return row


def _rollout(task: int, repeat: int, reward: Any, reply: str = "Verdict: PASS") -> dict[str, Any]:
    content = [{"type": "output_text", "text": reply}]
    return {
        "_ng_task_index": task,
        "_ng_rollout_index": repeat,
        "reward": reward,
        "response": {"output": [{"type": "message", "role": "assistant", "content": content}]},
    }


def _write(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _materialized(path: Path, rows: list[dict[str, Any]], repeats: int) -> Path:
    return _write(
        path,
        [
            {**row, "_ng_task_index": index, "_ng_rollout_index": repeat}
            for index, row in enumerate(rows)
            for repeat in range(repeats)
        ],
    )


def _split(tmp_path: Path, rows: list[dict[str, Any]], rollouts: list[dict[str, Any]]) -> Path:
    repeats = len(rollouts) // len(rows)
    materialized = _materialized(tmp_path / "materialized.jsonl", rows, repeats)
    output = tmp_path / "evaluation"
    assert (
        main([str(_write(tmp_path / "rollouts.jsonl", rollouts)), str(materialized), str(output)])
        == 0
    )
    return output


def test_rows_pass_only_when_every_rollout_scores_one(tmp_path: Path) -> None:
    rollouts = [
        _rollout(1, 1, 1.0, "Verdict: PASS"),
        _rollout(0, 0, 1.0),
        _rollout(1, 0, 0.0, "Verdict: FAIL"),
        _rollout(0, 1, 1.0),
    ]
    output = _split(tmp_path, [_request("a"), _request("b")], rollouts)

    assert _lines(output / "passed.jsonl") == [
        {
            "uuid": "a",
            "problem": "question a",
            "_reward": 1.0,
            "_success_rate": 1.0,
            "_replies": ["Verdict: PASS"] * 2,
        }
    ]
    assert _lines(output / "failed.jsonl") == [
        {
            "uuid": "b",
            "problem": "question b",
            "_reward": 0.5,
            "_success_rate": 0.5,
            "_replies": ["Verdict: FAIL", "Verdict: PASS"],
        }
    ]
    assert _lines(output / "reward_split.json") == [
        {"rows": 2, "passed": 1, "failed": 1, "unscored": 0, "pass_rate": 1.0}
    ]


def test_rows_sharing_a_prompt_keep_their_own_verdicts(tmp_path: Path) -> None:
    rows = [_request("a", "same question"), _request("b", "same question")]
    output = _split(tmp_path, rows, [_rollout(0, 0, 1.0), _rollout(1, 0, 0.0, "Verdict: FAIL")])

    assert [row["uuid"] for row in _lines(output / "passed.jsonl")] == ["a"]
    assert [row["uuid"] for row in _lines(output / "failed.jsonl")] == ["b"]


def test_a_rollout_without_a_finite_reward_fails_its_row_as_unscored(tmp_path: Path) -> None:
    rows = [_request("a"), _request("b"), _request("c")]
    rollouts = [_rollout(0, 0, None), _rollout(1, 0, float("nan")), _rollout(2, 0, True)]
    output = _split(tmp_path, rows, rollouts)

    assert [row["uuid"] for row in _lines(output / "failed.jsonl")] == ["a", "b", "c"]
    assert _lines(output / "reward_split.json") == [
        {"rows": 3, "passed": 0, "failed": 3, "unscored": 3, "pass_rate": 1.0}
    ]


def test_an_unfinished_empty_or_masked_reply_counts_as_unscored(tmp_path: Path) -> None:
    rows = [_request("a"), _request("b"), _request("c")]
    truncated = _rollout(0, 0, 0.0, reply="Reason: the answer")
    truncated["response"]["status"] = "incomplete"
    masked = {**_rollout(2, 0, 1.0), "mask_sample": True}
    output = _split(tmp_path, rows, [truncated, _rollout(1, 0, 1.0, reply=""), masked])

    assert [row["uuid"] for row in _lines(output / "failed.jsonl")] == ["a", "b", "c"]
    assert _lines(output / "reward_split.json") == [
        {"rows": 3, "passed": 0, "failed": 3, "unscored": 3, "pass_rate": 1.0}
    ]


def test_pass_rate_sets_the_share_of_rollouts_that_must_succeed(tmp_path: Path) -> None:
    materialized = _materialized(tmp_path / "materialized.jsonl", [_request("a")], repeats=2)
    rollouts = _write(
        tmp_path / "rollouts.jsonl", [_rollout(0, 0, 1.0), _rollout(0, 1, 0.0, "Verdict: FAIL")]
    )
    for options, passed in (("{}", 0), ('{"pass_rate": 0.5}', 1)):
        output = tmp_path / f"passed-{passed}"
        assert main([str(rollouts), str(materialized), str(output), "--options", options]) == 0
        assert len(_lines(output / "passed.jsonl")) == passed
    for options in ('{"pass_rate": 0}', '{"pass_rate": 1.5}', '{"threshold": 0.5}'):
        with pytest.raises(SystemExit):
            main([str(rollouts), str(materialized), str(tmp_path / "bad"), "--options", options])


def test_a_partial_collection_is_not_split(tmp_path: Path) -> None:
    materialized = _materialized(
        tmp_path / "materialized.jsonl", [_request("a"), _request("b")], repeats=2
    )
    rollouts = _write(
        tmp_path / "rollouts.jsonl", [_rollout(0, 0, 1.0), _rollout(0, 1, 1.0), _rollout(1, 0, 1.0)]
    )
    output = tmp_path / "evaluation"

    with pytest.raises(SystemExit, match="1 row"):
        main([str(rollouts), str(materialized), str(output)])
    assert not output.exists()
