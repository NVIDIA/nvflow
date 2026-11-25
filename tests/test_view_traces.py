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
"""Tests for the rollout-trace viewer's record parsing, index, search, and file list."""

from __future__ import annotations

import http.client
import json
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

from scripts import view_traces
from scripts.view_traces import (
    JsonlReader,
    TraceIndex,
    list_jsonl_files,
    make_handler,
    parse_record,
    search_lines,
)

# Two parallel searches whose outputs arrive in the other order, then a submit.
PARALLEL_CALLS = [
    {"type": "reasoning", "summary": [{"type": "summary_text", "text": "think"}]},
    {
        "type": "function_call",
        "name": "search",
        "call_id": "a",
        "arguments": '{"query": "\\"exact phrase\\""}',
    },
    {"type": "function_call", "name": "search", "call_id": "b", "arguments": '{"query": "other"}'},
    {"type": "function_call_output", "call_id": "b", "output": '{"results": "[{\\"id\\": 1}]"}'},
    {
        "type": "function_call_output",
        "call_id": "a",
        "output": '{"results": "line one\\nline two"}',
    },
    {"type": "function_call", "name": "submit", "call_id": "c", "arguments": "{}"},
    {"type": "function_call_output", "call_id": "c", "output": "done"},
]


def _rollout(task: int, answer: int, reward: float, output: list | None = None) -> dict:
    return {
        "responses_create_params": {
            "input": [{"role": "user", "content": "Question?", "type": "message"}],
            "tools": [
                {"type": "function", "name": "search"},
                {"type": "function", "name": "submit"},
            ],
        },
        "response": {
            "output": PARALLEL_CALLS if output is None else output,
            "metadata": {"stop_reason": "done_tool", "steps": "2"},
            "usage": {"output_tokens": 12, "output_tokens_details": {"reasoning_tokens": 5}},
        },
        "judge_text": "fine",
        "failure_kind": None,
        "reward": reward,
        "_ng_task_index": task,
        "_ng_rollout_index": answer,
    }


def _wait_ready(index: TraceIndex) -> dict:
    for _ in range(200):
        snapshot = index.snapshot()
        if snapshot["status"] == "ready":
            return snapshot
        time.sleep(0.01)
    raise AssertionError("index never finished")


def test_header_shows_every_field_and_the_response_metadata():
    header = parse_record(_rollout(3, 1, 1.0))["header"]

    assert next(iter(header)) == "reward"
    assert header["judge_text"] == "fine"
    assert header["_ng_task_index"] == 3
    assert "failure_kind" not in header
    assert "response" not in header and "responses_create_params" not in header
    assert header["response.metadata.stop_reason"] == "done_tool"
    assert header["response.usage.output_tokens_details.reasoning_tokens"] == 5
    assert header["tools offered"] == "search, submit"


def test_tool_outputs_name_the_call_they_answer_and_steps_count_model_turns():
    steps = parse_record(_rollout(0, 0, 0.0))["steps"]

    outputs = [(step["title"], step["call"]) for step in steps if step["kind"] == "tool_output"]
    assert outputs == [("search", 2), ("search", 1), ("submit", 3)]
    assert [step["turn"] for step in steps if step["kind"] == "tool_call"] == [1, 1, 2]
    assert steps[0]["kind"] == "user" and steps[0]["turn"] == 0


def test_json_packed_in_tool_strings_is_unpacked_but_quoted_text_is_kept():
    steps = parse_record(_rollout(0, 0, 0.0))["steps"]

    first_call = next(step for step in steps if step["kind"] == "tool_call")
    outputs = {step["call"]: step["content"] for step in steps if step["kind"] == "tool_output"}
    assert json.loads(first_call["content"]) == {"query": '"exact phrase"'}
    assert json.loads(outputs[2]) == {"results": [{"id": 1}]}
    assert json.loads(outputs[1]) == {"results": "line one\nline two"}
    assert outputs[3] == "done"


def test_records_without_a_trace_fall_back_to_raw_json():
    parsed = parse_record({"reward": 0.5, "note": "verdict"})

    assert parsed == {
        "schema": "generic",
        "header": {"reward": 0.5},
        "raw": {"reward": 0.5, "note": "verdict"},
    }


def test_index_orders_records_by_question_then_answer(tmp_path: Path):
    path = tmp_path / "rollouts.jsonl"
    records = [_rollout(1, 0, 0.0), _rollout(0, 1, 1.0), _rollout(0, 0, 0.5), {"note": "no trace"}]
    path.write_text("".join(json.dumps(record) + "\n" for record in records) + "\n")

    snapshot = _wait_ready(TraceIndex(path))

    assert snapshot["rows"] == [
        [2, 0, 0, 0.5],
        [1, 0, 1, 1.0],
        [0, 1, 0, 0.0],
        [3, None, None, None],
    ]


def test_a_growing_file_is_indexed_from_where_it_stopped(tmp_path: Path, monkeypatch):
    path = tmp_path / "rollouts.jsonl"
    path.write_text(json.dumps(_rollout(0, 0, 1.0)) + "\n")
    index = TraceIndex(path)
    reader = JsonlReader(path)
    assert _wait_ready(index)["rows"] == [[0, 0, 0, 1.0]]
    assert reader.get(1) is None

    summarized: list[int] = []
    original = view_traces.summarize_line

    def summarize(line_no: int, line: bytes) -> list:
        summarized.append(line_no)
        return original(line_no, line)

    monkeypatch.setattr(view_traces, "summarize_line", summarize)
    with path.open("a") as stream:
        stream.write(json.dumps(_rollout(1, 0, 0.0)) + "\n" + '{"_ng_task_index": 2')
    index.grew()
    reader.grew()

    assert _wait_ready(index)["rows"] == [[0, 0, 0, 1.0], [1, 1, 0, 0.0]]
    assert summarized == [1]
    assert reader.get(1)["_ng_task_index"] == 1
    assert reader.get(2) is None

    with path.open("a") as stream:
        stream.write(', "_ng_rollout_index": 0')
    reader.grew()
    assert reader.get(2) is None
    with path.open("a") as stream:
        stream.write("}\n")
    reader.grew()
    assert reader.get(2) == {"_ng_task_index": 2, "_ng_rollout_index": 0}


def test_requests_must_name_a_loopback_host(tmp_path: Path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]

    def status(host: str) -> int:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        connection.request("GET", "/api/files", headers={"Host": host})
        response = connection.getresponse()
        response.read()
        connection.close()
        return response.status

    try:
        assert status(f"localhost:{port}") == 200
        assert status(f"127.0.0.1:{port}") == 200
        assert status(f"[::1]:{port}") == 200
        assert status(f"rebound.example:{port}") == 403
    finally:
        server.shutdown()


def test_search_matches_raw_json_lines_ignoring_case(tmp_path: Path):
    path = tmp_path / "rollouts.jsonl"
    path.write_text('{"a": "Time budget exhausted"}\n{"a": "ok"}\n{"a": "time BUDGET"}\n')

    assert search_lines(path, "time budget") == ([0, 2], False)
    assert search_lines(path, "time budget", limit=1) == ([0], True)


def test_file_list_skips_inputs_and_empty_files(tmp_path: Path):
    outputs = tmp_path / "outputs"
    prepared = outputs / "rollouts" / "preprocessed_datasets"
    prepared.mkdir(parents=True)
    (outputs / "rollouts.jsonl").write_text("{}\n")
    (outputs / "rollouts_failures.jsonl").write_text("")
    (outputs / "rollouts_materialized_inputs.jsonl").write_text("{}\n")
    (prepared / "train.jsonl").write_text("{}\n")

    assert list_jsonl_files(tmp_path) == ["outputs/rollouts.jsonl"]
