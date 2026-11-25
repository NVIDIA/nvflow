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

import json

import pytest
from nvfinance.utils.sdg.postprocess_genselect import postprocess_genselect
from nvfinance.utils.sdg.prepare_genselect_data import merge_jsonl_files
from nvfinance.utils.sft.extract_selected_trajectories import (
    extract_messages,
    extract_selected_trajectories,
)
from nvfinance.utils.shared.dataset_transformer import transform_record


def _rollout(problem: str, rollout_index: int, answer: str) -> dict:
    prompt = f"Agent instructions\n\nQuestion:\n{problem}\n"
    return {
        "_ng_task_index": 0,
        "_ng_rollout_index": rollout_index,
        "problem": problem,
        "responses_create_params": {"input": [{"role": "user", "content": prompt}]},
        "response": {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Search the filing."}],
                },
                {
                    "type": "function_call",
                    "name": "sec_filing_search",
                    "call_id": f"search-{rollout_index}",
                    "arguments": '{"ticker":"NVDA"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": f"search-{rollout_index}",
                    "output": "filing result",
                },
                {
                    "type": "function_call",
                    "name": "submit_final_result",
                    "call_id": f"final-{rollout_index}",
                    "arguments": json.dumps({"final_result": answer}),
                },
            ]
        },
    }


@pytest.mark.parametrize("field", ["summary", "content"])
def test_keeps_every_reasoning_segment_of_a_turn(field):
    record = _rollout("What was revenue?", 0, "42")
    segments = [{"text": "Find the filing."}, {"text": "Read the income statement."}]
    record["response"]["output"].insert(0, {"type": "reasoning", field: segments})

    messages = extract_messages(record)

    assert messages[2]["content"] == (
        "<think>\nFind the filing.\n\nRead the income statement.\n\nSearch the filing.\n</think>"
    )


def test_calls_made_together_share_one_assistant_message():
    record = _rollout("What was revenue?", 0, "42")
    output = record["response"]["output"]

    def call(call_id: str) -> dict:
        return {
            "type": "function_call",
            "name": "parse_html_page",
            "call_id": call_id,
            "arguments": "{}",
        }

    def result(call_id: str) -> dict:
        return {"type": "function_call_output", "call_id": call_id, "output": "page"}

    output[3:3] = [
        call("a"),
        call("b"),
        result("a"),
        result("b"),
        {"type": "reasoning", "summary": [{"text": "Read one more."}]},
        call("c"),
        result("c"),
    ]
    output.insert(2, call("parallel"))

    messages = extract_messages(record)

    calls = [
        [c["id"] for c in m.get("tool_calls", [])] for m in messages if m["role"] == "assistant"
    ]
    assert calls == [["search-0", "parallel"], ["a", "b"], ["c"], []]
    assert [m["role"] for m in messages[2:]] == [
        "assistant",
        "tool",
        "assistant",
        "tool",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]


def test_a_call_made_with_the_submission_leaves_its_thinking_to_the_answer():
    record = _rollout("What was revenue?", 0, "42")
    record["response"]["output"][3:3] = [
        {"type": "reasoning", "summary": [{"text": "Check one page, then answer."}]},
        {"type": "function_call", "name": "parse_html_page", "call_id": "late", "arguments": "{}"},
    ]

    messages = extract_messages(record)

    assert [m["role"] for m in messages[1:]] == ["user", "assistant", "tool", "assistant"]
    assert [c["id"] for m in messages for c in m.get("tool_calls", [])] == ["search-0"]
    assert messages[-1]["content"] == "<think>\nCheck one page, then answer.\n</think>\n\n42"


def test_extracts_exact_genselect_candidate_not_shortest(tmp_path):
    problem = "What was revenue?"
    deepseek = tmp_path / "deepseek.jsonl"
    qwen = tmp_path / "qwen.jsonl"
    selected = tmp_path / "selected.jsonl"
    output = tmp_path / "messages.jsonl"

    deepseek.write_text(json.dumps(_rollout(problem, 0, "wrong")) + "\n")
    qwen.write_text(
        "\n".join(
            [
                json.dumps(_rollout(problem, 0, "short but not selected")),
                json.dumps(_rollout(problem, 1, "selected answer")),
            ]
        )
        + "\n"
    )
    selected.write_text(
        json.dumps(
            {
                "problem": problem,
                "generation": "selected answer",
                "question_type": "Numerical Reasoning",
                "selected_original_seed_index": 2,
                "selected_candidate_reference": {
                    "source_file_index": 1,
                    "rollout_index": 1,
                    "task_index": 0,
                },
            }
        )
        + "\n"
    )

    written, missing = extract_selected_trajectories(selected, [deepseek, qwen], output)

    assert (written, missing) == (1, 0)
    record = json.loads(output.read_text())
    assert record["expected_answer"] == "selected answer"
    assert record["selected_candidate_reference"]["rollout_index"] == 1
    assert record["messages"][-1]["content"].endswith("selected answer")


def test_genselect_preserves_selected_model_and_repeat_after_shuffle(tmp_path):
    problem = "What was revenue?"
    deepseek = tmp_path / "deepseek.jsonl"
    qwen = tmp_path / "qwen.jsonl"
    prepared = tmp_path / "prepared.jsonl"
    judged = tmp_path / "judged.jsonl"
    selected = tmp_path / "selected.jsonl"

    deepseek.write_text(
        "\n".join(
            [
                json.dumps(_rollout(problem, 0, "deepseek-0")),
                json.dumps(_rollout(problem, 1, "deepseek-1")),
            ]
        )
        + "\n"
    )
    qwen.write_text(
        "\n".join(
            [
                json.dumps(_rollout(problem, 0, "qwen-0")),
                json.dumps(_rollout(problem, 1, "qwen-selected")),
            ]
        )
        + "\n"
    )

    merge_jsonl_files(
        [deepseek, qwen],
        prepared,
        strip_context=True,
        shuffle_solutions=True,
        shuffle_seed=42,
    )
    prepared_record = json.loads(prepared.read_text())
    selected_position = prepared_record["generations_list"].index("qwen-selected")
    prepared_record["generation"] = f"Judgment: {selected_position}"
    judged.write_text(json.dumps(prepared_record) + "\n")

    postprocess_genselect(judged, selected)

    selected_record = json.loads(selected.read_text())
    assert selected_record["context"] == "N/A"
    assert selected_record["generation"] == "qwen-selected"
    transformed, error = transform_record(selected_record, "inline", "none")
    assert error is None
    assert transformed is not None
    assert transformed["generation"] == "qwen-selected"
    assert selected_record["selected_candidate_reference"] == {
        "source_file_index": 1,
        "rollout_index": 1,
        "task_index": 0,
    }
