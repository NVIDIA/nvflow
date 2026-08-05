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
"""Tests for HopChain SFT reasoning-trace postprocessing."""

from __future__ import annotations

import json

from nvflow.recipes.multimodal.utils.hopchain_sft_trace_filter_postprocess import (
    main as filter_main,
)
from nvflow.recipes.multimodal.utils.hopchain_sft_trace_filter_postprocess import (
    parse_judge_decision,
)
from nvflow.recipes.multimodal.utils.hopchain_sft_trace_generation_postprocess import (
    main as generation_main,
)
from nvflow.recipes.multimodal.utils.hopchain_sft_trace_generation_postprocess import (
    parse_trace_response,
)


def test_parse_trace_response_uses_server_side_reasoning_trace() -> None:
    reasoning_trace, final_answer, generation_text = parse_trace_response(
        {
            "reasoning_content": "I count the relevant items, then add them.",
            "generation": ("<response><final_answer>7</final_answer></response>"),
        }
    )

    assert reasoning_trace == "I count the relevant items, then add them."
    assert final_answer == "7"
    assert "<final_answer>7</final_answer>" in generation_text


def test_parse_trace_response_extracts_answer_from_shared_xml_prompt() -> None:
    reasoning_trace, final_answer, _ = parse_trace_response(
        {
            "reasoning_content": "Captured hidden reasoning trace.",
            "generation": ("<response>\n  <final_answer>7</final_answer>\n</response>"),
        }
    )

    assert reasoning_trace == "Captured hidden reasoning trace."
    assert final_answer == "7"


def test_generation_postprocess_keeps_only_correct_answers(tmp_path, monkeypatch) -> None:
    raw_output_path = tmp_path / "raw_output.jsonl"
    correct_output_path = tmp_path / "final_result.jsonl"
    incorrect_output_path = tmp_path / "incorrect_answers.jsonl"
    summary_path = tmp_path / "summary.json"

    def raw_generation(query_id: str, sample_index: int, answer: str) -> dict:
        return {
            "_metadata": {
                "query_id": query_id,
                "sample_index": sample_index,
                "k": 3,
                "question": "How many items are shown?",
                "hypothetical_answer": "7",
                "record": {"query_id": query_id, "image_id": "image-1"},
            },
            "reasoning_content": f"trace {sample_index}",
            "generation": (f"<response><final_answer>{answer}</final_answer></response>"),
        }

    raw_outputs = [
        raw_generation("q1", 0, "7"),
        raw_generation("q1", 1, "8"),
    ]
    raw_output_path.write_text("\n".join(json.dumps(output) for output in raw_outputs) + "\n")

    monkeypatch.setattr(
        "sys.argv",
        [
            "hopchain_sft_trace_generation_postprocess",
            "--input",
            str(raw_output_path),
            "--output",
            str(correct_output_path),
            "--incorrect-output",
            str(incorrect_output_path),
            "--summary",
            str(summary_path),
            "--model",
            "qwen3.5-397b-a17b",
        ],
    )
    generation_main()

    correct_rows = [
        json.loads(line) for line in correct_output_path.read_text().splitlines() if line.strip()
    ]
    incorrect_rows = [
        json.loads(line) for line in incorrect_output_path.read_text().splitlines() if line.strip()
    ]
    summary = json.loads(summary_path.read_text())

    assert [row["sample_index"] for row in correct_rows] == [0]
    assert [row["sample_index"] for row in incorrect_rows] == [1]
    assert summary["answer_match_count"] == 1
    assert summary["incorrect_answer_count"] == 1


def _candidate(
    *,
    query_id: str,
    sample_index: int,
    answer_matches: bool = True,
) -> dict:
    return {
        "query_id": query_id,
        "sft_sample_id": f"{query_id}-sample-{sample_index}",
        "sample_index": sample_index,
        "k": 3,
        "model": "qwen3.5-397b-a17b",
        "question": "How many items are shown?",
        "hypothetical_answer": "7",
        "final_answer": "7" if answer_matches else "8",
        "normalized_hypothetical_answer": "7",
        "normalized_final_answer": "7" if answer_matches else "8",
        "answer_matches_hypothetical": answer_matches,
        "reasoning_trace": f"trace {sample_index}",
        "raw_generation": "7",
        "source_record": {
            "image_id": "image-1",
            "image_fullpath": "/tmp/image.png",
            "query_metadata": {
                "reasoning_hops": [
                    {
                        "hop_number": 1,
                        "hop_type": "count",
                        "description": "Count the items.",
                        "output": "7",
                    }
                ]
            },
        },
        "generation_stats": {},
    }


def _judge_output(sft_sample_id: str, verdict: str) -> dict:
    return {
        "_metadata": {"sft_sample_id": sft_sample_id},
        "generation": json.dumps(
            {
                "reasoning": "Detailed hop audit: covers the hops.",
                "verdict": verdict,
            }
        ),
    }


def test_filter_postprocess_selects_only_one_passing_trace_per_query(tmp_path, monkeypatch) -> None:
    candidates_path = tmp_path / "candidates.jsonl"
    judge_path = tmp_path / "judge_output.jsonl"
    output_dir = tmp_path / "filtered"
    summary_path = output_dir / "summary.json"

    candidates = [
        _candidate(query_id="q1", sample_index=1),
        _candidate(query_id="q1", sample_index=2),
        _candidate(query_id="q2", sample_index=0),
    ]
    candidates_path.write_text("\n".join(json.dumps(candidate) for candidate in candidates) + "\n")

    judge_outputs = [
        _judge_output("q1-sample-1", "pass"),
        _judge_output("q1-sample-2", "pass"),
        _judge_output("q2-sample-0", "fail"),
    ]
    judge_path.write_text("\n".join(json.dumps(output) for output in judge_outputs) + "\n")

    monkeypatch.setattr(
        "sys.argv",
        [
            "hopchain_sft_trace_filter_postprocess",
            "--candidates",
            str(candidates_path),
            "--judge-output",
            str(judge_path),
            "--output-dir",
            str(output_dir),
            "--summary",
            str(summary_path),
        ],
    )
    filter_main()

    selected_rows = [
        json.loads(line)
        for line in (output_dir / "sft_output.jsonl").read_text().splitlines()
        if line.strip()
    ]
    kept_rows = [
        json.loads(line)
        for line in (output_dir / "kept_output.jsonl").read_text().splitlines()
        if line.strip()
    ]
    summary = json.loads(summary_path.read_text())

    assert [row["sft_sample_id"] for row in selected_rows] == ["q1-sample-1"]
    assert {row["sft_sample_id"] for row in kept_rows} == {"q1-sample-1", "q1-sample-2"}
    assert summary["answer_mismatch_candidates"] == 0
    assert summary["judge_pass_candidates"] == 2
    assert summary["selected_sft_count"] == 1


def test_filter_postprocess_parses_json_judge_response(tmp_path, monkeypatch) -> None:
    candidates_path = tmp_path / "candidates.jsonl"
    judge_path = tmp_path / "judge_output.jsonl"
    output_dir = tmp_path / "filtered"
    summary_path = output_dir / "summary.json"

    candidates_path.write_text(json.dumps(_candidate(query_id="q1", sample_index=0)) + "\n")
    judge_path.write_text(
        json.dumps(
            {
                "_metadata": {"sft_sample_id": "q1-sample-0"},
                "generation": json.dumps(
                    {
                        "reasoning": "Detailed hop audit: hop 1 is covered and no steps are skipped.",
                        "verdict": "pass",
                    }
                ),
            }
        )
        + "\n"
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "hopchain_sft_trace_filter_postprocess",
            "--candidates",
            str(candidates_path),
            "--judge-output",
            str(judge_path),
            "--output-dir",
            str(output_dir),
            "--summary",
            str(summary_path),
        ],
    )
    filter_main()

    selected_rows = [
        json.loads(line)
        for line in (output_dir / "sft_output.jsonl").read_text().splitlines()
        if line.strip()
    ]

    assert len(selected_rows) == 1
    assert selected_rows[0]["sft_trace_judge"]["reasoning"].startswith("Detailed hop audit")
    assert "failure_reasons" not in selected_rows[0]["sft_trace_judge"]


def test_parse_judge_decision_reads_json_object() -> None:
    decision = parse_judge_decision(
        {
            "generation": (
                '{"reasoning": "Detailed hop audit: hop 1 is covered, then hop 2 uses that output.", '
                '"verdict": "pass"}'
            )
        }
    )

    assert decision.verdict == "pass"
    assert decision.reasoning.startswith("Detailed hop audit")


def test_parse_judge_decision_ignores_reasoning_content() -> None:
    decision = parse_judge_decision(
        {
            "reasoning_content": (
                'The schema is {"reasoning": "example", "verdict": "pass or fail"}, '
                "but this is hidden reasoning and must not be parsed."
            ),
            "generation": ('{"reasoning": "Actual visible JSON audit.", "verdict": "fail"}'),
        }
    )

    assert decision.verdict == "fail"
    assert decision.reasoning == "Actual visible JSON audit."
