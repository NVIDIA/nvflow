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
"""The question-pool clean stage: fields, dates, rejections, and its plan."""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from nvfinance.stages.sdg.clean_question_pool import CleanQuestionPoolStage
from nvfinance.utils.sdg.clean_question_pool import clean_pool

from nvflow.execution import PlanningContext, TargetBinding

ACCESSION = "0000320193-23-000106"
GYM_ROW = {
    "uuid": "rl-1",
    "expected_answer": "383.3 billion",
    "file_path0": f"AAPL/10-K/2023/{ACCESSION}/item7/part.txt",
    "responses_create_params": {
        "input": [
            {
                "role": "user",
                "content": "You should answer as if the current date is March 1, 2025.\n\n"
                "Question:\nWhat was Apple's total net sales in fiscal 2023?",
            }
        ],
        "tools": [{"name": "sec_filing_search"}],
    },
    "agent_ref": {"name": "finance_sec_search_simple_agent"},
}
SFT_ROW = {
    "id": "sft-1",
    "answer": "12%",
    "current_date": "2024-05-02",
    "messages": [
        {"role": "system", "content": "You are an analyst."},
        {"role": "user", "content": "How much did Microsoft's cloud revenue grow in fiscal 2023?"},
        {"role": "assistant", "content": "12%"},
    ],
}
FILING_ROW = {
    "problem": "What was Apple's research and development expense in fiscal 2023?",
    "expected_answer": "29.9 billion",
    "file_path0": f"AAPL/10-K/2023/{ACCESSION}/item8/part.txt",
}
REJECTED = [
    {"problem": "What was Apple's gross margin in 2023?"},
    {"expected_answer": "5"},
    {"problem": "How did the company respond to supply chain pressures?", "expected_answer": "x"},
]


def _write(path: Path, records: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _run(tmp_path: Path, **settings: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import pandas as pd

    parquet = tmp_path / "sec_metadata.parquet"
    pd.DataFrame({"accession_number": [ACCESSION], "filing_date": ["2023-11-03"]}).to_parquet(
        parquet
    )
    inputs = [
        {
            "path": str(_write(tmp_path / "rl.jsonl", [GYM_ROW, FILING_ROW, *REJECTED])),
            "source": "rl",
        },
        {"path": str(_write(tmp_path / "sft.jsonl", [SFT_ROW])), "source": "sft"},
    ]
    output = tmp_path / "clean"
    report = clean_pool(inputs, output, {"sec_metadata_parquet": str(parquet), **settings})
    return _lines(output / "clean.jsonl"), report


def test_records_gain_question_answer_date_and_filing_fields(tmp_path: Path) -> None:
    rows, _report = _run(tmp_path)
    by_uuid = {row["uuid"]: row for row in rows}

    gym = by_uuid["rl-1"]
    assert gym["problem"] == "What was Apple's total net sales in fiscal 2023?"
    assert (gym["current_date"], gym["date_source"]) == ("March 1, 2025", "prompt")
    assert gym["filing_key"] == ACCESSION
    assert gym["source"] == "rl"
    assert "responses_create_params" not in gym and "agent_ref" not in gym

    sft = by_uuid["sft-1"]
    assert sft["problem"] == "How much did Microsoft's cloud revenue grow in fiscal 2023?"
    assert sft["expected_answer"] == "12%"
    assert (sft["current_date"], sft["date_source"]) == ("May 02, 2024", "record")
    assert (sft["source"], sft["filing_key"]) == ("sft", None)
    assert "messages" not in sft

    [filing] = [row for row in rows if row["question_source"] == "field"]
    days = random.Random(filing["uuid"]).randint(1, 60)
    expected = (datetime(2023, 11, 3) + timedelta(days=days)).strftime("%B %d, %Y")
    assert (filing["current_date"], filing["date_source"]) == (expected, "filing")
    assert len(filing["uuid"]) == 32


def test_every_row_gets_its_question_and_one_spelling_of_its_type(tmp_path: Path) -> None:
    pool = [
        {
            "uuid": "a",
            "question": "What was Apple's revenue in 2023?",
            "expected_answer": "1",
            "question_type": "Market_Analysis",
        },
        {
            "uuid": "b",
            "problem": "What was Apple's net income in 2023?",
            "expected_answer": "2",
            "question_type": " Market  Analysis ",
        },
        {"uuid": "c", "problem": "What was Apple's gross margin in 2023?", "expected_answer": "3"},
    ]
    output = tmp_path / "clean"
    clean_pool([{"path": str(_write(tmp_path / "pool.jsonl", pool)), "source": "rl"}], output, {})
    rows = {row["uuid"]: row for row in _lines(output / "clean.jsonl")}

    assert {uuid: row["question_type"] for uuid, row in rows.items()} == {
        "a": "Market Analysis",
        "b": "Market Analysis",
        "c": None,
    }
    assert all(row["question"] == row["problem"] for row in rows.values())


def test_unmatched_filings_use_the_fallback_date(tmp_path: Path) -> None:
    rows, _report = _run(tmp_path, sec_metadata_parquet=None, fallback_current_date="2025-04-07")

    [filing] = [row for row in rows if row["question_source"] == "field"]
    assert (filing["current_date"], filing["date_source"]) == ("April 07, 2025", "fallback")


def test_rejections_keep_their_reason(tmp_path: Path) -> None:
    _rows, report = _run(tmp_path)

    rejected = _lines(tmp_path / "clean" / "rejected.jsonl")
    reasons = [row["_rejected"].split(":")[0] for row in rejected]
    assert reasons == ["no answer", "no question", "vague reference"]
    assert report["files"]["rl:rl.jsonl"]["kept"] == 2
    assert report["files"]["rl:rl.jsonl"]["rejected: vague reference"] == 1


def test_max_rows_reads_only_the_first_rows_of_each_file(tmp_path: Path) -> None:
    rows, report = _run(tmp_path, max_rows=1)

    assert [row["uuid"] for row in rows] == ["rl-1", "sft-1"]
    assert report["files"]["rl:rl.jsonl"] == {
        "date: prompt": 1,
        "kept": 1,
        "question: prompt": 1,
        "with filing_key": 1,
    }


def test_sample_rows_draws_the_same_sample_for_a_seed(tmp_path: Path) -> None:
    rows, report = _run(tmp_path, sample_rows=3, sample_seed=7)
    again, _report = _run(tmp_path, sample_rows=3, sample_seed=7)

    read = sum(
        sum(n for key, n in tally.items() if key == "kept" or key.startswith("rejected"))
        for tally in report["files"].values()
    )
    assert read == 3
    assert [row["uuid"] for row in rows] == [row["uuid"] for row in again]
    everything, _report = _run(tmp_path, sample_rows=100)
    assert len(everything) == 3


def test_answers_linking_a_filing_after_their_date_are_rejected(tmp_path: Path) -> None:
    import pandas as pd

    parquet = tmp_path / "sec_metadata.parquet"
    pd.DataFrame(
        {
            "accession_number": ["0001193125-25-054877", "0001001250-25-000045"],
            "filing_date": ["2025-03-14", "2025-05-01"],
        }
    ).to_parquet(parquet)

    def row(uuid: str, accession: str) -> dict[str, Any]:
        link = f"https://www.sec.gov/Archives/edgar/data/1/{accession.replace('-', '')}/doc.htm"
        return {
            "uuid": uuid,
            "problem": f"What did Apple report in filing {accession}?",
            "expected_answer": f"42. Source: {link}",
            "current_date": "2025-04-07",
        }

    pool = _write(
        tmp_path / "pool.jsonl",
        [
            row("before", "0001193125-25-054877"),
            row("after", "0001001250-25-000045"),
            row("next-year", "0000950170-26-000001"),
            row("undated", "0000950170-25-000009"),
        ],
    )
    output = tmp_path / "clean"
    report = clean_pool(
        [{"path": str(pool), "source": "rl"}], output, {"sec_metadata_parquet": str(parquet)}
    )

    assert [row["uuid"] for row in _lines(output / "clean.jsonl")] == ["before", "undated"]
    assert {row["uuid"]: row["_rejected"] for row in _lines(output / "rejected.jsonl")} == {
        "after": "links a later filing: 0001001250-25-000045 filed 2025-05-01",
        "next-year": "links a later filing: 0000950170-26-000001 filed in 2026",
    }
    assert report["files"]["rl:pool.jsonl"]["rejected: links a later filing"] == 2
    assert report["undated_links"] == 1


def test_a_trace_gets_the_date_in_its_system_prompt(tmp_path: Path) -> None:
    system = {
        "role": "system",
        "content": "You should answer all questions as if the current date is 2021-01-22.",
    }

    def trace(uuid: str, answer: str) -> dict[str, Any]:
        return {
            "uuid": uuid,
            "question": f"What did Apple report for item {uuid}?",
            "expected_answer": answer,
            "messages": [system, {"role": "user", "content": "What did Apple report?"}],
        }

    link = "https://www.sec.gov/Archives/edgar/data/1/000032019325000001/doc.htm"
    pool = _write(tmp_path / "sft.jsonl", [trace("dated", "42"), trace("later", f"42. {link}")])
    output = tmp_path / "clean"
    clean_pool([{"path": str(pool), "source": "sft"}], output, {})

    [row] = _lines(output / "clean.jsonl")
    assert (row["uuid"], row["current_date"], row["date_source"]) == (
        "dated",
        "January 22, 2021",
        "prompt",
    )
    assert "messages" not in row
    [rejected] = _lines(output / "rejected.jsonl")
    assert rejected["_rejected"] == "links a later filing: 0000320193-25-000001 filed in 2025"


def test_an_id_shared_by_two_questions_fails_the_run(tmp_path: Path) -> None:
    revenue = {"id": "7", "problem": "What was Apple's revenue in 2023?", "expected_answer": "1"}
    income = {**revenue, "problem": "What was Apple's net income in 2023?"}
    inputs = [
        {"path": str(_write(tmp_path / "a.jsonl", [revenue])), "source": "sft"},
        {"path": str(_write(tmp_path / "b.jsonl", [revenue, income])), "source": "sft"},
    ]
    output = tmp_path / "clean"

    with pytest.raises(ValueError, match=r"1 id\(s\) belong to more than one question, such as 7"):
        clean_pool(inputs, output, {})
    assert not (output / "clean.jsonl").exists()

    clean_pool(inputs, output, {"id_fields": []})
    assert len({row["uuid"] for row in _lines(output / "clean.jsonl")}) == 2


def _context(config: dict[str, Any]) -> PlanningContext:
    return PlanningContext(
        stage="clean",
        config=config,
        launcher_config=config,
        target=TargetBinding.resolve("test"),
        expname="clean",
        run_after=(),
    )


CONFIG = {
    "output_dir": "/out/clean",
    "inputs": [{"path": "/pool/rl.jsonl", "source": "rl"}],
    "sec_metadata_parquet": "/sec/metadata.parquet",
}


def test_plan_runs_the_worker_on_one_cpu_task() -> None:
    CleanQuestionPoolStage().validate_config(CONFIG)
    [task] = CleanQuestionPoolStage().plan(_context(CONFIG)).tasks

    command = task.commands[0].command
    assert "-m nvfinance.utils.sdg.clean_question_pool" in command
    assert "/sec/metadata.parquet" in command
    assert task.commands[0].container == "nemo-skills"
    assert task.resources[0].gpus_per_node == 0
    assert {output.name: output.path for output in task.outputs} == {
        "clean": "/out/clean/clean.jsonl",
        "rejected": "/out/clean/rejected.jsonl",
        "report": "/out/clean/report.json",
    }


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"inputs": CONFIG["inputs"]}, "output_dir"),
        ({"output_dir": "/o", "inputs": []}, "inputs"),
        ({"output_dir": "/o", "inputs": [{"path": "/a.jsonl"}]}, "source"),
        ({**CONFIG, "question_fields": "problem"}, "question_fields"),
        ({**CONFIG, "jitter_min_days": 9, "jitter_max_days": 3}, "jitter"),
        ({**CONFIG, "max_rows": 0}, "max_rows"),
        ({**CONFIG, "max_rows": True}, "max_rows"),
        ({**CONFIG, "sample_rows": 0}, "sample_rows"),
        ({**CONFIG, "sample_rows": True}, "sample_rows"),
        ({**CONFIG, "sample_seed": "7"}, "sample_seed"),
        ({**CONFIG, "max_rows": 5, "sample_rows": 5}, "not both"),
        ({**CONFIG, "sample_row": 500}, "unknown clean stage option"),
        ({**CONFIG, "fallback_current_date": "April 7, 2025"}, "YYYY-MM-DD"),
        ({**CONFIG, "sec_metadata_parquet": ""}, "sec_metadata_parquet"),
    ],
)
def test_stage_config_fails_closed(config: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        CleanQuestionPoolStage().validate_config(config)
