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
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from nvfinance.stages.sft.select_teacher_traces import SelectTeacherTracesStage
from nvfinance.utils.sft.extract_selected_trajectories import extract_messages
from nvfinance.utils.sft.select_teacher_traces import rollout_facts, select_teacher_traces

from nvflow.execution import PlanningContext, TargetBinding

PROMPT = "Answer as the finance agent.\n\nQuestion:\nWhat was the company's revenue?"
EARLIER = "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/a.htm"
LATER = "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/b.htm"


def _rollout(
    task: int,
    index: int,
    *,
    passed: bool,
    calls: list[tuple[str, dict[str, Any], Any]],
    output_tokens: int = 100,
    submitted: bool = True,
) -> dict[str, Any]:
    output: list[dict[str, Any]] = []
    for number, (name, arguments, result) in enumerate(calls):
        call_id = f"call-{number}"
        output.append(
            {
                "type": "function_call",
                "name": name,
                "call_id": call_id,
                "arguments": json.dumps(arguments),
            }
        )
        output.append({"type": "function_call_output", "call_id": call_id, "output": result})
    if submitted:
        output.append(
            {
                "type": "function_call",
                "name": "submit_final_result",
                "call_id": "final",
                "arguments": json.dumps({"final_result": "42 billion"}),
            }
        )
    return {
        "_ng_task_index": task,
        "_ng_rollout_index": index,
        "reward": 1.0 if passed else 0.0,
        "responses_create_params": {"input": [{"role": "user", "content": PROMPT}]},
        "response": {
            "output": output,
            "usage": {"input_tokens": 1000, "output_tokens": output_tokens},
        },
    }


def _search(*listings: tuple[str, str]) -> tuple[str, dict[str, Any], str]:
    results = [{"filingUrl": url, "filedAt": filed} for url, filed in listings]
    return ("edgar_search", {"query": "revenue"}, json.dumps({"results": json.dumps(results)}))


def _open(url: str) -> tuple[str, dict[str, Any], str]:
    return ("parse_html_page", {"url": url, "key": "page"}, "saved")


def _collection(folder: Path, rows: list[dict[str, Any]], rollouts: list[dict[str, Any]]) -> Path:
    folder.mkdir(parents=True)
    inputs = [
        {**row, "_ng_task_index": task, "_ng_rollout_index": index}
        for task, row in enumerate(rows)
        for index in range(sum(rollout["_ng_task_index"] == task for rollout in rollouts))
    ]
    for name, records in (
        ("rollouts_materialized_inputs.jsonl", inputs),
        ("rollouts.jsonl", rollouts),
    ):
        (folder / name).write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )
    return folder


def _row(uuid: str, source: str = "rl") -> dict[str, Any]:
    return {
        "uuid": uuid,
        "question": "What was the company's revenue?",
        "problem": "What was the company's revenue?",
        "expected_answer": "7 billion",
        "current_date": "March 01, 2024",
        "source": source,
        "question_type": "lookup",
    }


def test_each_question_keeps_its_most_efficient_passing_trace(tmp_path: Path) -> None:
    lookup = _search((EARLIER, "2023-11-02"))
    efficient = _rollout(0, 1, passed=True, calls=[lookup])
    web_off = _collection(
        tmp_path / "web_off",
        [_row("q0"), _row("q1", source="sft")],
        [
            _rollout(1, 1, passed=False, calls=[lookup], submitted=False),
            efficient,
            _rollout(1, 0, passed=False, calls=[lookup]),
            _rollout(0, 0, passed=True, calls=[lookup, _open(EARLIER), _open(EARLIER)]),
        ],
    )
    web_on = _collection(
        tmp_path / "web_on",
        [_row("q2")],
        [
            _rollout(0, 0, passed=True, calls=[lookup], output_tokens=300),
            _rollout(0, 1, passed=True, calls=[lookup], output_tokens=200),
        ],
    )

    report = select_teacher_traces(
        {"web_off": web_off, "web_on": web_on}, tmp_path / "out", teacher="Kimi-K3.1-Pro"
    )

    traces = [json.loads(line) for line in (tmp_path / "out/traces.jsonl").read_text().splitlines()]
    assert [(trace["uuid"], trace["rollout_index"]) for trace in traces] == [("q0", 1), ("q2", 1)]
    assert {trace["expected_answer"] for trace in traces} == {"7 billion"}
    assert traces[0]["messages"] == extract_messages(efficient)
    assert traces[0]["messages"][-1] == {"role": "assistant", "content": "42 billion"}
    assert (traces[0]["tool_calls"], traces[0]["solved"], traces[0]["rollouts"]) == (2, 2, 2)
    assert report["total"] | {"tool_calls": None} == {
        "questions": 3,
        "solved_questions": 2,
        "traces": 2,
        "rollouts": 6,
        "passing_rollouts": 4,
        "rollout_pass_rate": 0.6667,
        "input_tokens": 6000,
        "output_tokens": 900,
        "tool_calls": None,
        "rollouts_opening_later_filings": 0,
        "passing_rollouts_opening_later_filings": 0,
        "rollouts_opening_web_without_web_search": 0,
        "passing_rollouts_opening_web_without_web_search": 0,
        "rollouts_without_submitted_answer": 1,
    }
    assert report["total"]["tool_calls"] == {
        "edgar_search": 6,
        "parse_html_page": 2,
        "submit_final_result": 5,
    }
    assert set(report["by_dataset_and_source"]) == {"web_off/rl", "web_off/sft", "web_on/rl"}
    lines = (tmp_path / "out/questions.jsonl").read_text().splitlines()
    questions = [json.loads(line) for line in lines]
    assert [question["trace_rollout"] for question in questions] == [1, None, 1]
    assert [[r["rollout_index"] for r in q["rollouts"]] for q in questions] == [[0, 1]] * 3


def test_a_filing_filed_after_the_question_counts_as_a_later_filing() -> None:
    as_of = date(2024, 3, 1)
    listed = _search((EARLIER, "2023-11-02"), (LATER, "2024-05-03"))
    unlisted = "https://www.sec.gov/Archives/edgar/data/320193/000032019325000001/c.htm"
    nineties = "https://www.sec.gov/Archives/edgar/data/320193/000091205799000123/d.htm"

    def later(*calls: tuple[str, dict[str, Any], str]) -> bool:
        rollout = _rollout(0, 0, passed=True, calls=list(calls))
        return bool(rollout_facts(rollout, as_of)["opened_later_filing"])

    assert later(listed, _open(LATER))
    assert not later(listed, _open(EARLIER))
    assert later(listed, _open(unlisted))
    assert not later(_open(EARLIER))
    assert not later(_open(nineties))
    assert not rollout_facts(_rollout(0, 0, passed=True, calls=[_open(LATER)]), None)[
        "opened_later_filing"
    ]


def test_a_page_off_sec_gov_counts_only_when_web_search_was_not_offered() -> None:
    def opened_web(url: str, *, web_search: bool) -> bool:
        rollout = _rollout(0, 0, passed=True, calls=[_open(url)])
        tools = ["edgar_search", "parse_html_page", "retrieve_information", "submit_final_result"]
        rollout["responses_create_params"]["tools"] = [
            {"type": "function", "name": name}
            for name in [*tools, *(["web_search"] if web_search else [])]
        ]
        return bool(rollout_facts(rollout, None)["opened_web_without_web_search"])

    financials = "https://stockanalysis.com/stocks/syf/financials/"
    assert opened_web(financials, web_search=False)
    assert opened_web("https://html.duckduckgo.com/html/?q=revenue", web_search=False)
    assert not opened_web(financials, web_search=True)
    assert not opened_web(EARLIER, web_search=False)
    assert not opened_web("https://efts.sec.gov/LATEST/search-index?q=revenue", web_search=False)


def test_a_trace_avoids_answers_that_opened_a_later_filing(tmp_path: Path) -> None:
    lookup = _search((EARLIER, "2023-11-02"), (LATER, "2024-05-03"))
    folder = _collection(
        tmp_path / "web_off",
        [_row("q0"), _row("q1")],
        [
            _rollout(0, 0, passed=True, calls=[lookup, _open(LATER)]),
            _rollout(0, 1, passed=True, calls=[lookup, _open(EARLIER), _open(EARLIER)]),
            _rollout(1, 0, passed=True, calls=[lookup, _open(LATER)]),
        ],
    )

    select_teacher_traces({"web_off": folder}, tmp_path / "out", teacher="Kimi-K3.1-Pro")

    traces = [json.loads(line) for line in (tmp_path / "out/traces.jsonl").read_text().splitlines()]
    assert [(t["uuid"], t["rollout_index"], t["opened_later_filing"]) for t in traces] == [
        ("q0", 1, False),
        ("q1", 0, True),
    ]


def test_a_trace_avoids_answers_that_opened_the_web_without_web_search(tmp_path: Path) -> None:
    lookup = _search((EARLIER, "2023-11-02"))
    web = "https://stockanalysis.com/stocks/aapl/financials/"
    folder = _collection(
        tmp_path / "web_off",
        [_row("q0"), _row("q1")],
        [
            _rollout(0, 0, passed=True, calls=[lookup, _open(web)]),
            _rollout(0, 1, passed=True, calls=[lookup, _open(EARLIER), _open(EARLIER)]),
            _rollout(1, 0, passed=True, calls=[lookup, _open(web)]),
        ],
    )

    select_teacher_traces({"web_off": folder}, tmp_path / "out", teacher="Kimi-K3.1-Pro")

    traces = [json.loads(line) for line in (tmp_path / "out/traces.jsonl").read_text().splitlines()]
    opened = [(t["uuid"], t["rollout_index"], t["opened_web_without_web_search"]) for t in traces]
    assert opened == [("q0", 1, False), ("q1", 0, True)]


def test_a_collection_missing_rollouts_fails(tmp_path: Path) -> None:
    folder = _collection(
        tmp_path / "web_off",
        [_row("q0")],
        [_rollout(0, 0, passed=True, calls=[_open(EARLIER)])],
    )
    (folder / "rollouts.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(SystemExit, match="don't match the inputs"):
        select_teacher_traces({"web_off": folder}, tmp_path / "out", teacher="Kimi-K3.1-Pro")


CONFIG = {
    "output_dir": "/out/traces",
    "dataset": "web_off",
    "rollouts": "/teacher/finance_sec_search/outputs",
    "teacher": "Kimi-K3.1-Pro",
}


def _context(config: dict[str, Any]) -> PlanningContext:
    return PlanningContext(
        stage="select_traces",
        config=config,
        launcher_config=config,
        target=TargetBinding.resolve("test"),
        expname="select_traces",
        run_after=(),
    )


def test_the_stage_opens_up_the_shared_cache_before_selecting() -> None:
    config = {**CONFIG, "shared_cache_dir": "/sec/cache"}
    SelectTeacherTracesStage().validate_config(config)
    [task] = SelectTeacherTracesStage().plan(_context(config)).tasks

    command = task.commands[0].command
    assert command.index(
        "find /sec/cache -user \"$(id -u)\" '(' -type d -exec chmod 2775 {} + "
        "-o -type f -exec chmod 664 {} + ')' || true"
    ) < command.index("nvfinance.utils.sft.select_teacher_traces")
    assert '{"web_off": "/teacher/finance_sec_search/outputs"}' in command
    assert task.commands[0].container == "nemo-skills"
    assert {output.path for output in task.outputs} == {
        "/out/traces/traces.jsonl",
        "/out/traces/questions.jsonl",
        "/out/traces/report.json",
    }
    [plain] = SelectTeacherTracesStage().plan(_context(CONFIG)).tasks
    assert "chmod" not in plain.commands[0].command


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({**CONFIG, "collections": {}}, "unknown select_teacher_traces stage option"),
        ({**CONFIG, "dataset": ""}, "dataset"),
        ({**CONFIG, "rollouts": ""}, "rollouts"),
        ({**CONFIG, "teacher": ""}, "teacher"),
        ({**CONFIG, "shared_cache_dir": ""}, "shared_cache_dir"),
        ({key: value for key, value in CONFIG.items() if key != "output_dir"}, "output_dir"),
    ],
)
def test_stage_config_fails_closed(config: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SelectTeacherTracesStage().validate_config(config)
