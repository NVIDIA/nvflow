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

"""Keep each question's most efficient passing teacher trace, and report on the teacher run.

Each dataset's folder holds its Gym collection's analyzed ``rollouts.jsonl`` and
``rollouts_materialized_inputs.jsonl``, joined on ``_ng_task_index``. A rollout
passes when its reward is 1.0. A question's trace is a passing rollout that
opened no later filing if it has one, then one that didn't open the web without
web search, and of those the one with the fewest tool calls, then the fewest
output tokens: chat messages with the question's uuid, reference answer, and
as-of date. Writes:

- ``traces.jsonl``: one trace for each question a teacher solved;
- ``questions.jsonl``: every question, with each rollout's outcome and cost;
- ``report.json``: solve rates, costs, later-filing and web counts, and rollouts
  that never submitted an answer (out of steps, context, or time), by dataset
  and source.

A rollout opened a later filing when a page it read with ``parse_html_page``
was filed after the question's as-of date: by the date its ``edgar_search``
results gave that page or, for a page they didn't list, by the year in the
page's accession number. A rollout opened the web without web search when it
read a page outside sec.gov while ``web_search`` wasn't among its tools, as in
web-off datasets; a rollout offered ``web_search`` is expected to open such
pages and never counts. The marks only rank a question's passing rollouts, so a
question with one keeps it whatever its marks; every trace carries its marks, to
filter on later. Costs count the agent's own model calls:
``retrieve_information`` sends its requests to the teacher separately, and Gym
leaves their tokens out of a rollout's usage.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterator
from datetime import date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import orjson

from nvfinance.utils.sft.extract_selected_trajectories import extract_messages, tool_arguments
from nvflow.utils.jsonl import iter_jsonl, write_jsonl

TASK_INDEX = "_ng_task_index"
ROLLOUT_INDEX = "_ng_rollout_index"
ROLLOUTS = "rollouts.jsonl"
INPUTS = "rollouts_materialized_inputs.jsonl"
FINAL_ANSWER_TOOL = "submit_final_result"
WEB_SEARCH_TOOL = "web_search"
_ROW_KEYS = (
    "uuid",
    "question",
    "problem",
    "expected_answer",
    "current_date",
    "source",
    "question_type",
)
_QUESTION_KEYS = ("uuid", "question", "current_date", "source", "question_type")
# EDGAR folder names hold the accession number: filer id, 2-digit year, sequence.
_ACCESSION = re.compile(r"/Archives/edgar/data/\d+/(\d{10})(\d{2})(\d{6})(?:/|$)", re.IGNORECASE)
_DATE_FORMATS = ("%B %d, %Y", "%Y-%m-%d")


def _as_of(value: Any) -> date | None:
    for form in _DATE_FORMATS:
        try:
            return datetime.strptime(str(value), form).date()
        except ValueError:
            continue
    return None


def _listed_filings(value: Any, found: dict[str, date]) -> None:
    """Add every ``filingUrl``/``filedAt`` pair in an ``edgar_search`` result to *found*.

    Gym returns the result list JSON-encoded in the output's ``results`` string.
    """
    if isinstance(value, str):
        if value.lstrip()[:1] not in ("[", "{"):
            return
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return
    if isinstance(value, dict):
        filed = _as_of(value.get("filedAt"))
        if isinstance(value.get("filingUrl"), str) and filed is not None:
            found[value["filingUrl"]] = filed
        for nested in value.values():
            _listed_filings(nested, found)
    elif isinstance(value, list):
        for nested in value:
            _listed_filings(nested, found)


def _filed_after(url: str, listed: dict[str, date], as_of: date) -> bool:
    if url in listed:
        return listed[url] > as_of
    match = _ACCESSION.search(url)
    if match is None:
        return False
    # EDGAR began in 1993, so two-digit years from 93 are in the 1990s.
    year = int(match.group(2))
    return year + (1900 if year >= 93 else 2000) > as_of.year


def _sec_page(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host == "sec.gov" or host.endswith(".sec.gov")


def _offered_tools(rollout: dict[str, Any]) -> set[str]:
    tools = (rollout.get("responses_create_params") or {}).get("tools") or []
    return {
        str(tool.get("name") or (tool.get("function") or {}).get("name"))
        for tool in tools
        if isinstance(tool, dict)
    }


def rollout_facts(rollout: dict[str, Any], as_of: date | None) -> dict[str, Any]:
    """Return a rollout's outcome, tool use, token counts, and later-filing and web flags."""
    response = rollout.get("response") or {}
    output = response.get("output") or []
    calls = [item for item in output if item.get("type") == "function_call"]
    names = {item.get("call_id"): item.get("name") for item in calls}
    listed: dict[str, date] = {}
    for item in output:
        if item.get("type") == "function_call_output" and names.get(item.get("call_id")) == (
            "edgar_search"
        ):
            try:
                _listed_filings(json.loads(item.get("output") or "null"), listed)
            except (TypeError, json.JSONDecodeError):
                continue
    opened = [
        str(tool_arguments(item.get("arguments")).get("url") or "")
        for item in calls
        if item.get("name") == "parse_html_page"
    ]
    usage = response.get("usage") or {}
    reward = rollout.get("reward")
    return {
        "rollout_index": rollout.get(ROLLOUT_INDEX),
        "passed": reward == 1.0 and not rollout.get("mask_sample"),
        "judge_rating": rollout.get("judge_rating"),
        "tool_calls": len(calls),
        "tools": dict(Counter(str(item.get("name")) for item in calls)),
        "submitted_answer": any(item.get("name") == FINAL_ANSWER_TOOL for item in calls),
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "opened_later_filing": as_of is not None
        and any(url and _filed_after(url, listed, as_of) for url in opened),
        "opened_web_without_web_search": WEB_SEARCH_TOOL not in _offered_tools(rollout)
        and any(url and not _sec_page(url) for url in opened),
    }


def _task(record: dict[str, Any]) -> str:
    return json.dumps(record[TASK_INDEX], sort_keys=True)


def _with_offsets(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield each JSONL record with the byte offset of its line."""
    with path.open("rb") as stream:
        offset = 0
        for line in stream:
            if line.strip():
                yield offset, orjson.loads(line)
            offset += len(line)


Outcomes = dict[str, list[tuple[dict[str, Any], int | None]]]


def _collection(folder: Path) -> tuple[dict[str, dict[str, Any]], Outcomes]:
    """Return each task's row and its rollouts' facts, checking the collection is complete.

    A rollout that could become a trace keeps its byte offset in ``rollouts.jsonl``
    instead of its content, so a collection never has to fit in memory.
    """
    rows: dict[str, dict[str, Any]] = {}
    expected: Counter[str] = Counter()
    for row in iter_jsonl(folder / INPUTS, on_error="raise"):
        task = _task(row)
        expected[task] += 1
        rows.setdefault(task, {key: row.get(key) for key in _ROW_KEYS})
    mismatch = f"{folder}: the rollouts don't match the inputs one to one; rerun it"
    outcomes: Outcomes = defaultdict(list)
    for offset, rollout in _with_offsets(folder / ROLLOUTS):
        task = _task(rollout)
        if task not in rows:
            raise SystemExit(mismatch)
        facts = rollout_facts(rollout, _as_of(rows[task]["current_date"]))
        usable = facts["passed"] and extract_messages(rollout) is not None
        outcomes[task].append((facts, offset if usable else None))
    if any(len(outcomes.get(task, ())) != count for task, count in expected.items()):
        raise SystemExit(mismatch)
    return rows, outcomes


def _summary(questions: list[dict[str, Any]]) -> dict[str, Any]:
    rollouts = [rollout for question in questions for rollout in question["rollouts"]]
    tools: Counter[str] = Counter()
    for rollout in rollouts:
        tools.update(rollout["tools"])
    passed = [rollout for rollout in rollouts if rollout["passed"]]
    return {
        "questions": len(questions),
        "solved_questions": sum(question["solved"] > 0 for question in questions),
        "traces": sum(question["trace_rollout"] is not None for question in questions),
        "rollouts": len(rollouts),
        "passing_rollouts": len(passed),
        "rollout_pass_rate": round(len(passed) / len(rollouts), 4) if rollouts else None,
        "input_tokens": sum(rollout["input_tokens"] for rollout in rollouts),
        "output_tokens": sum(rollout["output_tokens"] for rollout in rollouts),
        "tool_calls": dict(sorted(tools.items())),
        "rollouts_opening_later_filings": sum(r["opened_later_filing"] for r in rollouts),
        "passing_rollouts_opening_later_filings": sum(r["opened_later_filing"] for r in passed),
        "rollouts_opening_web_without_web_search": sum(
            r["opened_web_without_web_search"] for r in rollouts
        ),
        "passing_rollouts_opening_web_without_web_search": sum(
            r["opened_web_without_web_search"] for r in passed
        ),
        "rollouts_without_submitted_answer": sum(not r["submitted_answer"] for r in rollouts),
    }


def select_teacher_traces(
    datasets: dict[str, Path], output_dir: Path, *, teacher: str
) -> dict[str, Any]:
    """Write ``traces.jsonl``, ``questions.jsonl``, and ``report.json``; return the report."""
    questions: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    with write_jsonl(output_dir / "traces.jsonl") as traces:
        for name, folder in datasets.items():
            rows, outcomes = _collection(folder)
            with (folder / ROLLOUTS).open("rb") as stream:
                for task, row in rows.items():
                    ranked = sorted(outcomes[task], key=lambda pair: pair[0]["rollout_index"] or 0)
                    best = min(
                        ((facts, offset) for facts, offset in ranked if offset is not None),
                        key=lambda pair: (
                            pair[0]["opened_later_filing"],
                            pair[0]["opened_web_without_web_search"],
                            pair[0]["tool_calls"],
                            pair[0]["output_tokens"],
                        ),
                        default=None,
                    )
                    question = {key: row[key] for key in _QUESTION_KEYS}
                    question.update(
                        dataset=name,
                        solved=sum(facts["passed"] for facts, _offset in ranked),
                        trace_rollout=None if best is None else best[0]["rollout_index"],
                        rollouts=[facts for facts, _offset in ranked],
                    )
                    questions.append(question)
                    if best is None:
                        continue
                    facts, offset = best
                    stream.seek(offset)
                    traces.write(
                        {
                            "messages": extract_messages(orjson.loads(stream.readline())),
                            **row,
                            "dataset": name,
                            "teacher": teacher,
                            "rollout_index": facts["rollout_index"],
                            "tool_calls": facts["tool_calls"],
                            "output_tokens": facts["output_tokens"],
                            "opened_later_filing": facts["opened_later_filing"],
                            "opened_web_without_web_search": facts["opened_web_without_web_search"],
                            "solved": question["solved"],
                            "rollouts": len(ranked),
                        }
                    )
    with write_jsonl(output_dir / "questions.jsonl") as writer:
        for question in questions:
            writer.write(question)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for question in questions:
        groups[f"{question['dataset']}/{question['source']}"].append(question)
    report = {
        "teacher": teacher,
        "total": _summary(questions),
        "by_dataset_and_source": {key: _summary(group) for key, group in sorted(groups.items())},
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--datasets", required=True, help='JSON {"name": "<analyzed outputs folder>"}'
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--teacher", required=True, help="the teacher model, recorded on traces")
    args = parser.parse_args(argv)
    datasets = json.loads(args.datasets)
    if not isinstance(datasets, dict) or not datasets:
        parser.error("--datasets must map names to folders")
    report = select_teacher_traces(
        {str(name): Path(folder) for name, folder in datasets.items()},
        args.output_dir,
        teacher=args.teacher,
    )
    print(json.dumps(report["total"], sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
