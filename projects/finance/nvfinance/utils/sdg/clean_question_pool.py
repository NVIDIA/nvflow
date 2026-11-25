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
"""Normalize question/answer pools into one file before dedup and decontamination.

Each record keeps its fields, except ``drop_fields``, and gains ``problem`` and
``question`` (the same text), ``expected_answer``, ``uuid``, ``source``,
``question_type`` (underscores read as spaces; null when the record has none),
``filing_key`` (the ``file_path0`` accession), and ``current_date``: the
record's own date, else its prompt's, else its filing's SEC date plus the GRPO
jitter, else the fallback. Records
without a question or answer, that say "the company" without naming one, or
whose answer links an SEC filing filed after their date go to ``rejected.jsonl``.
A linked filing's date comes from ``sec_metadata_parquet``; one the parquet
doesn't list counts as later only when its accession year is later, and the
report counts those from the record's own year as ``undated_links``. An id
shared by two different questions fails the run; ``id_fields: []`` derives ids
from the questions instead. For development runs, ``max_rows`` reads only
the first rows of each file, and ``sample_rows`` keeps that many records drawn
at random with ``sample_seed`` across the files in proportion to their sizes;
set one or the other.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime, timedelta
from itertools import islice
from pathlib import Path
from typing import Any

from nvflow.utils import setup_logger

logger = setup_logger(__name__)

DEFAULTS: dict[str, Any] = {
    "question_fields": ["problem", "question"],
    "answer_fields": ["expected_answer", "answer", "ground_truth"],
    "id_fields": ["uuid", "id"],
    "drop_fields": ["responses_create_params", "messages", "agent_ref"],
    "question_marker": r"Question:\s*",
    "company_field": "company_name",
    "reject_vague_references": True,
    "sec_metadata_parquet": None,
    "parquet_accession_column": "accession_number",
    "parquet_filing_date_column": "filing_date",
    "jitter_min_days": 1,
    "jitter_max_days": 60,
    "fallback_current_date": "2025-04-07",
    "max_rows": None,
    "sample_rows": None,
    "sample_seed": 0,
}
_PROMPT_DATE = re.compile(r"current date is ([^.\n]+)", re.IGNORECASE)
_ACCESSION = re.compile(r"^\d{10}-\d{2}-\d{6}$")
_FILING_URL = re.compile(r"sec\.gov/Archives/edgar/data/\d+/(\d{10})-?(\d{2})-?(\d{6})\b", re.I)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(part.get("text", "") for part in content if isinstance(part, Mapping))
    return ""


def _messages(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    request = record.get("responses_create_params")
    if isinstance(request, Mapping):
        if isinstance(request.get("input"), str):
            return [{"role": "user", "content": request["input"]}]
        if isinstance(request.get("input"), list):
            return [item for item in request["input"] if isinstance(item, Mapping)]
    conversation = record.get("messages")
    if isinstance(conversation, list):
        return [item for item in conversation if isinstance(item, Mapping)]
    return []


def _first(record: Mapping[str, Any], fields: Sequence[str]) -> Any:
    for name in fields:
        value = record.get(name)
        if value not in (None, ""):
            return value
    return None


def question(record: Mapping[str, Any], settings: Mapping[str, Any]) -> tuple[str, str]:
    """Return the question and where it came from."""
    value = _first(record, settings["question_fields"])
    if isinstance(value, str) and value.strip():
        return value.strip(), "field"
    users = [message for message in _messages(record) if message.get("role") == "user"]
    if users:
        text = _text(users[-1].get("content"))
        parts = re.split(settings["question_marker"], text)
        if len(parts) > 1 and parts[-1].strip():
            return parts[-1].strip(), "prompt"
    conversation = record.get("messages")
    if isinstance(conversation, list):
        for message in conversation:
            if isinstance(message, Mapping) and message.get("role") == "user":
                text = _text(message.get("content")).strip()
                if text:
                    return text, "messages"
    return "", "missing"


def accession(record: Mapping[str, Any]) -> str | None:
    """The accession number in ``{ticker}/{form}/{year}/{accession}/...``."""
    path = record.get("file_path0")
    if not isinstance(path, str):
        return None
    parts = path.split("/")
    return parts[3] if len(parts) > 3 and _ACCESSION.match(parts[3]) else None


class DateResolver:
    """The record's own date, its prompt's date, its filing's date, or the fallback."""

    def __init__(self, settings: Mapping[str, Any]):
        from nvfinance.utils.rl.prompt_template_applier import (
            _LONG_DATE_FMT,
            _normalize_current_date,
        )

        self.long_format = _LONG_DATE_FMT
        self.normalize = _normalize_current_date
        self.fallback = _normalize_current_date(str(settings["fallback_current_date"]))
        self.minimum = int(settings["jitter_min_days"])
        self.maximum = int(settings["jitter_max_days"])
        self.filing_dates: dict[str, str] = {}
        self.undated_links = 0
        if settings.get("sec_metadata_parquet"):
            import pandas as pd

            columns = [settings["parquet_accession_column"], settings["parquet_filing_date_column"]]
            frame = pd.read_parquet(settings["sec_metadata_parquet"], columns=columns)
            for key, value in zip(frame[columns[0]], frame[columns[1]], strict=True):
                self.filing_dates.setdefault(str(key), str(value)[:10])

    def resolve(self, record: Mapping[str, Any], uuid: str) -> tuple[str, str]:
        own = record.get("current_date")
        if isinstance(own, str) and own.strip():
            return self.normalize(own.strip()), "record"
        for message in _messages(record):
            if match := _PROMPT_DATE.search(_text(message.get("content"))):
                return self.normalize(match.group(1).strip()), "prompt"
        filing = self.filing_dates.get(accession(record) or "")
        if filing:
            try:
                start = datetime.strptime(filing, "%Y-%m-%d")
            except ValueError:
                start = None
            if start is not None:
                days = random.Random(uuid).randint(self.minimum, self.maximum)
                return (start + timedelta(days=days)).strftime(self.long_format), "filing"
        return self.fallback, "fallback"

    def later_link(self, answer: str, current_date: str) -> str | None:
        """The first SEC filing ``answer`` links that was filed after ``current_date``."""
        try:
            today = datetime.strptime(current_date, self.long_format)
        except ValueError:
            return None
        for filer, year_part, sequence in _FILING_URL.findall(answer):
            number = f"{filer}-{year_part}-{sequence}"
            year = int(year_part) + (1900 if int(year_part) >= 90 else 2000)
            try:
                filed = datetime.strptime(self.filing_dates.get(number, ""), "%Y-%m-%d")
            except ValueError:
                if year > today.year:
                    return f"{number} filed in {year}"
                if year == today.year:
                    self.undated_links += 1
                continue
            if filed > today:
                return f"{number} filed {filed:%Y-%m-%d}"
        return None


def _question_type(record: Mapping[str, Any]) -> str | None:
    value = record.get("question_type")
    if not isinstance(value, str) or not value.strip():
        return None
    return " ".join(value.replace("_", " ").split())


def _vague_reference(
    problem: str, record: Mapping[str, Any], settings: Mapping[str, Any]
) -> str | None:
    if not settings["reject_vague_references"]:
        return None
    from nvfinance.utils.rl.regex_prefilter_questions import _should_drop

    company = record.get(settings["company_field"])
    drop, reason = _should_drop(problem, company if isinstance(company, str) else "")
    return reason if drop else None


def clean_record(
    record: Mapping[str, Any],
    source: str,
    settings: Mapping[str, Any],
    dates: DateResolver,
) -> tuple[dict[str, Any] | None, str]:
    """Return the cleaned record, or None and the reason it was rejected."""
    problem, question_source = question(record, settings)
    if not problem:
        return None, "no question"
    answer = _first(record, settings["answer_fields"])
    if answer is None:
        return None, "no answer"
    if reason := _vague_reference(problem, record, settings):
        return None, f"vague reference: {reason}"
    identifier = _first(record, settings["id_fields"])
    uuid = (
        str(identifier)
        if identifier is not None
        else hashlib.sha256(f"{source}\n{problem}".encode()).hexdigest()[:32]
    )
    current_date, date_source = dates.resolve(record, uuid)
    expected_answer = answer if isinstance(answer, str) else json.dumps(answer)
    if later := dates.later_link(expected_answer, current_date):
        return None, f"links a later filing: {later}"
    cleaned = {key: value for key, value in record.items() if key not in settings["drop_fields"]}
    cleaned.update(
        {
            "problem": problem,
            "question": problem,
            "expected_answer": expected_answer,
            "uuid": uuid,
            "source": source,
            "question_type": _question_type(record),
            "current_date": current_date,
            "date_source": date_source,
            "question_source": question_source,
            "filing_key": accession(record),
        }
    )
    return cleaned, ""


def _rows(path: Path, keep: set[int] | None = None) -> Iterator[dict[str, Any]]:
    """The file's records, or only those at the ``keep`` positions among its records."""
    position = -1
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            position += 1
            if keep is not None and position not in keep:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{number}: not a JSON object")
            yield record


def _sample(paths: Sequence[Path], rows: int | None, seed: int) -> Sequence[set[int] | None]:
    """Record positions to keep in each file: ``rows`` drawn at random across the files."""
    if rows is None:
        return [None] * len(paths)
    sizes = []
    for path in paths:
        with path.open("rb") as stream:
            sizes.append(sum(1 for line in stream if line.strip()))
    chosen = random.Random(seed).sample(range(sum(sizes)), min(rows, sum(sizes)))
    keep: list[set[int]] = [set() for _ in paths]
    for index in chosen:
        for file, size in enumerate(sizes):
            if index < size:
                keep[file].add(index)
                break
            index -= size
    return keep


def clean_pool(
    inputs: Sequence[Mapping[str, str]],
    output_dir: Path,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Write ``clean.jsonl``, ``rejected.jsonl``, and ``report.json``."""
    from nvflow.stages.data.chunks import atomic_output, atomic_write_json, file_evidence

    settings = {**DEFAULTS, **settings}
    dates = DateResolver(settings)
    counts: dict[str, Counter[str]] = {}
    questions: dict[str, str] = {}
    shared_ids: dict[str, None] = {}
    with (
        atomic_output(output_dir / "clean.jsonl") as clean,
        atomic_output(output_dir / "rejected.jsonl") as rejected,
    ):
        paths = [Path(item["path"]) for item in inputs]
        sample = _sample(paths, settings["sample_rows"], int(settings["sample_seed"]))
        for item, path, keep in zip(inputs, paths, sample, strict=True):
            source = item["source"]
            tally = counts.setdefault(f"{source}:{path.name}", Counter())
            for record in islice(_rows(path, keep), settings["max_rows"]):
                cleaned, reason = clean_record(record, source, settings, dates)
                if cleaned is None:
                    tally[f"rejected: {reason.split(':')[0]}"] += 1
                    row = {**record, "_rejected": reason, "_source_file": item["path"]}
                    rejected.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                    continue
                tally["kept"] += 1
                tally[f"date: {cleaned['date_source']}"] += 1
                tally[f"question: {cleaned['question_source']}"] += 1
                tally["with filing_key" if cleaned["filing_key"] else "without filing_key"] += 1
                if questions.setdefault(cleaned["uuid"], cleaned["problem"]) != cleaned["problem"]:
                    shared_ids[cleaned["uuid"]] = None
                clean.write(json.dumps(cleaned, ensure_ascii=False, default=str) + "\n")
        if shared_ids:
            raise ValueError(
                f"{len(shared_ids)} id(s) belong to more than one question, such as "
                f"{', '.join(list(shared_ids)[:3])}; set 'id_fields: []' to derive ids "
                "from the questions"
            )
    report = {
        "files": {name: dict(sorted(tally.items())) for name, tally in counts.items()},
        "inputs": [{**item, **file_evidence(Path(item["path"]))} for item in inputs],
        "undated_links": dates.undated_links,
        "settings": settings,
    }
    atomic_write_json(output_dir / "report.json", report)
    for name, tally in counts.items():
        logger.info(f"{name}: {dict(sorted(tally.items()))}")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--inputs", required=True, help="JSON list of {path, source}")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--settings", default="{}", help="JSON overrides of the defaults")
    args = parser.parse_args(argv)
    clean_pool(json.loads(args.inputs), args.output_dir, json.loads(args.settings))
    return 0


if __name__ == "__main__":
    sys.exit(main())
