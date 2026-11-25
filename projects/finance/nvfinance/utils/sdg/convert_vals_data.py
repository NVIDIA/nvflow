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
"""Convert VALS Finance Agent JSONL into template-based SDG seed data."""

# TODO: Generalize this VALS-specific adapter into a configurable
# benchmark-to-TSDG seed converter. Field mappings, dataset ID prefixes,
# document-cardinality extraction, and the company universe should come from
# workflow configuration so new benchmarks do not require new converter code.

from __future__ import annotations

import argparse
import json
from io import StringIO
from pathlib import Path

import pandas as pd

from nvflow.utils import setup_logger

logger = setup_logger(__name__)


def _create_sp500_company_info(output_file: str) -> None:
    """Write the broad S&P 500 substitution pool used by the VALS campaign."""
    import requests
    from bs4 import BeautifulSoup

    response = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers={"User-Agent": "NVFlowFinanceSDG/1.0 (https://github.com/NVIDIA/nvflow)"},
        timeout=60,
    )
    response.raise_for_status()
    table = BeautifulSoup(response.text, "html.parser").find("table", {"id": "constituents"})
    if table is None:
        raise RuntimeError("S&P 500 constituent table was not found")
    companies = pd.read_html(StringIO(str(table)))[0]
    companies = companies[["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]].rename(
        columns={
            "Symbol": "Ticker",
            "Security": "Company",
            "GICS Sector": "Sector",
            "GICS Sub-Industry": "SubIndustry",
        }
    )
    destination = Path(output_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    companies.to_csv(destination, index=False, sep="\t")
    logger.info("Wrote %d S&P 500 companies to %s", len(companies), destination)


def _seed_record(index: int, record: dict) -> dict:
    """Normalize one raw or Gym-prepared VALS record."""
    metadata = record.get("metadata") or {}
    question = record.get("input") or record.get("question") or ""
    answer = record.get("target") or record.get("expected_answer") or ""
    question_type = (
        metadata.get("question_category")
        or record.get("Question Type")
        or record.get("question_type")
        or "Unknown"
    )
    documents = metadata.get("documents_required_to_answers") or record.get(
        "documents_required_to_answers", ""
    )
    rubric = metadata.get("checks") or record.get("Rubric") or record.get("rubric") or []

    # The public Gym-prepared VALS data does not expose SEC accession numbers.
    # The template generator only uses their cardinality to choose a one- or
    # two-company substitution. Preserve multi-document cardinality when the
    # original benchmark metadata is available; otherwise use one placeholder.
    document_count = max(1, len([line for line in str(documents).splitlines() if line.strip()]))
    placeholders = ";".join(f"unknown-{document_index}" for document_index in range(document_count))

    return {
        "QID": f"VALS-{index:04d}",
        "original_question": str(question),
        "ground_truth": str(answer),
        "question_type": str(question_type),
        "accession_number": placeholders,
        "form_types": ";".join(["10-K"] * document_count),
        "report_dates": ";".join(["2024-12-31"] * document_count),
        "item": ";".join(["Item 7"] * document_count),
        "vals_documents": str(documents),
        "vals_checks": json.dumps(rubric, ensure_ascii=False),
    }


def convert_vals_data(input_file: str, output_file: str, company_info_file: str) -> None:
    """Write normalized VALS seeds and the company pool used for substitution."""
    input_path = Path(input_file)
    records: list[dict]
    with input_path.open(encoding="utf-8") as source:
        first = source.read(1)
        source.seek(0)
        if input_path.suffix == ".jsonl" or first == "{":
            records = [json.loads(line) for line in source if line.strip()]
        else:
            loaded = json.load(source)
            if not isinstance(loaded, list):
                raise ValueError("VALS JSON input must contain a list of records")
            records = loaded

    seeds = [_seed_record(index, record) for index, record in enumerate(records)]
    if not all(seed["original_question"] for seed in seeds):
        raise ValueError("every VALS record must contain a non-empty question")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as destination:
        for seed in seeds:
            destination.write(json.dumps(seed, ensure_ascii=False) + "\n")
    logger.info("Wrote %d VALS seed questions to %s", len(seeds), output_path)

    company_path = Path(company_info_file)
    if company_path.exists() and company_path.stat().st_size:
        logger.info("Using existing company information: %s", company_path)
    else:
        _create_sp500_company_info(str(company_path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--company_info_file", required=True)
    args = parser.parse_args()
    convert_vals_data(args.input_file, args.output_file, args.company_info_file)


if __name__ == "__main__":
    main()
