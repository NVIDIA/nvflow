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
"""Prepare exhaustive or sampled company/year combinations for TSDG."""

import argparse
import json
import random
from itertools import combinations

import jsonlines
import pandas as pd

from nvflow.utils import setup_logger

logger = setup_logger(__name__)

keep_key_list = [
    "QID",
    "original_question",
    "question_type",
    "accession_number",
    "item",
    "form_types",
    "report_dates",
]

DEFAULT_FORM_WEIGHTS = {"10-K": 1.0, "10-Q": 1.0}


def _load_company_data(company_info_file):
    """Load companies and construct same-subindustry comparison pairs."""
    company_info_df = pd.read_csv(company_info_file, sep="\t")
    all_companies = list(zip(company_info_df["Company"], company_info_df["Ticker"], strict=True))
    subindustry_groups = company_info_df.groupby("SubIndustry")[["Company", "Ticker"]].apply(
        lambda rows: list(zip(rows["Company"], rows["Ticker"], strict=True))
    )
    same_subindustry_pairs = []
    for companies in subindustry_groups:
        same_subindustry_pairs.extend(combinations(companies, 2))
    return company_info_df, all_companies, same_subindustry_pairs


def generate_company_year_combinations_for_question(
    entry, company_ticker_tuples, start_year, end_year
):
    """Generate all company/year combinations for a single question."""
    output = []
    for (company_a, ticker_a), (company_b, ticker_b) in company_ticker_tuples:
        for year in range(start_year, end_year + 1):
            new_entry = {}
            for key in keep_key_list:
                # Only copy key if it exists in the entry (for backwards compatibility)
                if key in entry:
                    new_entry[key] = entry[key]
            new_entry["company_a"] = company_a
            new_entry["ticker_a"] = ticker_a
            new_entry["company_b"] = company_b
            new_entry["ticker_b"] = ticker_b
            new_entry["year"] = year
            output.append(new_entry)
    return output


def generate_company_year_combinations_for_all_questions(
    input_file, company_info_file, start_year, end_year, output_file
):
    """
    Generate all company/year combinations for all questions.

    Args:
        input_file: Path to input jsonl file with seed questions
        company_info_file: Path to company info tsv file
        start_year: Start year for combinations
        end_year: End year for combinations
        output_file: Path to output jsonl file
    """
    logger.info("Input file: %s", input_file)
    logger.info("Company info file: %s", company_info_file)

    # Read the files
    company_info_df = pd.read_csv(company_info_file, sep="\t")
    with jsonlines.open(input_file) as reader:
        input_data = list(reader)

    logger.info("Using %d input entries and %d companies", len(input_data), len(company_info_df))
    logger.info("Year range: %d - %d", start_year, end_year)

    # Group companies by subindustry
    subindustry_groups = company_info_df.groupby("SubIndustry")[["Company", "Ticker"]].apply(
        lambda x: list(zip(x["Company"], x["Ticker"], strict=True))
    )

    # Generate all distinct pairs for each subindustry
    company_ticker_pairs_same_subindustry = []
    for companies in subindustry_groups:
        if len(companies) > 1:
            pairs = list(combinations(companies, 2))
            company_ticker_pairs_same_subindustry.extend(pairs)

    # Generate single company tuples (for questions with single accession number)
    companies_w_tickers = [
        ((company, ticker), (None, None))
        for company, ticker in zip(
            company_info_df["Company"], company_info_df["Ticker"], strict=True
        )
    ]

    logger.info(
        "Generated %d company pairs (same subindustry)", len(company_ticker_pairs_same_subindustry)
    )
    logger.info("Generated %d single company entries", len(companies_w_tickers))

    # Generate combinations for all questions
    total_output = 0
    with jsonlines.open(output_file, mode="w") as writer:
        for entry in input_data:
            unique_accession_numbers = set(entry["accession_number"].split(";"))
            if len(unique_accession_numbers) > 1:
                # Comparison questions - use company pairs
                output = generate_company_year_combinations_for_question(
                    entry, company_ticker_pairs_same_subindustry, start_year, end_year
                )
            else:
                # Single company questions
                output = generate_company_year_combinations_for_question(
                    entry, companies_w_tickers, start_year, end_year
                )
            total_output += len(output)
            for new_entry in output:
                writer.write(new_entry)

    logger.info("Written %d questions to: %s", total_output, output_file)


def sample_question_data(
    input_file,
    company_info_file,
    start_year,
    end_year,
    output_file,
    total_samples=10000,
    seed=42,
    form_weights=None,
):
    """Sample a fixed-size, reproducible set of seed/company/year/form combinations."""
    rng = random.Random(seed)
    effective_form_weights = form_weights or DEFAULT_FORM_WEIGHTS
    form_types = sorted(effective_form_weights)
    weights = [effective_form_weights[form_type] for form_type in form_types]

    company_info, all_companies, same_subindustry_pairs = _load_company_data(company_info_file)
    with jsonlines.open(input_file) as reader:
        input_data = list(reader)
    if not input_data:
        raise ValueError("input seed file is empty")
    if not all_companies:
        raise ValueError("company information file is empty")

    logger.info(
        "Sampling %d combinations from %d questions and %d companies (seed=%d)",
        total_samples,
        len(input_data),
        len(company_info),
        seed,
    )

    years = list(range(start_year, end_year + 1))
    with jsonlines.open(output_file, mode="w") as writer:
        for _ in range(total_samples):
            entry = rng.choice(input_data)
            unique_accessions = set(entry.get("accession_number", "unknown").split(";"))
            is_multi_company = len(unique_accessions) > 1
            new_entry = {key: entry[key] for key in keep_key_list if key in entry}
            new_entry["year"] = rng.choice(years)

            sampled_form = rng.choices(form_types, weights=weights, k=1)[0]
            new_entry["form_types"] = ";".join([sampled_form] * len(unique_accessions))

            if is_multi_company and same_subindustry_pairs:
                (company_a, ticker_a), (company_b, ticker_b) = rng.choice(same_subindustry_pairs)
            else:
                company_a, ticker_a = rng.choice(all_companies)
                company_b, ticker_b = None, None
            new_entry.update(
                company_a=company_a,
                ticker_a=ticker_a,
                company_b=company_b,
                ticker_b=ticker_b,
            )
            writer.write(new_entry)

    logger.info("Written %d sampled questions to: %s", total_samples, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_file", type=str, required=True, help="Input seed questions jsonl file"
    )
    parser.add_argument(
        "--company_info_file", type=str, required=True, help="Company info tsv file"
    )
    parser.add_argument("--start_year", type=int, required=True, help="Start year for combinations")
    parser.add_argument("--end_year", type=int, required=True, help="End year for combinations")
    parser.add_argument("--output_file", type=str, required=True, help="Output jsonl file")
    parser.add_argument("--mode", choices=["exhaustive", "sample"], default="exhaustive")
    parser.add_argument("--total_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--form_weights", type=str)
    args = parser.parse_args()

    if args.mode == "sample":
        sample_question_data(
            args.input_file,
            args.company_info_file,
            args.start_year,
            args.end_year,
            args.output_file,
            total_samples=args.total_samples,
            seed=args.seed,
            form_weights=json.loads(args.form_weights) if args.form_weights else None,
        )
    else:
        generate_company_year_combinations_for_all_questions(
            args.input_file,
            args.company_info_file,
            args.start_year,
            args.end_year,
            args.output_file,
        )
