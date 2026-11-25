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
"""Create seed data files for template-based SDG pipeline."""

import argparse
import os
import random
from io import StringIO

import jsonlines
import pandas as pd

from nvflow.utils import setup_logger

logger = setup_logger(__name__)

DEFAULT_RANDOM_SEED = 42


def create_input_file(output_path, sec_identity):
    """Create the input file with SecQue data and metadata from SEC filings."""
    from datasets import load_dataset
    from edgar import get_by_accession_number, set_identity

    logger.info("Creating input file from HuggingFace SecQue dataset...")

    # Load the SecQue dataset
    ds = load_dataset("nogabenyoash/SecQue")

    # Process dataset entries
    data = []
    acc_no_mapping = {}
    for i in range(len(ds["train"])):
        entry = ds["train"][i]
        new_entry = {}
        new_entry["original_question"] = entry["Question"]
        new_entry["ground_truth"] = entry["ground_truth_answer"]
        for key in ["QID", "accession_number", "item", "question_type"]:
            new_entry[key] = entry[key]
        data.append(new_entry)
        accession_numbers = entry["accession_number"].split(";")
        for acc_no in accession_numbers:
            acc_no_mapping[acc_no] = {}

    logger.info("Loaded %d entries from SecQue dataset", len(data))
    logger.info("Found %d unique accession numbers", len(acc_no_mapping))

    # Set identity for SEC EDGAR API
    logger.info("Using SEC identity: %s", sec_identity)
    set_identity(sec_identity)

    # Fetch filing metadata for each accession number
    logger.info("Fetching filing metadata from SEC EDGAR API...")
    for i, (k, v) in enumerate(acc_no_mapping.items()):
        if (i + 1) % 10 == 0:
            logger.info("  Processed %d/%d accession numbers...", i + 1, len(acc_no_mapping))
        filing = get_by_accession_number(k)
        v["report_date"] = str(filing.period_of_report)
        v["form_type"] = filing.form

    # Add form types and report dates to each entry
    for entry in data:
        accession_numbers = entry["accession_number"].split(";")
        form_types = []
        report_dates = []
        for acc_no in accession_numbers:
            form_types.append(acc_no_mapping[acc_no]["form_type"])
            report_dates.append(acc_no_mapping[acc_no]["report_date"])
        entry["form_types"] = ";".join(form_types)
        entry["report_dates"] = ";".join(report_dates)

    # Sort and save to output file
    data = sorted(data, key=lambda x: x.get("original_question", ""))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with jsonlines.open(output_path, "w") as writer:
        for entry in data:
            writer.write(entry)

    logger.info("Created input file with %d entries: %s", len(data), output_path)
    return data


def create_company_info_file(output_path):
    """Create the company info file with SP100 companies and their GICS sectors."""
    import requests
    from bs4 import BeautifulSoup

    logger.info("Creating company info file from Wikipedia...")

    # Fetch SP500 companies from Wikipedia
    sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "CompanyClassifierBot/1.0 (https://example.com/contact; "
            "mailto:youremail@example.com) requests library"
        )
    }

    logger.info("Fetching S&P 500 companies from Wikipedia...")
    resp = requests.get(sp500_url, headers=headers)
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", {"id": "constituents"})
    sp500_df = pd.read_html(StringIO(str(table)))[0]
    sp500_df = sp500_df[["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]].rename(
        columns={
            "Symbol": "Ticker",
            "Security": "Company",
            "GICS Sector": "Sector",
            "GICS Sub-Industry": "SubIndustry",
        }
    )

    # Fetch SP100 companies from Wikipedia
    logger.info("Fetching S&P 100 companies from Wikipedia...")
    sp100_url = "https://en.wikipedia.org/wiki/S%26P_100"
    resp = requests.get(sp100_url, headers=headers)
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", {"id": "constituents"})
    sp100_df = pd.read_html(StringIO(str(table)))[0]
    sp100_df = sp100_df.rename(columns={"Symbol": "Ticker", "Name": "Company"})
    sp100_companies = set(sp100_df.Ticker.tolist())

    # Filter SP500 to SP100 and exclude GOOGL (keep only GOOG)
    sp100_final = sp500_df[sp500_df.Ticker.isin(sp100_companies)]
    sp100_final = sp100_final[sp100_final.Ticker != "GOOGL"]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sp100_final.to_csv(output_path, index=False, sep="\t")

    logger.info("Created company info file with %d companies: %s", len(sp100_final), output_path)
    return sp100_final


def create_seed_data(
    output_file,
    company_info_file,
    sec_identity,
    filter_company_list=None,
    num_seed_questions=None,
):
    """
    Create seed data files for SDG pipeline.

    Args:
        output_file: Path to write the seed questions jsonl file
        company_info_file: Path to write the company info tsv file
        sec_identity: SEC EDGAR identity string (e.g., "Name email@example.com")
        filter_company_list: Optional list of company tickers to filter to
        num_seed_questions: Optional number of questions to sample (with fixed seed)
    """
    logger.info("=" * 60)
    logger.info("Creating Seed Data for Template-Based SDG Pipeline")
    logger.info("=" * 60)

    # Check if input file already exists
    if os.path.exists(output_file):
        logger.info("Input file already exists: %s", output_file)
        logger.info("Loading existing data...")
        with jsonlines.open(output_file) as reader:
            data = list(reader)
        logger.info("Loaded %d entries from existing file", len(data))
    else:
        logger.info("Input file does not exist: %s", output_file)
        logger.info("Will create from HuggingFace SecQue dataset + SEC EDGAR API...")
        data = create_input_file(output_file, sec_identity)

    # Check if company info file already exists
    if os.path.exists(company_info_file):
        logger.info("Company info file already exists: %s", company_info_file)
        logger.info("Loading existing data...")
        company_df = pd.read_csv(company_info_file, sep="\t")
        logger.info("Loaded %d companies from existing file", len(company_df))
    else:
        logger.info("Company info file does not exist: %s", company_info_file)
        logger.info("Will create from Wikipedia S&P 100 data...")
        company_df = create_company_info_file(company_info_file)

    # Filter companies if specified
    if filter_company_list:
        filter_set = set(filter_company_list)
        original_count = len(company_df)
        company_df = company_df[company_df.Ticker.isin(filter_set)]
        logger.info("Filtered companies: %d -> %d", original_count, len(company_df))
        logger.info("  Companies kept: %s", ", ".join(sorted(company_df.Ticker.tolist())))

        # Re-save filtered company info
        company_df.to_csv(company_info_file, index=False, sep="\t")
        logger.info("  Updated company info file: %s", company_info_file)

    # Sample questions if specified
    if num_seed_questions and len(data) > num_seed_questions:
        original_count = len(data)
        random.seed(DEFAULT_RANDOM_SEED)
        data = random.sample(data, num_seed_questions)
        logger.info(
            "Sampled questions: %d -> %d (seed=%d)", original_count, len(data), DEFAULT_RANDOM_SEED
        )

        # Re-save filtered data
        with jsonlines.open(output_file, "w") as writer:
            for entry in data:
                writer.write(entry)
        logger.info("  Updated input file: %s", output_file)

    logger.info("=" * 60)
    logger.info("Seed Data Creation Complete")
    logger.info("=" * 60)
    logger.info("  Input file: %s (%d questions)", output_file, len(data))
    logger.info("  Company info: %s (%d companies)", company_info_file, len(company_df))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create seed data for template-based SDG pipeline")
    parser.add_argument(
        "--output_file", type=str, required=True, help="Output path for seed questions jsonl"
    )
    parser.add_argument(
        "--company_info_file", type=str, required=True, help="Output path for company info tsv"
    )
    parser.add_argument(
        "--sec_identity",
        type=str,
        required=True,
        help="SEC EDGAR identity string (e.g., 'YourCompany your.email@example.com')",
    )
    parser.add_argument(
        "--filter_company_list",
        type=str,
        default=None,
        help="Comma-separated list of company tickers to filter to (e.g., 'AAPL,GOOG,MSFT')",
    )
    parser.add_argument(
        "--num_seed_questions",
        type=int,
        default=None,
        help="Number of seed questions to sample (uses fixed seed for reproducibility)",
    )
    args = parser.parse_args()

    # Parse filter_company_list from comma-separated string
    filter_company_list = None
    if args.filter_company_list:
        filter_company_list = [t.strip() for t in args.filter_company_list.split(",")]

    create_seed_data(
        output_file=args.output_file,
        company_info_file=args.company_info_file,
        sec_identity=args.sec_identity,
        filter_company_list=filter_company_list,
        num_seed_questions=args.num_seed_questions,
    )
