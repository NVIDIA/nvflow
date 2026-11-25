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
import argparse
import os
import re
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from functools import partial
from typing import Any

import jsonlines
import pandas as pd
import tiktoken
from markdownify import markdownify as md
from tqdm import tqdm

from nvflow.utils import setup_logger

logger = setup_logger(__name__)

FORM_10K_ITEMS = [
    "1",
    "1A",
    "1B",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "7A",
    "8",
    "9",
    "9A",
    "9B",
    "9C",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
]
FORM_10Q_ITEMS = [
    "part1item1",
    "part1item2",
    "part1item3",
    "part1item4",
    "part2item1",
    "part2item1a",
    "part2item2",
    "part2item3",
    "part2item4",
    "part2item5",
    "part2item6",
    "part1item9a",
    "part1item9b",
    "part1item9c",
    "part1item10",
    "part1item11",
    "part1item12",
    "part1item13",
    "part1item14",
    "part1item15",
]


def get_filename_from_item_name(item_name: str, form_type: str) -> str | None:
    """Get the filename from the item name. Also depends on the form_type"""
    match = re.match(r"ITEM\s+(\d+[A-Z]?)", item_name.upper().strip())
    if not match:
        return None

    item_num = match.group(1)

    if form_type == "10-K":
        filename = f"{item_num.upper()}.html"
    elif form_type == "10-Q":
        if item_num.upper() == "1A":
            filename = "part2item1a.html"
        else:
            filename = f"part1item{item_num.lower()}.html"
    else:
        filename = None
    return filename


def get_previous_filename_from_filename(full_filename: str, form_type: str) -> str | None:
    filename, extension = os.path.splitext(os.path.basename(full_filename))
    folder = os.path.dirname(full_filename)
    if form_type == "10-K":
        index = FORM_10K_ITEMS.index(filename)
        if index == 0:
            return None
        new_filename = FORM_10K_ITEMS[index - 1]
    elif form_type == "10-Q":
        index = FORM_10Q_ITEMS.index(filename)
        if index == 0:
            return None
        new_filename = FORM_10Q_ITEMS[index - 1]
    else:
        return None
    new_fullfile = f"{folder}/{new_filename}{extension}"
    if file_exists(new_fullfile):  # Changed from os.path.exists
        return new_fullfile
    else:
        return None


def convert_html_to_markdown(html: str) -> str:
    """Convert to markdown for easier comparison"""
    markdown = md(html, heading_style="ATX", strip=["script", "style"])
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    return markdown.strip()


def truncate_section_markdown(markdown: str, max_length: int, num_files: int) -> tuple[str, bool]:
    """
    Truncate the markdown text to a maximum length.
    Args:
        markdown: The markdown text to truncate.
        max_length: The maximum length of the markdown text - same as max context length of GPT-OSS.
        num_files: The number of files which will be concatenated together.

    Returns:
        truncated_markdown: The truncated markdown text.
        was_truncated: A boolean indicating if the text was truncated.

    """
    max_length = max_length // num_files
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(markdown)
    if len(tokens) <= max_length:
        return markdown, False
    return enc.decode(tokens[:max_length]), True


def get_closest_filing_month_accession_number(
    orig_report_date: str, ticker: str, year: str, form_type: str, filings_metadata: pd.DataFrame
) -> str | None:
    """Get the accession number of the filing with the closest filing month."""
    orig_report_date = datetime.strptime(orig_report_date, "%Y-%m-%d")
    orig_report_date_month = orig_report_date.month
    closest_filing_month_accession_number = None
    closest_filing_month_diff = float("inf")
    # Filter by ticker and form_type, then extract year from report_date
    subset_metadata = filings_metadata[
        (filings_metadata["ticker"] == ticker) & (filings_metadata["form_type"] == form_type)
    ]

    # Further filter by year extracted from report_date
    subset_metadata = subset_metadata[subset_metadata["report_date"].str[:4] == str(year)]
    for _, row in subset_metadata.iterrows():
        filing_date = datetime.strptime(row["report_date"], "%Y-%m-%d")
        filing_date_month = filing_date.month
        filing_month_diff = abs(filing_date_month - orig_report_date_month)
        if filing_month_diff < closest_filing_month_diff:
            closest_filing_month_diff = filing_month_diff
            closest_filing_month_accession_number = row["accession_number"]
    return closest_filing_month_accession_number


def get_accession_number(
    ticker: str, year: str, form_type: str, filings_metadata: pd.DataFrame
) -> str | None:
    """Get the accession number of the filing for this ticker and year."""
    subset_metadata = filings_metadata[
        (filings_metadata["ticker"] == ticker) & (filings_metadata["form_type"] == form_type)
    ]
    subset_metadata = subset_metadata[subset_metadata["report_date"].str[:4] == str(year)]
    if len(subset_metadata) == 0:
        return None
    return subset_metadata["accession_number"].iloc[-1]


def get_matching_section_filepath(
    form_type: str,
    orig_report_date: str,
    filings_metadata: pd.DataFrame,
    ticker: str,
    year: str,
    orig_item_name: str,
    filings_folder: str,
) -> str | None:
    """Get the filepath to the matching section of the document."""
    filename = get_filename_from_item_name(orig_item_name, form_type)
    if not filename:
        return None
    if form_type == "10-K":
        # There is only one 10-K per year, so we can just use the filename
        new_accession_number = get_accession_number(ticker, year, form_type, filings_metadata)
        if not new_accession_number:
            return None
        full_filename = (
            f"{filings_folder}/{ticker}/{form_type}/{year}/{new_accession_number}/{filename}"
        )
        if file_exists(full_filename):  # Changed from os.path.exists
            return full_filename
        else:
            new_filename = get_previous_filename_from_filename(full_filename, form_type)
            if new_filename:
                return new_filename
            else:
                return None
    elif form_type == "10-Q":
        # Since there are multiple 10-Qs, we take the one with the closest filing month (because that indicates filing in the same quarter of a different year)
        closest_filing_month_accession_number = get_closest_filing_month_accession_number(
            orig_report_date, ticker, year, form_type, filings_metadata
        )
        full_filename = f"{filings_folder}/{ticker}/{form_type}/{year}/{closest_filing_month_accession_number}/{filename}"
        if file_exists(full_filename):  # Changed from os.path.exists
            return full_filename
        else:
            new_filename = get_previous_filename_from_filename(full_filename, form_type)
            if new_filename:
                return new_filename
            else:
                return None
    else:
        return None


# Global cache per worker - built on demand
_worker_file_cache: OrderedDict[str, str] = OrderedDict()  # filepath -> markdown content
_worker_nonexistent_files: set[str] = set()  # Track files known to not exist
_MAX_CACHE_SIZE = 100  # Keep only 100 most recent files in memory


def get_cached_markdown(filepath: str) -> str:
    """Get markdown from cache (LRU with max size), loading from disk if needed"""
    global _worker_file_cache

    if filepath in _worker_file_cache:
        # Move to end (most recently used)
        _worker_file_cache.move_to_end(filepath)
        return _worker_file_cache[filepath]

    # Load and cache on first use
    with open(filepath) as f:
        html = f.read()
    markdown = convert_html_to_markdown(html)

    # Add to cache
    _worker_file_cache[filepath] = markdown

    # Evict oldest if cache too large
    if len(_worker_file_cache) > _MAX_CACHE_SIZE:
        _worker_file_cache.popitem(last=False)  # Remove oldest

    return markdown


def file_exists(filepath: str) -> bool:
    """
    Check if file exists, using cached knowledge of non-existent files.
    If we've already checked this file and it didn't exist, return False immediately.
    Otherwise, check with os.path.exists and cache the result if it doesn't exist.
    """
    global _worker_nonexistent_files

    # Fast path: we know it doesn't exist
    if filepath in _worker_nonexistent_files:
        return False

    # Check filesystem
    exists = os.path.exists(filepath)

    # Cache negative results
    if not exists:
        _worker_nonexistent_files.add(filepath)

    return exists


def process_question_item(
    question_item: dict[str, Any],
    filings_metadata: pd.DataFrame,
    filings_folder: str,
    token_limit: int,
) -> dict[str, Any] | None:
    """Process a single question item with lazy file caching"""

    # Extract parameters
    form_types = question_item["form_types"].split(";")
    sdg_companies = [question_item["ticker_a"], question_item["ticker_b"]]
    sdg_year = question_item["year"]
    orig_item_names = question_item["item"].split(";")
    orig_report_dates = question_item["report_dates"].split(";")

    if len(form_types) != len(orig_item_names):
        return None

    # Get filepaths - called ONCE per question
    filepaths = []
    for i in range(len(form_types)):
        form_type = form_types[i]
        orig_report_date = orig_report_dates[i]
        orig_item_name = orig_item_names[i]
        sdg_company = sdg_companies[i]
        if not sdg_company or sdg_company == "" or sdg_company == "None":
            return None
        filepath = get_matching_section_filepath(
            form_type=form_type,
            orig_report_date=orig_report_date,
            filings_metadata=filings_metadata,
            ticker=sdg_company,
            year=sdg_year,
            orig_item_name=orig_item_name,
            filings_folder=filings_folder,
        )
        if not filepath:
            return None
        filepaths.append(filepath)

    # Build context using cached files (lazy loading)
    full_context = []
    num_files = len(filepaths)

    for filepath in filepaths:
        # Get from cache (loads on first access)
        context = get_cached_markdown(filepath)
        context, _ = truncate_section_markdown(context, token_limit, num_files)
        full_context.append(context)

    full_context = "\n\n".join(full_context)
    filepath_str = ";".join(filepaths)

    return {**question_item, "context": full_context, "filepath": filepath_str}


def process_question_chunk(
    questions: list[dict[str, Any]],
    filings_metadata: pd.DataFrame,
    filings_folder: str,
    token_limit: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Process a chunk of questions in a single worker (maximizes cache reuse)

    Returns:
        results: List of processed questions with results or errors
        stats: Dictionary with success/failure counts
    """
    results = []
    stats = {"success": 0, "failed": 0}

    for question_item in questions:
        try:
            result = process_question_item(
                question_item,
                filings_metadata,
                filings_folder,
                token_limit,
            )
            if result:
                results.append(result)
                stats["success"] += 1
            else:
                stats["failed"] += 1
        except Exception:
            stats["failed"] += 1

    return results, stats


def map_questions_to_context(
    input_file: str,
    output_file: str,
    filings_dir: str,
    filings_metadata: str,
    token_limit: int,
    checkpoint_every: int = 500,
    num_workers: int = 32,
) -> None:
    """Map questions to contexts with lazy caching and auto-resume"""

    filings_metadata_df = pd.read_parquet(filings_metadata)

    # Load all questions
    logger.info("Loading questions...")
    with jsonlines.open(input_file) as reader:
        all_questions = list(reader)
    logger.info("Loaded %d questions", len(all_questions))

    # Check for existing progress
    processed_questions = set()
    if os.path.exists(output_file):
        logger.info("Found existing output file, resuming...")
        with jsonlines.open(output_file) as reader:
            for item in reader:
                question = item["problem"]
                if question:
                    processed_questions.add(question)
        logger.info("Already processed: %d questions", len(processed_questions))

    # Filter to unprocessed (use "problem" field to match what was added to processed_questions)
    questions_to_process = [q for q in all_questions if q["problem"] not in processed_questions]

    if not questions_to_process:
        logger.info("All questions already processed!")
        return

    logger.info("Processing %d remaining questions...", len(questions_to_process))

    # Sort questions by file lookup keys to maximize cache hits per worker
    logger.info("Sorting questions by file lookup keys...")
    questions_to_process.sort(
        key=lambda q: (
            q.get("ticker_a", ""),
            q.get("year", ""),
            q.get("form_types", ""),
            q.get("report_dates", ""),
            q.get("item", ""),
        )
    )
    logger.info("Questions sorted")

    # Split into equal chunks for each worker
    chunk_size = len(questions_to_process) // num_workers
    remainder = len(questions_to_process) % num_workers

    chunks = []
    start_idx = 0
    for i in range(num_workers):
        # Distribute remainder across first few chunks
        extra = 1 if i < remainder else 0
        end_idx = start_idx + chunk_size + extra
        if start_idx < len(questions_to_process):
            chunks.append(questions_to_process[start_idx:end_idx])
        start_idx = end_idx

    logger.info("Split into %d chunks: %s", len(chunks), [len(c) for c in chunks])
    logger.info("Using %d workers with sorted chunks for maximum cache reuse...", num_workers)

    process_func = partial(
        process_question_chunk,
        filings_metadata=filings_metadata_df,
        filings_folder=filings_dir,
        token_limit=token_limit,
    )

    completed_count = 0
    success_count = 0
    failed_count = 0
    checkpoint_buffer = []

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with jsonlines.open(output_file, mode="a") as writer:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit each chunk to a worker
            futures = {
                executor.submit(process_func, chunk): idx for idx, chunk in enumerate(chunks)
            }

            # Create progress bar for total questions, not chunks
            with tqdm(
                total=len(questions_to_process), desc="Processing questions", unit="q"
            ) as pbar:
                # Process results as they complete
                for future in as_completed(futures):
                    chunk_idx = futures[future]
                    try:
                        results, stats = future.result()  # Returns list of results and stats

                        success_count += stats["success"]
                        failed_count += stats["failed"]

                        # Update progress bar by number of questions in this chunk
                        pbar.update(len(results))
                        pbar.set_postfix(
                            {"success": success_count, "failed": failed_count}, refresh=True
                        )

                        logger.info(
                            "Chunk %d/%d complete: %d success, %d failed",
                            chunk_idx + 1,
                            len(chunks),
                            stats["success"],
                            stats["failed"],
                        )

                        # Only write successful results (those with filepath)
                        for result in results:
                            if result.get("filepath"):  # Only add if filepath exists
                                checkpoint_buffer.append(result)
                                completed_count += 1

                                # Checkpoint periodically
                                if len(checkpoint_buffer) >= checkpoint_every:
                                    writer.write_all(checkpoint_buffer)
                                    checkpoint_buffer = []
                                    logger.info(
                                        "  Checkpointed %d results so far...", completed_count
                                    )

                    except Exception as exc:
                        logger.error("Chunk %d processing error: %s", chunk_idx + 1, exc)

            # Write remaining buffer
            if checkpoint_buffer:
                writer.write_all(checkpoint_buffer)

    total_in_file = len(processed_questions) + completed_count
    logger.info("Complete! Processed %d questions this run", completed_count)
    logger.info("  - Success: %d", success_count)
    logger.info("  - Failed: %d", failed_count)
    logger.info("Total in output: %d questions", total_in_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map questions to filing contexts")
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to input questions file (jsonl)"
    )
    parser.add_argument(
        "--filings_metadata", type=str, required=True, help="Path to filings metadata Parquet file"
    )
    parser.add_argument(
        "--filings_dir", type=str, required=True, help="Directory containing filings data"
    )
    parser.add_argument("--output_file", type=str, required=True, help="Path to output JSONL file")
    parser.add_argument(
        "--token_limit", type=int, default=3000, help="Token limit for LLM/chunked filing context"
    )
    args = parser.parse_args()

    # Call the mapping function
    map_questions_to_context(
        input_file=args.input_file,
        output_file=args.output_file,
        filings_dir=args.filings_dir,
        filings_metadata=args.filings_metadata,
        token_limit=args.token_limit,
    )
