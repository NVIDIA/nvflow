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
"""Parallel section extraction from downloaded SEC filings."""

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from nvflow.utils import setup_logger

from ..index.corpus import resolve_source_path
from .extractors import Sec8KExtractor, Sec10KExtractor, Sec10QExtractor
from .parquet_io import write_parquet_atomic

logger = setup_logger(__name__)

logging.captureWarnings(True)
logging.getLogger("py.warnings").setLevel(logging.ERROR)

# Extraction over a full corpus runs for hours; without periodic flushes a
# preemption loses every section row written so far and the work repeats.
_FLUSH_EVERY_FILINGS = 2000


def _normalize_form_type(form_type: str) -> str:
    return form_type.upper().replace("-", "")


def is_section_supported(form_type: str) -> bool:
    """Whether :func:`get_extractor_class` can return an extractor for *form_type*."""
    return _normalize_form_type(form_type).startswith(("10K", "10Q", "8K"))


def get_extractor_class(form_type: str) -> type:
    """Return the appropriate extractor class for *form_type*.

    Handles amendment suffixes like 10-K/A, 10-Q/A, 8-K/A.
    """
    form_type_normalized = _normalize_form_type(form_type)

    if form_type_normalized.startswith("10K"):
        return Sec10KExtractor
    elif form_type_normalized.startswith("10Q"):
        return Sec10QExtractor
    elif form_type_normalized.startswith("8K"):
        return Sec8KExtractor
    else:
        raise ValueError(f"Unsupported form type: {form_type}")


def _extract_section_worker(filing_metadata: dict, artifact_root: str) -> tuple[list, str]:
    """Worker function for parallel section extraction."""
    primary_doc_path = resolve_source_path(filing_metadata["file_location"], Path(artifact_root))
    html_content = primary_doc_path.read_text(encoding="utf-8")

    extractor_class = get_extractor_class(filing_metadata["form_type"])
    sections = extractor_class().extract_sections(html_content=html_content)
    section_metadatas: list = []

    for section_data in sections.values():
        filename = section_data.get("filename")
        if not filename or not section_data.get("html_content"):
            continue
        section_path = primary_doc_path.parent / f"{filename}.html"
        section_html = section_data["html_content"]
        size = len(section_html.encode("utf-8"))
        section_path.write_text(section_html, encoding="utf-8")
        section_metadata = filing_metadata.copy()
        section_metadata["file_type"] = "section"
        section_metadata["file_location"] = str(section_path)
        section_metadata["size"] = size
        section_metadatas.append(section_metadata)

    log_message = (
        f"Extracted {len(section_metadatas)} sections: "
        f"ticker: {filing_metadata['ticker']}, "
        f"form type: {filing_metadata['form_type']}, "
        f"accession number: {filing_metadata['accession_number']}"
    )
    return section_metadatas, log_message


def _write_sections(
    metadata_df: pd.DataFrame, section_metadatas: list, metadata_path: str
) -> pd.DataFrame:
    """Append section rows to the metadata parquet and return the updated frame."""
    if not section_metadatas:
        return metadata_df
    metadata_df = pd.concat([metadata_df, pd.DataFrame(section_metadatas)])
    write_parquet_atomic(metadata_df, metadata_path)
    return metadata_df


def extract_sections(config: dict) -> None:
    """Extract sections from all downloaded filings that haven't been processed yet."""
    metadata_path = os.path.join(config["output_dir"], config["metadata_filename"])
    if not os.path.exists(metadata_path):
        raise ValueError("Metadata file not found, cannot extract sections")
    metadata_df = pd.read_parquet(metadata_path)

    sections_df = metadata_df[metadata_df["file_type"] == "section"]
    if not sections_df.empty:
        extracted_filings = set(
            sections_df[["ticker", "accession_number", "form_type"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
    else:
        extracted_filings = set()

    primary_documents_df = metadata_df[metadata_df["file_type"] == "primary_document"]
    total_primary_docs = len(primary_documents_df)

    # Worker results are consumed through a bare future.result(), so a single
    # form without an extractor would abort the whole run.
    supported_mask = primary_documents_df["form_type"].map(is_section_supported).astype(bool)
    skipped_count = int((~supported_mask).sum())
    primary_documents_df = primary_documents_df[supported_mask]

    if extracted_filings:
        mask = ~primary_documents_df.apply(
            lambda row: (row["ticker"], row["accession_number"], row["form_type"])
            in extracted_filings,
            axis=1,
        )
        primary_documents_df = primary_documents_df[mask]

    primary_documents = primary_documents_df.to_dict(orient="records")

    already_extracted = total_primary_docs - skipped_count - len(primary_documents)
    logger.info("Found %d total primary documents", total_primary_docs)
    logger.info("Skipping %d primary documents whose form type has no extractor", skipped_count)
    logger.info("Already extracted sections for %d filings", already_extracted)
    logger.info("Processing %d new filings", len(primary_documents))

    if len(primary_documents) == 0:
        logger.info("No new filings to extract sections from. Skipping section extraction.")
        return

    filing_section_metadatas: list = []
    log_file = os.path.join(config["output_dir"], "section_extraction.log")
    if not os.path.exists(log_file):
        with open(log_file, "w") as fp:
            pass

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(_extract_section_worker, fm, config["output_dir"]): fm
            for fm in primary_documents
        }
        completed = 0
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting sections"):
            section_metadatas, log_message = future.result()
            filing_section_metadatas.extend(section_metadatas)
            with open(log_file, "a") as fp:
                fp.write(log_message + "\n")

            completed += 1
            if completed % _FLUSH_EVERY_FILINGS == 0:
                metadata_df = _write_sections(metadata_df, filing_section_metadatas, metadata_path)
                filing_section_metadatas.clear()

    _write_sections(metadata_df, filing_section_metadatas, metadata_path)
