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
"""Download SEC filings (primary documents + exhibits) from EDGAR."""

import os
import re
import time
from datetime import date, datetime
from urllib.parse import urljoin

import httpx
import pandas as pd
from edgar import Company, Filing, set_identity
from tqdm import tqdm

from nvflow.utils import setup_logger

from .parquet_io import write_parquet_atomic

logger = setup_logger(__name__)

_MAX_RETRIES = 3


def fix_image_urls(url: str, html_content: str) -> str:
    """Rewrite relative image URLs in filing HTML to absolute SEC URLs."""
    if not url:
        return html_content

    img_pattern = r'<img\s+([^>]*?)src=["\']([^"\']+)["\']([^>]*?)>'

    def replace_img_src(match):
        before_src = match.group(1)
        src_value = match.group(2)
        after_src = match.group(3)
        if not src_value.startswith(("http://", "https://", "//", "data:")):
            absolute_url = urljoin(url, src_value)
            return f'<img {before_src}src="{absolute_url}"{after_src}>'
        return match.group(0)

    return re.sub(img_pattern, replace_img_src, html_content, flags=re.IGNORECASE)


def download_filings(config: dict) -> pd.DataFrame:
    """Download filings from SEC EDGAR for all configured tickers."""
    identity_info = config["identity_info"]
    set_identity(identity_info["email_address"])
    output_dir = os.path.join(config["output_dir"], config["data_dir"])
    os.makedirs(output_dir, exist_ok=True)

    start_year, end_year = config["start_year"], config["end_year"]
    form_types = config["form_types"]
    metadata_path = os.path.join(config["output_dir"], config["metadata_filename"])
    metadata_df = pd.DataFrame()
    acc_nos_to_skip: set = set()

    if os.path.exists(metadata_path):
        logger.info("Loading metadata from %s", metadata_path)
        metadata_df = pd.read_parquet(metadata_path)
        acc_nos_to_skip = set(metadata_df["accession_number"].values.tolist())
    else:
        logger.info("No metadata found at %s, not skipping any filings", metadata_path)

    for ticker in config["tickers"]:
        try:
            company = Company(ticker)
        except Exception as e:
            logger.warning("Skipping ticker %s: %s", ticker, e)
            continue
        logger.info("Downloading filings for %s from %d to %d", ticker, start_year, end_year)
        all_filings = company.get_filings(form=form_types, year=range(start_year, end_year + 1))
        filings_to_download = [f for f in all_filings if f.accession_no not in acc_nos_to_skip]
        logger.info(
            "Downloading %d filings and skipping %d filings for %s",
            len(filings_to_download),
            len(all_filings) - len(filings_to_download),
            ticker,
        )

        if filings_to_download:
            ticker_metadatas: list[dict] = []
            for filing in tqdm(filings_to_download, desc="Downloading filings"):
                try:
                    ticker_metadatas.extend(download_filing(filing, output_dir, ticker))
                except Exception as e:
                    # One unusable filing must not end a run that has hundreds of
                    # tickers left to fetch.
                    logger.warning(
                        "Skipping %s (%s): %s",
                        getattr(filing, "accession_no", "unknown accession"),
                        getattr(filing, "form", "unknown form"),
                        e,
                    )

            # Written once per ticker, never per filing: an interrupted run then
            # records none of this ticker's accessions and re-downloads them whole.
            ticker_metadata_df = pd.DataFrame(ticker_metadatas)
            metadata_df = pd.concat([metadata_df, ticker_metadata_df])
            write_parquet_atomic(metadata_df, metadata_path)
            logger.info("Saved metadata for %s to %s", ticker, metadata_path)

    return metadata_df


def filing_year(filing: Filing) -> int:
    """Year to file this filing under.

    Forms that cover no reporting period, such as DEF 14A and FWP, leave
    ``report_date`` empty, so the filing date stands in for it.
    """
    for value in (filing.report_date, filing.filing_date):
        if isinstance(value, date):
            return value.year
        text = str(value or "").strip()
        if text:
            return datetime.strptime(text[:10], "%Y-%m-%d").year
    raise ValueError(f"{filing.accession_no} has neither a report date nor a filing date")


_HTML_ROOT = re.compile(r"<html[\s>]", re.IGNORECASE)


def primary_document_html(filing: Filing) -> str | None:
    """Primary document HTML, read from the filing's full submission text file.

    Inline XBRL primary documents open with an XML declaration, and edgartools
    before 5.61 answers that by re-fetching the document through the filing's
    index page, a URL SEC now refuses with 403 Access Denied.
    """
    html = filing.sgml().html()
    if html and html.startswith("<?xml") and _HTML_ROOT.search(html):
        return html
    return filing.html()


def download_filing(filing: Filing, output_dir: str, ticker: str) -> list[dict]:
    """Download a single filing (primary document + exhibits)."""
    filing_metadatas: list[dict] = []

    year = filing_year(filing)
    # Convert 8-K/A (amendment) to 8-K_A to avoid extra nested folder
    form_folder = filing.form.replace("/", "_")
    filing_dir = os.path.join(output_dir, ticker, form_folder, str(year), filing.accession_no)
    filing_filename = os.path.join(filing_dir, "primary-document.html")
    os.makedirs(filing_dir, exist_ok=True)

    for attempt in range(_MAX_RETRIES):
        try:
            filing_url = filing.filing_url
            html = primary_document_html(filing)
            break
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.TimeoutException) as e:
            if attempt == _MAX_RETRIES - 1:
                logger.warning(
                    "Failed to download %s after %d attempts: %s",
                    filing.accession_no,
                    _MAX_RETRIES,
                    e,
                )
                return []
            logger.warning(
                "Timeout on attempt %d/%d for %s, retrying...",
                attempt + 1,
                _MAX_RETRIES,
                filing.accession_no,
            )
            time.sleep(2**attempt)
        except Exception as e:
            logger.warning(
                "Unexpected error, skipping %s (%s %s): %s",
                filing.accession_no,
                filing.form,
                filing.report_date,
                e,
            )
            return []

    if isinstance(html, bytes):
        html = html.decode("utf-8")
    html = fix_image_urls(filing_url, html)
    with open(filing_filename, "w", encoding="utf-8") as fp:
        fp.write(html)

    filing_metadata = {
        "accession_number": filing.accession_no,
        "form_type": filing.form,
        "report_date": filing.report_date,
        "filing_date": filing.filing_date,
        "url": filing_url,
        "size": filing.size,
        "file_type": "primary_document",
        "file_location": filing_filename,
        "ticker": ticker,
        "company_name": filing.company,
    }
    filing_metadatas.append(filing_metadata)

    # Download exhibits
    exhibits = [
        att
        for att in filing.attachments
        if att.document_type.lower().startswith("ex-") and att.document.endswith((".html", ".htm"))
    ]
    os.makedirs(os.path.join(filing_dir, "exhibits"), exist_ok=True)
    for exhibit in exhibits:
        exhibit_filename = os.path.join(filing_dir, f"exhibits/{exhibit.document_type}.html")
        exhibit_url = exhibit.url

        # Reset per exhibit: a failure must not fall through to the write below
        # and record the previous exhibit's HTML under this exhibit's name.
        exhibit_html = None
        for attempt in range(_MAX_RETRIES):
            try:
                exhibit_html = exhibit.download()
                break
            except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.TimeoutException):
                if attempt == _MAX_RETRIES - 1:
                    logger.warning(
                        "Failed to download exhibit %s for %s after %d attempts, skipping...",
                        exhibit.document_type,
                        filing.accession_no,
                        _MAX_RETRIES,
                    )
                    break
                logger.warning(
                    "Timeout on exhibit %s, attempt %d/%d, retrying...",
                    exhibit.document_type,
                    attempt + 1,
                    _MAX_RETRIES,
                )
                time.sleep(2**attempt)
            except Exception as e:
                logger.warning(
                    "Unexpected error, skipping exhibit %s for %s: %s",
                    exhibit.document_type,
                    filing.accession_no,
                    e,
                )
                break

        if exhibit_html is None:
            continue

        if isinstance(exhibit_html, bytes):
            try:
                exhibit_html = exhibit_html.decode("utf-8")
            except UnicodeDecodeError:
                exhibit_html = exhibit_html.decode("latin-1", errors="replace")
        exhibit_html = fix_image_urls(exhibit_url, exhibit_html)
        with open(exhibit_filename, "w", encoding="utf-8") as fp:
            fp.write(exhibit_html)

        exhibit_metadata = filing_metadata.copy()
        exhibit_metadata["url"] = exhibit_url
        exhibit_metadata["size"] = len(exhibit_html.encode("utf-8"))
        exhibit_metadata["file_type"] = "exhibit"
        exhibit_metadata["file_location"] = exhibit_filename
        filing_metadatas.append(exhibit_metadata)

    return filing_metadatas
