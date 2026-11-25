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
"""CLI entry point for SEC filing download and section extraction.

Invoked by the NVFlow stage as::

    python -m nvfinance.utils.download.download_sec_filings \\
        --tickers "AAPL MSFT" --forms 10-K 10-Q "DEF 14A" \\
        --start_year 2020 --end_year 2024 \\
        --output_dir /workspace/outputs/... \\
        --sec_email you@example.com --sec_company YourCompany
"""

import argparse
import sys
from time import time

from nvflow.utils import setup_logger

from .download_filings import download_filings
from .section_extractor import extract_sections

logger = setup_logger(__name__)


def _format_elapsed(seconds: float) -> str:
    if seconds >= 60:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    return f"{seconds:.1f}s"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and parse SEC filings")
    parser.add_argument("--tickers", type=str, required=True, help="Space-separated ticker symbols")
    # nargs, not a space-separated string: EDGAR form types such as "DEF 14A"
    # contain spaces and must survive as one argument.
    parser.add_argument(
        "--forms",
        type=str,
        nargs="+",
        default=["10-K", "10-Q", "8-K"],
        help="Form types, one argument each",
    )
    parser.add_argument("--start_year", type=int, required=True)
    parser.add_argument("--end_year", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sec_email", type=str, required=True, help="Email for SEC EDGAR identity")
    parser.add_argument(
        "--sec_company", type=str, default="", help="Company name for SEC EDGAR identity"
    )
    parser.add_argument("--download_only", action="store_true", help="Skip section extraction")
    parser.add_argument("--extract_only", action="store_true", help="Skip downloading")

    args = parser.parse_args()

    tickers = args.tickers.split()
    forms = args.forms

    config = {
        "tickers": tickers,
        "form_types": forms,
        "start_year": args.start_year,
        "end_year": args.end_year,
        "output_dir": args.output_dir,
        "identity_info": {"email_address": args.sec_email, "company_name": args.sec_company},
        "data_dir": "data",
        "metadata_filename": "sec_metadata.parquet",
    }

    logger.info("Tickers: %s", tickers)
    logger.info("Forms: %s", forms)
    logger.info("Years: %d - %d", args.start_year, args.end_year)
    logger.info("Output: %s", args.output_dir)
    logger.info("SEC Identity: %s <%s>", args.sec_company, args.sec_email)

    try:
        if not args.extract_only:
            logger.info("Downloading filings for tickers: %s", tickers)
            t0 = time()
            download_filings(config)
            logger.info("Downloading filings took %s", _format_elapsed(time() - t0))

        if not args.download_only:
            logger.info("Extracting sections for all tickers")
            t0 = time()
            extract_sections(config)
            logger.info("Extracting sections took %s", _format_elapsed(time() - t0))

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error("Error: %s", e, exc_info=True)
        sys.exit(1)

    logger.info("Done!")


if __name__ == "__main__":
    main()
