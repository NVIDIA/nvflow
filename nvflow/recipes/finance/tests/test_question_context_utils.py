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
"""Tests for SDG utility functions."""

import os
from pathlib import Path

import pandas as pd

from nvflow.recipes.finance.utils.shared.question_context_utils import (
    get_closest_filing_month_accession_number,
    get_matching_section_filepath,
    process_question_item,
)


class TestGetClosestFilingMonthAccessionNumber:
    """Test suite for get_closest_filing_month_accession_number function."""

    def test_closest_match(self):
        """Test getting the accession number of the filing with the closest filing month."""

        # Create test data
        filings_metadata = pd.DataFrame(
            {
                "ticker": ["COMPANY3", "COMPANY3", "COMPANY3", "COMPANY3", "COMPANY2"],
                "accession_number": ["0001", "0002", "0003", "0004", "0005"],
                "report_date": [
                    "2021-10-01",
                    "2020-11-01",
                    "2020-05-01",
                    "2020-09-01",
                    "2020-09-01",
                ],
                "form_type": ["10-Q", "10-Q", "10-Q", "10-Q", "10-Q"],
            }
        )

        # Test parameters
        orig_report_date = "2024-09-01"
        ticker = "COMPANY3"
        form_type = "10-Q"
        year = 2020

        # Call the function
        result = get_closest_filing_month_accession_number(
            orig_report_date=orig_report_date,
            ticker=ticker,
            year=year,
            form_type=form_type,
            filings_metadata=filings_metadata,
        )

        # Assert the result
        # The function should find the MSFT filing from 2020 with the closest month to September (month 9)
        # Candidates after filtering (MSFT, 10-Q, 2020):
        #   - 0002: 2020-11-01 (month 11, diff = 2)
        #   - 0003: 2020-05-01 (month 5, diff = 4)
        #   - 0004: 2020-09-01 (month 9, diff = 0) <- closest match
        assert result == "0004"

    def test_no_match(self):
        """Test behavior when no matching filings exist."""

        # Create test data with no matching entries
        filings_metadata = pd.DataFrame(
            {
                "ticker": ["COMPANY2", "COMPANY2"],
                "accession_number": ["0001", "0002"],
                "report_date": ["2020-09-01", "2020-12-01"],
                "form_type": ["10-Q", "10-Q"],
            }
        )

        # Test parameters - looking for MSFT but only GOOG exists
        orig_report_date = "2024-09-01"
        ticker = "COMPANY3"
        form_type = "10-Q"
        year = 2020

        # Call the function
        result = get_closest_filing_month_accession_number(
            orig_report_date=orig_report_date,
            ticker=ticker,
            year=year,
            form_type=form_type,
            filings_metadata=filings_metadata,
        )

        # Should return None when no match is found
        assert result is None

    def test_multiple_same_month(self):
        """Test behavior when multiple filings have the same month."""

        # Create test data with multiple filings in the same month (would never happen in reality)
        filings_metadata = pd.DataFrame(
            {
                "ticker": ["COMPANY3", "COMPANY3", "COMPANY3"],
                "accession_number": ["0001", "0002", "0003"],
                "report_date": ["2020-09-01", "2020-09-15", "2020-11-01"],
                "form_type": ["10-Q", "10-Q", "10-Q"],
            }
        )

        # Test parameters
        orig_report_date = "2024-09-01"
        ticker = "COMPANY3"
        form_type = "10-Q"
        year = 2020

        # Call the function
        result = get_closest_filing_month_accession_number(
            orig_report_date=orig_report_date,
            ticker=ticker,
            year=year,
            form_type=form_type,
            filings_metadata=filings_metadata,
        )

        # Should return one of the September filings (both have diff = 0)
        # The function returns the first one it encounters
        assert result in ["0001", "0002"]


class TestGetMatchingSectionFilepath:
    """Test suite for get_matching_section_filepath function."""

    def test_basic_10k_match(self):
        """Test the most basic case: finding a 10-K section file that exists."""

        # Create test data
        filings_metadata = pd.DataFrame(
            {
                "ticker": ["COMPANY2", "COMPANY1"],
                "accession_number": ["000001", "000000"],
                "report_date": ["2024-09-09", "2024-09-09"],
                "form_type": ["10-K", "10-K"],
            }
        )

        # Test parameters
        form_type = "10-K"
        orig_report_date = "2024-09-09"
        ticker = "COMPANY2"
        year = "2024"
        orig_item_name = "Item 7: the description of item 7"

        # Get the test data directory path (relative to this test file)
        test_dir = Path(__file__).parent
        filings_folder = str(test_dir / "data")

        # Call the function
        result = get_matching_section_filepath(
            form_type=form_type,
            orig_report_date=orig_report_date,
            filings_metadata=filings_metadata,
            ticker=ticker,
            year=year,
            orig_item_name=orig_item_name,
            filings_folder=filings_folder,
        )

        # Expected path
        expected_path = f"{filings_folder}/COMPANY2/10-K/2024/000001/7.html"

        # Assert the result
        assert result == expected_path
        assert os.path.exists(result)


class TestProcessQuestionItem:
    """Test suite for process_question_item function."""

    def test_basic_process_question_item(self):
        """Test processing a question item with two companies and two items."""

        # Create test data
        filings_metadata = pd.DataFrame(
            {
                "ticker": ["COMPANY2", "COMPANY1"],
                "accession_number": ["000001", "000000"],
                "report_date": ["2024-09-09", "2024-09-09"],
                "form_type": ["10-K", "10-K"],
            }
        )

        # Test question item
        question_item = {
            "item": "Item 7. Some description;Item 7.some other description",
            "accession_number": "0000034088-24-000050;0000093410-24-000040",
            "form_types": "10-K;10-K",
            "report_dates": "2024-09-09;2024-09-09",
            "company_a": "Company2",
            "ticker_a": "COMPANY2",
            "company_b": "Company1",
            "ticker_b": "COMPANY1",
            "year": 2024,
        }

        # Get the test data directory path
        test_dir = Path(__file__).parent
        filings_folder = str(test_dir / "data")
        token_limit = 500

        # Call the function
        result = process_question_item(
            question_item=question_item,
            filings_metadata=filings_metadata,
            filings_folder=filings_folder,
            token_limit=token_limit,
        )

        # Assert the result contains all original fields
        assert result is not None
        assert result["item"] == question_item["item"]
        assert result["accession_number"] == question_item["accession_number"]
        assert result["ticker_a"] == question_item["ticker_a"]
        assert result["ticker_b"] == question_item["ticker_b"]
        assert result["year"] == question_item["year"]

        # Assert new fields are added
        assert "filepath" in result
        assert "context" in result

        # Verify filepaths
        expected_goog_path = f"{filings_folder}/COMPANY2/10-K/2024/000001/7.html"
        expected_aapl_path = f"{filings_folder}/COMPANY1/10-K/2024/000000/7.html"
        expected_filepath = f"{expected_goog_path};{expected_aapl_path}"
        assert result["filepath"] == expected_filepath

        # Verify context contains markdown content from both files
        assert (
            result["context"]
            == "COMPANY2 10-K Item 7\n\n# Item 7: Management's Discussion and Analysis\n\nThis is a dummy COMPANY2 10-K Item 7 document for testing purposes.\n\nCOMPANY1 10-K Item 7\n\n# Item 7: Management's Discussion and Analysis\n\nThis is a dummy COMPANY1 10-K Item 7 document for testing purposes."
        )

    def test_process_question_item_single_item_broadcast(self):
        """Comparison question with two form_types but a SINGLE item value.

        SecQue copies `item` verbatim as one value even for multi-accession
        (comparison) questions, while create_seed_data builds one form_type /
        report_date per accession. The single item must be broadcast across the
        filings so the question maps, instead of being dropped (returning None).
        """

        # Create test data
        filings_metadata = pd.DataFrame(
            {
                "ticker": ["COMPANY2", "COMPANY1"],
                "accession_number": ["000001", "000000"],
                "report_date": ["2024-09-09", "2024-09-09"],
                "form_type": ["10-K", "10-K"],
            }
        )

        # Two form_types (per accession), but a SINGLE item and report_date.
        question_item = {
            "item": "Item 7. Some description",
            "accession_number": "0000034088-24-000050;0000093410-24-000040",
            "form_types": "10-K;10-K",
            "report_dates": "2024-09-09",
            "company_a": "Company2",
            "ticker_a": "COMPANY2",
            "company_b": "Company1",
            "ticker_b": "COMPANY1",
            "year": 2024,
        }

        # Get the test data directory path
        test_dir = Path(__file__).parent
        filings_folder = str(test_dir / "data")
        token_limit = 500

        # Call the function
        result = process_question_item(
            question_item=question_item,
            filings_metadata=filings_metadata,
            filings_folder=filings_folder,
            token_limit=token_limit,
        )

        # Without broadcasting this returned None; now the single item maps to both filings.
        assert result is not None

        expected_company2_path = f"{filings_folder}/COMPANY2/10-K/2024/000001/7.html"
        expected_company1_path = f"{filings_folder}/COMPANY1/10-K/2024/000000/7.html"
        expected_filepath = f"{expected_company2_path};{expected_company1_path}"
        assert result["filepath"] == expected_filepath
        assert "context" in result

    def test_process_question_item_no_match(self):
        """Test processing a question item with two companies and two items but one of the company items is missing from filings."""

        # Create test data
        filings_metadata = pd.DataFrame(
            {
                "ticker": ["COMPANY2", "COMPANY1"],
                "accession_number": ["000001", "000000"],
                "report_date": ["2024-09-09", "2024-09-09"],
                "form_type": ["10-K", "10-K"],
            }
        )

        # Test question item
        question_item = {
            "item": "Item 7. Some description;Item 8.some other description",
            "accession_number": "0000034088-24-000050;0000093410-24-000040",
            "form_types": "10-K;10-K",
            "report_dates": "2024-09-09;2024-09-09",
            "company_a": "Company2",
            "ticker_a": "COMPANY2",
            "company_b": "Company1",
            "ticker_b": "COMPANY1",
            "year": 2024,
        }

        # Get the test data directory path
        test_dir = Path(__file__).parent
        filings_folder = str(test_dir / "data")
        token_limit = 500

        # Call the function
        result = process_question_item(
            question_item=question_item,
            filings_metadata=filings_metadata,
            filings_folder=filings_folder,
            token_limit=token_limit,
        )

        # Assert the result contains all original fields
        assert result is None
