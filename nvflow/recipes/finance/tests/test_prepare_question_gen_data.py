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
"""Tests for prepare_question_gen_data - generating company/year combinations."""

import jsonlines
import pandas as pd

from nvflow.recipes.finance.utils.sdg.prepare_question_gen_data import (
    generate_company_year_combinations_for_all_questions,
)

# Run like: uv run pytest tests/test_prepare_question_gen_data.py -v


class TestPrepareQuestionGenData:
    """Test the prepare_question_gen_data functionality."""

    def test_generate_company_year_combinations_full_workflow(self, tmp_path):
        """Test generating company/year combinations for questions."""

        # Create dummy company info CSV file
        company_data = {
            "Ticker": ["GOOG", "META", "AAPL", "DELL"],
            "Company": ["Google", "Meta", "Apple", "Dell"],
            "Sector": [
                "Communication Services",
                "Communication Services",
                "Information Technology",
                "Information Technology",
            ],
            "SubIndustry": [
                "Interactive Media & Services",
                "Interactive Media & Services",
                "Technology Hardware, Storage & Peripherals",
                "Technology Hardware, Storage & Peripherals",
            ],
        }
        company_df = pd.DataFrame(company_data)
        company_info_file = tmp_path / "company_info.tsv"
        company_df.to_csv(company_info_file, sep="\t", index=False)

        # Create input JSONL file with questions
        input_data = [
            {
                "QID": 1,
                "question": "Revenue for Amazon",
                "question_type": "Analysis",
                "accession_number": "0000;0000",
                "item": "Item 1",
            },
            {
                "QID": 2,
                "question": "Revenue for Nvidia",
                "question_type": "Analysis",
                "accession_number": "0000;11111",
                "item": "Item 1",
            },
        ]

        input_file = tmp_path / "input_questions.jsonl"
        with jsonlines.open(input_file, "w") as writer:
            for row in input_data:
                writer.write(row)

        # Define output file
        output_file = tmp_path / "output_questions.jsonl"

        # Run the function
        start_year = 2022
        end_year = 2022
        generate_company_year_combinations_for_all_questions(
            str(input_file), str(company_info_file), start_year, end_year, str(output_file)
        )

        # Read and validate output
        output_rows = []
        with jsonlines.open(output_file) as reader:
            for row in reader:
                output_rows.append(row)

        # Expected: qid=1 should have 4 entries (single company each)
        # Expected: qid=2 should have 2 entries (company pairs within same SubIndustry)
        assert len(output_rows) == 6, f"Expected 6 output rows, got {len(output_rows)}"

        # Validate qid=1 entries (single accession number -> single companies)
        qid1_rows = [row for row in output_rows if row["QID"] == 1]
        assert len(qid1_rows) == 4, "Expected 4 rows for qid=1"

        # Check that all expected companies appear for qid=1
        qid1_companies = {row["company_a"] for row in qid1_rows}
        assert qid1_companies == {"Google", "Meta", "Apple", "Dell"}

        # Validate structure of qid=1 entries
        for row in qid1_rows:
            assert row["QID"] == 1
            assert row["question"] == "Revenue for Amazon"
            assert row["question_type"] == "Analysis"
            assert row["accession_number"] == "0000;0000"
            assert row["item"] == "Item 1"
            assert row["company_a"] in ["Google", "Meta", "Apple", "Dell"]
            assert row["ticker_a"] in ["GOOG", "META", "AAPL", "DELL"]
            assert row["company_b"] is None
            assert row["ticker_b"] is None
            assert row["year"] == 2022

        # Validate qid=2 entries (multiple accession numbers -> company pairs)
        qid2_rows = [row for row in output_rows if row["QID"] == 2]
        assert len(qid2_rows) == 2, "Expected 2 rows for qid=2"

        # Check that we have the expected company pairs from same SubIndustry
        # Communication Services: (GOOG, META)
        # Information Technology: (AAPL, DELL)
        qid2_pairs = {(row["ticker_a"], row["ticker_b"]) for row in qid2_rows}
        expected_pairs = {("GOOG", "META"), ("AAPL", "DELL")}
        assert qid2_pairs == expected_pairs, f"Expected pairs {expected_pairs}, got {qid2_pairs}"

        # Validate structure of qid=2 entries
        for row in qid2_rows:
            assert row["QID"] == 2
            assert row["question"] == "Revenue for Nvidia"
            assert row["question_type"] == "Analysis"
            assert row["accession_number"] == "0000;11111"
            assert row["item"] == "Item 1"
            assert row["company_a"] is not None
            assert row["ticker_a"] is not None
            assert row["company_b"] is not None
            assert row["ticker_b"] is not None
            assert row["year"] == 2022

    def test_generate_combinations_multiple_years(self, tmp_path):
        """Test that multiple years generate appropriate combinations."""

        # Create simple company info file
        company_data = {
            "Ticker": ["AAPL", "MSFT"],
            "Company": ["Apple", "Microsoft"],
            "Sector": ["Information Technology", "Information Technology"],
            "SubIndustry": ["Technology", "Technology"],
        }
        company_df = pd.DataFrame(company_data)
        company_info_file = tmp_path / "company_info.tsv"
        company_df.to_csv(company_info_file, sep="\t", index=False)

        # Create input with single question
        input_data = [
            {
                "QID": 1,
                "question": "Test question",
                "question_type": "Analysis",
                "accession_number": "0000",
                "item": "Item 1",
            }
        ]

        input_file = tmp_path / "input.jsonl"
        with jsonlines.open(input_file, "w") as writer:
            for row in input_data:
                writer.write(row)

        output_file = tmp_path / "output.jsonl"

        # Test with year range 2022-2024 (3 years)
        generate_company_year_combinations_for_all_questions(
            str(input_file), str(company_info_file), 2022, 2024, str(output_file)
        )

        # Read output
        output_rows = []
        with jsonlines.open(output_file) as reader:
            for row in reader:
                output_rows.append(row)

        # Expected: 2 companies × 3 years = 6 entries
        assert len(output_rows) == 6, (
            f"Expected 6 rows (2 companies × 3 years), got {len(output_rows)}"
        )

        # Check years
        years = sorted({row["year"] for row in output_rows})
        assert years == [2022, 2023, 2024], f"Expected years [2022, 2023, 2024], got {years}"

        # Check that each company appears 3 times (once per year)
        apple_rows = [row for row in output_rows if row["ticker_a"] == "AAPL"]
        msft_rows = [row for row in output_rows if row["ticker_a"] == "MSFT"]
        assert len(apple_rows) == 3
        assert len(msft_rows) == 3
