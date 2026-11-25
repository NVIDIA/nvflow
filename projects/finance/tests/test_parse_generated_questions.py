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
"""Tests for parse_generated_questions - parsing generation outputs."""

import jsonlines
from nvfinance.utils.sdg.parse_generated_questions import (
    parse_generations,
)

# Run like: uv run pytest projects/finance/tests/test_parse_generated_questions.py -v


class TestParseGenerations:
    """Test the parse_generations function with a full workflow."""

    def test_parse_generations_full_workflow(self, tmp_path):
        """Test parsing generations from input JSONL to output JSONL."""
        # Create test input data
        input_data = [
            {
                "generation": '{\n   "questions": [\n      "How does AT&T\'s Provision Coverage Ratio for 2021 reflect its strategy for handling bad debts?",\n      "What can be inferred about AT&T\'s bad debt management from its 2021 Provision Coverage Ratio?",\n      "In what way does the 2021 Provision Coverage Ratio indicate AT&T\'s approach to managing non-performing loans?"\n   ]\n}',
                "QID": "q_an001",
                "original_question": "The first question",
                "company_a": "AT&T",
                "ticker_a": "T",
                "company_b": None,
                "ticker_b": None,
                "year": 2021,
            },
            {
                "generation": '{\n   "questions": [\n      "How does Apple\'s Provision Coverage Ratio for 2022 reflect its strategy for handling bad debts?",\n      "The second question with duplicates",\n      "What can be inferred about AT&T\'s bad debt management from its 2021 Provision Coverage Ratio?"\n   ]\n}',
                "QID": "q_an002",
                "original_question": "The second question with duplicates",
                "company_a": "Apple",
                "ticker_a": "AAPL",
                "company_b": "Google",
                "ticker_b": "GOOG",
                "year": 2022,
            },
        ]

        # Create temporary input file
        input_file = tmp_path / "test_input.jsonl"
        with jsonlines.open(input_file, "w") as writer:
            for row in input_data:
                writer.write(row)

        # Define output file
        output_file = tmp_path / "test_output.jsonl"
        log_file = tmp_path / "test_output_log.txt"

        # Run the parse_generations function
        parse_generations(str(input_file), str(output_file))

        # Read and validate output
        output_rows = []
        with jsonlines.open(output_file) as reader:
            for row in reader:
                output_rows.append(row)

        # Validate output
        assert len(output_rows) == 4, "Expected 4 output rows"

        # Check first two rows (from first input)
        for i in range(2):
            assert output_rows[i]["QID"] == "q_an001"
            assert output_rows[i]["original_question"] == "The first question"
            assert output_rows[i]["company_a"] == "AT&T"
            assert output_rows[i]["ticker_a"] == "T"
            assert output_rows[i]["year"] == 2021
        assert (
            output_rows[0]["problem"]
            == "How does AT&T's Provision Coverage Ratio for 2021 reflect its strategy for handling bad debts?"
        )
        assert (
            output_rows[1]["problem"]
            == "What can be inferred about AT&T's bad debt management from its 2021 Provision Coverage Ratio?"
        )
        assert (
            output_rows[2]["problem"]
            == "In what way does the 2021 Provision Coverage Ratio indicate AT&T's approach to managing non-performing loans?"
        )

        assert output_rows[3]["QID"] == "q_an002"
        assert output_rows[3]["original_question"] == "The second question with duplicates"
        assert (
            output_rows[3]["problem"]
            == "How does Apple's Provision Coverage Ratio for 2022 reflect its strategy for handling bad debts?"
        )
        assert output_rows[3]["company_a"] == "Apple"
        assert output_rows[3]["ticker_a"] == "AAPL"
        assert output_rows[3]["company_b"] == "Google"
        assert output_rows[3]["ticker_b"] == "GOOG"
        assert output_rows[3]["year"] == 2022

        # Check log file
        with open(log_file) as f:
            log_content = f.read()
            assert "Number of original questions: 2" in log_content
            assert "Number of failed to parse: 0" in log_content
            assert "Number of generated questions: 6" in log_content
            assert "Number of generated questions after deduplication: 4" in log_content

    def test_parse_generations_with_missing_generation_field(self, tmp_path):
        """Test handling of rows without generation field."""
        input_data = [
            {
                "question_type": "Analysis",
                "original_question": "question without generation",
            },
            {
                "question_type": "Analysis",
                "original_question": "empty generation field",
                "generation": "",
            },
            {
                "question_type": "Analysis",
                "original_question": "empty list of generations",
                "generation": '{\n   "questions": []\n}',
            },
            {
                "question_type": "Analysis",
                "original_question": "question with bad json",
                "generation": "This is not valid JSON at all",
            },
        ]

        input_file = tmp_path / "test_input_missing.jsonl"
        with jsonlines.open(input_file, "w") as writer:
            for row in input_data:
                writer.write(row)

        output_file = tmp_path / "test_output_missing.jsonl"
        log_file = tmp_path / "test_output_missing_log.txt"

        # Should not crash
        parse_generations(str(input_file), str(output_file))

        # Output should be empty
        with jsonlines.open(output_file) as reader:
            output_rows = list(reader)
        assert len(output_rows) == 0

        # Log file should contain error message
        assert log_file.exists()

        with open(log_file) as f:
            log_content = f.read()
            assert "No generation field found" in log_content
            assert "Failed to parse generation for" in log_content
            assert "Number of original questions: 4" in log_content
            assert "Number of failed to parse: 3" in log_content
            assert "Number of generated questions: 0" in log_content
            assert "Number of generated questions after deduplication: 0" in log_content
