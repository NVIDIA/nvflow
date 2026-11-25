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
"""Tests for VALS-to-template-SDG seed normalization."""

import json

from nvfinance.utils.sdg.convert_vals_data import _seed_record


def test_seed_record_accepts_gym_prepared_vals_schema() -> None:
    seed = _seed_record(
        7,
        {
            "question": "Did Example Corp beat its revenue guidance?",
            "expected_answer": "Yes.",
            "question_type": "Beat or Miss",
            "rubric": [{"criterion": "Compares guidance with actual revenue"}],
        },
    )

    assert seed["QID"] == "VALS-0007"
    assert seed["original_question"].startswith("Did Example Corp")
    assert seed["ground_truth"] == "Yes."
    assert seed["question_type"] == "Beat or Miss"
    assert seed["accession_number"] == "unknown-0"
    assert json.loads(seed["vals_checks"]) == [
        {"criterion": "Compares guidance with actual revenue"}
    ]


def test_seed_record_preserves_raw_vals_document_cardinality() -> None:
    seed = _seed_record(
        1,
        {
            "input": "Compare two filings.",
            "target": "Comparison.",
            "metadata": {
                "question_category": "Complex Retrieval",
                "documents_required_to_answers": "A 10-K 2023\nA 8-K 2024",
                "checks": ["Uses both filings"],
            },
        },
    )

    assert seed["accession_number"] == "unknown-0;unknown-1"
    assert seed["form_types"] == "10-K;10-K"
    assert seed["report_dates"] == "2024-12-31;2024-12-31"
    assert seed["question_type"] == "Complex Retrieval"
