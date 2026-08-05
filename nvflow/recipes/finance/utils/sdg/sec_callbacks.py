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
"""SEC-specific callbacks for the DG-SDG pipeline.

These callbacks inject SEC filing domain knowledge into the generic
``nvflow.lib.sdg.document_grounded`` library functions.
"""

from typing import Any


def construct_context(result: dict[str, Any], file_type: str = "10-K") -> str:
    """Build SEC filing context string with company/year/section headers.

    Handles 4 cases based on how many companies and sections are present:
    - Case 1: 1 company, 1 section
    - Case 2: 1 company, 2 sections
    - Case 3: 2 companies, 2 sections
    - Case 4: 2 companies, 3 sections
    """
    try:
        if "item_section2" in result:
            return (
                f"**{result.get('year', '')} {result.get('company_name0', '')} {file_type}**\n\n"
                f"**Part of {result['item_section0']}**\n\n"
                f"{result.get('content0', '')}\n\n"
                f"**{result.get('year', '')} {result.get('company_name1', '')} {file_type}**\n\n"
                f"**Part of {result['item_section1']}**\n\n"
                f"{result.get('content1', '')}\n\n"
                f"**Part of {result['item_section2']}**\n\n"
                f"{result.get('content2', '')}\n\n"
            )
        elif "company_name0" in result:
            return (
                f"**{result.get('year', '')} {result['company_name0']} {file_type}**\n\n"
                f"**Part of {result['item_section0']}**\n\n"
                f"{result.get('content0', '')}\n\n"
                f"**{result.get('year', '')} {result['company_name1']} {file_type}**\n\n"
                f"**Part of {result['item_section1']}**\n\n"
                f"{result.get('content1', '')}\n\n"
            )
        elif "item_section1" in result:
            return (
                f"**{result.get('year', '')} {result.get('company_name', '')} {file_type}**\n\n"
                f"**Part of {result['item_section0']}**\n\n"
                f"{result.get('content0', '')}\n\n"
                f"**Part of {result['item_section1']}**\n\n"
                f"{result.get('content1', '')}\n\n"
            )
        elif "item_section0" in result:
            return (
                f"**{result.get('year', '')} {result.get('company_name', '')} {file_type}**\n\n"
                f"**Part of {result['item_section0']}**\n\n"
                f"{result.get('content0', '')}\n\n"
            )
    except Exception:
        return ""
    return ""


def sec_context_builder(record: dict[str, Any]) -> str:
    """Context builder callback for ``construct_question_generate_input``.

    Determines the SEC form type (10-K / 10-Q / 8-K), then delegates to
    ``construct_context``. Prefers the record's ``file_type`` field (set by the
    preprocess stage); falls back to inferring from ``file_path0``.
    """
    file_type = record.get("file_type")
    if file_type not in ("10-K", "10-Q", "8-K"):
        file_path0 = record.get("file_path0", "")
        if "8-K" in file_path0 or "8-k" in file_path0:
            file_type = "8-K"
        elif "10-K" in file_path0 or "10-k" in file_path0:
            file_type = "10-K"
        else:
            file_type = "10-Q"
    return construct_context(record, file_type)
