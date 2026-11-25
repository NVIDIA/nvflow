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
"""8-K Form Extractor."""

import re

import sec_parser as sp
from sec_parser.processing_steps.individual_semantic_element_extractor.individual_semantic_element_extractor import (
    IndividualSemanticElementExtractor,
)
from sec_parser.processing_steps.individual_semantic_element_extractor.single_element_checks.top_section_title_check import (
    TopSectionTitleCheck,
)
from sec_parser.processing_steps.top_section_manager_for_10q import TopSectionManagerFor10Q

from .base import BaseSecExtractor


class Sec8KExtractor(BaseSecExtractor):
    """
    Extractor for 8-K current reports.

    8-K has a different structure from 10-K/10-Q:
    - No traditional "Parts"
    - Event-driven with decimal-notation items (e.g., Item 1.01, Item 7.01)

    Filename format: {major}-{minor} (e.g., "7-1", "9-1")
    """

    def _create_parser(self):
        def without_10q_related_steps():
            all_steps = sp.Edgar10QParser().get_default_steps()
            steps_without_top_section_manager = [
                step for step in all_steps if not isinstance(step, TopSectionManagerFor10Q)
            ]

            def get_checks_without_top_section_title_check():
                all_checks = sp.Edgar10QParser().get_default_single_element_checks()
                return [
                    check for check in all_checks if not isinstance(check, TopSectionTitleCheck)
                ]

            return [
                (
                    IndividualSemanticElementExtractor(
                        get_checks=get_checks_without_top_section_title_check
                    )
                    if isinstance(step, IndividualSemanticElementExtractor)
                    else step
                )
                for step in steps_without_top_section_manager
            ]

        return sp.Edgar10QParser(get_steps=without_10q_related_steps)

    def get_expected_part(self, item_num: str) -> str:
        return "Unknown"

    def _extract_highest_item_number(self, title: str) -> str:
        title_upper = title.upper().strip()
        if title_upper in ("SIGNATURE", "SIGNATURES"):
            return "SIGNATURE"
        return super()._extract_highest_item_number(title)

    def get_filename(self, item_num: str, part: str) -> str:
        if item_num == "SIGNATURE":
            return "signature"

        decimal_match = re.match(r"(\d+)\.0?(\d+)", item_num)
        if decimal_match:
            return f"{decimal_match.group(1)}-{decimal_match.group(2)}"

        concat_match = re.match(r"(\d)0?(\d+)", item_num)
        if concat_match:
            return f"{concat_match.group(1)}-{concat_match.group(2)}"

        if "-" in item_num:
            return item_num

        return item_num

    def _is_section_title(self, title: str) -> bool:
        title_upper = title.upper().strip()
        if title_upper in ("SIGNATURE", "SIGNATURES"):
            return True
        return super()._is_section_title(title)
