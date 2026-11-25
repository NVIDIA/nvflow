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
"""10-K Form Extractor."""

import sec_parser as sp
from sec_parser.processing_steps.top_section_manager_for_10q import TopSectionManagerFor10Q

from ..constants import PARTS_10K
from .base import BaseSecExtractor


class Sec10KExtractor(BaseSecExtractor):
    """
    Extractor for 10-K annual reports.

    Part structure:
    - Part I: Items 1, 1A, 1B, 1C, 2, 3, 4
    - Part II: Items 5, 6, 7, 7A, 8, 9, 9A, 9B, 9C
    - Part III: Items 10, 11, 12, 13, 14
    - Part IV: Item 15

    Filename format: Simple item numbers (e.g., "1", "7A", "15")
    """

    def _create_parser(self):
        """Create a parser for 10-K with TopSectionManagerFor10Q removed."""

        def without_10q_manager():
            all_steps = sp.Edgar10QParser().get_default_steps()
            return [step for step in all_steps if not isinstance(step, TopSectionManagerFor10Q)]

        return sp.Edgar10QParser(get_steps=without_10q_manager)

    def get_expected_part(self, item_num: str) -> str:
        item_num = item_num.upper()
        for part_name, items in PARTS_10K.items():
            if item_num in items:
                return part_name
        return "Unknown"

    def get_filename(self, item_num: str, part: str) -> str:
        return item_num
