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
"""10-Q Form Extractor."""

import sec_parser as sp

from .base import BaseSecExtractor


class Sec10QExtractor(BaseSecExtractor):
    """
    Extractor for 10-Q quarterly reports.

    Part structure (items can appear in multiple parts!):
    - Part I:  Items 1, 2, 3, 4
    - Part II: Items 1, 1A, 2, 3, 4, 5, 6

    Filename format: part prefix + item (e.g., "part1item1", "part2item1a")
    Note: Part I Item 1 and Part II Item 1 are DIFFERENT sections.
    """

    def _create_parser(self):
        return sp.Edgar10QParser()

    def _is_part_ii_only_item(self, item_num: str) -> bool:
        return item_num.upper() in ["1A", "5", "6"]

    def get_expected_part(self, item_num: str) -> str:
        if self._is_part_ii_only_item(item_num):
            return "Part II"
        return "Unknown"

    def get_filename(self, item_num: str, part: str) -> str:
        if "II" in part:
            part_num = "2"
        elif "I" in part:
            part_num = "1"
        else:
            part_num = "2" if self._is_part_ii_only_item(item_num) else "1"

        item_lower = item_num.lower()
        return f"part{part_num}item{item_lower}"
