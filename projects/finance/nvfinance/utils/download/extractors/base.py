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
"""Base SEC Extractor - shared logic for all form types."""

import re
from abc import ABC, abstractmethod

from sec_parser.semantic_elements.top_section_title import TopSectionTitle


class BaseSecExtractor(ABC):
    """
    Base class for SEC document section extractors.

    Provides shared functionality for document parsing, section boundary detection,
    part assignment, and content extraction.  Subclasses implement form-specific
    logic (parser config, section validation, filename mapping, part structure).
    """

    def __init__(self, user_agent: str = None):
        self.parser = self._create_parser()

        self.part_patterns = {
            "Part I": re.compile(r"\bPART\s+I\b", re.IGNORECASE),
            "Part II": re.compile(r"\bPART\s+II\b", re.IGNORECASE),
            "Part III": re.compile(r"\bPART\s+III\b", re.IGNORECASE),
            "Part IV": re.compile(r"\bPART\s+IV\b", re.IGNORECASE),
        }

        self.user_agent = user_agent or "MyCompany/1.0 (contact@mycompany.com)"
        self.headers = {"User-Agent": self.user_agent}

    # ------------------------------------------------------------------
    # Abstract interface – implemented by each form-specific extractor
    # ------------------------------------------------------------------

    @abstractmethod
    def _create_parser(self):
        """Return a configured sec-parser instance for this form type."""
        pass

    @abstractmethod
    def get_expected_part(self, item_num: str) -> str:
        """Return the expected Part name for *item_num*, or ``"Unknown"``."""
        pass

    @abstractmethod
    def get_filename(self, item_num: str, part: str) -> str:
        """Return a filename (without extension) for the given section."""
        pass

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _clean_title(self, title: str) -> str:
        """Normalize all whitespace (including ``\\xa0``) to regular spaces."""
        return " ".join(title.split())

    def _extract_toc_mappings(self, elements: list) -> dict[str, str]:
        """Extract item mappings from the Table of Contents."""
        toc_mappings: dict[str, str] = {}

        for element in elements:
            if element.__class__.__name__ == "TableOfContentsElement":
                if hasattr(element, "text") and element.text:
                    toc_text = element.text
                    parts = re.split(r"(?:Page)?Item\s+", toc_text)

                    for part in parts[1:]:
                        match = re.match(r"(\d+[A-Z]?)\.\s*([^.]+)\.", part, re.IGNORECASE)
                        if match:
                            item_num = match.group(1).upper()
                            item_title = match.group(2).strip()
                            if item_title and not item_title[0].isdigit():
                                if item_num not in toc_mappings:
                                    toc_mappings[item_num] = item_title

                    if toc_mappings:
                        sorted(
                            toc_mappings.items(),
                            key=lambda x: (
                                int(re.match(r"(\d+)", x[0]).group(1)),
                                x[0],
                            ),
                        )

                break

        return toc_mappings

    def _find_section_start_for_toc_item(
        self, elements: list, item_num: str, item_title: str
    ) -> int:
        """Find where a TOC item's content starts in the document."""
        if item_num == "1":
            for i, elem in enumerate(elements):
                if hasattr(elem, "text") and elem.text:
                    text_upper = elem.text.upper()
                    if (
                        "CONSOLIDATED STATEMENTS OF INCOME" in text_upper
                        or "CONSOLIDATED BALANCE SHEETS" in text_upper
                        or "NOTES TO CONSOLIDATED FINANCIAL STATEMENTS" in text_upper
                    ):
                        if elem.__class__.__name__ in ["TitleElement", "TopSectionTitle"]:
                            return i

        elif item_num == "2":
            for i, elem in enumerate(elements):
                if hasattr(elem, "text") and elem.text:
                    text_upper = elem.text.upper()
                    if (
                        ("INTRODUCTION" in text_upper and len(elem.text.strip()) < 50)
                        or ("EXECUTIVE OVERVIEW" in text_upper)
                        or ("MANAGEMENT'S DISCUSSION" in text_upper)
                    ):
                        if elem.__class__.__name__ in ["TitleElement", "TopSectionTitle"]:
                            return i

        for i, elem in enumerate(elements):
            if hasattr(elem, "text") and elem.text:
                if re.search(rf"\bItem\s+{re.escape(item_num)}\b", elem.text, re.IGNORECASE):
                    return i

        return -1

    # ------------------------------------------------------------------
    # Main extraction entry point
    # ------------------------------------------------------------------

    def extract_sections(self, html_content: str) -> dict[str, dict]:
        """Extract sections from an SEC document.

        Args:
            html_content: Raw HTML of the filing.

        Returns:
            ``{section_key: {content, part, filename, …}}``
        """
        elements = self.parser.parse(html_content)
        toc_mappings = self._extract_toc_mappings(elements)

        section_boundaries: list = []
        part_boundaries: dict = {}

        # First pass: locate Part headers (keep only the first / best occurrence)
        for i, element in enumerate(elements):
            if hasattr(element, "text") and element.text:
                title_clean = self._clean_title(element.text)
                detected_part = self._detect_part(title_clean)
                if detected_part:
                    if detected_part not in part_boundaries:
                        part_boundaries[detected_part] = i
                    else:
                        current_idx = part_boundaries[detected_part]
                        current_elem = elements[current_idx]
                        current_is_title = (
                            isinstance(current_elem, TopSectionTitle)
                            or current_elem.__class__.__name__ == "TitleElement"
                        )
                        new_is_title = (
                            isinstance(element, TopSectionTitle)
                            or element.__class__.__name__ == "TitleElement"
                        )
                        if new_is_title and not current_is_title:
                            part_boundaries[detected_part] = i

        # Second pass: find sections and assign parts
        for i, element in enumerate(elements):
            is_section = isinstance(element, TopSectionTitle) or (
                element.__class__.__name__ == "TitleElement"
            )
            is_text_element_item = (
                element.__class__.__name__ == "TextElement"
                and hasattr(element, "text")
                and element.text
                and re.match(r"^\s*Item\s*\d+[A-Z]?\.", element.text.strip(), re.IGNORECASE)
            )

            if (is_section or is_text_element_item) and hasattr(element, "text") and element.text:
                title_clean = self._clean_title(element.text)

                if self._is_likely_cross_reference(elements, i):
                    continue

                if self._is_section_title(title_clean):
                    assigned_part = "Unknown"
                    sorted_parts = sorted(part_boundaries.items(), key=lambda x: x[1], reverse=True)
                    for part_name, part_index in sorted_parts:
                        if i >= part_index:
                            assigned_part = part_name
                            break

                    item_num = self._extract_highest_item_number(title_clean)
                    if item_num:
                        expected_part = self.get_expected_part(item_num)
                        if expected_part != "Unknown" and assigned_part != expected_part:
                            assigned_part = expected_part

                    section_boundaries.append(
                        {
                            "index": i,
                            "title": title_clean,
                            "part": assigned_part,
                            "element": element,
                            "element_type": element.__class__.__name__,
                        }
                    )

        # Add missing items from TOC
        if toc_mappings:
            for item_num, item_title in toc_mappings.items():
                expected_part = self.get_expected_part(item_num)
                already_found = any(
                    re.search(
                        rf"\bITEM\s+{re.escape(item_num)}\b",
                        str(b.get("title", "")),
                        re.IGNORECASE,
                    )
                    and b.get("part") == expected_part
                    for b in section_boundaries
                )

                if not already_found:
                    idx = self._find_section_start_for_toc_item(elements, item_num, item_title)
                    if idx >= 0:
                        section_boundaries.append(
                            {
                                "index": idx,
                                "title": f"Item {item_num}. {item_title}",
                                "part": expected_part,
                                "element": elements[idx],
                                "element_type": "TOC-derived",
                            }
                        )

        section_boundaries.sort(key=lambda x: x["index"])
        section_boundaries = self._deduplicate_sections(section_boundaries)

        # Extract content for each section
        sections: dict[str, dict] = {}
        for j, boundary in enumerate(section_boundaries):
            end_idx = (
                section_boundaries[j + 1]["index"]
                if j + 1 < len(section_boundaries)
                else len(elements)
            )
            start_idx = boundary["index"]

            # Stop at a Part boundary that falls between start and end
            for _part_name, part_idx in part_boundaries.items():
                if start_idx < part_idx < end_idx:
                    end_idx = part_idx
                    break

            if end_idx - start_idx == 1:
                continue

            section_elements = elements[start_idx:end_idx]

            item_num = self._extract_highest_item_number(boundary["title"])
            if item_num:
                section_title = f"ITEM {item_num}"
                if boundary["part"] != "Unknown":
                    section_key = f"{boundary['part']} - {section_title}"
                else:
                    section_key = section_title
                filename = self.get_filename(item_num, boundary["part"])
            else:
                section_title = boundary["title"]
                section_key = section_title
                filename = None

            html_content_out = self._extract_html_content(section_elements)

            sections[section_key] = {
                "title": section_title,
                "original_title": boundary["title"],
                "part": boundary["part"],
                "element_type": boundary["element_type"],
                "start_index": start_idx,
                "end_index": end_idx,
                "element_count": len(section_elements),
                "html_content": html_content_out,
                "filename": filename,
            }

        return sections

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def _deduplicate_sections(self, section_boundaries: list[dict]) -> list[dict]:
        """Deduplicate sections with same item number, keeping the best candidate."""
        seen_items: dict[tuple, int] = {}
        boundaries_to_skip: set = set()
        item_with_description_pattern = re.compile(r"^\s*ITEMS?\s*\d+[A-Z]?\.\s*\w", re.IGNORECASE)

        for idx, boundary in enumerate(section_boundaries):
            item_num = self._extract_highest_item_number(boundary["title"])
            if not item_num:
                continue

            part = boundary["part"]
            if part == "Unknown":
                part = self.get_expected_part(item_num)
                boundary["part"] = part

            key = (part, item_num)
            if key not in seen_items:
                seen_items[key] = idx
                continue

            prev_idx = seen_items[key]
            prev_boundary = section_boundaries[prev_idx]

            element_type_priority = {
                "TopSectionTitle": 3,
                "TitleElement": 2,
                "TextElement": 1,
                "TOC-derived": 0,
            }

            prev_priority = element_type_priority.get(prev_boundary["element_type"], 0)
            curr_priority = element_type_priority.get(boundary["element_type"], 0)

            prev_is_subsection = any(sep in prev_boundary["title"] for sep in ["––", "—", "- "])
            curr_is_subsection = any(sep in boundary["title"] for sep in ["––", "—", "- "])

            if curr_is_subsection and not prev_is_subsection:
                boundaries_to_skip.add(idx)
                continue
            elif prev_is_subsection and not curr_is_subsection:
                boundaries_to_skip.add(prev_idx)
                seen_items[key] = idx
                continue

            prev_has_desc = bool(item_with_description_pattern.match(prev_boundary["title"]))
            curr_has_desc = bool(item_with_description_pattern.match(boundary["title"]))

            if curr_has_desc and not prev_has_desc:
                boundaries_to_skip.add(prev_idx)
                seen_items[key] = idx
                continue
            elif prev_has_desc and not curr_has_desc:
                boundaries_to_skip.add(idx)
                continue

            if curr_priority > prev_priority:
                boundaries_to_skip.add(prev_idx)
                seen_items[key] = idx
            elif prev_priority > curr_priority:
                boundaries_to_skip.add(idx)
            elif boundary["index"] < prev_boundary["index"]:
                boundaries_to_skip.add(prev_idx)
                seen_items[key] = idx
            else:
                boundaries_to_skip.add(idx)

        return [b for i, b in enumerate(section_boundaries) if i not in boundaries_to_skip]

    # ------------------------------------------------------------------
    # Item / Part detection helpers
    # ------------------------------------------------------------------

    def _extract_highest_item_number(self, title: str) -> str:
        """Extract the highest item number from a title like ``'ITEMS 10, 11, 12 and 13'``."""
        title_upper = title.upper()
        title_upper = re.sub(r"ITEM\s*(\d+)\(A\)", r"ITEM \1A", title_upper)

        if not title_upper.startswith("ITEM"):
            matches = re.findall(
                r"ITEM[S]?\s*(\d+(?:\.\d+)?[A-Za-z]?)(?:\b|\.|\:)", title_upper, re.IGNORECASE
            )
            if not matches:
                return ""
            matches = [m.upper() for m in matches]
        else:
            prefix_match = re.match(
                r"ITEM[S]?\s*([\d\s,andABC]+?)\.(?!\d)", title_upper, re.IGNORECASE
            )
            if prefix_match:
                matches = re.findall(r"(\d+(?:\.\d+)?[A-Za-z]?)", prefix_match.group(1))
                matches = [m.upper() for m in matches]
            else:
                matches = re.findall(
                    r"ITEM[S]?\s*(\d+(?:\.\d+)?[A-Za-z]?)", title_upper, re.IGNORECASE
                )
                matches = [m.upper() for m in matches]

        if not matches:
            return ""

        def sort_key(item):
            num_match = re.match(r"(\d+)", item)
            return int(num_match.group(1)) if num_match else 0

        return max(matches, key=sort_key)

    def _detect_part(self, title: str) -> str | None:
        """Detect which Part this title represents (strict header matching)."""
        title_upper = title.upper().strip()

        if not (title_upper.startswith("PART ") or title_upper.startswith("PARTS ")):
            return None
        if len(title) > 100:
            return None

        for part_name in ["Part IV", "Part III", "Part II", "Part I"]:
            if self.part_patterns[part_name].search(title):
                return part_name

        return None

    def _is_likely_cross_reference(self, elements: list, current_index: int) -> bool:
        """True when an Item mention is a cross-reference rather than a real header.

        Requires **both** signals: previous TextElement without ending punctuation
        AND next TextElement starting lowercase.
        """
        if current_index <= 0 or current_index >= len(elements) - 1:
            return False

        current_elem = elements[current_index]
        prev_elem = elements[current_index - 1]
        next_elem = elements[current_index + 1]

        if isinstance(current_elem, TopSectionTitle):
            return False

        if hasattr(current_elem, "text") and current_elem.text:
            text_upper = current_elem.text.strip().upper()
            if re.match(r"^\s*ITEMS?\s*\d+[A-Z]?", text_upper):
                prev_ok = (
                    prev_elem.__class__.__name__ == "TextElement"
                    and hasattr(prev_elem, "text")
                    and prev_elem.text
                    and prev_elem.text.strip()[-1] not in ".!?"
                )
                next_ok = (
                    next_elem.__class__.__name__ == "TextElement"
                    and hasattr(next_elem, "text")
                    and next_elem.text
                    and next_elem.text.strip()[0].islower()
                )
                return prev_ok and next_ok

        return False

    def _is_section_title(self, title: str) -> bool:
        """Determine if *title* represents a section we want to extract."""
        title_upper = title.upper().strip()

        if "CONTINUED" in title_upper:
            return False
        if re.search(r"\bITEM\s*\d+\([b-z]\)", title_upper, re.IGNORECASE):
            return False

        is_part_header = self._detect_part(title) is not None
        has_item = bool(re.search(r"\bITEM[S]?\s*\d+", title_upper))

        if is_part_header:
            return has_item

        if has_item:
            if title_upper.startswith("ITEM"):
                if re.search(r"\bFORM\s+[\d\-]+[A-Z]*\b", title_upper):
                    return False
                return True
            elif len(title) > 200:
                return False
            return True

        return False

    # ------------------------------------------------------------------
    # Content extraction
    # ------------------------------------------------------------------

    def _extract_text_content(self, elements: list) -> str:
        """Extract plain text from a list of elements."""
        text_parts = []
        for element in elements:
            if hasattr(element, "text") and element.text:
                text_parts.append(element.text.strip())
        return "\n".join(filter(None, text_parts))

    def _extract_html_content(self, elements: list) -> str:
        """Extract HTML content from a list of elements."""
        html_parts = []
        for element in elements:
            if hasattr(element, "html_tag") and element.html_tag:
                try:
                    html_out = element.html_tag.get_source_code()
                    if (
                        html_out
                        and isinstance(html_out, str)
                        and "<" in html_out
                        and ">" in html_out
                    ):
                        html_parts.append(html_out)
                except Exception:
                    continue
        return "\n".join(html_parts)
