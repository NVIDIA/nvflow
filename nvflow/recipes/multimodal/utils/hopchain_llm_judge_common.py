#!/usr/bin/env python
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
"""Shared helpers for HopChain LLM judge stages."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

XML_TAG_TEMPLATE = r"<{tag_name}>\s*(.*?)\s*</{tag_name}>"
NUMBER_PATTERN = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


class ParsedLLMJudgeResponse(BaseModel):
    """Parsed judge response content."""

    reasoning: str = ""
    final_answer: str
    normalized_answer: str
    confidence: Literal["high", "medium", "low", "unknown"] = "unknown"


def load_prompt_template(prompt_path: str | Path) -> str:
    """Load a text prompt template from disk."""
    return Path(prompt_path).read_text()


def extract_xml_tag(text: str, tag_name: str, last: bool = False) -> str | None:
    """Extract an XML tag body from text.

    By default returns the first match. Pass last=True to return the last match,
    which is useful when a reasoning trace may contain multiple draft answers before
    the committed final answer at the end.
    """
    matches = re.findall(
        XML_TAG_TEMPLATE.format(tag_name=re.escape(tag_name)),
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not matches:
        return None
    extracted = (matches[-1] if last else matches[0]).strip()
    return extracted or None


def normalize_answer_text(answer: str) -> str:
    """Normalize judge answers so reconciler compares stable values."""
    stripped = " ".join(answer.strip().split())
    if not stripped:
        return ""

    numeric_candidate = stripped.replace(",", "")
    if NUMBER_PATTERN.fullmatch(stripped):
        return _normalize_decimal_string(numeric_candidate)

    numeric_matches = NUMBER_PATTERN.findall(stripped)
    if len(numeric_matches) == 1:
        return _normalize_decimal_string(numeric_matches[0].replace(",", ""))

    return stripped.casefold()


def parse_judge_response_text(response_text: str) -> ParsedLLMJudgeResponse:
    """Parse the XML response produced by the LLM judge prompt.

    Uses last=True for final_answer so that reasoning models whose thinking traces
    contain draft answers yield the committed answer at the end. Responses with one
    answer tag also work because their first and last matches are identical.
    """
    final_answer = extract_xml_tag(response_text, "final_answer", last=True)
    if final_answer is None:
        raise ValueError("Missing <final_answer> tag in judge response")

    confidence = (extract_xml_tag(response_text, "confidence") or "unknown").strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "unknown"

    reasoning = extract_xml_tag(response_text, "reasoning") or ""
    normalized_answer = normalize_answer_text(final_answer)
    if not normalized_answer:
        raise ValueError("Judge response final answer is empty after normalization")

    return ParsedLLMJudgeResponse(
        reasoning=reasoning,
        final_answer=final_answer.strip(),
        normalized_answer=normalized_answer,
        confidence=confidence,
    )


def _normalize_decimal_string(value: str) -> str:
    """Normalize numeric strings into a stable canonical representation."""
    try:
        normalized = Decimal(value).normalize()
    except InvalidOperation:
        return value

    if normalized == normalized.to_integral():
        return str(normalized.quantize(Decimal("1")))
    return format(normalized, "f").rstrip("0").rstrip(".")
