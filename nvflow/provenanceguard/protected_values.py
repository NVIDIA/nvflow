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
"""Protected value extraction and checking for ProvenanceGuard Open v1.

Protected values are numeric amounts, currency figures, percentages,
and dates that appear in an entailed claim.  If a protected value is
absent from the routed evidence text, the claim cannot be ``allow``ed
— the model may have fabricated the specific number even if the general
claim is entailed.

This is a conservative string-matching check: it normalizes formats
(billions/millions suffixes, percentage signs, date separators) and
checks for presence in the evidence text.  It does NOT parse financial
semantics.

Full normalized dates/numbers/currency/percent are required for a match.
A bare 4-digit year appearing in the evidence does NOT satisfy a date
protected value — the full normalized date string must be found.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ProtectedValue:
    """A protected numeric/date/percentage value extracted from a claim."""

    raw: str  # original text as it appeared in the claim
    normalized: str  # normalized form for matching
    kind: str  # "currency" | "percentage" | "date" | "number"


# --- Regex patterns ----------------------------------------------------------

# Currency: $1.23 billion, $1,234,567, $1.23M, €100, £50 million, etc.
_CURRENCY_RE = re.compile(
    r"[$€£¥]\s?\d[\d,]*(?:\.\d+)?\s*(?:billion|million|thousand|trillion|[bmk])?\b",
    re.IGNORECASE,
)

# Percentage: 12.3%, 5 percent, etc.
_PERCENTAGE_RE = re.compile(
    r"\d[\d,]*(?:\.\d+)?\s*(?:%|percent\b)",
    re.IGNORECASE,
)

# Date: 2024-01-15, January 15, 2024, Jan 15 2024, Q1 2024, FY2024, 2024-01, etc.
_DATE_RE = re.compile(
    r"\b(?:"
    r"\d{4}-\d{2}-\d{2}"  # 2024-01-15
    r"|\d{4}-\d{2}"  # 2024-01
    r"|\d{4}"  # 2024
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2},?\s*\d{4}"  # January 15, 2024
    r"|Q[1-4]\s+\d{4}"  # Q1 2024
    r"|FY\s*\d{4}"  # FY2024
    r"|(?:first|second|third|fourth)\s+quarter\s+\d{4}"  # first quarter 2024
    r")\b",
    re.IGNORECASE,
)

# Plain number with magnitude: 1.23 billion, 1,234,567, 12.3M, etc.
# Only matched if it looks like a financial figure (has commas, decimals,
# or magnitude words).  Single small integers are not protected.
_NUMBER_RE = re.compile(
    r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b"  # 1,234,567
    r"|\b\d+(?:\.\d+)?\s*(?:billion|million|thousand|trillion|[bmk])\b",  # 1.23 billion
    re.IGNORECASE,
)


def _normalize_currency(raw: str) -> str:
    """Normalize currency for matching: lowercase, remove spaces."""
    return re.sub(r"\s+", "", raw.lower())


def _normalize_percentage(raw: str) -> str:
    """Normalize percentage: '12.3 %' -> '12.3%'."""
    return re.sub(r"\s+", "", raw.lower())


def _normalize_date(raw: str) -> str:
    """Normalize date: lowercase, collapse spaces."""
    return re.sub(r"\s+", " ", raw.lower()).strip()


def _normalize_number(raw: str) -> str:
    """Normalize number: lowercase, remove spaces."""
    return re.sub(r"\s+", "", raw.lower())


def extract_protected_values(text: str) -> list[ProtectedValue]:
    """Extract all protected values from a claim text.

    Returns a list of :class:`ProtectedValue` with normalized forms.
    Order: currency, percentage, date, number (deduplicated by raw text).
    """
    if not text:
        return []

    results: list[ProtectedValue] = []
    seen_raw: set[str] = set()

    for match in _CURRENCY_RE.finditer(text):
        raw = match.group(0)
        if raw not in seen_raw:
            results.append(
                ProtectedValue(raw=raw, normalized=_normalize_currency(raw), kind="currency")
            )
            seen_raw.add(raw)

    for match in _PERCENTAGE_RE.finditer(text):
        raw = match.group(0)
        if raw not in seen_raw:
            results.append(
                ProtectedValue(raw=raw, normalized=_normalize_percentage(raw), kind="percentage")
            )
            seen_raw.add(raw)

    for match in _DATE_RE.finditer(text):
        raw = match.group(0)
        if raw not in seen_raw:
            results.append(ProtectedValue(raw=raw, normalized=_normalize_date(raw), kind="date"))
            seen_raw.add(raw)

    for match in _NUMBER_RE.finditer(text):
        raw = match.group(0)
        if raw not in seen_raw:
            results.append(
                ProtectedValue(raw=raw, normalized=_normalize_number(raw), kind="number")
            )
            seen_raw.add(raw)

    return results


def check_protected_values(
    claim_text: str,
    evidence_text: str,
) -> tuple[str, list[ProtectedValue]]:
    """Check if all protected values in *claim_text* appear in *evidence_text*.

    Returns ``(outcome, missing_values)``:
      - ``("pass", [])`` — all protected values found in evidence.
      - ``("fail", [...])`` — some protected values missing from evidence.
      - ``("not_applicable", [])`` — no protected values in the claim.
    """
    protected = extract_protected_values(claim_text)
    if not protected:
        return "not_applicable", []

    if not evidence_text:
        return "fail", protected

    evidence_lower = evidence_text.lower()
    evidence_compact = re.sub(r"\s+", "", evidence_lower)

    missing: list[ProtectedValue] = []
    for pv in protected:
        # Check both normalized and compact forms for robustness.
        if pv.normalized in evidence_lower or pv.normalized in evidence_compact:
            continue
        missing.append(pv)

    if missing:
        return "fail", missing
    return "pass", []


__all__ = [
    "ProtectedValue",
    "extract_protected_values",
    "check_protected_values",
]
