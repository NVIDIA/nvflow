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
"""Protected value extraction and checking for GroundingVerifier.

Protected values are numeric amounts, currency figures, percentages,
and dates that appear in an entailed claim.  If a protected value is
absent from the routed evidence text, the claim cannot be ``allow``ed
— the model may have fabricated the specific number even if the general
claim is entailed.

This is a conservative value-matching check. It compares equivalent scaled
amounts numerically and recognizes explicit fiscal-year aliases, but does NOT
parse general financial semantics.

Full dates remain strict. A bare 4-digit year appearing in evidence does NOT
satisfy a full date protected value.

Common financial metric names are also canonicalized. A claim that assigns an
otherwise supported value to a different metric is rejected deterministically
instead of relying on an NLI model to distinguish similar finance sentences.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation


@dataclass(frozen=True)
class ProtectedValue:
    """A protected numeric/date/percentage value extracted from a claim."""

    raw: str  # original text as it appeared in the claim
    normalized: str  # normalized form for matching
    kind: str  # "currency" | "percentage" | "date" | "number"


# Currency: $1.23 billion, $1,234,567, $1.23M, €100, £50 million, etc.
_CURRENCY_RE = re.compile(
    r"[$€£¥]\s?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*"
    r"(?:billion|million|thousand|trillion|[bmk])?\b",
    re.IGNORECASE,
)

# Percentage: 12.3%, 5 percent, etc.
_PERCENTAGE_RE = re.compile(
    r"-?\d[\d,]*(?:\.\d+)?\s*(?:%|percent\b)",
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

_FINANCIAL_METRICS = {
    "net_income": re.compile(r"\b(?:net income|net earnings|net profit)\b", re.IGNORECASE),
    "total_assets": re.compile(r"\btotal assets\b", re.IGNORECASE),
    "revenue": re.compile(r"\b(?:total )?revenues?\b|\bnet sales\b", re.IGNORECASE),
    "operating_income": re.compile(
        r"\b(?:operating income|income from operations)\b", re.IGNORECASE
    ),
    "operating_cash_flow": re.compile(
        r"\b(?:operating cash flow|cash flows? from operating activities|"
        r"net cash provided by operating activities)\b",
        re.IGNORECASE,
    ),
    "free_cash_flow": re.compile(r"\bfree cash flow\b", re.IGNORECASE),
    "gross_profit": re.compile(r"\bgross profit\b", re.IGNORECASE),
    "research_and_development": re.compile(r"\b(?:research and development|R&D)\b", re.IGNORECASE),
}


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


def _numeric_value(value: ProtectedValue) -> Decimal | None:
    """Return a canonical numeric value for comparable protected types."""
    raw = value.raw.lower().replace(",", "").strip()
    raw = re.sub(r"^[\s$€£¥]+", "", raw)
    raw = re.sub(r"\s*(?:%|percent)\s*$", "", raw)
    match = re.fullmatch(
        r"(-?\d+(?:\.\d+)?)\s*(trillion|billion|million|thousand|[tbmk])?",
        raw,
    )
    if not match:
        return None
    try:
        number = Decimal(match.group(1))
    except InvalidOperation:
        return None
    scale = {
        "t": Decimal("1e12"),
        "trillion": Decimal("1e12"),
        "b": Decimal("1e9"),
        "billion": Decimal("1e9"),
        "m": Decimal("1e6"),
        "million": Decimal("1e6"),
        "k": Decimal("1e3"),
        "thousand": Decimal("1e3"),
    }.get((match.group(2) or "").lower(), Decimal(1))
    return number * scale


def _date_matches(value: ProtectedValue, evidence_text: str) -> bool:
    """Match fiscal-year aliases without weakening full-date protection."""
    exact = value.normalized
    evidence_lower = evidence_text.lower()
    if exact in evidence_lower:
        return True
    fiscal_year = re.fullmatch(r"fy\s*(20\d{2})", value.raw, re.IGNORECASE)
    if fiscal_year:
        year = fiscal_year.group(1)
        return bool(re.search(rf"\b(?:fy\s*{year}|fiscal year\s+{year})\b", evidence_lower))
    return False


def _value_matches(value: ProtectedValue, evidence_text: str) -> bool:
    evidence_lower = evidence_text.lower()
    evidence_compact = re.sub(r"\s+", "", evidence_lower)
    if value.normalized in evidence_lower or value.normalized in evidence_compact:
        return True
    if value.kind == "date":
        return _date_matches(value, evidence_text)
    if value.kind not in {"currency", "number", "percentage"}:
        return False
    expected = _numeric_value(value)
    if expected is None:
        return False
    candidates = extract_protected_values(evidence_text)
    comparable_kinds = (
        {"currency", "number"} if value.kind in {"currency", "number"} else {value.kind}
    )
    return any(
        candidate.kind in comparable_kinds and _numeric_value(candidate) == expected
        for candidate in candidates
    )


def extract_protected_values(text: str) -> list[ProtectedValue]:
    """Extract normalized values in currency, percentage, date, number order."""
    if not text:
        return []

    results: list[ProtectedValue] = []
    seen_raw: set[str] = set()
    patterns = (
        ("currency", _CURRENCY_RE, _normalize_currency),
        ("percentage", _PERCENTAGE_RE, _normalize_percentage),
        ("date", _DATE_RE, _normalize_date),
        ("number", _NUMBER_RE, _normalize_number),
    )
    for kind, pattern, normalize in patterns:
        for match in pattern.finditer(text):
            raw = match.group(0)
            if raw in seen_raw:
                continue
            seen_raw.add(raw)
            results.append(ProtectedValue(raw, normalize(raw), kind))

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

    missing = [value for value in protected if not _value_matches(value, evidence_text)]
    return ("fail", missing) if missing else ("pass", [])


def has_financial_metric_mismatch(claim_text: str, evidence_text: str) -> bool:
    """Return whether a claim names a financial metric absent from evidence."""
    claim_metrics = extract_financial_metrics(claim_text)
    if not claim_metrics:
        return False
    evidence_metrics = extract_financial_metrics(evidence_text)
    if evidence_metrics:
        return not claim_metrics.issubset(evidence_metrics)
    # Metric-less evidence is only a deterministic mismatch when it repeats the
    # claim's protected values; otherwise NLI remains responsible for relevance.
    outcome, _ = check_protected_values(claim_text, evidence_text)
    return outcome == "pass"


def extract_financial_metrics(text: str) -> set[str]:
    """Return canonical financial metric names found in text."""
    return {name for name, pattern in _FINANCIAL_METRICS.items() if pattern.search(text)}


def protected_numeric_value(value: ProtectedValue) -> Decimal | None:
    """Return the exact numeric value represented by a protected value."""
    return _numeric_value(value)
