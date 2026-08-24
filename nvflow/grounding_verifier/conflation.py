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
"""Deterministic explicit SEC source-attribution checks."""

from __future__ import annotations

import re
from urllib.parse import urlsplit, urlunsplit

from nvflow.grounding_verifier.types import AtomicClaim, EvidenceChunk

_SEC_URL_RE = re.compile(r"https?://(?:www\.)?sec\.gov/\S+", re.IGNORECASE)
_ATTRIBUTION_RE = re.compile(
    r"(?:according to\s+)?(?:SEC\s+)?(?:accession|CIK)(?:\s+number)?\s+"
    r"[\d-]+(?:\s+(?:reports?|reported|states?|stated)\s+that)?",
    re.IGNORECASE,
)
_GENERIC_SEC_ATTRIBUTION_RE = re.compile(
    r"\s+as\s+reported\s+in\s+(?:the\s+)?SEC\s+filings?",
    re.IGNORECASE,
)
_TICKER_RE = re.compile(r"\(([A-Z][A-Z0-9.]{0,7})\)")
_CORPORATE_SUFFIX_RE = re.compile(r"\b(?:inc|corp|corporation|ltd|llc)\b", re.IGNORECASE)
_COMPANY_STOPWORDS = {"co", "company", "corp", "corporation", "inc", "incorporated", "ltd", "llc"}


def _digits(value: str) -> str:
    return re.sub(r"\D", "", value)


def _source_fields(source_id: str | None) -> dict[str, str]:
    if not source_id:
        return {}
    return {
        key: value
        for part in source_id.split(":")
        if "=" in part
        for key, value in [part.split("=", 1)]
    }


def _canonical_url(url: str) -> str:
    parts = urlsplit(url.rstrip(".,;"))
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, "", ""))


def content_for_routing(text: str) -> str:
    """Remove explicit SEC attribution tokens before semantic routing."""
    content = _SEC_URL_RE.sub(" ", text)
    content = _ATTRIBUTION_RE.sub(" ", content)
    content = _GENERIC_SEC_ATTRIBUTION_RE.sub(" ", content)
    return " ".join(content.split()) or text


def explicit_source_match(claim: AtomicClaim, chunk: EvidenceChunk) -> bool | None:
    """Compare an explicit SEC citation with a source, or return ``None``.

    This intentionally covers only identifiers stated in the claim. It does not
    infer attribution from company names or replace token-level alignment.
    """
    fields = _source_fields(chunk.source_id)
    compared = False
    stated: dict[str, set[str]] = {"cik": set(), "accession": set()}
    for identifier in claim.stated_sec_ids:
        kind, _, value = identifier.partition(":")
        if kind in stated:
            stated[kind].add(_digits(value))

    for kind, expected in stated.items():
        actual = _digits(fields.get(kind, ""))
        if expected and actual:
            compared = True
            if actual not in expected:
                return False

    if claim.stated_sec_urls and chunk.sec_url:
        compared = True
        routed_url = _canonical_url(chunk.sec_url)
        if routed_url not in {_canonical_url(url) for url in claim.stated_sec_urls}:
            return False
    return True if compared else None


def has_explicit_source_conflation(claim: AtomicClaim, chunk: EvidenceChunk) -> bool:
    """Return true when an explicit SEC citation disagrees with routed evidence."""
    return explicit_source_match(claim, chunk) is False


def has_explicit_entity_conflation(claim: AtomicClaim, chunk: EvidenceChunk) -> bool:
    """Compare explicit company/ticker mentions with SEC source metadata."""
    claim_tickers = set(_TICKER_RE.findall(claim.text))
    if claim_tickers and chunk.sec_ticker and chunk.sec_ticker.upper() not in claim_tickers:
        return True

    if not chunk.sec_company_name or not _CORPORATE_SUFFIX_RE.search(claim.text):
        return False
    company_tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", chunk.sec_company_name.lower())
        if token not in _COMPANY_STOPWORDS
    }
    claim_lower = claim.text.lower()
    return bool(company_tokens) and not any(
        re.search(rf"\b{re.escape(token)}\b", claim_lower) for token in company_tokens
    )
