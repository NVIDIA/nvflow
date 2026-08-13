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
"""Deterministic rule-based claim decomposer (v1 default).

This conservative splitter uses sentence boundaries and conjunctions. It may
over-merge compound claims or split mid-clause.

Claim IDs are deterministic SHA-256 hashes of the claim text and the
source sentence, so re-evaluation of the same answer yields stable IDs.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

from nvflow.grounding_verifier.types import AtomicClaim

# SEC filing URL pattern — used to extract stated SEC IDs/URLs from claim text.
_SEC_URL_RE = re.compile(
    r"https?://(?:www\.)?sec\.gov/[^\s\"'<>]+",
    re.IGNORECASE,
)

# CIK pattern — 10-digit zero-padded numbers.
_CIK_RE = re.compile(r"\b(\d{10})\b")

# Accession number pattern — e.g. 0001811414-25-000010
_ACCESSION_RE = re.compile(r"\b(\d{10}-\d{2}-\d{6})\b")

# Keep common corporate suffixes intact ("Apple Inc. reported ...").
_SENTENCE_END_RE = re.compile(
    r"(?<!Inc\.)(?<!Corp\.)(?<!Ltd\.)(?<!Co\.)(?<=[.!?])\s+|\n+",
    re.IGNORECASE,
)

# Compound claim splitters within a sentence.
_COMPOUND_SPLIT_RE = re.compile(
    r"\s*;\s*|\s+--\s+|,?\s+(?:although|but(?:\s+also)?|however|whereas)\s+",
    re.IGNORECASE,
)
_REFUTED_SUFFIX_RE = re.compile(r",?\s+which\s+contradicts?\b.*$", re.IGNORECASE)
_META_EVIDENCE_RE = re.compile(
    r"\b(?:provided|stated|found)\s+in\s+(?:the\s+)?evidence\s+card|"
    r"\bevidence\s+card\s+(?:with\s+[^.]+\s+)?(?:provides|states)\b",
    re.IGNORECASE,
)
_CITED_EVIDENCE_RE = re.compile(r"^cited evidence\s*:", re.IGNORECASE)


def _is_meta_evidence_fragment(text: str) -> bool:
    """Ignore provenance narration that contains no submitted factual value."""
    has_factual_value = bool(re.search(r"[$€£¥%]|\b\d{1,3}(?:,\d{3})+\b", text))
    return bool(_CITED_EVIDENCE_RE.search(text)) or (
        bool(_META_EVIDENCE_RE.search(text)) and not has_factual_value
    )


def _stable_claim_id(text: str, source_sentence: str) -> str:
    """Deterministic 16-char hex hash for a claim."""
    raw = f"{text}|{source_sentence}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _extract_stated_sec_ids(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Extract SEC CIKs and accession numbers mentioned in claim text."""
    accessions = tuple(m.group(1) for m in _ACCESSION_RE.finditer(text))
    # An accession begins with ten digits, but that prefix is not necessarily
    # the issuer CIK. Remove accession spans before looking for standalone CIKs.
    without_accessions = _ACCESSION_RE.sub(" ", text)
    ids = [f"cik:{m.group(1)}" for m in _CIK_RE.finditer(without_accessions)]
    ids.extend(f"accession:{accession}" for accession in accessions)
    urls = tuple(m.group(0) for m in _SEC_URL_RE.finditer(text))
    return tuple(ids), urls


class RuleBasedDecomposer:
    """Deterministic sentence/claim splitter.

    Splits the answer into sentences, then further splits compound
    claims on semicolons, em-dashes, and ``"but also"`` connectors.
    Filters out empty fragments and very short noise (< 3 words).
    """

    def decompose(self, answer: str) -> Sequence[AtomicClaim]:
        if not answer or not answer.strip():
            return []

        # Normalize whitespace but preserve sentence structure.
        text = answer.strip()

        # Split into sentences.
        sentences = [s.strip() for s in _SENTENCE_END_RE.split(text) if s.strip()]
        if not sentences:
            sentences = [text]

        claims: list[AtomicClaim] = []
        for sentence in sentences:
            # Further split compound claims.
            fragments = _COMPOUND_SPLIT_RE.split(sentence)
            for frag in fragments:
                frag = _REFUTED_SUFFIX_RE.sub("", frag).strip()
                if not frag:
                    continue
                if _is_meta_evidence_fragment(frag):
                    continue
                # Keep short clauses created by an explicit compound split
                # (for example, "Acme won"); otherwise skip short noise.
                if len(frag.split()) < 3 and len(fragments) == 1:
                    continue
                sec_ids, sec_urls = _extract_stated_sec_ids(frag)
                claim_id = _stable_claim_id(frag, sentence)
                claims.append(
                    AtomicClaim(
                        claim_id=claim_id,
                        text=frag,
                        source_sentence=sentence,
                        stated_sec_ids=sec_ids,
                        stated_sec_urls=sec_urls,
                    )
                )

        return claims
