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
"""Domain data types for ProvenanceGuard Open v1.

All types are frozen dataclasses so they are hashable and safe to share
across threads.  The ``to_dict`` / ``serialize`` helpers produce the JSON
shape written to sidecar files.

This is an **uncalibrated open approximation** of the research
ProvenanceGuard system.  The evidence basis is model-retrieved trace
excerpts, NOT primary SEC filing verification.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvidenceChunk:
    """A single evidence span extracted from a tool-call trace.

    ``content_basis`` is always ``"retrieval_model_excerpt"`` — the text
    came from a model retrieval call (``retrieve_information``), not
    from primary source verification.

    ``source_id`` is a stable canonical SEC identifier
    (``sec:cik=<...>:accession=<...>:doc=<...>``) when the chunk can be
    correlated to a specific filing.  When multiple storage keys are
    combined in one retrieval output and cannot be separated,
    ``source_ids`` holds all candidate IDs and ``attribution_state`` is
    ``"unavailable"``.
    """

    chunk_id: str
    text: str
    content_basis: str = "retrieval_model_excerpt"
    source_id: str | None = None
    source_ids: tuple[str, ...] = ()
    sec_url: str | None = None
    sec_accession: str | None = None
    sec_document: str | None = None
    sec_cik: str | None = None
    storage_keys: tuple[str, ...] = ()
    char_range: tuple[int, int] | None = None
    tool_call_id: str | None = None
    tool_result_id: str | None = None
    attribution_state: str = "available"

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        if self.char_range is not None:
            d["char_range"] = {"start": self.char_range[0], "end": self.char_range[1]}
        d["source_ids"] = list(self.source_ids)
        d["storage_keys"] = list(self.storage_keys)
        return d


@dataclass(frozen=True)
class AtomicClaim:
    """A sub-sentence atomic claim extracted from the final answer.

    ``claim_id`` is a deterministic hash of the claim text and source
    sentence, so re-evaluation of the same answer yields stable IDs.

    ``stated_sec_ids`` / ``stated_sec_urls`` capture SEC identifiers or
    URLs the claim explicitly references in its text.
    """

    claim_id: str
    text: str
    source_sentence: str
    stated_sec_ids: tuple[str, ...] = ()
    stated_sec_urls: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["stated_sec_ids"] = list(self.stated_sec_ids)
        d["stated_sec_urls"] = list(self.stated_sec_urls)
        return d


@dataclass(frozen=True)
class ClaimVerdict:
    """Per-claim outcome with routing, NLI, and protected-value detail.

    ``raw_nli_label`` is the direct output of the NLI model
    (``entailment`` / ``neutral`` / ``contradiction``).

    ``final_label`` applies policy adjustments:
      - ``no_source`` when no evidence chunk was available for routing.
      - ``protected_value_mismatch`` when protected numeric/date/percentage
        values in an entailed claim are absent from routed evidence.
    """

    claim_id: str
    claim_text: str
    routed_source_id: str | None = None
    routed_source_ids: tuple[str, ...] = ()
    routed_attribution_state: str = "available"
    routed_chunk_id: str | None = None
    routing_score: float = 0.0
    routing_margin: float = 0.0
    raw_nli_label: str = "neutral"
    raw_nli_probabilities: tuple[tuple[str, float], ...] = ()
    final_label: str = "neutral"
    protected_value_outcome: str = "not_applicable"
    evidence_excerpt: str = ""
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "routed_source_id": self.routed_source_id,
            "routed_source_ids": list(self.routed_source_ids),
            "routed_attribution_state": self.routed_attribution_state,
            "routed_chunk_id": self.routed_chunk_id,
            "routing_score": self.routing_score,
            "routing_margin": self.routing_margin,
            "raw_nli_label": self.raw_nli_label,
            "raw_nli_probabilities": dict(self.raw_nli_probabilities),
            "final_label": self.final_label,
            "protected_value_outcome": self.protected_value_outcome,
            "evidence_excerpt": self.evidence_excerpt,
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class Decision:
    """Top-level allow / block / unavailable result for one rollout row."""

    status: str  # allow | block | unavailable
    reason: str
    verdicts: tuple[ClaimVerdict, ...] = ()
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "verdicts": [v.to_dict() for v in self.verdicts],
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class NLIResult:
    """Raw NLI model output for a single (premise, hypothesis) pair."""

    label: str  # entailment | neutral | contradiction
    score: float
    probabilities: tuple[tuple[str, float], ...] = ()
