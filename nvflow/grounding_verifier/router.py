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
"""Embedding-centroid source router for GroundingVerifier.

For each unique source (identified by ``source_id`` or ``chunk_id``),
computes the mean embedding of its chunks (a centroid).  Each atomic
claim is routed to the source whose centroid has the highest cosine
similarity.  The margin between top-1 and top-2 is recorded so the
pipeline can flag ambiguous routing.

Uses the injectable :class:`~nvflow.grounding_verifier.protocols.Embedder`
protocol.  No lexical/hash runtime fallback — if the embedder fails,
the evaluator must produce an ``unavailable`` decision.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from nvflow.grounding_verifier.conflation import content_for_routing
from nvflow.grounding_verifier.protected_values import (
    check_protected_values,
    extract_protected_values,
)
from nvflow.grounding_verifier.protocols import Embedder, RoutedClaim
from nvflow.grounding_verifier.types import AtomicClaim, EvidenceChunk


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(x * x for x in a[:n]))
    nb = math.sqrt(sum(x * x for x in b[:n]))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _vec_add_inplace(acc: list[float], v: Sequence[float]) -> None:
    if len(acc) != len(v):
        if not acc:
            acc.extend(float(x) for x in v)
            return
        raise ValueError(f"embedding dim mismatch: acc={len(acc)} v={len(v)}")
    for i, x in enumerate(v):
        acc[i] += float(x)


def _vec_scale(v: list[float], k: float) -> list[float]:
    return [x * k for x in v]


def _routing_key(chunk: EvidenceChunk) -> str:
    if chunk.attribution_state == "unavailable":
        return f"_unattributable:{chunk.chunk_id}"
    return chunk.source_id or chunk.chunk_id


def _group_evidence(
    evidence: Sequence[EvidenceChunk],
) -> dict[str, list[EvidenceChunk]]:
    groups: dict[str, list[EvidenceChunk]] = {}
    for chunk in evidence:
        groups.setdefault(_routing_key(chunk), []).append(chunk)
    return groups


def _build_centroids(
    keys: Sequence[str],
    chunk_keys: Sequence[str],
    vectors: Sequence[Sequence[float]],
) -> dict[str, list[float]]:
    centroids: dict[str, list[float]] = {key: [] for key in keys}
    counts = dict.fromkeys(keys, 0)
    for key, vector in zip(chunk_keys, vectors, strict=True):
        _vec_add_inplace(centroids[key], vector)
        counts[key] += 1
    return {key: _vec_scale(centroids[key], 1.0 / (counts[key] or 1)) for key in keys}


def _routing_candidates(
    claim_text: str,
    keys: Sequence[str],
    groups: dict[str, list[EvidenceChunk]],
) -> Sequence[str]:
    if not extract_protected_values(claim_text):
        return keys
    matches = [
        key
        for key in keys
        if any(check_protected_values(claim_text, chunk.text)[0] == "pass" for chunk in groups[key])
    ]
    return matches or keys


def _matching_chunks(
    claim_text: str,
    chunks_with_vectors: Sequence[tuple[EvidenceChunk, Sequence[float]]],
) -> Sequence[tuple[EvidenceChunk, Sequence[float]]]:
    if not extract_protected_values(claim_text):
        return chunks_with_vectors
    matches = [
        item
        for item in chunks_with_vectors
        if check_protected_values(claim_text, item[0].text)[0] == "pass"
    ]
    return matches or chunks_with_vectors


class EmbeddingSourceRouter:
    """Cosine-on-centroids router.  Returns top-1 with margin to top-2."""

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    def route(
        self,
        claims: Sequence[AtomicClaim],
        evidence: Sequence[EvidenceChunk],
    ) -> Sequence[RoutedClaim]:
        if not claims or not evidence:
            return []

        groups = _group_evidence(evidence)
        keys = list(groups)
        keyed_chunks = [(key, chunk) for key in keys for chunk in groups[key]]
        evidence_texts = [chunk.text for _, chunk in keyed_chunks]
        claim_texts = [content_for_routing(claim.text) for claim in claims]

        embeddings = self._embedder.embed(evidence_texts + claim_texts)
        if len(embeddings) != len(evidence_texts) + len(claim_texts):
            raise RuntimeError("embedder returned wrong number of vectors")

        evidence_vectors = embeddings[: len(evidence_texts)]
        claim_vectors = embeddings[len(evidence_texts) :]
        centroids = _build_centroids(keys, [key for key, _ in keyed_chunks], evidence_vectors)
        chunks_by_key: dict[str, list[tuple[EvidenceChunk, Sequence[float]]]] = {
            key: [] for key in keys
        }
        for (key, chunk), vector in zip(keyed_chunks, evidence_vectors, strict=True):
            chunks_by_key[key].append((chunk, vector))

        routed: list[RoutedClaim] = []
        for claim, vector in zip(claims, claim_vectors, strict=True):
            claim_text = content_for_routing(claim.text)
            candidate_keys = _routing_candidates(claim_text, keys, groups)
            scored = sorted(
                ((key, _cosine(vector, centroids[key])) for key in candidate_keys),
                key=lambda item: item[1],
                reverse=True,
            )
            top_key, top_score = scored[0]
            margin = top_score - (scored[1][1] if len(scored) > 1 else 0.0)
            chunk = max(
                _matching_chunks(claim_text, chunks_by_key[top_key]),
                key=lambda item: _cosine(vector, item[1]),
            )[0]
            routed.append(
                RoutedClaim(
                    claim=claim,
                    chunk=chunk,
                    score=float(top_score),
                    margin=float(margin),
                )
            )
        return routed
