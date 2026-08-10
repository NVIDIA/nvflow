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
"""Embedding-centroid source router for ProvenanceGuard Open v1.

For each unique source (identified by ``source_id`` or ``chunk_id``),
computes the mean embedding of its chunks (a centroid).  Each atomic
claim is routed to the source whose centroid has the highest cosine
similarity.  The margin between top-1 and top-2 is recorded so the
pipeline can flag ambiguous routing.

Uses the injectable :class:`~nvflow.provenanceguard.protocols.Embedder`
protocol.  No lexical/hash runtime fallback — if the embedder fails,
the evaluator must produce an ``unavailable`` decision.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from nvflow.provenanceguard.protocols import Embedder, RoutedClaim
from nvflow.provenanceguard.types import AtomicClaim, EvidenceChunk


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

        # Group evidence by routing key (source_id, or chunk_id as fallback).
        groups: dict[str, list[EvidenceChunk]] = {}
        key_order: list[str] = []
        for chunk in evidence:
            # Composite/unattributable chunks (attribution_state == "unavailable")
            # have source_id=None on purpose. Using chunk_id directly would be
            # fine since IDs are unique, but we prefix with "_unattributable:" so
            # the group key can never collide with a real source_id and the
            # intent is explicit when debugging.
            if chunk.attribution_state == "unavailable":
                key = f"_unattributable:{chunk.chunk_id}"
            else:
                key = chunk.source_id or chunk.chunk_id
            if key not in groups:
                groups[key] = []
                key_order.append(key)
            groups[key].append(chunk)

        # Embed evidence + claims in one batch.
        all_evidence_texts: list[str] = []
        chunk_keys: list[str] = []
        for key in key_order:
            for c in groups[key]:
                all_evidence_texts.append(c.text)
                chunk_keys.append(key)
        claim_texts = [c.text for c in claims]

        embeddings = self._embedder.embed(all_evidence_texts + claim_texts)
        if len(embeddings) != len(all_evidence_texts) + len(claim_texts):
            raise RuntimeError("embedder returned wrong number of vectors")

        ev_vecs = embeddings[: len(all_evidence_texts)]
        cl_vecs = embeddings[len(all_evidence_texts) :]

        # Per-key centroid.
        centroids: dict[str, list[float]] = {k: [] for k in key_order}
        counts: dict[str, int] = dict.fromkeys(key_order, 0)
        for key, vec in zip(chunk_keys, ev_vecs, strict=True):
            _vec_add_inplace(centroids[key], vec)
            counts[key] += 1
        for key in key_order:
            n = counts[key] or 1
            centroids[key] = _vec_scale(centroids[key], 1.0 / n)

        # Representative chunk per key (longest text, deterministic).
        rep_chunk: dict[str, EvidenceChunk] = {
            key: max(groups[key], key=lambda c: len(c.text)) for key in key_order
        }

        routed: list[RoutedClaim] = []
        for claim, cv in zip(claims, cl_vecs, strict=True):
            scored = [(key, _cosine(cv, centroids[key])) for key in key_order]
            scored.sort(key=lambda kv: kv[1], reverse=True)
            top_key, top_score = scored[0]
            margin = top_score - (scored[1][1] if len(scored) > 1 else 0.0)
            routed.append(
                RoutedClaim(
                    claim=claim,
                    chunk=rep_chunk[top_key],
                    score=float(top_score),
                    margin=float(margin),
                )
            )
        return routed


__all__ = ["EmbeddingSourceRouter"]
