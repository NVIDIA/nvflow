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
"""Protocol interfaces for GroundingVerifier.

These protocols enable dependency injection so that unit tests can use
deterministic fake decomposer / embedder / NLI implementations with
``HF_HUB_OFFLINE=1`` and ``TRANSFORMERS_OFFLINE=1``.

No model download or real model execution is performed during import.
All concrete implementations that touch Hugging Face libraries do so
lazily — only when their methods are called.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from nvflow.grounding_verifier.types import AtomicClaim, EvidenceChunk, NLIResult


class ClaimDecomposer(Protocol):
    """Split an assistant answer into atomic claims.

    The default deterministic splitter may over-merge compound claims or
    split mid-clause.
    """

    def decompose(self, answer: str) -> Sequence[AtomicClaim]: ...


class Embedder(Protocol):
    """Embed text into fixed-dimensional vectors for routing.

    Default public model: ``sentence-transformers/all-MiniLM-L6-v2``.
    No lexical/hash runtime fallback — if model loading fails, the
    evaluator returns ``unavailable``.
    """

    @property
    def dimension(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


class NLIScorer(Protocol):
    """Score a (premise, hypothesis) pair with sequence-classification NLI.

    Default public model:
    ``MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli``.

    Returns entailment / neutral / contradiction with probabilities.
    No lexical/hash runtime fallback.
    """

    def score(self, *, premise: str, hypothesis: str) -> NLIResult: ...


@dataclass(frozen=True)
class RoutedClaim:
    """A claim paired with its best evidence chunk and routing scores."""

    claim: AtomicClaim
    chunk: EvidenceChunk
    score: float
    margin: float


class SourceRouter(Protocol):
    """Route each claim to its best-matching evidence chunk.

    Uses embedding cosine similarity (centroids per source) to find
    the top-1 evidence source, recording the margin to top-2 for
    ambiguity detection.
    """

    def route(
        self,
        claims: Sequence[AtomicClaim],
        evidence: Sequence[EvidenceChunk],
    ) -> Sequence[RoutedClaim]: ...
