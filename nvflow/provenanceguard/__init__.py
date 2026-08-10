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
"""ProvenanceGuard Open v1 — routing-nli-v1.

An uncalibrated, not paper-faithful open approximation of the research
ProvenanceGuard system.  Provides claim decomposition, embedding-centroid
source routing, DeBERTa NLI scoring, and protected-value checking to
produce per-rollout-row allow / block / unavailable verdicts.

All Hugging Face model loading is strictly lazy — no model download or
real model execution is performed at import time.
"""

from nvflow.provenanceguard.protocols import (
    ClaimDecomposer,
    Embedder,
    NLIScorer,
    RoutedClaim,
    SourceRouter,
)
from nvflow.provenanceguard.types import (
    AtomicClaim,
    ClaimVerdict,
    Decision,
    EvidenceChunk,
    NLIResult,
)

__all__ = [
    "AtomicClaim",
    "ClaimDecomposer",
    "ClaimVerdict",
    "Decision",
    "Embedder",
    "EvidenceChunk",
    "NLIScorer",
    "NLIResult",
    "RoutedClaim",
    "SourceRouter",
]
