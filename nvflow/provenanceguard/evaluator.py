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
"""Core evaluator for ProvenanceGuard Open v1 (routing-nli-v1).

Pipeline: decompose → route → NLI → protected-value check → decision.

Decision strictness (conservative):

- **allow**: every claim is entailed AND every protected value is
  found in routed evidence.
- **block**: any claim is contradiction, neutral, no_source, or
  protected_value_mismatch.
- **unavailable**: no evidence chunks were provided, no claims were
  extracted from the answer, or any NLI/model error prevented at
  least one claim from being fully evaluated.  If any claim has an
  error, the row is unavailable even if another claim is
  contradiction or neutral.

This is an **uncalibrated open approximation** — not paper-faithful.
Evidence is retrieval_model_excerpt only, never primary SEC verification.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from nvflow.provenanceguard.protected_values import check_protected_values
from nvflow.provenanceguard.protocols import (
    ClaimDecomposer,
    NLIScorer,
    SourceRouter,
)
from nvflow.provenanceguard.types import (
    AtomicClaim,
    ClaimVerdict,
    Decision,
    EvidenceChunk,
)

ALGORITHM_VERSION = "routing-nli-v1"


@dataclass(frozen=True)
class ProvenanceGuardConfig:
    """Configuration for the ProvenanceGuard evaluator.

    The fail-closed policy is **fixed and non-configurable**:
    contradiction, neutral, no_source, and protected_value_mismatch
    always block; any model/trace error yields unavailable.
    There are no ``block_on_*`` toggles — the policy cannot be
    disabled.

    Attributes:
        evidence_excerpt_length: Maximum characters of evidence text
            to include in each claim verdict for auditability.
        routing_model: Model ID used for embedding/routing (metadata).
        nli_model: Model ID used for NLI scoring (metadata).
    """

    evidence_excerpt_length: int = 500
    routing_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    nli_model: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": ALGORITHM_VERSION,
            "policy": "fixed_fail_closed",
            "evidence_excerpt_length": self.evidence_excerpt_length,
            "routing_model": self.routing_model,
            "nli_model": self.nli_model,
        }


class ProvenanceGuardEvaluator:
    """Orchestrates claim decomposition, routing, NLI, and protected-value checking.

    All model-bearing collaborators (``Embedder``, ``NLIScorer``) are
    injected so that tests can use deterministic fakes.  No Hugging Face
    imports happen during construction or evaluation of this class —
    that is the responsibility of ``HFEmbedder`` / ``HFNLI``, which load
    lazily.
    """

    def __init__(
        self,
        *,
        decomposer: ClaimDecomposer,
        router: SourceRouter,
        nli_scorer: NLIScorer,
        config: ProvenanceGuardConfig | None = None,
    ) -> None:
        self._decomposer = decomposer
        self._router = router
        self._nli = nli_scorer
        self._config = config or ProvenanceGuardConfig()

    @property
    def config(self) -> ProvenanceGuardConfig:
        return self._config

    def evaluate(
        self,
        answer: str,
        evidence: Sequence[EvidenceChunk],
    ) -> Decision:
        """Evaluate an answer against evidence and return a Decision.

        Args:
            answer: The final answer text submitted by the agent.
            evidence: Evidence chunks extracted from the tool-call trace.

        Returns:
            A :class:`~nvflow.provenanceguard.types.Decision` with status
            ``allow``, ``block``, or ``unavailable``.
        """
        errors: list[str] = []

        if not evidence:
            return Decision(
                status="unavailable",
                reason="no_evidence",
                errors=("No evidence chunks were provided.",),
            )

        claims = self._decomposer.decompose(answer)
        if not claims:
            return Decision(
                status="unavailable",
                reason="no_claims_extracted",
                errors=(),
            )

        try:
            routed = self._router.route(claims, evidence)
        except Exception as exc:
            return Decision(
                status="unavailable",
                reason="routing_error",
                errors=(f"Routing failed: {exc!s}",),
            )

        routed_by_claim_id: dict[str, Any] = {}
        for rc in routed:
            routed_by_claim_id[rc.claim.claim_id] = rc

        verdicts: list[ClaimVerdict] = []
        nli_errors = 0

        for claim in claims:
            verdict = self._evaluate_claim(claim, routed_by_claim_id, errors)
            verdicts.append(verdict)
            if verdict.errors:
                nli_errors += 1

        if nli_errors > 0:
            reason = "all_claims_errored" if nli_errors == len(claims) else "partial_nli_errors"
            return Decision(
                status="unavailable",
                reason=reason,
                verdicts=tuple(verdicts),
                errors=tuple(errors),
            )

        status, reason = self._aggregate(verdicts)
        return Decision(
            status=status,
            reason=reason,
            verdicts=tuple(verdicts),
            errors=tuple(errors),
        )

    def _evaluate_claim(
        self,
        claim: AtomicClaim,
        routed_by_claim_id: dict[str, Any],
        errors: list[str],
    ) -> ClaimVerdict:
        """Evaluate a single claim and return its verdict."""
        routed = routed_by_claim_id.get(claim.claim_id)

        if routed is None:
            return ClaimVerdict(
                claim_id=claim.claim_id,
                claim_text=claim.text,
                final_label="no_source",
                protected_value_outcome="not_applicable",
                errors=(),
            )

        chunk = routed.chunk
        excerpt = chunk.text[: self._config.evidence_excerpt_length]

        if chunk.attribution_state in ("unavailable", "composite", "unknown"):
            return ClaimVerdict(
                claim_id=claim.claim_id,
                claim_text=claim.text,
                routed_source_id=chunk.source_id,
                routed_source_ids=chunk.source_ids,
                routed_attribution_state=chunk.attribution_state,
                routed_chunk_id=chunk.chunk_id,
                routing_score=routed.score,
                routing_margin=routed.margin,
                final_label="no_source",
                protected_value_outcome="not_applicable",
                evidence_excerpt=excerpt,
                errors=(),
            )

        try:
            nli_result = self._nli.score(premise=chunk.text, hypothesis=claim.text)
        except Exception as exc:
            err = f"NLI error for claim {claim.claim_id}: {exc!s}"
            errors.append(err)
            return ClaimVerdict(
                claim_id=claim.claim_id,
                claim_text=claim.text,
                routed_source_id=chunk.source_id,
                routed_source_ids=chunk.source_ids,
                routed_attribution_state=chunk.attribution_state,
                routed_chunk_id=chunk.chunk_id,
                routing_score=routed.score,
                routing_margin=routed.margin,
                raw_nli_label="neutral",
                raw_nli_probabilities=(),
                final_label="neutral",
                protected_value_outcome="not_applicable",
                evidence_excerpt=excerpt,
                errors=(err,),
            )

        raw_label = nli_result.label
        pv_outcome, _missing = check_protected_values(claim.text, chunk.text)

        final_label = self._apply_policy(raw_label, pv_outcome)

        return ClaimVerdict(
            claim_id=claim.claim_id,
            claim_text=claim.text,
            routed_source_id=chunk.source_id,
            routed_source_ids=chunk.source_ids,
            routed_attribution_state=chunk.attribution_state,
            routed_chunk_id=chunk.chunk_id,
            routing_score=routed.score,
            routing_margin=routed.margin,
            raw_nli_label=raw_label,
            raw_nli_probabilities=nli_result.probabilities,
            final_label=final_label,
            protected_value_outcome=pv_outcome,
            evidence_excerpt=excerpt,
            errors=(),
        )

    def _apply_policy(self, raw_nli_label: str, pv_outcome: str) -> str:
        """Apply fixed fail-closed policy to the raw NLI label."""
        if raw_nli_label == "entailment":
            if pv_outcome == "fail":
                return "protected_value_mismatch"
            return "entailment"
        if raw_nli_label == "contradiction":
            return "contradiction"
        return "neutral"

    def _aggregate(self, verdicts: list[ClaimVerdict]) -> tuple[str, str]:
        """Aggregate per-claim verdicts into a top-level decision (fixed fail-closed)."""
        labels = [v.final_label for v in verdicts if not v.errors]
        if not labels:
            return "unavailable", "all_claims_errored"

        if any(lbl == "contradiction" for lbl in labels):
            return "block", "contradiction"
        if any(lbl == "protected_value_mismatch" for lbl in labels):
            return "block", "protected_value_mismatch"
        if any(lbl == "neutral" for lbl in labels):
            return "block", "neutral"
        if any(lbl == "no_source" for lbl in labels):
            return "block", "no_source"
        if all(lbl == "entailment" for lbl in labels):
            return "allow", "all_entailed"
        return "block", "unverifiable"


__all__ = [
    "ALGORITHM_VERSION",
    "ProvenanceGuardConfig",
    "ProvenanceGuardEvaluator",
]
