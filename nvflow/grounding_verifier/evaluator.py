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
"""Core evaluator for GroundingVerifier.

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

Evidence is retrieval_model_excerpt only, never primary SEC verification.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from nvflow.grounding_verifier.conflation import (
    content_for_routing,
    has_explicit_entity_conflation,
    has_explicit_source_conflation,
)
from nvflow.grounding_verifier.embedder import DEFAULT_EMBEDDING_MODEL
from nvflow.grounding_verifier.nli import DEFAULT_NLI_MODEL
from nvflow.grounding_verifier.protected_values import (
    check_protected_values,
    extract_protected_values,
    has_financial_metric_mismatch,
)
from nvflow.grounding_verifier.protocols import ClaimDecomposer, NLIScorer, SourceRouter
from nvflow.grounding_verifier.types import (
    AtomicClaim,
    ClaimVerdict,
    Decision,
    EvidenceChunk,
)

ALGORITHM_VERSION = "routing-nli-sec-attribution-v5"

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_DIRECT_SUPPORT_STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "approximately",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "had",
    "has",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "were",
}


def _fact_tokens(text: str) -> set[str]:
    """Return conservative lexical tokens for direct fact alignment."""
    return {
        token.lower()
        for token in _TOKEN_RE.findall(text)
        if token.lower() not in _DIRECT_SUPPORT_STOPWORDS
    }


def _has_direct_value_support(
    claim_text: str,
    evidence: EvidenceChunk,
    pv_outcome: str,
) -> bool:
    """Recognize near-verbatim numeric facts when an NLI model is under-confident.

    This is deliberately narrower than a generic lexical fallback: every protected
    value must already match, at least one non-date numeric value must be present,
    and the evidence must cover both the claim vocabulary and two descriptive
    (non-numeric) terms. A changed number therefore remains fail-closed.
    """
    if pv_outcome != "pass":
        return False
    protected = extract_protected_values(claim_text)
    if not any(value.kind in {"currency", "number", "percentage"} for value in protected):
        return False

    if not evidence.sec_company_name and not evidence.sec_ticker:
        return False
    claim_lower = claim_text.lower()
    company_tokens = {
        token
        for token in _fact_tokens(evidence.sec_company_name or "")
        if token not in {"co", "company", "corp", "corporation", "inc", "incorporated", "ltd"}
    }
    ticker = (evidence.sec_ticker or "").lower()
    has_entity_match = any(token in claim_lower for token in company_tokens) or (
        bool(ticker) and re.search(rf"\b{re.escape(ticker)}\b", claim_lower) is not None
    )
    if not has_entity_match and not re.search(r"\b(?:the company|registrant)\b", claim_lower):
        return False

    claim_tokens = _fact_tokens(claim_text)
    evidence_tokens = _fact_tokens(evidence.text)
    if not claim_tokens:
        return False

    overlap = claim_tokens & evidence_tokens
    descriptive_overlap = {token for token in overlap if not token.isdigit()}
    return len(descriptive_overlap) >= 2 and len(overlap) / len(claim_tokens) >= 0.8


@dataclass(frozen=True)
class GroundingVerifierConfig:
    """Configuration for the GroundingVerifier evaluator.

    The fail-closed policy is **fixed and non-configurable**:
    contradiction, no_source, and protected_value_mismatch always block;
    neutral blocks unless deterministic entity/metric/value checks establish
    direct support. Any model/trace error yields unavailable.
    There are no ``block_on_*`` toggles — the policy cannot be
    disabled.

    Attributes:
        evidence_excerpt_length: Maximum characters of evidence text
            to include in each claim verdict for auditability.
        routing_model: Model ID used for embedding/routing (metadata).
        nli_model: Model ID used for NLI scoring (metadata).
    """

    evidence_excerpt_length: int = 500
    routing_model: str = DEFAULT_EMBEDDING_MODEL
    nli_model: str = DEFAULT_NLI_MODEL

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": ALGORITHM_VERSION,
            "policy": "fixed_fail_closed",
            "evidence_excerpt_length": self.evidence_excerpt_length,
            "routing_model": self.routing_model,
            "nli_model": self.nli_model,
        }


class GroundingVerifierEvaluator:
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
        config: GroundingVerifierConfig | None = None,
    ) -> None:
        self._decomposer = decomposer
        self._router = router
        self._nli = nli_scorer
        self._config = config or GroundingVerifierConfig()

    @property
    def config(self) -> GroundingVerifierConfig:
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
            A :class:`~nvflow.grounding_verifier.types.Decision` with status
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

        routed_by_claim_id = {item.claim.claim_id: item for item in routed}

        verdicts: list[ClaimVerdict] = []
        for claim in claims:
            verdicts.append(self._evaluate_claim(claim, routed_by_claim_id, errors))

        nli_errors = sum(bool(verdict.errors) for verdict in verdicts)
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

    def verify_against_premise(self, answer: str, premise: str) -> Decision:
        """Require every submitted claim to be entailed by a canonical premise.

        This is used for derived facts, such as arithmetic or comparisons, for
        which callers can construct a deterministic premise from attributed
        source facts. No source routing or lexical entity vocabulary is used.
        """
        claims = self._decomposer.decompose(answer)
        if not claims:
            return Decision(status="unavailable", reason="no_claims_extracted")

        verdicts = []
        errors = []
        for claim in claims:
            claim_text = " ".join(claim.text.split())
            premise_text = " ".join(premise.split())
            if claim_text in premise_text:
                verdicts.append(
                    ClaimVerdict(
                        claim_id=claim.claim_id,
                        claim_text=claim.text,
                        raw_nli_label="entailment",
                        raw_nli_probabilities=(("entailment", 1.0),),
                        final_label="entailment",
                        evidence_excerpt=premise[: self._config.evidence_excerpt_length],
                    )
                )
                continue
            try:
                result = self._nli.score(premise=premise, hypothesis=claim.text)
            except Exception as exc:
                error = f"NLI error for claim {claim.claim_id}: {exc!s}"
                errors.append(error)
                verdicts.append(
                    ClaimVerdict(
                        claim_id=claim.claim_id,
                        claim_text=claim.text,
                        final_label="neutral",
                        evidence_excerpt=premise[: self._config.evidence_excerpt_length],
                        errors=(error,),
                    )
                )
                continue
            verdicts.append(
                ClaimVerdict(
                    claim_id=claim.claim_id,
                    claim_text=claim.text,
                    raw_nli_label=result.label,
                    raw_nli_probabilities=result.probabilities,
                    final_label=result.label,
                    evidence_excerpt=premise[: self._config.evidence_excerpt_length],
                )
            )

        if errors:
            return Decision(
                status="unavailable",
                reason="semantic_verification_error",
                verdicts=tuple(verdicts),
                errors=tuple(errors),
            )
        for label in ("contradiction", "neutral"):
            if any(verdict.final_label == label for verdict in verdicts):
                return Decision(
                    status="block",
                    reason=f"semantic_{label}",
                    verdicts=tuple(verdicts),
                )
        return Decision(status="allow", reason="semantic_entailment", verdicts=tuple(verdicts))

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
        routed_fields = {
            "claim_id": claim.claim_id,
            "claim_text": claim.text,
            "routed_source_id": chunk.source_id,
            "routed_source_ids": chunk.source_ids,
            "routed_attribution_state": chunk.attribution_state,
            "routed_chunk_id": chunk.chunk_id,
            "routing_score": routed.score,
            "routing_margin": routed.margin,
            "evidence_excerpt": excerpt,
        }

        if chunk.attribution_state in ("unavailable", "composite", "unknown"):
            return ClaimVerdict(
                final_label="no_source",
                protected_value_outcome="not_applicable",
                errors=(),
                **routed_fields,
            )

        if has_explicit_source_conflation(claim, chunk):
            return ClaimVerdict(
                final_label="conflation",
                protected_value_outcome="not_applicable",
                errors=(),
                **routed_fields,
            )

        if has_explicit_entity_conflation(claim, chunk):
            return ClaimVerdict(
                final_label="entity_conflation",
                protected_value_outcome="not_applicable",
                errors=(),
                **routed_fields,
            )

        factual_text = content_for_routing(claim.text)
        try:
            nli_result = self._nli.score(premise=chunk.text, hypothesis=factual_text)
        except Exception as exc:
            err = f"NLI error for claim {claim.claim_id}: {exc!s}"
            errors.append(err)
            return ClaimVerdict(
                raw_nli_label="neutral",
                raw_nli_probabilities=(),
                final_label="neutral",
                protected_value_outcome="not_applicable",
                errors=(err,),
                **routed_fields,
            )

        raw_label = nli_result.label
        pv_outcome, _missing = check_protected_values(factual_text, chunk.text)

        if has_financial_metric_mismatch(factual_text, chunk.text):
            return ClaimVerdict(
                raw_nli_label=raw_label,
                raw_nli_probabilities=nli_result.probabilities,
                final_label="financial_metric_mismatch",
                protected_value_outcome=pv_outcome,
                errors=(),
                **routed_fields,
            )

        final_label = self._apply_policy(raw_label, pv_outcome)
        if raw_label == "neutral" and _has_direct_value_support(factual_text, chunk, pv_outcome):
            final_label = "entailment"

        return ClaimVerdict(
            raw_nli_label=raw_label,
            raw_nli_probabilities=nli_result.probabilities,
            final_label=final_label,
            protected_value_outcome=pv_outcome,
            errors=(),
            **routed_fields,
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

        for blocking_label in (
            "conflation",
            "entity_conflation",
            "financial_metric_mismatch",
            "contradiction",
            "protected_value_mismatch",
            "neutral",
            "no_source",
        ):
            if blocking_label in labels:
                return "block", blocking_label
        if all(lbl == "entailment" for lbl in labels):
            return "allow", "all_entailed"
        return "block", "unverifiable"
