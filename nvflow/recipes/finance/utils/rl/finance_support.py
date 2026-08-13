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
"""Source-bound verification for common finance calculations and comparisons."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from nvflow.grounding_verifier.protected_values import (
    extract_financial_metrics,
    extract_protected_values,
    protected_numeric_value,
)
from nvflow.grounding_verifier.types import ClaimVerdict, Decision, EvidenceChunk

_YEAR_RE = re.compile(r"\b(?:fiscal year|FY)\s*(20\d{2})\b", re.IGNORECASE)
_PERCENT_RE = re.compile(r"(-?\d[\d,]*(?:\.\d+)?)\s*(?:%|percent\b)", re.IGNORECASE)
_REFUSAL_RE = re.compile(
    r"\b(?:insufficient evidence|cannot determine|not (?:found|available|provided)|"
    r"no .+ data|does not contain)\b",
    re.IGNORECASE,
)
_CLOSED_WORLD_RE = re.compile(
    r"\b(?:answer\s+)?(?:only\s+)?(?:using|from)\s+only\s+(?:the\s+)?(?:provided\s+)?"
    r"(?:evidence\s+)?cards?\b|"
    r"\busing\s+only\s+(?:the\s+)?(?:provided\s+)?(?:evidence\s+)?cards?\b|"
    r"\banswer\s+from\s+(?:the\s+)?cards?\b",
    re.IGNORECASE,
)
SemanticVerifier = Callable[[str, str], Decision]


@dataclass(frozen=True)
class _Answer:
    kind: str
    value: object
    citations: tuple[str, ...]
    explanation: str


@dataclass(frozen=True)
class _Fact:
    chunk: EvidenceChunk
    ticker: str
    metric: str
    year: int
    value: Decimal
    displayed_value: Decimal


def _parse_answer(answer: str, question: str) -> _Answer | None:
    try:
        payload = json.loads(answer)
    except (json.JSONDecodeError, TypeError):
        payload = None
    if isinstance(payload, dict):
        return _Answer(
            kind=str(payload.get("answer_type") or "").lower(),
            value=payload.get("value"),
            citations=tuple(str(item) for item in payload.get("evidence_ids") or ()),
            explanation=str(payload.get("explanation") or ""),
        )
    if _REFUSAL_RE.search(answer):
        return _Answer("insufficient_evidence", None, (), answer)
    if _operation(question) in {"yoy", "margin"}:
        values = _PERCENT_RE.findall(answer)
        if values:
            return _Answer(
                "calculation",
                values[-1],
                tuple(re.findall(r"https?://\S+", answer, re.IGNORECASE)),
                answer,
            )
    return None


def _operation(question: str) -> str | None:
    lower = question.lower()
    if "year-over-year" in lower and "percent change" in lower:
        return "yoy"
    if "gross margin" in lower:
        return "margin"
    if "highest" in lower:
        return "highest"
    if "lowest" in lower:
        return "lowest"
    return None


def _query_tickers(question: str, evidence: list[EvidenceChunk]) -> tuple[str, ...]:
    lower = question.lower()
    found = []
    for chunk in evidence:
        ticker = (chunk.sec_ticker or "").upper()
        names = _company_aliases(chunk)
        if ticker and (
            re.search(rf"\b{re.escape(ticker.lower())}\b", lower)
            or any(re.search(rf"\b{re.escape(name)}(?:'s)?\b", lower) for name in names)
        ):
            found.append(ticker)
    return tuple(dict.fromkeys(found))


def _company_aliases(chunk: EvidenceChunk) -> list[str]:
    return [
        word
        for word in re.findall(r"[a-z0-9]+", (chunk.sec_company_name or "").lower())
        if word not in {"co", "company", "corp", "corporation", "inc", "ltd", "llc", "plc"}
    ]


def _fact(chunk: EvidenceChunk, text: str | None = None) -> _Fact | None:
    if chunk.attribution_state != "available" or not chunk.source_id or not chunk.sec_ticker:
        return None
    text = text or chunk.text
    metrics = extract_financial_metrics(text)
    years = set(_YEAR_RE.findall(text))
    protected = extract_protected_values(text)
    currency = [item for item in protected if item.kind == "currency"]
    numeric = [item for item in protected if item.kind == "number"]
    candidates = currency or numeric
    values = {protected_numeric_value(item) for item in candidates} - {None}
    text_lower = text.lower()
    names = _company_aliases(chunk)
    entity_matches = (chunk.sec_ticker.lower() in text_lower) or any(
        re.search(rf"\b{re.escape(name)}\b", text_lower) for name in names
    )
    if len(metrics) != 1 or len(years) != 1 or len(values) != 1 or not entity_matches:
        return None
    displayed = {_literal_number(item) for item in candidates} - {None}
    if len(displayed) != 1:
        return None
    return _Fact(
        chunk,
        chunk.sec_ticker.upper(),
        next(iter(metrics)),
        int(next(iter(years))),
        next(iter(values)),
        next(iter(displayed)),
    )


def _facts(chunk: EvidenceChunk) -> list[_Fact]:
    """Extract every atomic fact while retaining the chunk's source identity."""
    return [
        fact
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", chunk.text)
        if (fact := _fact(chunk, sentence)) is not None
    ]


def _cites(answer: _Answer, facts: list[_Fact]) -> bool:
    if not answer.citations:
        return False

    def matches(citation: str, fact: _Fact) -> bool:
        accession = re.sub(r"\D", "", fact.chunk.sec_accession or "")
        return (
            citation == fact.chunk.chunk_id
            or citation == fact.chunk.source_id
            or citation.rstrip(".,;)") == (fact.chunk.sec_url or "")
            or bool(accession and re.sub(r"\D", "", citation) == accession)
        )

    return all(
        any(matches(citation, fact) for citation in answer.citations) for fact in facts
    ) and all(any(matches(citation, fact) for fact in facts) for citation in answer.citations)


def _decimal(value: object) -> Decimal | None:
    try:
        return Decimal(str(value).replace(",", ""))
    except (InvalidOperation, TypeError, ValueError):
        return None


def _literal_number(value) -> Decimal | None:
    """Return the displayed number without applying a magnitude-word scale."""
    match = re.search(r"-?\d[\d,]*(?:\.\d+)?", value.raw)
    return _decimal(match.group(0)) if match else None


def _verdict(
    status: str, reason: str, answer: _Answer, facts: list[_Fact] | None = None
) -> Decision:
    facts = facts or []
    return Decision(
        status=status,
        reason=reason,
        verdicts=(
            ClaimVerdict(
                claim_id="finance-support",
                claim_text=answer.explanation,
                routed_source_id=facts[0].chunk.source_id if len(facts) == 1 else None,
                routed_source_ids=tuple(fact.chunk.source_id or "" for fact in facts),
                final_label=reason,
                protected_value_outcome="pass" if status == "allow" else "fail",
                evidence_excerpt="\n".join(fact.chunk.text for fact in facts)[:500],
            ),
        ),
    )


def _values_are_supported(
    answer: _Answer, facts: list[_Fact], expected_result: Decimal | None = None
) -> bool:
    explanation = re.sub(r"\$\\(?:approx|times)\$", " ", answer.explanation)
    for value in extract_protected_values(explanation):
        if value.kind == "date":
            continue
        number = _literal_number(value)
        if number is None:
            continue
        if value.kind == "percentage":
            if expected_result is None or abs(number - expected_result) > Decimal("0.15"):
                return False
        elif all(number != fact.displayed_value for fact in facts):
            return False
    return True


def _metrics_are_supported(answer: _Answer, facts: list[_Fact]) -> bool:
    claimed = extract_financial_metrics(answer.explanation)
    supported = {fact.metric for fact in facts}
    return not claimed or claimed.issubset(supported)


def _years_are_supported(answer: _Answer, facts: list[_Fact]) -> bool:
    claimed = set(_YEAR_RE.findall(answer.explanation))
    return not claimed or claimed == {str(fact.year) for fact in facts}


def _target_metric(answer: _Answer, question: str) -> str:
    """Prefer the submitted claim, then fall back to an unambiguous question metric."""
    for text in (answer.explanation, question):
        metrics = extract_financial_metrics(text)
        if len(metrics) == 1:
            return next(iter(metrics))
    return ""


def _number(value: Decimal) -> str:
    return format(value, "f").rstrip("0").rstrip(".") or "0"


def _fact_claim(fact: _Fact) -> str:
    """Preserve the exact attributed excerpt, including displayed units."""
    return fact.chunk.text


def _premise(facts: list[_Fact], conclusion: str) -> str:
    excerpts = dict.fromkeys(_fact_claim(fact) for fact in facts)
    return " ".join([*excerpts, conclusion])


def _semantic_gate(
    answer: _Answer,
    premise: str,
    verify_semantics: SemanticVerifier,
) -> Decision | None:
    decision = verify_semantics(answer.explanation, premise)
    return None if decision.status == "allow" else decision


def _select(facts: list[_Fact], ticker: str, metric: str, year: int) -> _Fact | None:
    matches = [
        fact
        for fact in facts
        if fact.ticker == ticker and fact.metric == metric and fact.year == year
    ]
    return matches[0] if len(matches) == 1 else None


def _calculation(
    operation: str,
    answer: _Answer,
    question: str,
    facts: list[_Fact],
    tickers: tuple[str, ...],
    verify_semantics: SemanticVerifier,
) -> Decision:
    if len(tickers) != 1:
        return _verdict("block", "entity_conflation", answer)
    years = [int(year) for year in _YEAR_RE.findall(question)]
    ticker = tickers[0]
    selected: list[_Fact] = []
    if operation == "yoy" and len(years) >= 2:
        metric = _target_metric(answer, question)
        selected = [
            fact
            for year in (years[0], years[-1])
            if (fact := _select(facts, ticker, metric, year)) is not None
        ]
        expected = (
            (selected[1].value - selected[0].value) / abs(selected[0].value) * 100
            if len(selected) == 2 and selected[0].value
            else None
        )
    elif operation == "margin" and years:
        selected = [
            fact
            for metric in ("gross_profit", "revenue")
            if (fact := _select(facts, ticker, metric, years[-1])) is not None
        ]
        expected = (
            selected[0].value / selected[1].value * 100
            if len(selected) == 2 and selected[1].value
            else None
        )
    else:
        expected = None
    observed = _decimal(answer.value)
    if expected is None or observed is None:
        return _verdict("block", "calculation_inputs_missing", answer, selected)
    if not _cites(answer, selected):
        return _verdict("block", "source_conflation", answer, selected)
    if abs(observed - expected) > Decimal("0.15"):
        return _verdict("block", "calculation_mismatch", answer, selected)
    if not _metrics_are_supported(answer, selected):
        return _verdict("block", "metric_conflation", answer, selected)
    if not _years_are_supported(answer, selected):
        return _verdict("block", "temporal_conflation", answer, selected)
    if not _values_are_supported(answer, selected, expected):
        return _verdict("block", "protected_value_mismatch", answer, selected)
    company = selected[0].chunk.sec_company_name or ticker
    metric = selected[0].metric.replace("_", " ")
    if operation == "yoy":
        conclusion = (
            f"{company} ({ticker}) had a year-over-year percent change in {metric} "
            f"of {_number(expected)}% from fiscal year {selected[0].year} to fiscal year "
            f"{selected[1].year}."
        )
    else:
        conclusion = (
            f"{company} ({ticker}) had a gross margin of {_number(expected)}% for fiscal "
            f"year {selected[0].year}."
        )
    premise = _premise(selected, conclusion)
    if decision := _semantic_gate(answer, premise, verify_semantics):
        return decision
    return _verdict("allow", "calculation_verified", answer, selected)


def _comparison(
    operation: str,
    answer: _Answer,
    question: str,
    facts: list[_Fact],
    tickers: tuple[str, ...],
    verify_semantics: SemanticVerifier,
) -> Decision:
    years = [int(year) for year in _YEAR_RE.findall(question)]
    metric = _target_metric(answer, question)
    selected = [
        fact
        for ticker in tickers
        if years and (fact := _select(facts, ticker, metric, years[-1])) is not None
    ]
    if len(tickers) < 2 or len(selected) != len(tickers):
        return _verdict("block", "comparison_inputs_missing", answer, selected)
    if not _cites(answer, selected):
        return _verdict("block", "source_conflation", answer, selected)
    choose = max if operation == "highest" else min
    expected = choose(selected, key=lambda fact: fact.value).ticker
    if str(answer.value).upper() != expected:
        return _verdict("block", "comparison_mismatch", answer, selected)
    if not _metrics_are_supported(answer, selected):
        return _verdict("block", "metric_conflation", answer, selected)
    if not _years_are_supported(answer, selected):
        return _verdict("block", "temporal_conflation", answer, selected)
    if not _values_are_supported(answer, selected):
        return _verdict("block", "protected_value_mismatch", answer, selected)
    winner = next(fact for fact in selected if fact.ticker == expected)
    company = winner.chunk.sec_company_name or winner.ticker
    candidates = " and ".join(
        f"{fact.chunk.sec_company_name or fact.ticker} ({fact.ticker})" for fact in selected
    )
    premise = _premise(
        selected,
        f"{company} ({expected}) had the {operation} {metric.replace('_', ' ')} for "
        f"fiscal year {winner.year} among {candidates}.",
    )
    if decision := _semantic_gate(answer, premise, verify_semantics):
        return decision
    return _verdict("allow", "comparison_verified", answer, selected)


def _refusal(
    answer: _Answer,
    question: str,
    facts: list[_Fact],
    tickers: tuple[str, ...],
    operation: str | None,
    evidence: list[EvidenceChunk],
    verify_semantics: SemanticVerifier,
) -> Decision:
    if len(tickers) != 1:
        return _verdict("unavailable", "refusal_target_unparsed", answer)
    years = [int(year) for year in _YEAR_RE.findall(question)]
    metric = _target_metric(answer, question)
    if not metric or not years:
        return _verdict("unavailable", "refusal_target_unparsed", answer)
    year = years[-1]
    if operation in {"yoy", "margin"}:
        probe = _calculation(
            operation,
            _Answer("calculation", 0, (), ""),
            question,
            facts,
            tickers,
            verify_semantics,
        )
        if probe.reason != "calculation_inputs_missing":
            return _verdict("block", "false_refusal", answer)
    else:
        if _select(facts, tickers[0], metric, year):
            return _verdict("block", "false_refusal", answer)
        for chunk in evidence:
            if (chunk.sec_ticker or "").upper() != tickers[0]:
                continue
            if metric in extract_financial_metrics(chunk.text) and str(year) in chunk.text:
                return _verdict("unavailable", "refusal_evidence_ambiguous", answer)
    if not _CLOSED_WORLD_RE.search(question):
        return _verdict("unavailable", "refusal_scope_unverified", answer)
    if not any(fact.ticker == tickers[0] for fact in facts):
        return _verdict("unavailable", "refusal_coverage_unverified", answer)
    if any(
        value.kind in {"currency", "number", "percentage"}
        for value in extract_protected_values(answer.explanation)
    ):
        return _verdict("block", "unsupported_refusal_detail", answer)
    company = next(
        (fact.chunk.sec_company_name for fact in facts if fact.ticker == tickers[0]),
        None,
    ) or tickers[0]
    premise = _premise(
        [fact for fact in facts if fact.ticker == tickers[0]],
        f"Using only the provided evidence cards, {metric.replace('_', ' ')} for "
        f"{company} ({tickers[0]}) is not available for fiscal year {year}.",
    )
    if decision := _semantic_gate(answer, premise, verify_semantics):
        return decision
    return _verdict("allow", "supported_refusal", answer)


def evaluate_finance_support(
    answer_text: str,
    question: str | None,
    evidence: list[EvidenceChunk],
    verify_semantics: SemanticVerifier,
) -> Decision | None:
    """Verify supported finance reasoning, or return ``None`` for the core evaluator."""
    if not question:
        return None
    answer = _parse_answer(answer_text, question)
    if answer is None:
        return None
    facts = [fact for chunk in evidence for fact in _facts(chunk)]
    tickers = _query_tickers(question, evidence)
    operation = _operation(question)
    if answer.kind == "insufficient_evidence":
        return _refusal(
            answer, question, facts, tickers, operation, evidence, verify_semantics
        )
    if operation in {"yoy", "margin"}:
        return _calculation(operation, answer, question, facts, tickers, verify_semantics)
    if operation in {"highest", "lowest"}:
        return _comparison(operation, answer, question, facts, tickers, verify_semantics)
    return None
