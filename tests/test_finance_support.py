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
"""Focused regressions for source-bound finance reasoning."""

from __future__ import annotations

import json

import pytest

from nvflow.grounding_verifier.types import Decision, EvidenceChunk
from nvflow.recipes.finance.utils.rl.finance_support import evaluate_finance_support


def _fact(ticker: str, company: str, metric: str, year: int, value: int) -> EvidenceChunk:
    accession = {"AAPL": "000032019324000123", "MSFT": "000095017024087843"}[ticker]
    chunk_id = f"{ticker}_{metric.replace(' ', '')}_{year}"
    return EvidenceChunk(
        chunk_id=chunk_id,
        text=f"{company} ({ticker}) reported {metric} of ${value:,} for fiscal year {year}.",
        source_id=f"sec:cik=0000000001:accession={accession}:doc={ticker.lower()}.htm",
        sec_accession=accession,
        sec_document=f"{ticker.lower()}.htm",
        sec_cik="0000000001",
        sec_ticker=ticker,
        sec_company_name=company,
    )


def _answer(kind: str, value, citations: list[str], explanation: str) -> str:
    return json.dumps(
        {
            "answer_type": kind,
            "value": value,
            "unit": "percent" if kind == "calculation" else "ticker",
            "evidence_ids": citations,
            "explanation": explanation,
        }
    )


def _allow_semantics(_answer: str, _premise: str) -> Decision:
    return Decision(status="allow", reason="semantic_entailment")


def _block_semantics(_answer: str, _premise: str) -> Decision:
    return Decision(status="block", reason="semantic_contradiction")


@pytest.mark.parametrize(
    ("value", "citations", "explanation", "status"),
    [
        (-10, ["AAPL_netincome_2023", "AAPL_netincome_2024"], "Apple's change was -10%.", "allow"),
        (-20, ["AAPL_netincome_2023", "AAPL_netincome_2024"], "Apple's change was -20%.", "block"),
        (-10, ["AAPL_netincome_2024"], "Apple's change was -10%.", "block"),
        (
            -10,
            ["AAPL_netincome_2023", "AAPL_netincome_2024", "fabricated-source"],
            "Apple's change was -10%.",
            "block",
        ),
        (
            -10,
            ["AAPL_netincome_2023", "AAPL_netincome_2024"],
            "For Microsoft, net income changed by -10%.",
            "block",
        ),
        (
            -10,
            ["AAPL_netincome_2023", "AAPL_netincome_2024"],
            "Apple's change was -10%, and revenue was $1,234,567.",
            "block",
        ),
    ],
)
def test_calculation_is_value_source_and_entity_bound(value, citations, explanation, status):
    evidence = [
        _fact("AAPL", "Apple Inc.", "net income", 2023, 100_000_000),
        _fact("AAPL", "Apple Inc.", "net income", 2024, 90_000_000),
    ]
    decision = evaluate_finance_support(
        _answer("calculation", value, citations, explanation),
        "Using the evidence cards, what was AAPL's year-over-year percent change in net income "
        "from fiscal year 2023 to fiscal year 2024?",
        evidence,
        _block_semantics if "Microsoft" in explanation else _allow_semantics,
    )
    assert decision is not None and decision.status == status


@pytest.mark.parametrize(
    ("value", "citations", "explanation", "status"),
    [
        ("AAPL", ["AAPL_netincome_2024", "MSFT_netincome_2024"], "AAPL was highest.", "allow"),
        ("MSFT", ["AAPL_netincome_2024", "MSFT_netincome_2024"], "MSFT was highest.", "block"),
        ("AAPL", ["AAPL_netincome_2024"], "AAPL was highest.", "block"),
        ("AAPL", ["AAPL_netincome_2024", "MSFT_netincome_2024"], "MSFT was highest.", "block"),
    ],
)
def test_comparison_recomputes_all_cited_candidates(value, citations, explanation, status):
    evidence = [
        _fact("AAPL", "Apple Inc.", "net income", 2024, 100_000_000),
        _fact("MSFT", "Microsoft Corp", "net income", 2024, 90_000_000),
    ]
    decision = evaluate_finance_support(
        _answer("comparison", value, citations, explanation),
        "Among AAPL and MSFT, which ticker had the highest net income in fiscal year 2024?",
        evidence,
        _block_semantics if explanation.startswith("MSFT") else _allow_semantics,
    )
    assert decision is not None and decision.status == status


@pytest.mark.parametrize(
    "explanation",
    [
        "This gives Microsoft a net income change of -10%.",
        "The issuer is Microsoft; the net income change was -10%.",
        "Therefore, Microsoft posted a -10% net income change.",
    ],
)
def test_calculation_rejects_plain_language_entity_swaps(explanation):
    evidence = [
        _fact("AAPL", "Apple Inc.", "net income", 2023, 100_000_000),
        _fact("AAPL", "Apple Inc.", "net income", 2024, 90_000_000),
    ]
    decision = evaluate_finance_support(
        _answer(
            "calculation",
            -10,
            ["AAPL_netincome_2023", "AAPL_netincome_2024"],
            explanation,
        ),
        "What was AAPL's year-over-year percent change in net income from fiscal year 2023 "
        "to fiscal year 2024?",
        evidence,
        _block_semantics,
    )
    assert decision is not None and decision.status == "block"
    assert decision.reason == "semantic_contradiction"


@pytest.mark.parametrize(
    "explanation",
    [
        "Microsoft was the winner.",
        "The winner was Microsoft.",
        "Microsoft ranks first.",
        "The submitted answer is AAPL, although Microsoft won.",
    ],
)
def test_comparison_rejects_contradictory_winner_paraphrases(explanation):
    evidence = [
        _fact("AAPL", "Apple Inc.", "net income", 2024, 100_000_000),
        _fact("MSFT", "Microsoft Corp", "net income", 2024, 90_000_000),
    ]
    decision = evaluate_finance_support(
        _answer(
            "comparison",
            "AAPL",
            ["AAPL_netincome_2024", "MSFT_netincome_2024"],
            explanation,
        ),
        "Among AAPL and MSFT, which ticker had the highest net income in fiscal year 2024?",
        evidence,
        _block_semantics,
    )
    assert decision is not None and decision.status == "block"
    assert decision.reason == "semantic_contradiction"


@pytest.mark.parametrize(
    "explanation",
    [
        "Apple was the winner.",
        "The winner was Apple.",
        "Among AAPL and MSFT, AAPL ranked first.",
        "Apple Inc. (AAPL) had the highest net income.",
    ],
)
def test_comparison_allows_grounded_winner_paraphrases(explanation):
    evidence = [
        _fact("AAPL", "Apple Inc.", "net income", 2024, 100_000_000),
        _fact("MSFT", "Microsoft Corp", "net income", 2024, 90_000_000),
    ]
    decision = evaluate_finance_support(
        _answer(
            "comparison",
            "AAPL",
            ["AAPL_netincome_2024", "MSFT_netincome_2024"],
            explanation,
        ),
        "Among AAPL and MSFT, which ticker had the highest net income in fiscal year 2024?",
        evidence,
        _allow_semantics,
    )
    assert decision is not None and decision.status == "allow"


def test_comparison_normalizes_displayed_financial_units():
    smaller = _fact("AAPL", "Apple Inc.", "total assets", 2024, 93_020_840)
    larger = _fact("MSFT", "Microsoft Corp", "total assets", 2024, 137_012)
    evidence = [
        EvidenceChunk(
            **{
                **smaller.__dict__,
                "text": (
                    "Apple Inc. (AAPL) reported total assets of $93,020,840 thousand "
                    "for fiscal year 2024."
                ),
            }
        ),
        EvidenceChunk(
            **{
                **larger.__dict__,
                "text": (
                    "Microsoft Corp (MSFT) reported total assets of $137,012 million "
                    "for fiscal year 2024."
                ),
            }
        ),
    ]
    decision = evaluate_finance_support(
        _answer(
            "comparison",
            "MSFT",
            [smaller.chunk_id, larger.chunk_id],
            "Apple Inc. (AAPL) reported total assets of $93,020,840 thousand for fiscal "
            "year 2024. Microsoft Corp (MSFT) reported total assets of $137,012 million "
            "for fiscal year 2024. Microsoft Corp (MSFT) had the highest total assets.",
        ),
        "Among AAPL and MSFT, which ticker had the highest total assets in fiscal year 2024?",
        evidence,
        _allow_semantics,
    )
    assert decision is not None and decision.status == "allow"


def test_evidence_scoped_refusal_requires_the_requested_fact_to_be_absent():
    evidence = [_fact("AAPL", "Apple Inc.", "net income", 2024, 100_000_000)]
    question = "What was AAPL's free cash flow for fiscal year 2024? Answer from the cards."
    safe = _answer("insufficient_evidence", None, [], "Free cash flow was not found in the cards.")
    false = _answer("insufficient_evidence", None, [], "Net income was not found in the cards.")

    assert evaluate_finance_support(safe, question, evidence, _allow_semantics).status == "allow"
    assert (
        evaluate_finance_support(
            false,
            "What was AAPL's net income for fiscal year 2024? Answer from the cards.",
            evidence,
            _allow_semantics,
        ).status
        == "block"
    )


def test_multiple_atomic_facts_can_share_one_retrieval_chunk():
    first = _fact("AAPL", "Apple Inc.", "total assets", 2023, 100_000_000)
    evidence = [
        EvidenceChunk(
            **{
                **first.__dict__,
                "text": (
                    "APPLE INC. (AAPL) reported total assets of $100,000,000 for fiscal year 2023. "
                    "APPLE INC. (AAPL) reported total assets of $110,000,000 for fiscal year 2024."
                ),
            }
        )
    ]
    answer = _answer(
        "calculation",
        10,
        [first.chunk_id],
        "APPLE INC. (AAPL) reported total assets of $100,000,000 for fiscal year 2023. "
        "APPLE INC. (AAPL) reported total assets of $110,000,000 for fiscal year 2024. "
        "The year-over-year change was 10%.",
    )
    decision = evaluate_finance_support(
        answer,
        "What was AAPL's year-over-year percent change in total assets from fiscal year 2023 "
        "to fiscal year 2024?",
        evidence,
        _allow_semantics,
    )
    assert decision is not None and decision.status == "allow"


def test_all_caps_company_words_are_not_treated_as_ticker_mentions():
    evidence = [
        _fact("AAPL", "APPLE AIR LINES, INC.", "net income", 2023, 100_000_000),
        _fact("AAPL", "APPLE AIR LINES, INC.", "net income", 2024, 90_000_000),
    ]
    answer = _answer(
        "calculation",
        -10,
        ["AAPL_netincome_2023", "AAPL_netincome_2024"],
        "APPLE AIR LINES, INC. (AAPL) reported net income of $100,000,000 for fiscal year 2023. "
        "APPLE AIR LINES, INC. (AAPL) reported net income of $90,000,000 for fiscal year 2024. "
        "The year-over-year change was -10%.",
    )
    decision = evaluate_finance_support(
        answer,
        "What was AAPL's year-over-year percent change in net income from fiscal year 2023 "
        "to fiscal year 2024?",
        evidence,
        _allow_semantics,
    )
    assert decision is not None and decision.status == "allow"


def test_refusal_target_comes_from_answer_not_coverage_instructions():
    evidence = [_fact("AAPL", "Apple Inc.", "total assets", 2024, 100_000_000)]
    answer = _answer(
        "insufficient_evidence",
        None,
        [],
        "Using only the provided evidence cards for Apple Inc. (AAPL), free cash flow is not "
        "available for fiscal year 2024.",
    )
    question = (
        "Using only the evidence cards for Apple Inc. (AAPL), retrieve total assets as a "
        "coverage check, then report free cash flow for fiscal year 2024."
    )
    decision = evaluate_finance_support(answer, question, evidence, _allow_semantics)
    assert decision is not None and decision.status == "allow"


def test_closed_world_refusal_accepts_a_single_evidence_card():
    evidence = [_fact("AAPL", "Northstar Ltd.", "revenue", 2024, 100_000_000)]
    answer = _answer(
        "insufficient_evidence",
        None,
        [],
        "Using only the provided evidence card for Northstar Ltd. (AAPL), free cash flow "
        "is not available for fiscal year 2024.",
    )
    question = (
        "Using only the evidence card for Northstar Ltd. (AAPL), retrieve revenue as a "
        "coverage check, then report free cash flow for fiscal year 2024."
    )
    decision = evaluate_finance_support(answer, question, evidence, _allow_semantics)
    assert decision is not None and decision.status == "allow"


def test_calculation_rejects_a_supported_value_assigned_to_the_wrong_year():
    evidence = [
        _fact("AAPL", "Apple Inc.", "net income", 2023, 100_000_000),
        _fact("AAPL", "Apple Inc.", "net income", 2024, 90_000_000),
    ]
    answer = _answer(
        "calculation",
        -10,
        ["AAPL_netincome_2023", "AAPL_netincome_2024"],
        "Apple Inc. (AAPL) reported net income of $100,000,000 for fiscal year 2023. "
        "Apple Inc. (AAPL) reported net income of $90,000,000 for fiscal year 2025. "
        "The year-over-year change was -10%.",
    )
    decision = evaluate_finance_support(
        answer,
        "What was AAPL's year-over-year percent change in net income from fiscal year 2023 "
        "to fiscal year 2024?",
        evidence,
        _allow_semantics,
    )
    assert decision is not None and decision.status == "block"
    assert decision.reason == "temporal_conflation"
