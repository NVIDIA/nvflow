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
"""Golden + symmetry tests for the validate_questions regex_prefilter set.

The prefilter uses ONE source-of-truth set ``_NON_COMPANY_TOKENS``
(lowercase) consumed by both ``_has_ticker`` and
``_has_proper_noun_mid_sentence`` via case-fold comparison.  Prior to
unification the same concept was stored in two parallel sets in
different casings, and they drifted: a token like ``"INC"`` was in
``TICKER_DENYLIST`` but ``"Inc"`` was missing from
``_PROPER_NOUN_STOPWORDS``, so the proper-noun rescue incorrectly
treated the bare word ``"Inc"`` as a company reference.

These tests pin the unified contract:

1. The set contents are byte-pinned (``test_non_company_tokens_match_golden``).
2. Every token is in canonical lowercase, no apostrophes (the regex word
   boundary excludes apostrophes from emitted tokens, so storing
   ``"it's"`` or similar would be dead-letter).
3. Bidirectional symmetry: for every token, neither check rescues
   regardless of how the token is cased in source text.  This is the
   property that the casing-mirror bug violated and the test that would
   have caught it.
"""

from __future__ import annotations

import string

import pytest

from nvflow.recipes.finance.utils.rl.regex_prefilter_questions import (
    _NON_COMPANY_TOKENS,
    _has_proper_noun_mid_sentence,
    _has_ticker,
)

# Pinned literal value of _NON_COMPANY_TOKENS.  Update only with an
# accompanying re-run of the diff-test against fresh baselines and a
# manual review of newly-dropped rows -- this set governs which
# question-style strings get filtered out of the GRPO training corpus.
_GOLDEN = frozenset(
    {
        # Articles, prepositions, conjunctions, basic verbs, pronouns,
        # determiners.
        "a",
        "am",
        "an",
        "and",
        "as",
        "at",
        "be",
        "by",
        "do",
        "for",
        "if",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "from",
        "with",
        "but",
        "than",
        "then",
        "that",
        "this",
        "these",
        "those",
        "between",
        "among",
        "over",
        "under",
        "during",
        "before",
        "after",
        # Auxiliary / modal verbs.
        "are",
        "was",
        "were",
        "been",
        "does",
        "did",
        "has",
        "have",
        "had",
        "can",
        "could",
        "should",
        "would",
        "will",
        "shall",
        "may",
        "might",
        "must",
        # WH / interrogative starters.
        "how",
        "what",
        "why",
        "where",
        "when",
        "who",
        "which",
        "whom",
        "whose",
        # Imperative question starters.
        "given",
        "considering",
        "assuming",
        "suppose",
        "compare",
        "contrast",
        "explain",
        "discuss",
        "describe",
        "analyse",
        "analyze",
        "evaluate",
        "identify",
        "summarise",
        "summarize",
        "define",
        "calculate",
        "estimate",
        "find",
        "list",
        "name",
        "provide",
        "present",
        "show",
        "state",
        "using",
        # Sentence-context modifiers.
        "based",
        "non",
        # Corporate suffixes.
        "inc",
        "llc",
        "lp",
        "ltd",
        # Financial / corporate / general acronyms.
        "ai",
        "ceo",
        "cfo",
        "coo",
        "cto",
        "eps",
        "gdp",
        "sec",
        "go",
        "it",
        "its",
        "my",
        "no",
        "not",
        "our",
        "so",
        "up",
        "us",
        "usa",
        "we",
        # Single-letter / very short ticker-shaped tokens.
        "m",
        "md",
        "r",
        "qa",
        "qb",
        "qc",
        "qd",
        "qe",
    }
)


def test_non_company_tokens_match_golden() -> None:
    assert _NON_COMPANY_TOKENS == _GOLDEN


def test_all_entries_are_canonical_lowercase() -> None:
    """Entries must be lowercase ASCII letters (consumers compare via .lower())."""
    allowed = set(string.ascii_lowercase)
    for token in _NON_COMPANY_TOKENS:
        assert token, "empty token in _NON_COMPANY_TOKENS"
        assert token == token.lower(), f"non-lowercase entry: {token!r}"
        bad = set(token) - allowed
        assert not bad, f"token {token!r} contains non-letter chars {bad!r}"


@pytest.mark.parametrize("token", sorted(_NON_COMPANY_TOKENS))
def test_bidirectional_casing_symmetry(token: str) -> None:
    """Both checks must agree on every token regardless of source casing.

    Pre-unification this property was violated: ``"INC"`` was in
    ``TICKER_DENYLIST`` (so ``_has_ticker`` correctly skipped it) but
    ``"Inc"`` was NOT in ``_PROPER_NOUN_STOPWORDS`` (so
    ``_has_proper_noun_mid_sentence`` rescued it).  This regression
    test asserts the symmetry that closes that gap.

    Construction note: we wrap each token in a stub sentence so the
    consumer functions exercise a real regex-match path.  The sentence
    has NO other capitalised tokens, so a True return value can ONLY
    come from the token under test.
    """
    # _has_ticker only matches uppercase 1-5 char tokens; only test this
    # branch when the token shape is reachable by TICKER_RE.
    if 1 <= len(token) <= 5:
        sentence_ticker = f"the firm reported {token.upper()} last quarter"
        assert not _has_ticker(sentence_ticker), (
            f"_has_ticker rescued on uppercase {token.upper()!r} -- "
            "regression of the casing-mirror bug"
        )

    # _has_proper_noun_mid_sentence only matches 3+ char first-cap tokens.
    if len(token) >= 3:
        sentence_proper = f"the firm reported {token.title()} last quarter"
        assert not _has_proper_noun_mid_sentence(sentence_proper), (
            f"_has_proper_noun_mid_sentence rescued on title-case {token.title()!r} -- "
            "regression of the casing-mirror bug"
        )
        # Also exercise the all-caps form against the proper-noun check
        # (regex matches all-caps because [a-zA-Z] accepts uppercase).
        sentence_proper_caps = f"the firm reported {token.upper()} last quarter"
        assert not _has_proper_noun_mid_sentence(sentence_proper_caps), (
            f"_has_proper_noun_mid_sentence rescued on all-caps {token.upper()!r} -- "
            "regression of the casing-mirror bug"
        )
