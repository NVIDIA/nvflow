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

"""Drop criteria that restate another criterion in the same rubric.

This is deliberately not ``data.dedup``. That stage removes duplicate
*records* and cannot see inside one, and two of the rules here are outside what
MinHash similarity can express: a criterion that pins a different figure is
never a duplicate however alike it reads, and "implied by another" is a subset
relation rather than a symmetric one.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

# Relations that cost a criterion its place. ``regrades_same_number`` is
# reported but kept: two criteria can legitimately check the same figure from
# different angles, which is coverage rather than redundancy.
DROPPED_RELATIONS = frozenset({"near_duplicate", "implied_by_other"})

_SIMILARITY_THRESHOLD = 0.6

_NUMBER = re.compile(r"(?<![\w.])[$\u20ac\u00a3]?\d[\d,]*(?:\.\d+)?%?")

# Form designators ("10-K") and section references carry no quantity.
_NOT_A_QUANTITY = re.compile(r"\b\d{1,2}-[A-Z]\b|\bSection\s+\d+(\.\d+)*")

_STOPWORDS = frozenset(
    """a an and are as at be been by correctly does for from has have identify in including is
    it its list mention must not of on or report response set should state states stating that
    the their there these this to value values was were which with within answer must_pass
    criterion""".split()
)


def quantities(text: str) -> set[str]:
    """Return the figures a criterion pins, normalized for comparison.

    Bare years and one-digit values are excluded: they appear in almost every
    criterion and would make unrelated ones look like they check the same thing.
    """
    out: set[str] = set()
    for match in _NUMBER.findall(_NOT_A_QUANTITY.sub(" ", text or "")):
        value = match.lstrip("$\u20ac\u00a3").replace(",", "").rstrip("%")
        if "." in value:
            value = value.rstrip("0").rstrip(".")
        if not value or re.fullmatch(r"(19|20)\d\d", value) or len(value.replace(".", "")) < 2:
            continue
        out.add(value)
    return out


def content_words(text: str) -> set[str]:
    """Return the words that carry meaning, lowercased."""
    tokens = re.findall(r"[a-z0-9.$%]+", (text or "").lower())
    return {token for token in tokens if token not in _STOPWORDS and len(token) > 1}


def proper_nouns(text: str) -> set[str]:
    """Return the content words that appear inside a capitalized span."""
    out: set[str] = set()
    for span in re.findall(r"\b[A-Z][\w&'.\-]*", text or ""):
        out |= content_words(span)
    return out


def relation(first: str, second: str) -> str | None:
    """Classify how two criteria in one rubric relate, or ``None`` if unrelated.

    Criteria that read alike are not duplicates when the words they differ on
    carry the fact: a different figure, or a different named entity. Without
    that guard, a rubric checking a segment's revenue and the consolidated
    total loses one of them.
    """
    words_a, words_b = content_words(first), content_words(second)
    if not words_a or not words_b:
        return None

    numbers_a, numbers_b = quantities(first), quantities(second)
    if numbers_a and numbers_b and numbers_a != numbers_b:
        return None

    similarity = len(words_a & words_b) / len(words_a | words_b)
    if numbers_a and numbers_a == numbers_b:
        return "near_duplicate" if similarity >= _SIMILARITY_THRESHOLD else "regrades_same_number"

    if (words_a ^ words_b) & (proper_nouns(first) | proper_nouns(second)):
        return None
    if similarity >= _SIMILARITY_THRESHOLD:
        return "near_duplicate"
    if (
        len(words_a & words_b) == min(len(words_a), len(words_b))
        and abs(len(words_a) - len(words_b)) >= 2
    ):
        return "implied_by_other"
    return None


def dedup_criteria(
    rubric: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Return the rubric without restated criteria, and what was found.

    A criterion is never dropped if that would leave the rubric with no
    must-pass criterion: a rubric with nothing mandatory cannot fail an answer,
    which is worse than a rubric that repeats itself.
    """
    criteria = list(rubric)
    texts = [str(item.get("criterion") or "") for item in criteria]
    found: Counter[str] = Counter()
    drop: set[int] = set()

    for i in range(len(texts)):
        if i in drop:
            continue
        for j in range(i + 1, len(texts)):
            if j in drop:
                continue
            found_relation = relation(texts[i], texts[j])
            if found_relation:
                found[found_relation] += 1
                if found_relation in DROPPED_RELATIONS:
                    drop.add(j)

    if not drop:
        return criteria, found

    kept = [item for index, item in enumerate(criteria) if index not in drop]
    if not any(item.get("must_pass") for item in kept) and any(
        item.get("must_pass") for item in criteria
    ):
        first_must_pass = next(
            index for index, item in enumerate(criteria) if item.get("must_pass")
        )
        drop.discard(first_must_pass)
        kept = [item for index, item in enumerate(criteria) if index not in drop]

    found["criteria_dropped"] = len(criteria) - len(kept)
    return kept, found
