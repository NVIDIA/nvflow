#!/usr/bin/env python3
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
"""Phase 1 of the ``validate_questions`` GRPO stage: CPU-only regex prefilter.

Deterministic, intentionally narrow drop rule -- recall over precision, so
the LLM phase (Phase 2) is what catches subtler cases.

A record is dropped ONLY when ALL of the following hold:

1. ``problem`` contains a vague-reference phrase (``the company``/``the firm``/
   ``this filing``/...) -- see ``VAGUE_REFS``.
2. ``problem`` does NOT mention ``record["company_name"]`` (full name
   substring match, or first-token whole-word fallback).
3. ``problem`` has NO ticker-like uppercase token (``\\b[A-Z]{1,5}\\b``)
   outside ``TICKER_DENYLIST`` of common non-ticker acronyms.
4. ``problem`` has NO mid-sentence proper-noun token outside
   ``_PROPER_NOUN_STOPWORDS`` (rescues cases like ``company_name="ABNB"``
   but question text says ``"Airbnb"``).

If any check (2-4) passes, the record is kept.

Usage:
    python regex_prefilter_questions.py \\
        --input_file  .../final_result.jsonl \\
        --output_kept .../prefiltered.jsonl \\
        --output_dropped .../regex_dropped.jsonl \\
        --stats_file .../prefilter_stats.json
"""

import argparse
import os
import re
from pathlib import Path

import orjson

from nvflow.utils import setup_logger

logger = setup_logger(__name__)

WRITE_BUFFER_SIZE = 1000

# Vague company references that indicate a question may not be self-contained.
# All compared case-insensitively against the ``problem`` text.
VAGUE_REFS = (
    "the company",
    "the firm",
    "the entity",
    "the corporation",
    "the business",
    "the organization",
    "this filing",
    "this company",
)

# Ticker-like uppercase tokens of 1-5 letters.  A real company identifier
# if present (AAPL, NVDA, GOOGL, etc.).  We exclude a small deny-list of
# sentence-start and common English acronyms that match the pattern but
# aren't tickers in context.
TICKER_RE = re.compile(r"\b[A-Z]{1,5}\b")

# Tokens that LOOK like tickers but are common sentence-case words or
# well-known non-ticker acronyms -- used to avoid false "ticker present"
# matches on generic prose.  Kept conservative so we err on the side of
# recognising a ticker (and therefore keeping the record).
TICKER_DENYLIST = frozenset(
    {
        "A",
        "AI",
        "AM",
        "AN",
        "AND",
        "AS",
        "AT",
        "BE",
        "BY",
        "CEO",
        "CFO",
        "COO",
        "CTO",
        "DO",
        "EPS",
        "FOR",
        "GDP",
        "GO",
        "IF",
        "IN",
        "INC",
        "IS",
        "IT",
        "IT'S",
        "ITS",
        "LLC",
        "LP",
        "LTD",
        "M",
        "MD",
        "MY",
        "NO",
        "NOT",
        "OF",
        "ON",
        "OR",
        "OUR",
        "QA",
        "QB",
        "QC",
        "QD",
        "QE",
        "R",
        "SEC",
        "SO",
        "THE",
        "TO",
        "UP",
        "US",
        "USA",
        "WE",
        "WHY",
    }
)


def _has_vague_reference(text: str) -> bool:
    """True when ``text`` contains any vague-reference phrase (case-insensitive)."""
    lowered = text.lower()
    return any(ref in lowered for ref in VAGUE_REFS)


def _has_company_name(text: str, company_name: str) -> bool:
    """True when ``text`` mentions ``company_name``.

    Matches in order of specificity:
    1. Full name as substring (case-insensitive). E.g. ``company_name``
       ``"NVIDIA Corporation"`` matches if the full string appears.
    2. First token of ``company_name`` as a whole word (case-insensitive).
       E.g. ``company_name="Apple Inc."`` matches ``"Apple's 10-K"`` via the
       first-token fallback because the SDG data often stores the corporate
       suffix (``Inc.``, ``Corporation``, ...) while questions use the
       short form.

    Only falls back to the first token when it is at least 3 characters long
    and not a stopword, to avoid spurious matches on words like ``the``
    or single-letter corporate prefixes.
    """
    if not company_name:
        return False

    text_lower = text.lower()
    name_lower = company_name.lower()

    if name_lower in text_lower:
        return True

    tokens = company_name.split()
    if not tokens:
        return False

    first_token = tokens[0]
    if len(first_token) < 3 or first_token.lower() in {"the", "a", "an"}:
        return False

    pattern = r"\b" + re.escape(first_token) + r"\b"
    return bool(re.search(pattern, text, flags=re.IGNORECASE))


def _has_ticker(text: str) -> bool:
    """True when ``text`` contains at least one ticker-like uppercase token
    that is not in the deny-list of common non-ticker acronyms."""
    for token in TICKER_RE.findall(text):
        if token not in TICKER_DENYLIST:
            return True
    return False


# Mid-sentence capitalised words that are NOT proper-noun signals.  Used to
# stop ``_has_proper_noun_mid_sentence`` from treating sentence-start
# interrogatives and common English stopwords as implicit company references.
#
# Entries are stored in the exact case the regex will emit (title case, first
# char upper + rest lower), since the regex ``\b[A-Z][a-zA-Z]{2,}\b`` already
# requires the first character to be uppercase.  No case normalisation happens
# at match time -- all-caps tokens like "HOW" are extremely rare in SEC text
# and would fall through to the "unknown proper noun" branch (i.e. kept).
_PROPER_NOUN_STOPWORDS = frozenset(
    {
        # WH / interrogative starters
        "How",
        "What",
        "Why",
        "Where",
        "When",
        "Who",
        "Which",
        "Whom",
        "Whose",
        # Imperative question starters
        "Given",
        "Considering",
        "Assuming",
        "Suppose",
        "Compare",
        "Contrast",
        "Explain",
        "Discuss",
        "Describe",
        "Analyse",
        "Analyze",
        "Evaluate",
        "Identify",
        "Summarise",
        "Summarize",
        "Define",
        "Calculate",
        "Estimate",
        "Find",
        "List",
        "Name",
        "Provide",
        "Present",
        "Show",
        "State",
        "Using",
        # Auxiliary / modal verbs capitalised at sentence start
        "Is",
        "Are",
        "Was",
        "Were",
        "Am",
        "Be",
        "Been",
        "Does",
        "Do",
        "Did",
        "Has",
        "Have",
        "Had",
        "Can",
        "Could",
        "Should",
        "Would",
        "Will",
        "Shall",
        "May",
        "Might",
        "Must",
        # Prepositions / conjunctions often capitalised after a period
        "If",
        "In",
        "On",
        "At",
        "By",
        "For",
        "To",
        "From",
        "With",
        "Of",
        "And",
        "Or",
        "But",
        "As",
        "Than",
        "Then",
        "That",
        "This",
        "These",
        "Those",
        "Between",
        "Among",
        "Over",
        "Under",
        "During",
        "Before",
        "After",
        "Based",
        "Non",  # e.g. "Non-GAAP"
        # Generic sentence starters we've seen in SDG prompts
        "The",
        "A",
        "An",
    }
)

# Matches a capitalised token of 3+ letters (including all-caps like "NVDA"
# since ``[a-zA-Z]`` matches uppercase too).  Apostrophes terminate the
# match so "Airbnb's" captures "Airbnb".
_PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")


def _has_proper_noun_mid_sentence(text: str) -> bool:
    """True when ``text`` has a capitalised proper-noun token not in the
    sentence-starter stopword set.  Used as a recall-over-precision
    backstop when ``company_name`` is a ticker (e.g. ``"ABNB"``) but the
    question uses the full company name (e.g. ``"Airbnb"``), so neither
    ``_has_company_name`` nor ``_has_ticker`` catches the reference."""
    for token in _PROPER_NOUN_RE.findall(text):
        if token not in _PROPER_NOUN_STOPWORDS:
            return True
    return False


def _should_drop(problem: str, company_name: str) -> tuple[bool, str]:
    """Return (drop, reason). ``drop=True`` means the record should be dropped."""
    if not isinstance(problem, str) or not problem:
        return True, "empty_problem"

    has_vague = _has_vague_reference(problem)
    if not has_vague:
        return False, "no_vague_reference"

    has_name = _has_company_name(problem, company_name)
    if has_name:
        return False, "vague_reference_but_company_named"

    has_tkr = _has_ticker(problem)
    if has_tkr:
        return False, "vague_reference_but_ticker_present"

    if _has_proper_noun_mid_sentence(problem):
        return False, "vague_reference_but_proper_noun_present"

    return True, "vague_reference_no_identifier"


def prefilter(
    input_file: str,
    output_kept: str,
    output_dropped: str,
    stats_file: str,
    high_drop_threshold: float = 0.20,
) -> dict:
    """Run the Phase 1 regex prefilter.

    See module docstring for the rule.  Returns the stats dict that was
    written to ``stats_file``.
    """
    num_total = 0
    num_kept = 0
    num_dropped = 0
    num_empty_problem = 0

    kept_buffer: list[bytes] = []
    dropped_buffer: list[bytes] = []

    with (
        open(input_file, "rb") as reader,
        open(output_kept, "wb") as kept_writer,
        open(output_dropped, "wb") as dropped_writer,
    ):
        for line in reader:
            line = line.strip()
            if not line:
                continue

            num_total += 1

            try:
                row = orjson.loads(line)
            except orjson.JSONDecodeError as exc:
                # Malformed input -- conservatively drop with a reason so
                # downstream can see what happened, but don't crash the job.
                dropped_buffer.append(
                    orjson.dumps(
                        {
                            "_regex_drop_reason": "malformed_json",
                            "_regex_drop_error": str(exc),
                            "_regex_drop_raw": line.decode("utf-8", errors="replace")[:500],
                        }
                    )
                )
                num_dropped += 1
                continue

            problem = row.get("problem", "")
            company_name = row.get("company_name", "")

            drop, reason = _should_drop(problem, company_name)

            if reason == "empty_problem":
                num_empty_problem += 1

            if drop:
                num_dropped += 1
                dropped_row = dict(row)
                dropped_row["_regex_drop_reason"] = reason
                dropped_buffer.append(orjson.dumps(dropped_row))
            else:
                num_kept += 1
                kept_buffer.append(orjson.dumps(row))

            if len(kept_buffer) >= WRITE_BUFFER_SIZE:
                kept_writer.write(b"\n".join(kept_buffer) + b"\n")
                kept_buffer.clear()
            if len(dropped_buffer) >= WRITE_BUFFER_SIZE:
                dropped_writer.write(b"\n".join(dropped_buffer) + b"\n")
                dropped_buffer.clear()

        if kept_buffer:
            kept_writer.write(b"\n".join(kept_buffer) + b"\n")
        if dropped_buffer:
            dropped_writer.write(b"\n".join(dropped_buffer) + b"\n")

    drop_rate = num_dropped / num_total if num_total else 0.0
    high_drop_warning = drop_rate > high_drop_threshold

    stats = {
        "num_total": num_total,
        "num_kept": num_kept,
        "num_dropped": num_dropped,
        "num_empty_problem": num_empty_problem,
        "drop_rate": round(drop_rate, 6),
        "high_drop_threshold": high_drop_threshold,
        "high_drop_warning": high_drop_warning,
        "input_file": input_file,
        "output_kept": output_kept,
        "output_dropped": output_dropped,
    }

    # Atomic write: the skip-if-present path at the CLI entrypoint treats
    # stats_file's existence as "prior run completed", so the file must
    # only be visible when fully written.  A crash mid-write would
    # otherwise leave a truncated stats_file and silently trigger a skip
    # with stale audit data.
    tmp_stats_file = f"{stats_file}.tmp"
    with open(tmp_stats_file, "wb") as stats_writer:
        stats_writer.write(orjson.dumps(stats, option=orjson.OPT_INDENT_2))
    os.replace(tmp_stats_file, stats_file)

    logger.info("regex prefilter summary")
    logger.info(f"  total:   {num_total}")
    logger.info(f"  kept:    {num_kept}")
    logger.info(f"  dropped: {num_dropped} ({drop_rate * 100:.2f}%)")
    if num_empty_problem:
        logger.info(f"  empty_problem (dropped): {num_empty_problem}")
    if high_drop_warning:
        logger.warning(
            "drop rate %.2f%% exceeds threshold %.2f%% -- inspect %s before "
            "running downstream stages",
            drop_rate * 100,
            high_drop_threshold * 100,
            output_dropped,
        )

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 1 regex prefilter for GRPO validate_questions stage"
    )
    parser.add_argument(
        "--input_file",
        required=True,
        help="Input JSONL produced by SDG (expects 'problem' and 'company_name' fields).",
    )
    parser.add_argument(
        "--output_kept",
        required=True,
        help="Output JSONL of records that passed the regex filter.",
    )
    parser.add_argument(
        "--output_dropped",
        required=True,
        help="Output JSONL of records dropped by the regex filter, annotated with _regex_drop_reason.",
    )
    parser.add_argument(
        "--stats_file",
        required=True,
        help="Output JSON file with total / kept / dropped counts and drop rate.",
    )
    parser.add_argument(
        "--high_drop_threshold",
        type=float,
        default=0.20,
        help="Drop-rate above which the stats file records high_drop_warning=true (default 0.20).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force a re-run even if outputs are already on disk.  Default "
        "is skip-if-present: the script is a no-op when stats_file + "
        "output_kept are both already present, which keeps phase 1 rerun-"
        "safe (doesn't rewrite prefiltered.jsonl, so phase 2's resume by "
        "row index in output.jsonl-async stays consistent).",
    )
    args = parser.parse_args()

    if not args.force:
        stats_path = Path(args.stats_file)
        kept_path = Path(args.output_kept)
        if stats_path.exists() and kept_path.exists():
            # Parse the cached stats and verify the prior run was against
            # the same input we're being asked to filter.  Guards against
            # the footgun where upstream SDG swaps out final_result.jsonl
            # but step-0-validate-questions/ artefacts from the previous
            # source are left in place.
            cached_input: str | None = None
            try:
                cached_stats = orjson.loads(stats_path.read_bytes())
                cached_input = cached_stats.get("input_file")
            except orjson.JSONDecodeError as exc:
                logger.warning(
                    "regex prefilter: stats_file %s is corrupt (%s); re-running",
                    stats_path,
                    exc,
                )

            if cached_input == args.input_file:
                logger.info(
                    "regex prefilter skipped: outputs already present "
                    "(stats=%s, kept=%s; pass --force to re-run)",
                    stats_path,
                    kept_path,
                )
                raise SystemExit(0)

            if cached_input is not None:
                logger.warning(
                    "regex prefilter: stats_file %s was produced from input %r "
                    "but current invocation specifies %r; re-running",
                    stats_path,
                    cached_input,
                    args.input_file,
                )

    prefilter(
        args.input_file,
        args.output_kept,
        args.output_dropped,
        args.stats_file,
        high_drop_threshold=args.high_drop_threshold,
    )
