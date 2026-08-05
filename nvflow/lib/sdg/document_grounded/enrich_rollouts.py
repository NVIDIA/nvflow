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
"""Restore original SDG fields onto rollout output, then extract generations.

This is the SDG counterpart of ``nvflow.recipes.finance.utils.rl.enrich_rollouts``.
It is invoked as the ``enrich_module`` of :func:`nvflow.lib.rl.rollout.rollout`
(which runs it inside each seed's merge job, after chunks are concatenated):

    python -m nvflow.lib.sdg.document_grounded.enrich_rollouts <input.jsonl> <rollouts.jsonl>

NeMo-Gym rollout output keeps only gym fields (``response``, ``reward``,
``verifier``, ``match_details``, ``_ng_task_index``, ``agent_ref``) plus the
echoed ``responses_create_params``; the original SDG fields (``context``,
``problem``, ``company_name``, ...) are dropped.  For each rollout row this
script joins it back to its input row, merges the original fields in, and
extracts the assistant text into ``generation_key`` (plus ``reasoning_content``)
so downstream SDG steps keep working.

Join key: ``responses_api.compute_join_id`` = md5 of
``responses_create_params.input``.  Index-based joins are unsafe: ng_collect
assigns ``_ng_task_index`` per chunk, which collides across a merged file.

Strict mode (default) raises on any anomaly -- the silent-corruption guard
described in SKILL.md Gotcha #13:
  - duplicate join id among input rows (ambiguous prompts)
  - a rollout row whose join id matches no input row
  - rollout count != input count (stale cache / partial run / truncation)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from nvflow.lib.sdg.document_grounded.responses_api import (
    _extract_assistant_text,
    _extract_reasoning_text,
    compute_join_id,
)
from nvflow.utils import setup_logger

logger = setup_logger(__name__)


def _has_nonblank_line(path: str) -> bool:
    """True if *path* has at least one non-blank line (cheap, reads lazily)."""
    with open(path) as f:
        for line in f:
            if line.strip():
                return True
    return False


def _build_input_index(input_file: str) -> tuple[dict[str, int], int, int]:
    """Single streaming pass over *input_file* (which may be far larger than RAM).

    Returns ``(offsets, total, duplicate_ids)`` where:
      - ``offsets`` maps each row's join id -> the byte offset of its line, so a
        caller can later ``seek`` back to that one row without holding the file
        in memory.  Last write wins, mirroring the old ``by_id[jid] = row``.
      - ``total`` is the number of parseable rows (== old ``len(inputs)``).
      - ``duplicate_ids`` counts rows whose join id was already seen.

    Only the offset map lives in memory (a few hundred MB for millions of rows),
    never the row contents -- this is the whole point of the rewrite: the prior
    ``_load_jsonl(input_file)`` materialised the entire (200GB+) file as Python
    objects and OOM-killed the merge job.
    """
    offsets: dict[str, int] = {}
    total = 0
    duplicate_ids = 0
    dropped = 0
    # IMPORTANT: use binary mode so offsets are true byte offsets and remain
    # stable across file handles. Text-mode tell() returns an opaque cookie
    # that is not robust to reuse on a separately opened stream.
    with open(input_file, "rb") as f:
        while True:
            offset = f.tell()
            line_bytes = f.readline()
            if not line_bytes:
                break
            if not line_bytes.strip():
                continue
            try:
                row = json.loads(line_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                dropped += 1
                continue
            total += 1
            jid = compute_join_id(row)
            if jid in offsets:
                duplicate_ids += 1
            offsets[jid] = offset
    if dropped:
        logger.warning("Dropped %d malformed line(s) in %s", dropped, input_file)
    return offsets, total, duplicate_ids


def _read_row_at(f, offset: int) -> dict:
    """Read and parse the single JSONL row starting at byte *offset* of *f*."""
    f.seek(offset)
    line_bytes = f.readline()
    if not line_bytes.strip():
        raise json.JSONDecodeError("Expecting value", "", 0)
    return json.loads(line_bytes.decode("utf-8"))


def enrich(
    input_file: str,
    rollouts_file: str,
    *,
    generation_key: str = "generation",
    strict: bool = True,
) -> None:
    """Join rollout output back to input rows and extract generations in place.

    Streaming implementation: neither the input file nor the rollout file is
    loaded into memory in full.  The input file is scanned once to build a
    ``join_id -> byte_offset`` index; the rollout file is then streamed row by
    row, each matched input row is read back via ``seek``, and enriched output is
    written to a temp file that atomically replaces the original on success.
    Peak memory is the offset index plus a couple of rows -- independent of file
    size.

    Args:
        input_file: Responses-API JSONL fed to ng_collect_rollouts (carries the
            original SDG fields).
        rollouts_file: NeMo-Gym rollout output (rewritten in place with restored
            fields + extracted generation).
        generation_key: Field name to store the extracted assistant text.
        strict: Raise on duplicate join id / unmatched rollout / count mismatch.
    """
    # Empty rollout file is a no-op (mirrors the old ``if not rollouts: return``)
    # and lets us skip the expensive input scan entirely.
    if not _has_nonblank_line(rollouts_file):
        logger.warning("No rollouts to enrich in %s -- nothing to do.", rollouts_file)
        return

    offsets, input_total, duplicate_ids = _build_input_index(input_file)

    errors: list[str] = []
    if duplicate_ids:
        errors.append(
            f"{duplicate_ids} input row(s) share a join id "
            f"(identical responses_create_params.input -- ambiguous prompts)"
        )

    matched = 0
    unmatched = 0
    rollout_total = 0
    dropped = 0

    # Write to a temp sibling, then atomically replace -- the original rollout
    # file stays intact if anything (including the strict guard) fails.
    tmp_out = f"{rollouts_file}.enrich.tmp"
    with open(input_file, "rb") as fin, open(rollouts_file) as fr, open(tmp_out, "w") as fout:
        for line in fr:
            if not line.strip():
                continue
            try:
                rollout = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                continue
            rollout_total += 1

            jid = compute_join_id(rollout)
            offset = offsets.get(jid)
            if offset is None:
                unmatched += 1
                merged = dict(rollout)
            else:
                matched += 1
                # Original SDG fields as base; rollout fields (response, reward,
                # verifier, ...) take precedence and are never overwritten.
                src = _read_row_at(fin, offset)
                merged = {**src, **rollout}

            response = rollout.get("response", {}) or {}
            merged[generation_key] = _extract_assistant_text(response)
            reasoning = _extract_reasoning_text(response)
            if reasoning:
                merged["reasoning_content"] = reasoning

            # Snapshot the Responses-API original form (full request + response object) under
            # non-ALWAYS_DROP aliases so the per-stage trim keeps them.  ``setdefault``
            # makes this a no-op once a snapshot exists: A-gen records get the answer
            # truth here, while later gym stages (genselect judge, evaluate judge)
            # already carry the A-gen snapshot from their input and must NOT have it
            # overwritten with the judge's response.
            merged.setdefault("answer_response", response)
            merged.setdefault(
                "answer_responses_create_params", merged.get("responses_create_params")
            )

            fout.write(json.dumps(merged, ensure_ascii=False) + "\n")

    if dropped:
        logger.warning("Dropped %d malformed line(s) in %s", dropped, rollouts_file)

    if unmatched:
        errors.append(
            f"{unmatched}/{rollout_total} rollout(s) did not match any input row by join id"
        )
    if rollout_total != input_total:
        errors.append(
            f"rollout count ({rollout_total}) != input count ({input_total}); "
            f"stale rollout cache, partial run, or input changed mid-run "
            f"(see SKILL.md Gotcha #13)"
        )

    if errors:
        message = (
            "enrich_rollouts: alignment check failed.\n  - "
            + "\n  - ".join(errors)
            + f"\n\ninput_file={input_file}\nrollouts_file={rollouts_file}"
        )
        if strict:
            # Leave the original rollouts file untouched on failure.
            try:
                os.remove(tmp_out)
            except OSError:
                pass
            raise RuntimeError(message)
        logger.warning(message)

    os.replace(tmp_out, rollouts_file)

    logger.info(
        "Enriched %d/%d rollouts (matched=%d, unmatched=%d) -> %s",
        matched,
        rollout_total,
        matched,
        unmatched,
        rollouts_file,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_file", help="Responses-API input JSONL (carries original SDG fields)"
    )
    parser.add_argument("rollouts_file", help="Rollout output JSONL (rewritten in place)")
    parser.add_argument(
        "--generation_key",
        default="generation",
        help="Field to store extracted assistant text (default: generation)",
    )
    parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Warn instead of raising on join/count anomalies",
    )
    args = parser.parse_args(argv)
    enrich(
        args.input_file,
        args.rollouts_file,
        generation_key=args.generation_key,
        strict=args.strict,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
