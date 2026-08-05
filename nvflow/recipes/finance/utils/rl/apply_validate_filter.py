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
"""Phase 2 filter applier for the ``validate_questions`` GRPO stage.

Reads the parsed JSONL produced by ``parse_validate_responses.py`` and
splits it into two streams by the ``validate_tag`` field:

- Records with ``validate_tag == keep_tag`` (default ``"VALID"``) emit the
  *original* SDG record (looked up by ``problem`` in the raw SDG JSONL)
  to the kept output (normally ``final_result.jsonl``), which
  ``data_transformation`` reads next.  validate_questions is a pure
  row-filter: kept output bytes match the SDG input bytes for those rows.
- Records with other tags go to the dropped output for audit, with the
  nemo-skills "pollution" fields stripped.

A stats JSON is also written with total / kept / dropped counts, the
drop rate, and a non-fatal ``high_drop_warning`` flag.  Mirrors the SDG
``apply_answer_filter.py`` pattern.

``raw_sdg_path`` is REQUIRED.  An older code path used to fall back to
emitting the LLM-mutated record (with pollution stripped) when no SDG
file was provided, but that path produced rows whose ``reasoning_content``
came from the LLM provider rather than the SDG -- ``data_transformation``
rejects those.  The fallback was unreachable in practice (the only
caller, ``validate_questions.py``, has always passed ``--raw_sdg_source``)
so it was removed in S1.  The CLI now requires ``--raw_sdg_source`` and
the function raises :class:`MissingSdgRecordError` when a VALID row's
``problem`` is absent from the SDG file (a clear caller-side bug).
"""

import argparse
from pathlib import Path

import orjson

from nvflow.utils import setup_logger
from nvflow.utils.jsonl import iter_jsonl, write_jsonl, write_stats_json

logger = setup_logger(__name__)

# Non-SDG fields injected by nemo-skills generate() + the LLM provider +
# our parse step.  Stripped from the dropped audit stream so it stays
# readable; the kept stream emits original SDG bytes verbatim and so
# never sees pollution at all.
_STRIP_FIELDS = frozenset(
    {
        "generation",  # LLM classifier "Reason: ... Answer: VALID" text
        "finish_reason",
        "num_generated_tokens",
        "num_input_tokens",
        "generation_start_time",
        "generation_end_time",
        "generation_time",
        "serialized_output",
        "provider_specific_fields",
        "validate_tag",  # internal -- consumed by the split below
        "validate_explanation",
        "validate_parse_failed",
        "validate_parse_error",
    }
)


def _strip_pollution(row: dict) -> dict:
    for k in _STRIP_FIELDS:
        row.pop(k, None)
    return row


class MissingSdgRecordError(KeyError):
    """Raised when a kept row's ``problem`` is absent from the raw SDG file.

    Indicates an inconsistency between the parsed Phase 2 JSONL and the
    SDG file the caller pointed at -- e.g., the SDG file was regenerated
    after Phase 1 ran, or the wrong ``raw_sdg_source`` was passed.

    The historical code path silently emitted the LLM-mutated row when
    this happened, producing data that ``data_transformation`` then
    rejected with an opaque downstream error.  Failing fast here pins
    the diagnostic to the apply step where the inconsistency originated.
    """


def _load_raw_sdg_records(raw_sdg_path: str) -> dict[str, bytes]:
    """Build ``{problem -> original_record_bytes}`` from the raw SDG file.

    validate_questions is a pure row-filter: output records must be
    identical to input records, just fewer of them.  We index original
    records by ``problem`` so that apply_validate_filter can emit the
    original record (unchanged) for each VALID verdict.

    The stored bytes are the stripped line contents WITHOUT a trailing
    newline -- ``write_jsonl`` adds the line terminator on flush.
    """
    problem_to_record: dict[str, bytes] = {}
    # ``yield_error`` so we silently skip malformed lines (matches the
    # historical behaviour) without giving up the raw bytes for valid
    # rows. Re-open the file in raw byte mode to recover the source bytes
    # for each yielded row, since iter_jsonl only yields the parsed dict.
    with open(raw_sdg_path, "rb") as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                obj = orjson.loads(stripped)
            except orjson.JSONDecodeError:
                continue
            problem = obj.get("problem")
            if not problem:
                continue
            if problem not in problem_to_record:
                problem_to_record[problem] = stripped
    return problem_to_record


def apply_validate_filter(
    input_file: str,
    output_kept: str,
    output_dropped: str,
    stats_file: str,
    raw_sdg_path: str,
    keep_tag: str = "VALID",
    high_drop_threshold: float = 0.20,
) -> dict:
    """Split records by ``validate_tag`` into kept / dropped streams.

    Args:
        input_file: JSONL with ``validate_tag`` field on each row.
        output_kept: JSONL receiving ``keep_tag`` records (default VALID).
            Each kept row is the ORIGINAL SDG record bytes -- pure row-filter.
        output_dropped: JSONL receiving all other records, with
            ``_llm_drop_reason`` annotated and pollution fields stripped.
        stats_file: JSON with counts and drop-rate warning flag.
        raw_sdg_path: Path to the raw SDG JSONL file. Required.  Records
            are indexed by ``problem`` so the kept stream emits SDG-original
            bytes byte-for-byte (preserves the schema and ``reasoning_content``
            that ``data_transformation`` requires).
        keep_tag: tag value to keep (default ``"VALID"``).
        high_drop_threshold: fraction above which ``high_drop_warning``
            is set in the stats file (default 0.20).

    Raises:
        MissingSdgRecordError: A kept (VALID) row's ``problem`` is absent
            from the SDG file.  Always a caller bug (mismatched
            ``raw_sdg_path``) -- fail fast rather than silently emitting
            LLM-mutated bytes.

    Returns:
        The stats dict that was written to ``stats_file``.
    """
    num_total = 0
    num_kept = 0
    num_dropped = 0
    num_missing_tag = 0
    num_parse_failed = 0
    num_original_used = 0
    num_original_missing = 0

    logger.info("Loading raw SDG records from %s", raw_sdg_path)
    problem_to_record = _load_raw_sdg_records(raw_sdg_path)
    logger.info("  %d unique problems loaded from raw SDG", len(problem_to_record))

    with write_jsonl(output_kept) as kept_writer, write_jsonl(output_dropped) as dropped_writer:
        for row, parse_exc, _raw_line in iter_jsonl(input_file, on_error="yield_error"):
            num_total += 1

            if parse_exc is not None:
                # Parse error on a row that's already supposed to be
                # post-parse.  Count as a drop with a clear reason.
                num_dropped += 1
                dropped_writer.write(
                    {
                        "_llm_drop_reason": "malformed_parsed_json",
                        "_llm_drop_error": str(parse_exc),
                    }
                )
                continue

            assert row is not None  # narrow for type-checkers in yield_error mode

            # Per-row SDG lookup.  We do this for EVERY row (not just kept
            # ones) so ``num_original_records_used`` reports a Phase 1 / Phase 2
            # problem-key consistency check: a non-zero gap between this and
            # the post-parse-error row count is a strong signal that the SDG
            # file the caller pointed at doesn't match the Phase 1 inputs.
            problem = row.get("problem", "")
            original_record_bytes = problem_to_record.get(problem)
            if original_record_bytes is not None:
                num_original_used += 1
            else:
                num_original_missing += 1

            tag = row.get("validate_tag")
            if row.get("validate_parse_failed"):
                num_parse_failed += 1

            if tag is None:
                num_missing_tag += 1
                num_dropped += 1
                # Emit reason BEFORE strip so _llm_drop_reason survives
                # (it's not in _STRIP_FIELDS -- it's audit metadata on the
                # dropped-only stream, fine to keep).
                row["_llm_drop_reason"] = "missing_validate_tag"
                dropped_writer.write(_strip_pollution(row))
            elif tag == keep_tag:
                if original_record_bytes is None:
                    raise MissingSdgRecordError(
                        f"VALID row's problem {problem!r} not found in {raw_sdg_path}; "
                        "the raw SDG file is inconsistent with the parsed Phase 2 input. "
                        "Check that --raw_sdg_source points at the SDG file Phase 1 was run on."
                    )
                num_kept += 1
                kept_writer.write(original_record_bytes)
            else:
                num_dropped += 1
                row["_llm_drop_reason"] = f"tag={tag}"
                dropped_writer.write(_strip_pollution(row))

    drop_rate = num_dropped / num_total if num_total else 0.0
    high_drop_warning = drop_rate > high_drop_threshold

    stats = {
        "num_total": num_total,
        "num_kept": num_kept,
        "num_dropped": num_dropped,
        "num_missing_tag": num_missing_tag,
        "num_parse_failed_kept_as_valid": num_parse_failed,
        # Counts every row whose problem WAS found in the SDG file (kept
        # or dropped).  Together with num_original_records_missing the
        # totals add up to num_total - num_malformed_post_parse.
        "num_original_records_used": num_original_used,
        # Counts every row whose problem was NOT found in the SDG file.
        # In the validate_questions stage path this is always 0 because
        # for VALID rows a miss raises MissingSdgRecordError; for
        # dropped rows the SDG file is consistent in practice.  Kept as
        # a real counter (rather than hardcoded 0) so this still works
        # as a Phase 1 / Phase 2 input-consistency metric for callers
        # other than the VALID-only short-circuit (e.g., audit tools
        # invoking apply_validate_filter with a non-VALID keep_tag).
        "num_original_records_missing": num_original_missing,
        "drop_rate": round(drop_rate, 6),
        "keep_tag": keep_tag,
        "high_drop_threshold": high_drop_threshold,
        "high_drop_warning": high_drop_warning,
        "input_file": input_file,
        "output_kept": output_kept,
        "output_dropped": output_dropped,
        "raw_sdg_path": raw_sdg_path,
    }

    # Atomic write: a crash mid-write would otherwise leave a truncated
    # stats_file on disk and surprise downstream tools that consume it
    # (or, in resume scenarios, treat its existence as "prior run done").
    write_stats_json(stats_file, stats)

    logger.info("validate filter summary")
    logger.info(f"  total:        {num_total}")
    logger.info(f"  kept ({keep_tag}): {num_kept}")
    logger.info(f"  dropped:      {num_dropped} ({drop_rate * 100:.2f}%)")
    if num_missing_tag:
        logger.info(f"  missing tag (dropped): {num_missing_tag}")
    if num_parse_failed:
        logger.info(f"  parse-failures kept as VALID (recall bias): {num_parse_failed}")
    logger.info(f"  original records used: {num_original_used}  missing: {num_original_missing}")
    if high_drop_warning:
        logger.warning(
            "drop rate %.2f%% exceeds threshold %.2f%% -- inspect %s before proceeding",
            drop_rate * 100,
            high_drop_threshold * 100,
            output_dropped,
        )

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply VALID/INVALID filter from validate_questions Phase 2 parsed output"
    )
    parser.add_argument(
        "--input_file",
        required=True,
        help="Parsed JSONL produced by parse_validate_responses.py (must have validate_tag).",
    )
    parser.add_argument(
        "--output_kept",
        required=True,
        help="Output JSONL with only VALID records (normally final_result.jsonl).",
    )
    parser.add_argument(
        "--output_dropped",
        required=True,
        help="Output JSONL with dropped records annotated with _llm_drop_reason.",
    )
    parser.add_argument(
        "--stats_file",
        required=True,
        help="Output JSON file with filter counts and high_drop_warning flag.",
    )
    parser.add_argument(
        "--keep_tag",
        default="VALID",
        help="Tag value to keep (default VALID).",
    )
    parser.add_argument(
        "--high_drop_threshold",
        type=float,
        default=0.20,
        help="Drop-rate threshold for the high_drop_warning flag (default 0.20).",
    )
    parser.add_argument(
        "--raw_sdg_source",
        required=True,
        help=(
            "Directory of the raw SDG JSONL.  Required: kept records emit "
            "the original SDG bytes verbatim (preserves reasoning_content, "
            "key ordering, and float formatting that data_transformation "
            "expects)."
        ),
    )
    parser.add_argument(
        "--raw_sdg_filename",
        default="final_result.jsonl",
        help="Filename inside --raw_sdg_source (default: final_result.jsonl).",
    )
    args = parser.parse_args()

    raw_sdg_path = str(Path(args.raw_sdg_source) / args.raw_sdg_filename)

    apply_validate_filter(
        args.input_file,
        args.output_kept,
        args.output_dropped,
        args.stats_file,
        raw_sdg_path=raw_sdg_path,
        keep_tag=args.keep_tag,
        high_drop_threshold=args.high_drop_threshold,
    )
