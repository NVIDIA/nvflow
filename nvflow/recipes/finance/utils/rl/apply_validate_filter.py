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

- Records with ``validate_tag == keep_tag`` (default ``"VALID"``) go to
  the kept output (normally ``final_result.jsonl``), which
  ``data_transformation`` reads next.
- Records with other tags go to the dropped output for audit.

A stats JSON is also written with total / kept / dropped counts, the
drop rate, and a non-fatal ``high_drop_warning`` flag.  Mirrors the SDG
``apply_answer_filter.py`` pattern.
"""

import argparse
from pathlib import Path

import orjson

from nvflow.utils import setup_logger

logger = setup_logger(__name__)

WRITE_BUFFER_SIZE = 1000

# Non-SDG fields injected by nemo-skills generate() + the LLM provider +
# our parse step.  Stripped before writing so the output preserves the
# raw SDG schema (validate_questions is a pure row-filter).
# ``reasoning_content`` is NOT here -- it's a legitimate SDG field that
# gets overwritten by the LLM; the restore in apply_validate_filter puts
# the SDG-original value back, so we must not strip it.
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


def _load_raw_sdg_records(raw_sdg_path: str) -> dict[str, bytes]:
    """Build ``{problem -> original_record_bytes}`` from the raw SDG file.

    validate_questions is a pure row-filter: output records must be
    identical to input records, just fewer of them.  We index original
    records by ``problem`` so that apply_validate_filter can emit the
    original record (unchanged) for each VALID verdict.
    """
    problem_to_record: dict[str, bytes] = {}
    with open(raw_sdg_path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = orjson.loads(line)
            except orjson.JSONDecodeError:
                continue
            problem = obj.get("problem")
            if not problem:
                continue
            if problem not in problem_to_record:
                problem_to_record[problem] = line
    return problem_to_record


def apply_validate_filter(
    input_file: str,
    output_kept: str,
    output_dropped: str,
    stats_file: str,
    keep_tag: str = "VALID",
    high_drop_threshold: float = 0.20,
    raw_sdg_path: str | None = None,
) -> dict:
    """Split records by ``validate_tag`` into kept / dropped streams.

    Args:
        input_file: JSONL with ``validate_tag`` field on each row.
        output_kept: JSONL receiving ``keep_tag`` records (default VALID).
        output_dropped: JSONL receiving all other records, with
            ``_llm_drop_reason`` annotated.
        stats_file: JSON with counts and drop-rate warning flag.
        keep_tag: tag value to keep (default ``"VALID"``).
        high_drop_threshold: fraction above which ``high_drop_warning``
            is set in the stats file (default 0.20).
        raw_sdg_path: Optional path to the raw SDG JSONL file used to
            restore ``reasoning_content`` per record (see
            :func:`_load_raw_sdg_reasoning` for rationale).  When provided,
            the field is overwritten with the SDG-original value before
            the pollution strip, guaranteeing SDG-schema-faithful output.
            Recommended for GRPO (``dataset_transformer`` requires a
            non-empty ``reasoning_content`` on single-seed SDG records).

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

    problem_to_record: dict[str, bytes] | None = None
    if raw_sdg_path:
        logger.info("Loading raw SDG records from %s", raw_sdg_path)
        problem_to_record = _load_raw_sdg_records(raw_sdg_path)
        logger.info("  %d unique problems loaded from raw SDG", len(problem_to_record))

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
                # Parse error on a row that's already supposed to be
                # post-parse.  Count as a drop with a clear reason.
                num_dropped += 1
                dropped_buffer.append(
                    orjson.dumps(
                        {
                            "_llm_drop_reason": "malformed_parsed_json",
                            "_llm_drop_error": str(exc),
                        }
                    )
                )
                continue

            # Resolve the original SDG record for this problem.  When
            # raw_sdg_path is provided, the output is the ORIGINAL record
            # (unchanged) -- validate_questions is a pure row-filter.
            # When not provided, fall back to stripping nemo-skills
            # pollution from the generate() output (legacy behavior).
            original_record_bytes: bytes | None = None
            if problem_to_record is not None:
                original_record_bytes = problem_to_record.get(row.get("problem", ""))
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
                dropped_buffer.append(orjson.dumps(_strip_pollution(row)))
            elif tag == keep_tag:
                num_kept += 1
                if original_record_bytes is not None:
                    kept_buffer.append(original_record_bytes)
                else:
                    kept_buffer.append(orjson.dumps(_strip_pollution(row)))
            else:
                num_dropped += 1
                row["_llm_drop_reason"] = f"tag={tag}"
                dropped_buffer.append(orjson.dumps(_strip_pollution(row)))

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
        "num_missing_tag": num_missing_tag,
        "num_parse_failed_kept_as_valid": num_parse_failed,
        "num_original_records_used": num_original_used,
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

    with open(stats_file, "wb") as stats_writer:
        stats_writer.write(orjson.dumps(stats, option=orjson.OPT_INDENT_2))

    logger.info("validate filter summary")
    logger.info(f"  total:        {num_total}")
    logger.info(f"  kept ({keep_tag}): {num_kept}")
    logger.info(f"  dropped:      {num_dropped} ({drop_rate * 100:.2f}%)")
    if num_missing_tag:
        logger.info(f"  missing tag (dropped): {num_missing_tag}")
    if num_parse_failed:
        logger.info(f"  parse-failures kept as VALID (recall bias): {num_parse_failed}")
    if problem_to_record is not None:
        logger.info(
            f"  original records used: {num_original_used}  missing: {num_original_missing}"
        )
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
        default=None,
        help=(
            "Optional directory of the raw SDG JSONL.  When set, restores "
            "reasoning_content per record (nemo-skills' generate() overwrites "
            "it with the LLM provider's reasoning, which dataset_transformer "
            "rejects)."
        ),
    )
    parser.add_argument(
        "--raw_sdg_filename",
        default="final_result.jsonl",
        help="Filename inside --raw_sdg_source (default: final_result.jsonl).",
    )
    args = parser.parse_args()

    raw_sdg_path = (
        str(Path(args.raw_sdg_source) / args.raw_sdg_filename) if args.raw_sdg_source else None
    )

    apply_validate_filter(
        args.input_file,
        args.output_kept,
        args.output_dropped,
        args.stats_file,
        keep_tag=args.keep_tag,
        high_drop_threshold=args.high_drop_threshold,
        raw_sdg_path=raw_sdg_path,
    )
