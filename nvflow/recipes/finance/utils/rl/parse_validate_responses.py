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
"""Phase 2 response parser for the ``validate_questions`` GRPO stage.

Reads the LLM generation output (the ``generation`` field in
nemo-skills' ``output.jsonl``), extracts the final
``Answer: VALID`` / ``Answer: INVALID`` tag, and attaches it as
``validate_tag``.  Mirrors the SDG ``parse_filter_responses.py`` pattern.

The ``validate_tag`` field is an **internal tag** used by the downstream
``apply_validate_filter.py`` to split VALID from INVALID rows.  It and
other classifier/nemo-skills leftovers (``generation``, timing fields,
``serialized_output``, etc.) are stripped by ``apply_validate_filter.py``
before writing ``final_result.jsonl`` / ``llm_dropped.jsonl``.
``reasoning_content`` is restored from the raw SDG file there (not
stripped -- it is a legitimate SDG field).  Downstream consumers see an
SDG-schema-shaped record with the same fields as the input, just
filtered.

**Recall bias.** On parse failure (missing tag, empty generation, garbled
output) we default to ``validate_tag = "VALID"`` so the downstream filter
keeps the record.  Parse failures are counted separately in the companion
``*_parse_log.txt`` so unusual rates are visible.
"""

import argparse
import re

import orjson

from nvflow.utils import setup_logger

logger = setup_logger(__name__)

WRITE_BUFFER_SIZE = 1000

# Accepts "Answer: VALID" / "Answer: INVALID" (case-insensitive).  We take
# the LAST match to avoid false positives when the rubric/reasoning
# repeats the words earlier in the response.
_ANSWER_RE = re.compile(r"Answer:\s*(VALID|INVALID)", re.IGNORECASE)


def parse_validate_tag(generation_text: str) -> tuple[str | None, str | None, str | None]:
    """Extract VALID/INVALID tag from generation text.

    Returns:
        (tag, explanation, error_msg)

        - ``tag``: ``"VALID"`` or ``"INVALID"`` on success, ``None`` on parse failure.
        - ``explanation``: the text preceding the final ``Answer:`` line, or ``None``
          if too short.
        - ``error_msg``: description of why parsing failed, or ``None`` on success.

    On parse failure the CALLER defaults the tag to ``"VALID"`` (keep).
    """
    if not generation_text or not isinstance(generation_text, str):
        return None, None, "Empty or invalid generation text"

    matches = list(_ANSWER_RE.finditer(generation_text))
    if not matches:
        return (
            None,
            None,
            f"No 'Answer: VALID/INVALID' pattern found in: {generation_text[:200]}",
        )

    last = matches[-1]
    tag = last.group(1).upper()

    explanation = generation_text[: last.start()].strip()
    if not explanation or len(explanation) < 10:
        explanation = None

    return tag, explanation, None


def parse_validate_responses(input_file: str, output_file: str) -> None:
    """Parse nemo-skills generation output and attach validate_tag.

    Args:
        input_file: JSONL produced by nemo-skills ``generate()`` with a
            ``generation`` field on each row.
        output_file: JSONL with ``validate_tag`` (and optional
            ``validate_explanation``) attached to every row.
    """
    log_file = output_file.replace(".jsonl", "_parse_log.txt")

    num_total = 0
    num_parsed = 0
    num_parse_failed = 0
    num_valid = 0
    num_invalid = 0
    num_parse_failed_defaulted_valid = 0

    buffer: list[bytes] = []

    with (
        open(input_file, "rb") as reader,
        open(output_file, "wb") as writer,
        open(log_file, "w") as log_writer,
    ):
        for line in reader:
            line = line.strip()
            if not line:
                continue

            num_total += 1

            try:
                row = orjson.loads(line)
            except orjson.JSONDecodeError as exc:
                msg = f"Failed to parse JSON at entry {num_total}: {exc}"
                log_writer.write(msg + "\n")
                num_parse_failed += 1
                # Can't attach anything useful — skip this line entirely.
                continue

            generation = row.get("generation", "")

            tag, explanation, error = parse_validate_tag(generation)

            if tag is None:
                # Recall bias: default to VALID on parse failure so we
                # keep the record.  Log for visibility.
                num_parse_failed += 1
                num_parse_failed_defaulted_valid += 1
                row["validate_tag"] = "VALID"
                row["validate_parse_failed"] = True
                if error:
                    row["validate_parse_error"] = error[:300]
                log_writer.write(
                    f"entry {num_total}: parse failure -> defaulting to VALID ({error})\n"
                )
            else:
                num_parsed += 1
                row["validate_tag"] = tag
                if explanation:
                    row["validate_explanation"] = explanation
                if tag == "VALID":
                    num_valid += 1
                else:
                    num_invalid += 1

            buffer.append(orjson.dumps(row))
            if len(buffer) >= WRITE_BUFFER_SIZE:
                writer.write(b"\n".join(buffer) + b"\n")
                buffer.clear()

        if buffer:
            writer.write(b"\n".join(buffer) + b"\n")

        log_writer.write(f"\n{'=' * 60}\n")
        log_writer.write("VALIDATE PARSE SUMMARY\n")
        log_writer.write(f"{'=' * 60}\n")
        log_writer.write(f"Total entries:                    {num_total}\n")
        log_writer.write(f"Successfully parsed:              {num_parsed}\n")
        log_writer.write(f"Parse failures (defaulted VALID): {num_parse_failed_defaulted_valid}\n")
        log_writer.write(f"VALID:                            {num_valid}\n")
        log_writer.write(f"INVALID:                          {num_invalid}\n")
        if num_total:
            log_writer.write(f"Parse success rate: {num_parsed / num_total * 100:.2f}%\n")
        if num_parsed:
            log_writer.write(
                f"INVALID rate (of successfully parsed): {num_invalid / num_parsed * 100:.2f}%\n"
            )

    logger.info("validate parse summary")
    logger.info(f"  total:               {num_total}")
    logger.info(f"  parsed:              {num_parsed}")
    logger.info(f"  parse failed:        {num_parse_failed_defaulted_valid} (defaulted VALID)")
    logger.info(f"  VALID:               {num_valid}")
    logger.info(f"  INVALID:             {num_invalid}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse VALID/INVALID tags from LLM generation output (validate_questions Phase 2)"
    )
    parser.add_argument(
        "--input_file",
        required=True,
        help="Input JSONL with nemo-skills 'generation' field.",
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Output JSONL with validate_tag (and optional validate_explanation) attached.",
    )
    args = parser.parse_args()

    parse_validate_responses(args.input_file, args.output_file)
