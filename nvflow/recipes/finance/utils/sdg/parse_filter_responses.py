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
"""Parse filter responses from LLM generation output."""

import argparse
import re

import orjson

from nvflow.recipes.finance.utils.shared.metadata_utils import nest_metadata_fields
from nvflow.utils import setup_logger

# Initialize logger
logger = setup_logger(__name__)

WRITE_BUFFER_SIZE = 1000  # Write every 1000 records


def parse_filter_tag(generation_text):
    """Extract ANSWERABLE/UNANSWERABLE tag from generation field.

    Looks for the pattern "Answer: ANSWERABLE" or "Answer: UNANSWERABLE"
    and extracts the last occurrence to avoid false positives in reasoning text.

    Returns:
        tuple: (tag, explanation, error_msg)
            - tag: "ANSWERABLE" or "UNANSWERABLE" if found, None otherwise
            - explanation: The reasoning text before the final answer, None otherwise
            - error_msg: Error message if parsing failed, None otherwise
    """
    if not generation_text or not isinstance(generation_text, str):
        return None, None, "Empty or invalid generation text"

    # Look for "Answer: ANSWERABLE" or "Answer: UNANSWERABLE" (case insensitive)
    # Use finditer to get all matches, then take the last one
    pattern = r"Answer:\s*(ANSWERABLE|UNANSWERABLE)"
    matches = list(re.finditer(pattern, generation_text, re.IGNORECASE))

    if not matches:
        return (
            None,
            None,
            f"No 'Answer: ANSWERABLE/UNANSWERABLE' pattern found in: {generation_text[:200]}",
        )

    # Take the last match (most likely to be the final answer)
    last_match = matches[-1]
    tag = last_match.group(1).upper()  # Normalize to uppercase

    # Extract explanation (everything before the final Answer: line)
    answer_position = last_match.start()
    explanation = generation_text[:answer_position].strip()

    # If explanation is empty or very short, set to None
    if not explanation or len(explanation) < 10:
        explanation = None

    return tag, explanation, None


def parse_filter_responses(input_file, output_file):
    """Parse filter responses from LLM generation output.

    Args:
        input_file: Path to input JSONL file with 'generation' field
        output_file: Path to output JSONL file with 'filter_tag' and 'filter_explanation' fields
    """
    log_file = output_file.replace(".jsonl", "_parse_log.txt")
    num_total_entries = 0
    num_failed_to_parse = 0
    num_successfully_parsed = 0
    num_answerable = 0
    num_unanswerable = 0

    buffer = []

    with (
        open(input_file, "rb") as reader,
        open(output_file, "wb") as writer,
        open(log_file, "w") as log_writer,
    ):
        for line in reader:
            line = line.strip()
            if not line:
                continue

            num_total_entries += 1

            try:
                row = orjson.loads(line)
            except orjson.JSONDecodeError as e:
                msg = f"Failed to parse JSON at entry {num_total_entries}: {e}"
                log_writer.write(msg + "\n")
                num_failed_to_parse += 1
                continue

            try:
                filter_generation = row["filter_generation"]
            except KeyError:
                problem = row.get("problem", "unknown")
                msg = f"No filter_generation field found for entry {num_total_entries}: {problem}"
                log_writer.write(msg + "\n")
                num_failed_to_parse += 1
                continue

            filter_tag, explanation, error_msg = parse_filter_tag(filter_generation)

            if filter_tag:
                row["filter_tag"] = filter_tag
                if explanation:
                    row["filter_explanation"] = explanation
                num_successfully_parsed += 1

                if filter_tag == "ANSWERABLE":
                    num_answerable += 1
                else:
                    num_unanswerable += 1

                # Nest metadata fields before writing
                row = nest_metadata_fields(row, "filter_answer_metadata")

                # Add to buffer
                buffer.append(orjson.dumps(row))

                # Write buffer when it reaches threshold
                if len(buffer) >= WRITE_BUFFER_SIZE:
                    writer.write(b"\n".join(buffer) + b"\n")
                    buffer.clear()
            else:
                num_failed_to_parse += 1
                msg = f"Failed to parse generation for entry {num_total_entries}: {error_msg}"
                log_writer.write(msg + "\n")

        # Write remaining buffer
        if buffer:
            writer.write(b"\n".join(buffer) + b"\n")

        # Write summary to log
        log_writer.write(f"\n{'=' * 60}\n")
        log_writer.write("PARSE SUMMARY\n")
        log_writer.write(f"{'=' * 60}\n")
        log_writer.write(f"Total entries: {num_total_entries}\n")
        log_writer.write(f"Successfully parsed: {num_successfully_parsed}\n")
        log_writer.write(f"Failed to parse: {num_failed_to_parse}\n")
        log_writer.write(
            f"Success rate: {num_successfully_parsed / num_total_entries * 100:.2f}%\n"
        )
        log_writer.write(f"\nAnswerable: {num_answerable}\n")
        log_writer.write(f"Unanswerable: {num_unanswerable}\n")
        if num_successfully_parsed > 0:
            log_writer.write(
                f"Unanswerable rate: {num_unanswerable / num_successfully_parsed * 100:.2f}%\n"
            )

        # Log summary
        logger.info(f"\n{'=' * 60}")
        logger.info("PARSE SUMMARY")
        logger.info(f"{'=' * 60}")
        logger.info(f"Total entries: {num_total_entries}")
        logger.info(f"Successfully parsed: {num_successfully_parsed}")
        logger.info(f"Failed to parse: {num_failed_to_parse}")
        logger.info(f"Success rate: {num_successfully_parsed / num_total_entries * 100:.2f}%")
        logger.info(f"\nAnswerable: {num_answerable}")
        logger.info(f"Unanswerable: {num_unanswerable}")
        if num_successfully_parsed > 0:
            logger.info(
                f"Unanswerable rate: {num_unanswerable / num_successfully_parsed * 100:.2f}%"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse filter responses from LLM generation field and extract ANSWERABLE/UNANSWERABLE tags"
    )
    parser.add_argument(
        "--input_file", required=True, help="Input JSONL file with generation field"
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Output JSONL file with filter_tag and filter_explanation fields",
    )
    args = parser.parse_args()

    parse_filter_responses(args.input_file, args.output_file)
