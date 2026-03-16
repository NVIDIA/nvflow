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
import argparse
import os
import re
import sys

import orjson

from nvflow.recipes.finance.utils.shared.metadata_utils import nest_metadata_fields
from nvflow.utils import setup_logger

# Initialize logger
logger = setup_logger(__name__)


def extract_judgment_index(generation_text):
    """
    Extract the number after the last occurrence of "Judgement: " or "Judgment: " in the text.

    Args:
        generation_text: The generation field text

    Returns:
        int: The extracted index, or None if not found
    """
    # Remove all asterisks to simplify matching
    cleaned_text = generation_text.replace("*", "")

    # Find all occurrences of "Judgement: " or "Judgment: " followed by a number
    # Try both spellings
    matches = re.findall(r"Judgment:\s*(\d+)", cleaned_text)
    if not matches:
        matches = re.findall(r"Judgement:\s*(\d+)", cleaned_text)

    if matches:
        # Return the last match as an integer
        return int(matches[-1])

    return None


def postprocess_genselect(input_file, output_file):
    """
    Process GenSelect output to extract the selected solution based on judgment.

    Args:
        input_file: Path to input JSONL file with generation and solutions_list
        output_file: Path to output JSONL file with extracted solution
    """
    logger.info(f"Processing: {input_file}")
    logger.info(f"Output to: {output_file}")

    records_processed = 0
    records_skipped = 0

    with (
        open(input_file, encoding="utf-8") as f_in,
        open(output_file, "w", encoding="utf-8") as f_out,
    ):
        for line_num, line in enumerate(f_in, 1):
            try:
                record = orjson.loads(line.strip())

                # Extract judgment index from generation field
                generation = record.get("generation", "")
                reasoning_content = record.get("reasoning_content", "")
                generations_list = record.get("generations_list", [])
                reasonings_list = record.get("reasonings_list", [])

                if not generation:
                    logger.warning(f"Line {line_num} missing 'generation' field, skipping")
                    records_skipped += 1
                    continue

                if not generations_list:
                    logger.warning(f"Line {line_num} missing 'generations_list' field, skipping")
                    records_skipped += 1
                    continue

                if not reasonings_list:
                    logger.warning(f"Line {line_num} missing 'reasonings_list' field, skipping")
                    records_skipped += 1
                    continue

                # Extract the judgment index
                judgment_idx = extract_judgment_index(generation)

                if judgment_idx is None:
                    logger.warning(f"Line {line_num} could not extract judgment index, skipping")
                    records_skipped += 1
                    continue

                # Validate index is within bounds for both lists
                if judgment_idx < 0 or judgment_idx >= len(generations_list):
                    logger.warning(
                        f"Line {line_num} judgment index {judgment_idx} out of bounds "
                        f"(generations_list has {len(generations_list)} items), skipping"
                    )
                    records_skipped += 1
                    continue

                if judgment_idx >= len(reasonings_list):
                    logger.warning(
                        f"Line {line_num} judgment index {judgment_idx} out of bounds "
                        f"(reasonings_list has {len(reasonings_list)} items), skipping"
                    )
                    records_skipped += 1
                    continue

                # Extract the selected solution and reasoning
                selected_solution = generations_list[judgment_idx]
                selected_reasoning = reasonings_list[judgment_idx]

                # Create output record
                output_record = record.copy()

                # Save the GenSelect response before overwriting
                genselect_response = {}
                if generation:
                    genselect_response["generation"] = generation
                if reasoning_content:
                    genselect_response["reasoning_content"] = reasoning_content

                output_record["genselect_response"] = genselect_response

                # Nest metadata fields before writing
                output_record = nest_metadata_fields(output_record, "genselect_answers_metadata")

                # Replace generation and reasoning_content with selected ones
                output_record["generation"] = selected_solution
                output_record["reasoning_content"] = selected_reasoning
                output_record["selected_index"] = judgment_idx

                # Write to output
                f_out.write(orjson.dumps(output_record).decode("utf-8") + "\n")
                records_processed += 1

                if records_processed % 100 == 0:
                    logger.info(f"  Processed {records_processed} records...")

            except orjson.JSONDecodeError as e:
                logger.error(f"Error parsing line {line_num}: {e}")
                records_skipped += 1
                continue
            except Exception as e:
                logger.error(f"Error processing line {line_num}: {e}")
                records_skipped += 1
                continue

    logger.info("")
    logger.info("✓ Complete!")
    logger.info(f"  Records processed: {records_processed}")
    logger.info(f"  Records skipped: {records_skipped}")
    logger.info(f"  Output written to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Post-process GenSelect output to extract selected solutions."
    )
    parser.add_argument("--input_dir", required=True, help="Directory containing output.jsonl file")
    parser.add_argument("--output_file", required=True, help="Path for output JSONL file")
    args = parser.parse_args()

    # Look for output.jsonl in input_dir
    input_file = os.path.join(args.input_dir, "output.jsonl")

    if not os.path.exists(input_file):
        logger.error(f"✗ Error: Input file not found: {input_file}")
        sys.exit(1)

    output_file = args.output_file

    logger.info("GenSelect Post-processor")
    logger.info("=" * 50)
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")
    logger.info("=" * 50)

    try:
        postprocess_genselect(input_file, output_file)
    except Exception as e:
        logger.error(f"\n✗ Error: {e}")
        sys.exit(1)
