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
"""Post process document grounded sdg data by cleaning fields and creating subsets."""

import argparse
import json
import os

from nvflow.utils import setup_logger

logger = setup_logger(__name__)

# Fields to remove from the output
FIELDS_TO_REMOVE = {
    "solutions",
    "generations_list",
    "reasonings_list",
    "answer_0",
    "answer_1",
    "answer_2",
    "answer_3",
    "answer_4",
    "answer_reasoning_content_0",
    "answer_reasoning_content_1",
    "answer_reasoning_content_2",
    "answer_reasoning_content_3",
    "answer_reasoning_content_4",
    "num_solutions",
    "max_idx",
    "num_generated_tokens",
    "finish_reason",
    "generation_start_time",
    "generation_end_time",
    "generation_time",
    "genselect_response",
    "selected_index",
    "reasoning_content",
    "serialized_output",
}

# Fields to rename
FIELDS_TO_RENAME = {
    "reference_reasoning": "reasoning_content",
    "reference_answer": "answer",
}


def clean_record(record: dict) -> dict:
    """Remove unwanted fields and rename fields."""
    # Remove unwanted fields
    cleaned = {k: v for k, v in record.items() if k not in FIELDS_TO_REMOVE}

    # Rename fields
    for old_name, new_name in FIELDS_TO_RENAME.items():
        if old_name in cleaned:
            cleaned[new_name] = cleaned.pop(old_name)

    return cleaned


def is_medium_sft_eligible(record: dict) -> bool:
    """
    Check if a record is eligible for medium_sft_data.jsonl.

    Criteria:
    - difficulty_score is 1, 2, 3, or 4
    - For 10-K filings: exclude Risk_Factors questions
    - For 10-Q filings: only include Risk_Factors questions
    """
    difficulty_score = record.get("difficulty_score")
    if difficulty_score not in [1, 2, 3, 4]:
        return False

    file_path0 = record.get("file_path0", "")
    question_type = record.get("question_type", "")

    # For 10-K filings: exclude Risk_Factors
    if "10-K" in file_path0:
        if question_type == "Risk_Factors":
            return False

    # For 10-Q filings: only include Risk_Factors
    if "10-Q" in file_path0:
        if question_type != "Risk_Factors":
            return False

    return True


def is_hard_rl_eligible(record: dict) -> bool:
    """
    Check if a record is eligible for hard_rl_data.jsonl.

    Criteria:
    - difficulty_score is 0
    """
    return record.get("difficulty_score") == 0


def dgsdg_post_process(
    input_file: str,
    output_dir: str,
    seed: int = 42,
):
    """
    Post process document grounded sdg data by cleaning fields and creating subsets.

    Args:
        input_file: Path to input JSONL file (answers_with_difficulty.jsonl)
        output_dir: Directory to write output files
        seed: Random seed for reproducibility
    """
    logger.info(f"Post processing document grounded sdg data from: {input_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Random seed: {seed}")

    os.makedirs(output_dir, exist_ok=True)

    # First pass: clean all records and write full data
    full_output_file = os.path.join(output_dir, "full_data.jsonl")
    all_records = []

    logger.info("Pass 1: Cleaning records and writing full data...")

    with (
        open(input_file, encoding="utf-8") as f_in,
        open(full_output_file, "w", encoding="utf-8") as f_out,
    ):
        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: JSON decode error: {e}")
                continue

            cleaned = clean_record(record)
            all_records.append(cleaned)
            f_out.write(json.dumps(cleaned, ensure_ascii=False) + "\n")

            if line_num % 100000 == 0:
                logger.info(f"  Processed {line_num} lines...")

    total_records = len(all_records)
    logger.info(f"✓ Full data written: {total_records} records -> {full_output_file}")

    # Second pass: create final_result.jsonl and hard_rl_data.jsonl
    logger.info("")
    logger.info("Pass 2: Creating final_result.jsonl and hard_rl_data.jsonl...")

    medium_sft_file = os.path.join(output_dir, "final_result.jsonl")
    hard_rl_file = os.path.join(output_dir, "hard_rl_data.jsonl")

    medium_sft_count = 0
    hard_rl_count = 0

    with (
        open(medium_sft_file, "w", encoding="utf-8") as f_medium,
        open(hard_rl_file, "w", encoding="utf-8") as f_hard,
    ):
        for record in all_records:
            # Check for final_result.jsonl
            if is_medium_sft_eligible(record):
                f_medium.write(json.dumps(record, ensure_ascii=False) + "\n")
                medium_sft_count += 1

            # Check for hard_rl_data.jsonl
            if is_hard_rl_eligible(record):
                f_hard.write(json.dumps(record, ensure_ascii=False) + "\n")
                hard_rl_count += 1

    logger.info(f"  ✓ final_result.jsonl: {medium_sft_count} records")
    logger.info(f"  ✓ hard_rl_data.jsonl: {hard_rl_count} records")

    logger.info("")
    logger.info("=" * 60)
    logger.info("✓ Document grounded sdg data post processing complete!")
    logger.info(f"  Total records: {total_records}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info("  Files created:")
    logger.info(f"    - full_data.jsonl ({total_records} records)")
    logger.info(f"    - final_result.jsonl ({medium_sft_count} records)")
    logger.info(f"    - hard_rl_data.jsonl ({hard_rl_count} records)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Post process document grounded sdg data by cleaning fields and creating subsets."
    )
    parser.add_argument(
        "--input_file",
        required=True,
        help="Path to input JSONL file (answers_with_difficulty.jsonl)",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write output files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        exit(1)

    dgsdg_post_process(args.input_file, args.output_dir, args.seed)
