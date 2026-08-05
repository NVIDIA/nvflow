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
"""Parse evaluate_answers LLM responses to extract answerable and correct tags."""

import argparse
import json
import os
import re
import sys

from nvflow.utils import setup_logger

logger = setup_logger(__name__)


def parse_evaluation(generation_text: str) -> tuple[str | None, str | None, str | None]:
    """
    Extract answerable and correct tags from evaluation response.

    This is the single source of truth for evaluation parsing across
    the DG-SDG pipeline (used by both evaluate and aggregate modules).

    Args:
        generation_text: The generation field text

    Returns:
        tuple: (answerable, correct, error_msg)
            - answerable: "YES" or "NO" or None if not found
            - correct: "YES" or "NO" or None if not found
            - error_msg: Error message if parsing failed, None otherwise
    """
    if not generation_text:
        return None, None, "Empty generation text"

    json_pattern = r'\{[^{}]*"answerable"\s*:\s*"(YES|NO)"[^{}]*"correct"\s*:\s*"(YES|NO)"[^{}]*\}'
    match = re.search(json_pattern, generation_text, re.IGNORECASE)

    if match:
        answerable = match.group(1).upper()
        correct = match.group(2).upper()
        return answerable, correct, None

    json_pattern_rev = (
        r'\{[^{}]*"correct"\s*:\s*"(YES|NO)"[^{}]*"answerable"\s*:\s*"(YES|NO)"[^{}]*\}'
    )
    match = re.search(json_pattern_rev, generation_text, re.IGNORECASE)

    if match:
        correct = match.group(1).upper()
        answerable = match.group(2).upper()
        return answerable, correct, None

    try:
        json_start = generation_text.rfind("{")
        json_end = generation_text.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            json_str = generation_text[json_start:json_end]
            data = json.loads(json_str)
            answerable = str(data.get("answerable", "")).upper()
            correct = str(data.get("correct", "")).upper()
            if answerable in ("YES", "NO") and correct in ("YES", "NO"):
                return answerable, correct, None
    except (json.JSONDecodeError, KeyError):
        pass

    return None, None, f"Could not parse evaluation from: {generation_text[:200]}..."


def parse_evaluate_responses(input_file: str, output_file: str):
    """
    Process evaluation output to extract answerable and correct tags.

    Args:
        input_file: Path to input JSONL file with evaluate_generation field
        output_file: Path to output JSONL file with parsed tags
    """
    logger.info(f"Processing: {input_file}")
    logger.info(f"Output to: {output_file}")

    log_file = output_file.replace(".jsonl", "_parse_errors.log")

    num_total = 0
    num_parsed = 0
    num_failed = 0

    with (
        open(input_file, encoding="utf-8") as f_in,
        open(output_file, "w", encoding="utf-8") as f_out,
        open(log_file, "w", encoding="utf-8") as log_writer,
    ):
        for line_num, line in enumerate(f_in, 1):
            if not line.strip():
                continue

            num_total += 1

            try:
                row = json.loads(line.strip())
            except json.JSONDecodeError as e:
                log_writer.write(f"Line {line_num}: JSON decode error: {e}\n")
                num_failed += 1
                continue

            evaluate_generation = row.get("evaluate_generation", "")
            if not evaluate_generation:
                problem = row.get("problem", "unknown")
                log_writer.write(f"Line {line_num}: No evaluate_generation field for: {problem}\n")
                num_failed += 1
                continue

            answerable, correct, error_msg = parse_evaluation(evaluate_generation)

            if answerable is None or correct is None:
                problem = row.get("problem", "unknown")
                log_writer.write(f"Line {line_num}: {error_msg} for: {problem}\n")
                num_failed += 1
                continue

            row["answerable"] = answerable
            row["correct"] = correct

            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            num_parsed += 1

            if num_parsed % 100 == 0:
                logger.info(f"  Parsed {num_parsed} entries...")

    logger.info("")
    logger.info("Complete!")
    logger.info(f"  Total entries: {num_total}")
    logger.info(f"  Successfully parsed: {num_parsed}")
    logger.info(f"  Failed to parse: {num_failed}")
    logger.info(f"  Output: {output_file}")
    if num_failed > 0:
        logger.info(f"  Parse errors logged to: {log_file}")


def parse_multi_seed_responses(input_dir: str, num_seeds: int = 5):
    """
    Parse evaluation responses for multiple seed files.

    For each output-rsN.jsonl file, creates a corresponding parsed-rsN.jsonl file.

    Args:
        input_dir: Directory containing output-rs*.jsonl files
        num_seeds: Number of random seeds used (default 5)
    """
    logger.info(f"Parsing multi-seed responses from: {input_dir}")
    logger.info(f"Number of seeds: {num_seeds}")

    for seed_idx in range(num_seeds):
        input_file = os.path.join(input_dir, f"output-rs{seed_idx}.jsonl")
        output_file = os.path.join(input_dir, f"parsed-rs{seed_idx}.jsonl")

        if not os.path.exists(input_file):
            logger.warning(f"Seed file not found: {input_file}")
            continue

        logger.info(f"Processing seed {seed_idx}: {input_file}")
        parse_evaluate_responses(input_file, output_file)

    logger.info("Completed parsing all seed files")


def filter_incorrect_answers(input_file: str, output_file: str):
    """
    Filter out incorrect answers, keeping all correct ones (both answerable and unanswerable).

    Args:
        input_file: Path to input JSONL with 'correct' and 'answerable' fields
        output_file: Path to output JSONL with only correct answers
    """
    logger.info(f"Filtering incorrect answers from: {input_file}")
    logger.info(f"Output to: {output_file}")

    num_total = 0
    num_kept = 0
    num_filtered = 0
    stats = {"answerable_correct": 0, "unanswerable_correct": 0}

    with (
        open(input_file, encoding="utf-8") as f_in,
        open(output_file, "w", encoding="utf-8") as f_out,
    ):
        for line in f_in:
            if not line.strip():
                continue

            num_total += 1

            try:
                row = json.loads(line.strip())
            except json.JSONDecodeError:
                num_filtered += 1
                continue

            correct = row.get("correct", "").upper()
            answerable = row.get("answerable", "").upper()

            if correct == "YES":
                f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                num_kept += 1

                if answerable == "YES":
                    stats["answerable_correct"] += 1
                else:
                    stats["unanswerable_correct"] += 1
            else:
                num_filtered += 1

    logger.info("")
    logger.info("Complete!")
    logger.info(f"  Total entries: {num_total}")
    logger.info(f"  Kept (correct): {num_kept}")
    logger.info(f"    - Answerable & Correct: {stats['answerable_correct']}")
    logger.info(f"    - Unanswerable & Correct: {stats['unanswerable_correct']}")
    logger.info(f"  Filtered (incorrect): {num_filtered}")
    logger.info(f"  Output: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse and filter evaluate_answers responses.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p1 = subparsers.add_parser("parse", help="Parse evaluation responses")
    p1.add_argument("--input_file", required=True, help="Input JSONL with evaluate_generation")
    p1.add_argument("--output_file", required=True, help="Output JSONL with parsed tags")

    p2 = subparsers.add_parser("filter", help="Filter out incorrect answers")
    p2.add_argument("--input_file", required=True, help="Input JSONL with correct/answerable tags")
    p2.add_argument("--output_file", required=True, help="Output JSONL with only correct answers")

    p3 = subparsers.add_parser("parse_multi_seed", help="Parse multi-seed evaluation responses")
    p3.add_argument("--input_dir", required=True, help="Directory with output-rs*.jsonl files")
    p3.add_argument("--num_seeds", type=int, default=5, help="Number of random seeds")

    args = parser.parse_args()

    if args.command == "parse":
        if not os.path.exists(args.input_file):
            logger.error(f"Input file not found: {args.input_file}")
            sys.exit(1)
        parse_evaluate_responses(args.input_file, args.output_file)
    elif args.command == "filter":
        if not os.path.exists(args.input_file):
            logger.error(f"Input file not found: {args.input_file}")
            sys.exit(1)
        filter_incorrect_answers(args.input_file, args.output_file)
    elif args.command == "parse_multi_seed":
        if not os.path.isdir(args.input_dir):
            logger.error(f"Input directory not found: {args.input_dir}")
            sys.exit(1)
        parse_multi_seed_responses(args.input_dir, args.num_seeds)
