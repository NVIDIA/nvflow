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
"""Utilities for difficulty estimation stage."""

import argparse
import json
import os
import re
from collections import defaultdict

from nvflow.utils import setup_logger

logger = setup_logger(__name__)


def prepare_difficulty_input(input_file: str, output_file: str):
    """
    Prepare input for difficulty estimation by creating records for small model to answer.

    The input file should have 'problem', 'context', and 'generation' (reference answer).
    We keep the reference answer for later comparison.

    Args:
        input_file: Path to evaluated answers JSONL
        output_file: Path to output JSONL for small model generation
    """
    logger.info(f"Preparing difficulty estimation input from: {input_file}")
    logger.info(f"Output to: {output_file}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    count = 0
    with (
        open(input_file, encoding="utf-8") as f_in,
        open(output_file, "w", encoding="utf-8") as f_out,
    ):
        for line in f_in:
            if not line.strip():
                continue

            try:
                record = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            # Keep original fields and rename generation to reference_answer
            output_record = record.copy()
            output_record["reference_answer"] = record.get("generation", "")
            # Remove generation so small model generates fresh answer
            if "generation" in output_record:
                del output_record["generation"]
            if "reasoning_content" in output_record:
                output_record["reference_reasoning"] = output_record.pop("reasoning_content")

            f_out.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            count += 1

    logger.info(f"✓ Prepared {count} records for difficulty estimation")


def truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    """Truncate text to max_chars, returning (truncated_text, was_truncated)."""
    if len(text) <= max_chars:
        return text, False
    # Try to truncate at a sentence or word boundary
    truncated = text[:max_chars]
    # Find last period or newline for cleaner truncation
    last_period = truncated.rfind(".")
    last_newline = truncated.rfind("\n")
    cut_point = max(last_period, last_newline)
    if cut_point > max_chars * 0.7:  # Only use if we keep at least 70%
        truncated = truncated[: cut_point + 1]
    return truncated + "\n\n[... truncated ...]", True


def prepare_judge_input(input_dir: str, output_dir: str, max_answer_chars: int = 50000):
    """
    Prepare input for judge model by pairing reference answers with candidate answers.

    Creates separate output files for each seed to enable streaming aggregation.

    Args:
        input_dir: Directory containing output-rs*.jsonl files from small model
        output_dir: Directory to write judge_input_rs*.jsonl files
        max_answer_chars: Maximum characters for reference/candidate answers (default 50000)
    """
    logger.info(f"Preparing judge input from: {input_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Max answer chars: {max_answer_chars}")

    # Find all output files
    input_files = sorted(
        [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.startswith("output-rs") and f.endswith(".jsonl")
        ]
    )

    if not input_files:
        # Fallback to single output.jsonl
        single_file = os.path.join(input_dir, "output.jsonl")
        if os.path.exists(single_file):
            input_files = [single_file]
        else:
            raise FileNotFoundError(f"No output files found in {input_dir}")

    logger.info(f"Found {len(input_files)} input files")

    os.makedirs(output_dir, exist_ok=True)

    total_count = 0
    truncated_count = 0
    for input_file in input_files:
        # Extract seed number from filename (e.g., output-rs0.jsonl -> 0)
        seed_match = re.search(r"output-rs(\d+)\.jsonl", os.path.basename(input_file))
        seed_num = int(seed_match.group(1)) if seed_match else 0

        output_file = os.path.join(output_dir, f"judge_input_rs{seed_num}.jsonl")
        count = 0
        seed_truncated = 0

        with (
            open(input_file, encoding="utf-8") as f_in,
            open(output_file, "w", encoding="utf-8") as f_out,
        ):
            for line in f_in:
                if not line.strip():
                    continue

                try:
                    record = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                # Create judge input record
                output_record = record.copy()

                # Keep reference_answer as-is (don't truncate ground truth)
                output_record["reference_answer"] = record.get("reference_answer", "")

                # Truncate candidate_answer if too long
                cand_answer = record.get("generation", "")
                cand_answer, cand_truncated = truncate_text(cand_answer, max_answer_chars)
                output_record["candidate_answer"] = cand_answer

                output_record["seed_num"] = seed_num

                if cand_truncated:
                    seed_truncated += 1

                f_out.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                count += 1

        if seed_truncated > 0:
            logger.info(
                f"  Seed {seed_num}: {count} records ({seed_truncated} truncated) -> {output_file}"
            )
        else:
            logger.info(f"  Seed {seed_num}: {count} records -> {output_file}")
        total_count += count
        truncated_count += seed_truncated

    logger.info(f"✓ Prepared {total_count} records for judging ({len(input_files)} files)")
    if truncated_count > 0:
        logger.warning(
            f"  Truncated {truncated_count} records with candidate_answer > {max_answer_chars} chars"
        )


def aggregate_difficulty_scores(input_dir: str, output_file: str, num_seeds: int = 5):
    """
    Aggregate judge results to compute difficulty scores (streaming version).

    Processes all seed files line-by-line in parallel, assuming they have
    the same order (which is guaranteed by nemo_skills generate).

    Difficulty score = number of correct answers out of num_seeds attempts.
    Lower score = harder question.

    Args:
        input_dir: Directory containing output_rs*.jsonl files from judge
        output_file: Path to output JSONL with difficulty scores
        num_seeds: Number of random seeds used (default 5)
    """
    logger.info(f"Aggregating difficulty scores from: {input_dir}")
    logger.info(f"Output to: {output_file}")
    logger.info(f"Number of seeds: {num_seeds}")
    logger.info("Using streaming mode (line-by-line processing)")

    def get_key(record: dict) -> tuple:
        return (record.get("context", ""), record.get("problem", ""))

    def parse_judge_result(judge_gen: str) -> bool:
        """Parse judge generation to determine if answer is correct."""
        is_correct = "CORRECT" in judge_gen.upper() and "INCORRECT" not in judge_gen.upper()
        if not is_correct:
            match = re.search(r"Judgment:\s*(CORRECT|INCORRECT)", judge_gen, re.IGNORECASE)
            if match:
                is_correct = match.group(1).upper() == "CORRECT"
        return is_correct

    # Find all judge output files
    judge_files = []
    for seed_idx in range(num_seeds):
        # Try different naming patterns
        patterns = [
            os.path.join(input_dir, f"output_rs{seed_idx}.jsonl"),
            os.path.join(input_dir, f"output-rs{seed_idx}.jsonl"),
            os.path.join(input_dir, f"rs{seed_idx}", "output.jsonl"),
        ]
        found = False
        for pattern in patterns:
            if os.path.exists(pattern):
                judge_files.append(pattern)
                logger.info(f"Found seed {seed_idx}: {pattern}")
                found = True
                break
        if not found:
            logger.error(f"Judge output not found for seed {seed_idx}")
            raise FileNotFoundError(f"No judge output found for seed {seed_idx}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Stats
    num_total = 0
    num_key_mismatch = 0
    stats: dict[int, int] = defaultdict(int)

    # Open all files and process line by line
    file_handles = [open(f, encoding="utf-8") for f in judge_files]

    try:
        with open(output_file, "w", encoding="utf-8") as f_out:
            while True:
                # Read one line from each file
                lines: list[str | None] = []
                for fh in file_handles:
                    line = fh.readline()
                    if not line:
                        lines.append(None)
                    else:
                        lines.append(line.strip())

                # Check if all files are exhausted
                if all(line is None for line in lines):
                    break

                # Check if some files are exhausted but not all
                if any(line is None for line in lines):
                    logger.warning("Files have different number of lines!")
                    break

                # Skip empty lines
                if all(not line for line in lines):
                    continue

                num_total += 1

                # Parse all records
                records = []
                parse_error = False
                for line in lines:
                    if not line:
                        parse_error = True
                        break
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        parse_error = True
                        break

                if parse_error:
                    continue

                # Verify all records have the same key
                keys = [get_key(r) for r in records]
                if len(set(keys)) > 1:
                    num_key_mismatch += 1
                    if num_key_mismatch <= 5:
                        logger.warning(f"Key mismatch at line {num_total}: {keys}")
                    continue

                # Count correct answers across seeds
                correct_count = sum(
                    1 for r in records if parse_judge_result(r.get("judge_generation", ""))
                )

                # Create output record from first seed's base record
                base_record = records[0]
                output_record = {
                    k: v
                    for k, v in base_record.items()
                    if k not in ["candidate_answer", "seed_num", "judge_generation", "generation"]
                }
                output_record["difficulty_score"] = correct_count
                output_record["difficulty_total"] = num_seeds
                output_record["difficulty_ratio"] = correct_count / num_seeds

                f_out.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                stats[correct_count] += 1

                # Progress logging
                if num_total % 100000 == 0:
                    logger.info(f"Processed {num_total} records...")

    finally:
        for fh in file_handles:
            fh.close()

    logger.info("")
    logger.info(f"✓ Aggregated {num_total} problems with difficulty scores")
    if num_key_mismatch > 0:
        logger.warning(f"  Key mismatches (skipped): {num_key_mismatch}")
    logger.info(f"\nDifficulty distribution (correct out of {num_seeds}):")
    for score in sorted(stats.keys()):
        logger.info(f"  Score {score}: {stats[score]} problems")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Difficulty estimation utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Prepare input for small model
    p1 = subparsers.add_parser("prepare_input", help="Prepare input for small model generation")
    p1.add_argument("--input_file", required=True)
    p1.add_argument("--output_file", required=True)

    # Prepare input for judge
    p2 = subparsers.add_parser("prepare_judge", help="Prepare input for judge model")
    p2.add_argument("--input_dir", required=True)
    p2.add_argument("--output_dir", required=True)
    p2.add_argument(
        "--max_answer_chars",
        type=int,
        default=50000,
        help="Max chars for reference/candidate answers (default: 50000)",
    )

    # Aggregate scores
    p3 = subparsers.add_parser("aggregate", help="Aggregate difficulty scores")
    p3.add_argument("--input_dir", required=True)
    p3.add_argument("--output_file", required=True)
    p3.add_argument("--num_seeds", type=int, default=5)

    args = parser.parse_args()

    if args.command == "prepare_input":
        prepare_difficulty_input(args.input_file, args.output_file)
    elif args.command == "prepare_judge":
        prepare_judge_input(args.input_dir, args.output_dir, args.max_answer_chars)
    elif args.command == "aggregate":
        aggregate_difficulty_scores(args.input_dir, args.output_file, args.num_seeds)
