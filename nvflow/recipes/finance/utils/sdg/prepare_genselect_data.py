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
import glob
import os
import sys
from collections import defaultdict

import orjson

from nvflow.utils import setup_logger

# Initialize logger
logger = setup_logger(__name__)

# Skip if the solutions are too long, we use 120K which is approx 30K tokens because X 5 we may get 150K in the worst case, this is still okay for Qwen3-235b which has 262K context window
SKIP_LENGTH = 80_000
WRITE_BUFFER_SIZE = 1000  # Write every 1000 records

problem_key = "problem"


def merge_jsonl_files(input_files, output_file):
    """
    Merge multiple JSONL files by joining on 'problem' key.
    Creates separate 'solutions' and 'reasonings' lists for each problem.

    Args:
        input_files: List of 5 input JSONL file paths
        output_file: Output JSONL file path
    """

    logger.info("Phase 1: Reading and indexing files...")

    # Dictionary to store: problem -> {base_record, solutions_list}
    data = {}

    # Metrics tracking: problem -> number of solutions filtered
    filtered_per_problem = defaultdict(int)
    total_filtered = 0

    # Process each file
    for file_idx, filepath in enumerate(input_files, 1):
        logger.info(f"Processing file {file_idx}/{len(input_files)}: {filepath}")

        with open(filepath, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if line_num % 100000 == 0:
                    logger.info(f"  Processed {line_num} lines...")

                try:
                    record = orjson.loads(line.strip())
                    problem = record[problem_key]
                    generation = record.get("generation", "")
                    reasoning = record.get("reasoning_content", "")
                    if not reasoning and not generation:
                        logger.warning(
                            f"Line {line_num} question {problem} in {filepath} missing 'generation' and 'reasoning_content' keys"
                        )

                    if problem not in data:
                        # First time seeing this problem - initialize structure
                        base_record = {
                            k: v
                            for k, v in record.items()
                            if k != "generation" and k != "reasoning_content"
                        }
                        data[problem] = {
                            "base_record": base_record,
                            "generations": [],
                            "reasonings": [],
                        }

                    # Check solution length and filter if too long
                    if len(generation) > SKIP_LENGTH:
                        filtered_per_problem[problem] += 1
                        total_filtered += 1
                    else:
                        # Add solution and reasoning separately
                        data[problem]["generations"].append(generation)
                        data[problem]["reasonings"].append(reasoning)

                except orjson.JSONDecodeError as e:
                    logger.error(f"Error parsing line {line_num} in {filepath}: {e}")
                    continue

    logger.info(f"\nPhase 2: Writing merged output to {output_file}...")
    logger.info(f"Total unique problems: {len(data)}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Write merged data with buffering
    buffer = []

    with open(output_file, "wb") as out:  # Open in binary mode for faster writing
        for idx, (problem, problem_data) in enumerate(data.items(), 1):
            if idx % 100000 == 0:
                logger.info(f"  Written {idx} records...")

            base_record = problem_data["base_record"]
            generations = problem_data["generations"]
            reasonings = problem_data["reasonings"]

            # Create merged solutions field with only generations
            merged_solutions = "\n".join(
                f"Solution {i}:\n{generation}" for i, generation in enumerate(generations)
            )

            # Create output record
            output_record = base_record.copy()
            output_record["problem"] = problem
            output_record["solutions"] = merged_solutions
            output_record["generations_list"] = generations
            output_record["reasonings_list"] = reasonings
            output_record["max_idx"] = len(generations) - 1
            output_record["num_solutions"] = len(generations)

            # Add individual indexed fields for each solution
            for i, (generation, reasoning) in enumerate(zip(generations, reasonings, strict=False)):
                output_record[f"answer_{i}"] = generation
                output_record[f"answer_reasoning_content_{i}"] = reasoning

            # Add to buffer
            buffer.append(orjson.dumps(output_record))

            # Write buffer when it reaches threshold
            if len(buffer) >= WRITE_BUFFER_SIZE:
                out.write(b"\n".join(buffer) + b"\n")
                buffer.clear()

        # Write remaining buffer
        if buffer:
            out.write(b"\n".join(buffer) + b"\n")

    logger.info(f"\n✓ Complete! Written {len(data)} merged records to {output_file}")
    logger.info(
        f"  Average solutions per problem: {sum(len(p['generations']) for p in data.values()) / len(data):.2f}"
    )

    # Print filtering metrics
    logger.info(f"\n📊 Filtering Metrics (solutions exceeding {SKIP_LENGTH:,} characters):")
    logger.info(f"  Total generations filtered: {total_filtered}")
    logger.info(f"  Problems with filtered generations: {len(filtered_per_problem)}")

    # Count distribution of filtered solutions per problem
    filter_distribution = defaultdict(int)
    for count in filtered_per_problem.values():
        filter_distribution[count] += 1

    logger.info("\n  Distribution of filtered solutions per problem:")
    for num_filtered in sorted(filter_distribution.keys()):
        num_problems = filter_distribution[num_filtered]
        logger.info(f"    {num_filtered} generation(s) filtered: {num_problems} problem(s)")

    problems_with_no_filtering = len(data) - len(filtered_per_problem)
    if problems_with_no_filtering > 0:
        logger.info(f"    0 generations filtered: {problems_with_no_filtering} problem(s)")


if __name__ == "__main__":
    # Configure your file paths here

    parser = argparse.ArgumentParser(description="Merge JSONL files by 'problem' key.")
    parser.add_argument("--input_dir", help="Directory containing input JSONL files")
    parser.add_argument("--output_file", help="Path for merged output JSONL file")
    args = parser.parse_args()

    input_files = sorted(glob.glob(os.path.join(args.input_dir, "output-rs*.jsonl")))

    output_file = args.output_file

    logger.info("JSONL Merger - Joining on 'problem' key")
    logger.info("=" * 50)
    logger.info(f"Input files: {len(input_files)}")
    logger.info(f"Output file: {output_file}")
    logger.info("=" * 50)

    # Check if output file already exists and is non-empty
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        logger.info(f"\n✓ Output file already exists: {output_file}")
        logger.info("Skipping merge operation.")
        sys.exit(0)
    try:
        merge_jsonl_files(input_files, output_file)
    except Exception as e:
        logger.error(f"\n✗ Error: {e}")
        sys.exit(1)
