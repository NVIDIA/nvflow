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
"""Aggregate multi-seed evaluation results.

Memory-efficient two-pass version:
- Pass 1: Scan all files, extract key + correct/answerable for each record
- Filter: Find keys in all files with all correct=YES and consistent answerable
- Pass 2: Stream through ONE file (rs0), output records whose keys passed filtering
"""

import argparse
import hashlib
import json
import os
import sys

from nvflow.lib.sdg.document_grounded.evaluate import parse_evaluation
from nvflow.utils import setup_logger

logger = setup_logger(__name__)


def get_key(record: dict) -> tuple[str, str]:
    """Build key for each record (context + problem)."""
    return (record.get("context", ""), record.get("problem", ""))


def hash_key(key: tuple[str, str]) -> str:
    """Hash a key to save memory. Uses MD5 for speed."""
    key_str = f"{key[0]}|||{key[1]}"
    return hashlib.md5(key_str.encode("utf-8")).hexdigest()


def scan_file_with_eval(file_path: str) -> dict[str, tuple[str, str]]:
    """
    Pass 1: Scan a file and return dict of key_hash -> (correct, answerable).

    Args:
        file_path: Path to the JSONL file

    Returns:
        dict mapping key_hash to (correct, answerable) tuple
    """
    results: dict[str, tuple[str, str]] = {}
    line_count = 0
    parse_success = 0
    parse_fail = 0

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            line_count += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                parse_fail += 1
                continue

            key = get_key(record)
            key_hash = hash_key(key)

            evaluate_gen = record.get("evaluate_generation", "")
            answerable, correct, _ = parse_evaluation(evaluate_gen)

            if answerable is not None and correct is not None:
                results[key_hash] = (correct, answerable)
                parse_success += 1
            else:
                parse_fail += 1

            if line_count % 500000 == 0:
                logger.info(f"  Scanned {line_count} lines, {parse_success} parsed...")

    logger.info(f"  Total: {line_count} lines, parsed: {parse_success}, failed: {parse_fail}")
    return results


def aggregate_evaluate_results(
    input_dir: str,
    output_file: str,
    num_seeds: int = 5,
):
    """
    Aggregate evaluation results from multiple seeds (memory-efficient two-pass version).

    Keeps only records where:
    1. Key exists in ALL seed files
    2. All seeds have correct=YES
    3. All seeds have consistent answerable (all YES or all NO)

    Args:
        input_dir: Directory containing output-rs*.jsonl files
        output_file: Path to output JSONL with aggregated results
        num_seeds: Number of random seeds (default 5)
    """
    logger.info(f"Aggregating evaluation results from: {input_dir}")
    logger.info(f"Output to: {output_file}")
    logger.info(f"Number of seeds: {num_seeds}")
    logger.info("Using memory-efficient two-pass algorithm")

    output_files = []
    for seed_idx in range(num_seeds):
        output_file_path = os.path.join(input_dir, f"output-rs{seed_idx}.jsonl")
        if not os.path.exists(output_file_path):
            logger.error(f"Output file not found: {output_file_path}")
            sys.exit(1)
        output_files.append(output_file_path)
        logger.info(f"Found seed {seed_idx}: {output_file_path}")

    # Pass 1: Scan all files to extract key -> (correct, answerable)
    logger.info("")
    logger.info("=" * 60)
    logger.info("PASS 1: Scanning files to extract evaluation results")
    logger.info("=" * 60)

    all_evals: list[dict[str, tuple[str, str]]] = []

    for i, file_path in enumerate(output_files):
        logger.info(f"Scanning seed {i}: {file_path}")
        eval_dict = scan_file_with_eval(file_path)
        logger.info(f"  Found {len(eval_dict)} valid records")
        all_evals.append(eval_dict)

    # Find intersection and filter
    logger.info("")
    logger.info("=" * 60)
    logger.info("Finding common keys and filtering")
    logger.info("=" * 60)

    common_keys = set(all_evals[0].keys())
    for eval_dict in all_evals[1:]:
        common_keys = common_keys & set(eval_dict.keys())

    logger.info(f"Keys in each file: {[len(ed) for ed in all_evals]}")
    logger.info(f"Common keys (intersection): {len(common_keys)}")

    if len(common_keys) == 0:
        logger.error("No common keys found across all files!")
        sys.exit(1)

    passed_keys: dict[str, str] = {}
    num_filtered_correct = 0
    num_filtered_inconsistent = 0

    for key_hash in common_keys:
        evals = [all_evals[i][key_hash] for i in range(num_seeds)]
        correct_list = [e[0] for e in evals]
        answerable_list = [e[1] for e in evals]

        if not all(c == "YES" for c in correct_list):
            num_filtered_correct += 1
            continue

        all_yes = all(a == "YES" for a in answerable_list)
        all_no = all(a == "NO" for a in answerable_list)

        if not (all_yes or all_no):
            num_filtered_inconsistent += 1
            continue

        passed_keys[key_hash] = "YES" if all_yes else "NO"

    logger.info(f"Passed filtering: {len(passed_keys)}")
    logger.info(f"Filtered (not all correct): {num_filtered_correct}")
    logger.info(f"Filtered (inconsistent answerable): {num_filtered_inconsistent}")

    del all_evals
    del common_keys

    stats = {"answerable": 0, "unanswerable": 0}
    for ans in passed_keys.values():
        if ans == "YES":
            stats["answerable"] += 1
        else:
            stats["unanswerable"] += 1

    if len(passed_keys) == 0:
        logger.error("No records passed filtering!")
        sys.exit(1)

    # Pass 2: Stream through rs0 and output matching records
    logger.info("")
    logger.info("=" * 60)
    logger.info("PASS 2: Streaming output from seed 0")
    logger.info("=" * 60)

    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    num_output = 0
    line_count = 0

    with (
        open(output_files[0], encoding="utf-8") as f_in,
        open(output_file, "w", encoding="utf-8") as f_out,
    ):
        for line in f_in:
            line = line.strip()
            if not line:
                continue

            line_count += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            key = get_key(record)
            key_hash = hash_key(key)

            if key_hash not in passed_keys:
                continue

            answerable_value = passed_keys[key_hash]

            output_record = {
                k: v
                for k, v in record.items()
                if k not in ["correct", "answerable", "evaluate_generation"]
            }
            output_record["answerable"] = answerable_value

            f_out.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            num_output += 1

            if num_output % 100000 == 0:
                logger.info(f"  Output {num_output} records...")

    logger.info("")
    logger.info("=" * 60)
    logger.info("Aggregation complete!")
    logger.info("=" * 60)
    logger.info(f"  Records passed filtering: {len(passed_keys)}")
    logger.info(f"  Records output: {num_output}")
    logger.info(f"    - Answerable: {stats['answerable']}")
    logger.info(f"    - Unanswerable: {stats['unanswerable']}")
    logger.info(f"  Filtered (not all correct): {num_filtered_correct}")
    logger.info(f"  Filtered (inconsistent answerable): {num_filtered_inconsistent}")

    if num_output != len(passed_keys):
        logger.warning(
            f"  Warning: Output count ({num_output}) != passed keys ({len(passed_keys)})"
        )
        logger.warning("  Some keys in rs0 may have been skipped due to parse errors")

    logger.info(f"  Output: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate multi-seed evaluation results.")

    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing output-rs*.jsonl files",
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Output JSONL file with aggregated results",
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=5,
        help="Number of random seeds (default: 5)",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        logger.error(f"Input directory not found: {args.input_dir}")
        sys.exit(1)

    aggregate_evaluate_results(args.input_dir, args.output_file, args.num_seeds)
