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
"""Merge multi-seed answer files and post-process GenSelect output.

Combines the functionality of prepare_genselect_data.py and postprocess_genselect.py
into a single module with CLI subcommands.
"""

import argparse
import os
import random
import re
import sys
from collections import defaultdict

import orjson

from nvflow.integrations.nemo_gym.native_contract import NativeGymOutputLayout
from nvflow.stages.sdg.document_grounded.metadata import nest_metadata_fields
from nvflow.utils import setup_logger

logger = setup_logger(__name__)

SKIP_LENGTH = 80_000
WRITE_BUFFER_SIZE = 1000


# ---------------------------------------------------------------------------
# Merge multi-seed answer files
# ---------------------------------------------------------------------------


def merge_jsonl_files(
    input_files,
    output_file,
    shuffle_solutions=False,
    shuffle_seed=42,
    include_context=False,
):
    """
    Merge multiple JSONL files by joining on 'problem' key.
    Creates separate 'solutions' and 'reasonings' lists for each problem.

    Args:
        input_files: List of input JSONL file paths
        output_file: Output JSONL file path
        shuffle_solutions: Randomly reorder candidates per problem before judging.
        shuffle_seed: Seed for deterministic candidate shuffling.
        include_context: Embed each candidate's retrieval context in its solution block.
    """
    logger.info("Phase 1: Reading and indexing files...")

    data = {}
    filtered_per_problem = defaultdict(int)
    total_filtered = 0

    for file_idx, filepath in enumerate(input_files, 1):
        logger.info(f"Processing file {file_idx}/{len(input_files)}: {filepath}")

        with open(filepath, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if line_num % 100000 == 0:
                    logger.info(f"  Processed {line_num} lines...")

                try:
                    record = orjson.loads(line.strip())
                    problem = record["problem"]
                    generation = record.get("generation") or ""
                    reasoning = record.get("reasoning_content") or ""
                    # Full Responses-API original form of this candidate answer
                    # (snapshotted by enrich_rollouts under ``answer_response``).
                    # Collected into a per-problem list aligned with
                    # ``generations`` so postprocess can select the same index
                    # the genselect judge picks.
                    answer_response = record.get("answer_response")
                    if not reasoning and not generation:
                        logger.warning(
                            f"Line {line_num} question {problem} in {filepath} missing 'generation' and 'reasoning_content' keys"
                        )

                    if problem not in data:
                        base_record = {
                            k: v
                            for k, v in record.items()
                            if k not in ("generation", "reasoning_content", "answer_response")
                        }
                        data[problem] = {
                            "base_record": base_record,
                            "generations": [],
                            "reasonings": [],
                            "answer_responses": [],
                            "seed_indices": [],
                            "contexts": [],
                        }

                    if len(generation) > SKIP_LENGTH:
                        filtered_per_problem[problem] += 1
                        total_filtered += 1
                    else:
                        data[problem]["generations"].append(generation)
                        data[problem]["reasonings"].append(reasoning)
                        data[problem]["answer_responses"].append(answer_response)
                        data[problem]["seed_indices"].append(file_idx - 1)
                        data[problem]["contexts"].append(record.get("context") or "")

                except orjson.JSONDecodeError as e:
                    logger.error(f"Error parsing line {line_num} in {filepath}: {e}")
                    continue

    logger.info(f"\nPhase 2: Writing merged output to {output_file}...")
    logger.info(f"Total unique problems: {len(data)}")
    if shuffle_solutions:
        logger.info(f"Shuffling solutions per problem (seed={shuffle_seed})")
    if include_context:
        logger.info("Including per-candidate retrieval context in solution blocks")

    buffer = []
    rng = random.Random(shuffle_seed)

    # Write to a temp sibling, then atomically replace -- both this CLI's own
    # skip-if-exists check (below) and gym_genselect_answers's stage-level
    # `prep_exists` check treat this exact final path as "already prepared"
    # once it exists and is non-empty. A direct write would let a
    # killed/timed-out job leave a truncated-but-non-empty file that either
    # check mistakes for complete, silently feeding a short genselect input
    # downstream forever.
    tmp_output_file = f"{output_file}.tmp"
    with open(tmp_output_file, "wb") as out:
        for idx, (problem, problem_data) in enumerate(data.items(), 1):
            if idx % 100000 == 0:
                logger.info(f"  Written {idx} records...")

            base_record = problem_data["base_record"]
            generations = problem_data["generations"]
            reasonings = problem_data["reasonings"]
            answer_responses = problem_data["answer_responses"]
            seed_indices = problem_data["seed_indices"]
            contexts = problem_data["contexts"]

            if shuffle_solutions:
                candidates = list(
                    zip(
                        generations,
                        reasonings,
                        answer_responses,
                        seed_indices,
                        contexts,
                        strict=True,
                    )
                )
                if candidates:
                    rng.shuffle(candidates)
                    generations, reasonings, answer_responses, seed_indices, contexts = map(
                        list, zip(*candidates, strict=True)
                    )

            solution_parts = []
            for i, (generation, context) in enumerate(zip(generations, contexts, strict=True)):
                block = f"Solution {i}:"
                if include_context and context:
                    block += f"\n<context>\n{context}\n</context>"
                block += f"\n{generation}"
                solution_parts.append(block)
            merged_solutions = "\n".join(solution_parts)

            output_record = base_record.copy()
            if include_context:
                # Candidate contexts are embedded in the judge prompt; avoid retaining
                # the seed-0 context separately as the generic top-level context.
                output_record["context"] = ""
            output_record["problem"] = problem
            output_record["solutions"] = merged_solutions
            output_record["generations_list"] = generations
            output_record["reasonings_list"] = reasonings
            output_record["answer_responses_list"] = answer_responses
            if include_context:
                output_record["contexts_list"] = contexts
            if shuffle_solutions:
                # Maps judge-visible candidate position back to its source rollout file.
                output_record["original_seed_indices"] = seed_indices
            output_record["max_idx"] = len(generations) - 1
            output_record["num_solutions"] = len(generations)

            for i, (generation, reasoning, context) in enumerate(
                zip(generations, reasonings, contexts, strict=True)
            ):
                output_record[f"answer_{i}"] = generation
                output_record[f"answer_reasoning_content_{i}"] = reasoning
                if include_context:
                    output_record[f"answer_context_{i}"] = context

            buffer.append(orjson.dumps(output_record))

            if len(buffer) >= WRITE_BUFFER_SIZE:
                out.write(b"\n".join(buffer) + b"\n")
                buffer.clear()

        if buffer:
            out.write(b"\n".join(buffer) + b"\n")

    os.replace(tmp_output_file, output_file)

    logger.info(f"\nComplete! Written {len(data)} merged records to {output_file}")
    logger.info(
        f"  Average solutions per problem: {sum(len(p['generations']) for p in data.values()) / len(data):.2f}"
    )

    logger.info(f"\nFiltering Metrics (solutions exceeding {SKIP_LENGTH:,} characters):")
    logger.info(f"  Total generations filtered: {total_filtered}")
    logger.info(f"  Problems with filtered generations: {len(filtered_per_problem)}")

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


# ---------------------------------------------------------------------------
# Post-process GenSelect output
# ---------------------------------------------------------------------------


def extract_judgment_index(generation_text):
    """
    Extract the number after the last occurrence of "Judgement: " or "Judgment: " in the text.

    Args:
        generation_text: The generation field text

    Returns:
        int: The extracted index, or None if not found
    """
    cleaned_text = generation_text.replace("*", "")

    matches = re.findall(r"Judgment:\s*(\d+)", cleaned_text)
    if not matches:
        matches = re.findall(r"Judgement:\s*(\d+)", cleaned_text)

    if matches:
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

                generation = record.get("generation") or ""
                reasoning_content = record.get("reasoning_content") or ""
                generations_list = record.get("generations_list") or []
                reasonings_list = record.get("reasonings_list") or []
                answer_responses_list = record.get("answer_responses_list") or []

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

                judgment_idx = extract_judgment_index(generation)

                if judgment_idx is None:
                    logger.warning(f"Line {line_num} could not extract judgment index, skipping")
                    records_skipped += 1
                    continue

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

                original_seed_indices = record.get("original_seed_indices") or []
                selected_original_seed_index = (
                    original_seed_indices[judgment_idx]
                    if judgment_idx < len(original_seed_indices)
                    else judgment_idx
                )
                selected_solution = generations_list[judgment_idx]
                selected_reasoning = reasonings_list[judgment_idx]
                # Select the Responses-API original form of the same candidate the judge
                # picked.  Lenient: a missing/short list yields None rather than
                # dropping the record -- the truth object is a best-effort
                # enrichment, not a gate.
                selected_response = (
                    answer_responses_list[judgment_idx]
                    if judgment_idx < len(answer_responses_list)
                    else None
                )

                output_record = record.copy()

                genselect_response = {}
                if generation:
                    genselect_response["generation"] = generation
                if reasoning_content:
                    genselect_response["reasoning_content"] = reasoning_content

                output_record["genselect_response"] = genselect_response

                output_record = nest_metadata_fields(output_record, "genselect_answers_metadata")

                output_record["generation"] = selected_solution
                output_record["reasoning_content"] = selected_reasoning
                # Snapshot the genselect-picked answer + its reasoning into
                # ``reference_*`` so downstream LLM stages (evaluate_answers,
                # difficulty judge, etc.) can freely overwrite ``generation`` /
                # ``reasoning_content`` with judge output without losing the
                # original answer-pipeline reasoning we want to surface in the
                # final post-process record.
                #
                # ``postprocess.py.FIELDS_TO_RENAME`` restores these back at
                # the end (``reference_reasoning -> reasoning_content``,
                # ``reference_answer -> answer``).  ``difficulty.py`` uses
                # ``setdefault`` so it does not clobber these snapshots when
                # they already exist.
                output_record["reference_reasoning"] = selected_reasoning
                output_record["reference_answer"] = selected_solution
                # Snapshot the selected candidate's Responses-API truth + the
                # (shared) request params so post-process can emit them as the
                # final ``response`` / ``responses_create_params``.
                output_record["reference_response"] = selected_response
                output_record["reference_responses_create_params"] = record.get(
                    "answer_responses_create_params"
                )
                output_record["selected_index"] = judgment_idx
                output_record["selected_original_seed_index"] = selected_original_seed_index

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
    logger.info("Complete!")
    logger.info(f"  Records processed: {records_processed}")
    logger.info(f"  Records skipped: {records_skipped}")
    logger.info(f"  Output written to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge multi-seed answers and post-process GenSelect output."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p1 = subparsers.add_parser("merge", help="Merge JSONL files by 'problem' key")
    p1.add_argument("--input_dir", required=True, help="Directory containing input JSONL files")
    p1.add_argument("--output_file", required=True, help="Path for merged output JSONL file")
    p1.add_argument(
        "--shuffle_solutions",
        action="store_true",
        help="Deterministically shuffle candidates per problem before judging",
    )
    p1.add_argument(
        "--shuffle_seed",
        type=int,
        default=42,
        help="Random seed for --shuffle_solutions (default: 42)",
    )
    p1.add_argument(
        "--include_context",
        action="store_true",
        help="Embed each candidate's retrieval context in its solution block",
    )

    p2 = subparsers.add_parser("postprocess", help="Post-process GenSelect output")
    p2.add_argument("--input_dir", required=True, help="Directory containing output.jsonl file")
    p2.add_argument("--output_file", required=True, help="Path for output JSONL file")

    args = parser.parse_args()

    if args.command == "merge":
        canonical_rollouts = NativeGymOutputLayout.collection(
            args.input_dir
        ).analyzed_artifacts.rollouts.as_posix()
        input_files = [canonical_rollouts] if os.path.exists(canonical_rollouts) else []
        if not input_files:
            logger.error(f"No canonical rollouts.jsonl found in {args.input_dir}")
            sys.exit(1)

        if os.path.exists(args.output_file) and os.path.getsize(args.output_file) > 0:
            logger.info(f"Output file already exists: {args.output_file}")
            logger.info("Skipping merge operation.")
            sys.exit(0)

        try:
            merge_jsonl_files(
                input_files,
                args.output_file,
                shuffle_solutions=args.shuffle_solutions,
                shuffle_seed=args.shuffle_seed,
                include_context=args.include_context,
            )
        except Exception as e:
            logger.error(f"Error: {e}")
            sys.exit(1)

    elif args.command == "postprocess":
        input_file = os.path.join(args.input_dir, "output.jsonl")
        if not os.path.exists(input_file):
            logger.error(f"Input file not found: {input_file}")
            sys.exit(1)

        try:
            postprocess_genselect(input_file, args.output_file)
        except Exception as e:
            logger.error(f"Error: {e}")
            sys.exit(1)
