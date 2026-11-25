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
import json
import os
import random
import re
import sys
from collections import defaultdict

import orjson

from nvflow.utils import setup_logger

# Initialize logger
logger = setup_logger(__name__)


def _extract_question_from_rcp(rcp: dict) -> str:
    """Extract the question text from a raw rollout's responses_create_params.

    Mirrors extract_multiturn_data.extract_question_from_rcp so the extracted
    question is byte-identical to the flat ``problem`` field other seeds carry
    (verified: raw records merge with flattened ones on this key).
    """
    inp = rcp.get("input", []) if isinstance(rcp, dict) else []
    content = inp[0].get("content", "") if inp else ""
    if isinstance(content, list):
        content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
    content = str(content)
    m = re.search(r"Question:\s*\n(.*?)(?:\n\n|\Z)", content, re.DOTALL)
    if m:
        return m.group(1).strip()
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    return paragraphs[-1] if paragraphs else content.strip()


def _extract_answer_from_response(response: dict) -> str:
    """Extract the final answer from a raw rollout's response.output.

    Raw collect_rollouts records keep the answer inside the tool-calling
    trajectory (the ``submit_final_result`` function call), not in a flat
    ``generation`` field. Returns "" if no final answer is present.
    """
    output_items = response.get("output", []) if isinstance(response, dict) else []
    for item in output_items:
        if item.get("type") == "function_call" and item.get("name") == "submit_final_result":
            args = item.get("arguments", "")
            try:
                parsed = json.loads(args) if isinstance(args, str) else args
                if isinstance(parsed, dict):
                    result = str(parsed.get("final_result", ""))
                    if result:
                        return result
            except (json.JSONDecodeError, AttributeError, TypeError):
                pass
            return str(args) if args else ""
    return ""


# Skip if the solutions are too long, we use 120K which is approx 30K tokens because X 5 we may get 150K in the worst case, this is still okay for Qwen3-235b which has 262K context window
SKIP_LENGTH = 80_000
WRITE_BUFFER_SIZE = 1000  # Write every 1000 records
# Skip entire problem if total prompt chars exceed this. Qwen3-235B has 262K tokens; we reserve
# 16K for output → 245,760 input tokens max. Financial SEC data with dense numbers/tables tokenises
# at ~2.8-3.0 chars/token (not the usual 4). At 2.789 c/t: 245,760 × 2.789 = 685,579 chars.
# Use 650K for a comfortable 12K-token safety margin, accounting for template overhead (~1.6K chars).
MAX_TOTAL_CHARS = 650_000

problem_key = "problem"


def merge_jsonl_files(
    input_files,
    output_file,
    strip_context=False,
    shuffle_solutions=False,
    shuffle_seed=42,
    include_reasoning=False,
    include_context=False,
):
    """
    Merge multiple JSONL files by joining on 'problem' key.
    Creates separate 'solutions' and 'reasonings' lists for each problem.

    Args:
        input_files: List of input JSONL file paths
        output_file: Output JSONL file path
        strip_context: If True, set context to empty string in all output records
        shuffle_solutions: If True, randomly shuffle solutions per problem. Stores
            original_seed_indices so selected_index can be mapped back post-GenSelect.
        shuffle_seed: Global RNG seed for reproducibility when shuffle_solutions=True
        include_reasoning: If True, include each seed's reasoning chain in the solution
            block so GenSelect can evaluate chain-of-thought quality, not just final answers.
        include_context: If True, embed each seed's SEC retrieval context inside its
            solution block as <context>...</context>. All 8 seeds' retrieved filings are
            shown to GenSelect rather than only seed-0's context at the top level.
            Problems whose total prompt would exceed MAX_TOTAL_CHARS are skipped.
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
                    # Backward-compatible raw-rollout support: flattened records have
                    # a flat problem/generation; raw collect_rollouts records don't
                    # (the answer lives in response.output). Fall back to extracting
                    # both so GenSelect can run directly on raw rollouts.
                    problem = record.get(problem_key) or ""
                    if not problem:
                        problem = _extract_question_from_rcp(
                            record.get("responses_create_params", {})
                        )
                    if not problem:
                        logger.warning(
                            f"Line {line_num} in {filepath}: no '{problem_key}' and none "
                            "extractable from responses_create_params, skipping"
                        )
                        continue
                    generation = record.get("generation", "") or ""
                    if not generation or generation.strip() == "N/A":
                        extracted = _extract_answer_from_response(record.get("response", {}))
                        if extracted:
                            generation = extracted
                    reasoning = record.get("reasoning_content", "")
                    context = record.get("context", "") or ""
                    if not reasoning and not generation:
                        logger.warning(
                            f"Line {line_num} question {problem} in {filepath} missing 'generation' and 'reasoning_content' keys"
                        )

                    if problem not in data:
                        # First time seeing this problem - initialize structure
                        base_record = {
                            k: v
                            for k, v in record.items()
                            if k
                            not in {
                                "generation",
                                "reasoning_content",
                                # Raw agent rollouts can be very large. GenSelect only
                                # needs their extracted final answers; the selected full
                                # trajectory is recovered from the source rollout later.
                                "response",
                            }
                        }
                        data[problem] = {
                            "base_record": base_record,
                            "generations": [],
                            "reasonings": [],
                            "contexts": [],  # per-seed SEC retrieval context
                            # Candidate position before optional shuffling. For the
                            # historical layout (one seed per file) this is the seed
                            # index. It also works for native Gym files containing
                            # multiple repeats from one model.
                            "seed_indices": [],
                            # Exact source identity used to recover the selected full
                            # tool trajectory after GenSelect.
                            "candidate_references": [],
                        }

                    # Check solution length and filter if too long
                    if len(generation) > SKIP_LENGTH:
                        filtered_per_problem[problem] += 1
                        total_filtered += 1
                    else:
                        # Add solution, reasoning, and context separately
                        data[problem]["generations"].append(generation)
                        data[problem]["reasonings"].append(reasoning)
                        data[problem]["contexts"].append(context)
                        data[problem]["seed_indices"].append(len(data[problem]["seed_indices"]))
                        data[problem]["candidate_references"].append(
                            {
                                "source_file_index": file_idx - 1,
                                "rollout_index": record.get("_ng_rollout_index"),
                                "task_index": record.get("_ng_task_index"),
                            }
                        )

                except orjson.JSONDecodeError as e:
                    logger.error(f"Error parsing line {line_num} in {filepath}: {e}")
                    continue

    logger.info(f"\nPhase 2: Writing merged output to {output_file}...")
    logger.info(f"Total unique problems: {len(data)}")
    if shuffle_solutions:
        logger.info(f"Shuffling solutions per problem (seed={shuffle_seed})")
    if include_context:
        logger.info(
            f"Including per-seed SEC retrieval context in solution blocks (total prompt skip threshold: {MAX_TOTAL_CHARS:,} chars)"
        )

    rng = random.Random(shuffle_seed)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Write merged data with buffering
    buffer = []
    total_skipped_long = 0

    with open(output_file, "wb") as out:  # Open in binary mode for faster writing
        for idx, (problem, problem_data) in enumerate(data.items(), 1):
            if idx % 100000 == 0:
                logger.info(f"  Written {idx} records...")

            base_record = problem_data["base_record"]
            generations = problem_data["generations"]
            reasonings = problem_data["reasonings"]
            contexts = problem_data["contexts"]
            seed_indices = problem_data["seed_indices"]
            candidate_references = problem_data["candidate_references"]

            if shuffle_solutions:
                # Shuffle all four lists together using a per-problem seed derived from global rng
                combined = list(
                    zip(
                        generations,
                        reasonings,
                        contexts,
                        seed_indices,
                        candidate_references,
                        strict=False,
                    )
                )
                rng.shuffle(combined)
                generations, reasonings, contexts, seed_indices, candidate_references = map(
                    list, zip(*combined, strict=False)
                )

            # Build per-seed solution blocks based on requested flags
            solution_parts = []
            for i, (gen, reas, ctx) in enumerate(
                zip(generations, reasonings, contexts, strict=False)
            ):
                effective_ctx = "" if (strip_context or not include_context) else ctx
                block = f"Solution {i}:"
                if include_context and effective_ctx:
                    block += f"\n<context>\n{effective_ctx}\n</context>"
                if include_reasoning and reas:
                    block += f"\n<reasoning>\n{reas}\n</reasoning>"
                block += f"\n{gen}"
                solution_parts.append(block)
            merged_solutions = "\n".join(solution_parts)

            # Skip problem if total prompt is too long for the model's context window.
            # Count problem text + seed-0 top-level context + all solution blocks (which include
            # the per-seed contexts and reasonings). Template overhead is small (~5 K chars).
            if include_context:
                top_ctx = "" if strip_context else (base_record.get("context", "") or "")
                problem_text = base_record.get("problem", "") or ""
                total_chars = len(problem_text) + len(top_ctx) + len(merged_solutions)
                if total_chars > MAX_TOTAL_CHARS:
                    total_skipped_long += 1
                    continue

            # Create output record
            output_record = base_record.copy()
            if strip_context or include_context:
                # strip_context: no context anywhere.
                # include_context: per-seed context is embedded inside each solution block,
                # so clear the top-level field to avoid duplicating seed-0's retrieval.
                output_record["context"] = ""
            output_record["problem"] = problem
            output_record["solutions"] = merged_solutions
            output_record["generations_list"] = generations
            output_record["reasonings_list"] = reasonings
            output_record["max_idx"] = len(generations) - 1
            output_record["num_solutions"] = len(generations)
            if include_context:
                output_record["contexts_list"] = contexts
            if shuffle_solutions:
                # Maps shuffled position → the candidate's logical index before
                # shuffling. Exact rollout recovery uses candidate_references below.
                output_record["original_seed_indices"] = seed_indices
            output_record["candidate_references"] = candidate_references

            # Add individual indexed fields for each solution
            for i, (generation, reasoning, ctx) in enumerate(
                zip(generations, reasonings, contexts, strict=False)
            ):
                output_record[f"answer_{i}"] = generation
                output_record[f"answer_reasoning_content_{i}"] = reasoning
                if include_context:
                    output_record[f"answer_context_{i}"] = ctx

            # Add to buffer
            buffer.append(orjson.dumps(output_record))

            # Write buffer when it reaches threshold
            if len(buffer) >= WRITE_BUFFER_SIZE:
                out.write(b"\n".join(buffer) + b"\n")
                buffer.clear()

        # Write remaining buffer
        if buffer:
            out.write(b"\n".join(buffer) + b"\n")

    written = len(data) - total_skipped_long
    logger.info(f"\n✓ Complete! Written {written} merged records to {output_file}")
    if total_skipped_long:
        logger.info(
            f"  Skipped {total_skipped_long} problems whose total prompt exceeded {MAX_TOTAL_CHARS:,} chars"
        )
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
    parser.add_argument(
        "--input_dir", help="Directory containing output-rs*.jsonl files (single-run)"
    )
    parser.add_argument(
        "--input_files",
        nargs="+",
        help=(
            "Explicit list of rollout JSONL files to merge, in seed order. Use this for "
            "multi-model rollouts (e.g. 4 DeepSeek files + 4 Qwen files) or any set of files "
            "not co-located in one --input_dir. Order = seed index. Overrides --input_dir."
        ),
    )
    parser.add_argument("--output_file", help="Path for merged output JSONL file")
    parser.add_argument(
        "--strip_context",
        action="store_true",
        help="Set context to empty string in all output records",
    )
    parser.add_argument(
        "--shuffle_solutions",
        action="store_true",
        help="Randomly shuffle solution order per problem to test position bias",
    )
    parser.add_argument(
        "--shuffle_seed", type=int, default=42, help="RNG seed for shuffle_solutions (default: 42)"
    )
    parser.add_argument(
        "--include_reasoning",
        action="store_true",
        help="Include each seed's reasoning chain in the solution block for GenSelect",
    )
    parser.add_argument(
        "--include_context",
        action="store_true",
        help=(
            "Embed each seed's SEC retrieval context inside its solution block as "
            "<context>...</context>. All 8 seeds' retrieved filings are shown to GenSelect "
            "rather than only seed-0's context at the top level. "
            f"Problems whose total prompt exceeds {MAX_TOTAL_CHARS:,} chars are skipped."
        ),
    )
    args = parser.parse_args()

    # Two ways to supply rollouts:
    #   --input_files  explicit ordered list (multi-model: 4 DSV4 + 4 Qwen, etc.)
    #   --input_dir    glob output-rs*.jsonl from one directory (single-run baseline)
    if args.input_files:
        input_files = list(args.input_files)  # explicit order defines seed index
    elif args.input_dir:
        input_files = sorted(glob.glob(os.path.join(args.input_dir, "output-rs*.jsonl")))
    else:
        parser.error("provide either --input_files or --input_dir")

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
        merge_jsonl_files(
            input_files,
            output_file,
            strip_context=args.strip_context,
            shuffle_solutions=args.shuffle_solutions,
            shuffle_seed=args.shuffle_seed,
            include_reasoning=args.include_reasoning,
            include_context=args.include_context,
        )
    except Exception as e:
        logger.error(f"\n✗ Error: {e}")
        sys.exit(1)
