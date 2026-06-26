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
"""Transform SEC-QUE dataset to standard training format."""

import argparse
import json
import re
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from nvflow.utils import setup_logger

# Initialize logger
logger = setup_logger(__name__)

# Format descriptions for logging
SOURCE_FORMAT_DESC = {
    "separated": "reasoning model SDG (separate fields)",
    "think_tags": "reasoning model SDG (<think> tags in answer)",
    "inline": "non-reasoning model SDG (keep as-is)",
}

REASONING_MODE_DESC = {
    "thinking": "<think>\\n{reasoning}</think>\\n\\n{answer}",
    "natural": "{reasoning}\\n\\n{answer}",
    "none": "{answer} only",
}


def generate_uuid(problem: str, generation: str) -> str:
    """Generate a deterministic UUID based on problem and generation content.

    Uses UUID5 (SHA-1 based, namespace-based) to create a standard RFC 4122
    compliant UUID that is deterministic based on content.

    Args:
        problem: Problem/question text
        generation: Generation/answer text

    Returns:
        Standard UUID string (RFC 4122 compliant, 36 characters)
    """
    content = f"{problem}||{generation}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, content))


def transform_record(
    record: dict[str, Any],
    source_format: str = "separated",
    reasoning_mode: str = "none",
) -> tuple[dict[str, Any] | None, str | None]:
    """Transform a single record to the standard 6-field output format.

    Normalizes to: uuid, problem, context, reasoning_content, generation, question_type

    Args:
        record: Input record with fields: problem, context, generation,
            question_type, and optionally reasoning fields.
        source_format: How the source SDG data is structured:
            - "separated": Reasoning in answer_reasoning_content_{idx} or reasoning_content
            - "think_tags": Reasoning in <think> tags within generation
            - "inline": No reasoning separation needed
        reasoning_mode: How to format generation for training:
            - "thinking": Wrap reasoning in <think> tags (Qwen3 style)
            - "natural": Combine reasoning + answer without tags
            - "none": Answer only, discard reasoning

    Returns:
        (transformed_record, None) on success, or (None, error_message) on failure.
    """
    problem = record.get("problem")
    if not problem:
        return None, "Missing required field: problem"

    context = record.get("context")
    if not context:
        return None, "Missing required field: context"

    raw_generation = record.get("generation") or record.get("answer")
    if not raw_generation:
        return None, "Missing required field: generation (or answer)"

    # Reasoning field (for separated format)
    if source_format == "separated":
        selected_index = record.get("selected_index")
        has_indexed_fields = any(k.startswith("answer_reasoning_content_") for k in record)

        if has_indexed_fields:
            # Model reasoning is per-answer in answer_reasoning_content_{idx}.
            # Top-level reasoning_content is the judge's reasoning -- do NOT use it.
            if selected_index is None:
                return (
                    None,
                    "answer_reasoning_content_* fields present but selected_index is missing",
                )
            raw_reasoning = record.get(f"answer_reasoning_content_{selected_index}", "")
            if not raw_reasoning:
                return None, f"answer_reasoning_content_{selected_index} is empty or missing"
        else:
            raw_reasoning = record.get("reasoning_content", "")
            if not raw_reasoning:
                return None, "Missing required field: reasoning_content"
    else:
        raw_reasoning = ""

    # Question type (required for stratification)
    question_type = record.get("question_type", "unknown")

    # Parse source data based on source_format
    if source_format == "separated":
        reasoning = raw_reasoning
        answer = raw_generation

    elif source_format == "think_tags":
        # Parse <think>...</think> tags from answer
        match = re.search(r"<think>(.*?)</think>\s*(.*)", raw_generation, re.DOTALL)
        if match:
            reasoning = match.group(1).strip()
            answer = match.group(2).strip()
        else:
            # No <think> tags found - treat as answer only
            reasoning = ""
            answer = raw_generation

    else:  # inline
        # Non-reasoning SDG - keep as-is, no parsing
        reasoning = ""
        answer = raw_generation

    reasoning_stripped = reasoning.rstrip("\n")

    # Apply reasoning mode transformation (output format)
    if reasoning_mode == "thinking" and reasoning_stripped:
        # Wrap reasoning in <think> tags with newline for cross-model compatibility
        # Note: newline after <think> required for models like Nemotron-3-Nano-30B
        final_generation = f"<think>\n{reasoning_stripped}</think>\n\n{answer}"
    elif reasoning_mode == "natural" and reasoning_stripped:
        # Combine reasoning + answer without special tags
        final_generation = f"{reasoning_stripped}\n\n{answer}"
    else:
        # "none" or no reasoning available - answer only
        final_generation = answer

    # Generate uuid based on problem and generation content
    record_uuid = generate_uuid(problem, final_generation)

    # Build minimal output record (6 fields only)
    output = {
        "uuid": record_uuid,
        "problem": problem,
        "context": context,
        "reasoning_content": reasoning_stripped,
        "generation": final_generation,
        "question_type": question_type,
    }

    return output, None


def compute_length_statistics(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Compute length statistics for key fields."""
    fields = ["problem", "context", "reasoning_content", "generation"]
    summary = {}

    for field in fields:
        lengths = [len(str(r[field])) for r in records if r.get(field) is not None]
        missing = len(records) - len(lengths)

        if lengths:
            lengths_sorted = sorted(lengths)
            n = len(lengths)
            summary[field] = {
                "count": n,
                "missing": missing,
                "min": min(lengths),
                "max": max(lengths),
                "mean": sum(lengths) / n,
                "median": lengths_sorted[n // 2],
                "p95": lengths_sorted[int(n * 0.95)],
                "p99": lengths_sorted[int(n * 0.99)],
            }
        else:
            summary[field] = dict.fromkeys(
                ["count", "min", "max", "mean", "median", "p95", "p99"], 0
            )
            summary[field]["missing"] = missing

    return summary


def compute_question_type_distribution(records: list[dict[str, Any]]) -> dict[str, int]:
    """Compute distribution of question types."""
    return dict(Counter(r.get("question_type", "unknown") for r in records))


def compute_percentile(values: list[float], percentile: float) -> float:
    """Compute a percentile from a list of values.

    Args:
        values: List of numeric values
        percentile: Percentile to compute (0-100)

    Returns:
        Value at the specified percentile
    """
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int(len(sorted_values) * (percentile / 100.0))
    return sorted_values[min(index, len(sorted_values) - 1)]


def should_filter_record(
    record: dict[str, Any],
    context_min: float,
    context_max: float,
    reasoning_min: float,
    reasoning_max: float,
) -> tuple[bool, str | None]:
    """Check if record should be filtered based on length thresholds.

    Args:
        record: Transformed record to check
        context_min: Minimum context length
        context_max: Maximum context length
        reasoning_min: Minimum reasoning length
        reasoning_max: Maximum reasoning length

    Returns:
        Tuple of (should_filter, reason)
    """
    context = str(record.get("context", ""))
    reasoning = str(record.get("reasoning_content", ""))

    context_len = len(context)
    reasoning_len = len(reasoning)

    if context_len < context_min:
        return True, f"Context too short: {context_len} < {context_min:.0f}"

    if context_len > context_max:
        return True, f"Context too long: {context_len} > {context_max:.0f}"

    if reasoning_len < reasoning_min:
        return True, f"Reasoning too short: {reasoning_len} < {reasoning_min:.0f}"

    if reasoning_len > reasoning_max:
        return True, f"Reasoning too long: {reasoning_len} > {reasoning_max:.0f}"

    return False, None


def main():
    """Transform dataset from SEC-QUE format to training format."""
    parser = argparse.ArgumentParser(description="Transform SEC-QUE dataset")
    parser.add_argument("input_files", type=str, nargs="+", help="Input JSONL file(s)")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file")
    parser.add_argument(
        "--num_chunks",
        type=int,
        default=1,
        help="Number of chunks to split output into (default: 1). Chunks always saved to {output_dir}/chunks/",
    )
    parser.add_argument(
        "--filter_outliers",
        action="store_true",
        help="Enable outlier filtering based on percentiles",
    )
    parser.add_argument(
        "--context_min_percentile",
        type=float,
        default=1.0,
        help="Context min percentile (default: 1.0)",
    )
    parser.add_argument(
        "--context_max_percentile",
        type=float,
        default=99.0,
        help="Context max percentile (default: 99.0)",
    )
    parser.add_argument(
        "--reasoning_min_percentile",
        type=float,
        default=1.0,
        help="Reasoning min percentile (default: 1.0)",
    )
    parser.add_argument(
        "--reasoning_max_percentile",
        type=float,
        default=99.0,
        help="Reasoning max percentile (default: 99.0)",
    )
    parser.add_argument(
        "--source_format",
        type=str,
        choices=["separated", "think_tags", "inline"],
        default="separated",
        help="Source data format: 'separated' (separate fields), 'think_tags' (<think> in answer), 'inline' (non-reasoning, keep as-is)",
    )
    parser.add_argument(
        "--reasoning_mode",
        type=str,
        choices=["thinking", "natural", "none"],
        default="none",
        help="Output format: 'thinking' (Qwen3 <think> tags), 'natural' (reasoning + answer), 'none' (answer only)",
    )
    parser.add_argument(
        "--deduplicate_by_uuid",
        action="store_true",
        help="Drop exact (problem, generation) dups after outlier filter. "
        "Opt-in; defaults to OFF to match legacy behavior (SFT + pre-dedup GRPO runs).",
    )

    args = parser.parse_args()

    # Block invalid combination: inline + thinking
    if args.source_format == "inline" and args.reasoning_mode == "thinking":
        logger.error(
            "Invalid combination: source_format='inline' + reasoning_mode='thinking'. "
            "Cannot reliably add <think> tags to non-reasoning model SDG. "
            "Use reasoning_mode='natural' or 'none' for inline source format."
        )
        sys.exit(1)

    input_paths = [Path(f) for f in args.input_files]
    output_path = Path(args.output_file)

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process records
    transformed_records = []
    error_records = []
    total_records = 0

    logger.info(f"Processing {len(input_paths)} input file(s)")
    logger.info(f"Writing to: {output_path}")
    logger.info(f"Source format: {args.source_format} → {SOURCE_FORMAT_DESC[args.source_format]}")
    logger.info(
        f"Reasoning mode: {args.reasoning_mode} → {REASONING_MODE_DESC[args.reasoning_mode]}"
    )

    # Process each input file
    for file_idx, input_path in enumerate(input_paths, 1):
        logger.info(f"\n[File {file_idx}/{len(input_paths)}] Processing: {input_path}")

        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            continue

        file_records = 0
        with open(input_path, encoding="utf-8") as infile:
            for line_num, line in enumerate(infile, 1):
                total_records += 1
                file_records += 1
                try:
                    record = json.loads(line.strip())
                    transformed, error = transform_record(
                        record,
                        source_format=args.source_format,
                        reasoning_mode=args.reasoning_mode,
                    )

                    if transformed:
                        transformed_records.append(transformed)
                    else:
                        error_records.append(
                            {
                                "source_file": str(input_path),
                                "line_number": line_num,
                                "error": error,
                                "record": record,
                            }
                        )
                        logger.warning(f"[{input_path.name}:{line_num}] {error}")

                except json.JSONDecodeError as e:
                    error_msg = f"JSON decode error: {e}"
                    error_records.append(
                        {
                            "source_file": str(input_path),
                            "line_number": line_num,
                            "error": error_msg,
                            "raw_line": line,
                        }
                    )
                    logger.error(f"[{input_path.name}:{line_num}] {error_msg}")

        logger.info(f"  Processed {file_records:,} records from {input_path.name}")

    # Apply outlier filtering if enabled
    filtered_records = []

    if args.filter_outliers and transformed_records:
        logger.info("")
        logger.info("=" * 80)
        logger.info("OUTLIER FILTERING")
        logger.info("=" * 80)
        logger.info(f"Calculating percentiles from {len(transformed_records):,} records...")

        # Calculate percentiles from all transformed records
        context_lengths = [len(str(r.get("context", ""))) for r in transformed_records]
        reasoning_lengths = [len(str(r.get("reasoning_content", ""))) for r in transformed_records]

        context_min = compute_percentile(context_lengths, args.context_min_percentile)
        context_max = compute_percentile(context_lengths, args.context_max_percentile)
        reasoning_min = compute_percentile(reasoning_lengths, args.reasoning_min_percentile)
        reasoning_max = compute_percentile(reasoning_lengths, args.reasoning_max_percentile)

        logger.info("Context percentile thresholds:")
        logger.info(f"  P{args.context_min_percentile}: {context_min:,.0f} chars")
        logger.info(f"  P{args.context_max_percentile}: {context_max:,.0f} chars")
        logger.info("Reasoning percentile thresholds:")
        logger.info(f"  P{args.reasoning_min_percentile}: {reasoning_min:,.0f} chars")
        logger.info(f"  P{args.reasoning_max_percentile}: {reasoning_max:,.0f} chars")

        # Filter records based on percentile thresholds
        final_records = []
        for record in transformed_records:
            should_filter, reason = should_filter_record(
                record, context_min, context_max, reasoning_min, reasoning_max
            )

            if should_filter:
                filtered_records.append({"record": record, "reason": reason})
            else:
                final_records.append(record)

        pct_filtered = len(filtered_records) / len(transformed_records) * 100
        pct_kept = len(final_records) / len(transformed_records) * 100
        logger.info(f"Filtered out: {len(filtered_records):,} ({pct_filtered:.1f}%)")
        logger.info(f"Kept:         {len(final_records):,} ({pct_kept:.1f}%)")

        # Save filtered records to output directory
        if filtered_records:
            filtered_path = output_path.parent / "filtered_outliers.jsonl"
            with open(filtered_path, "w", encoding="utf-8") as filtfile:
                for filt in filtered_records:
                    filtfile.write(json.dumps(filt, ensure_ascii=False) + "\n")
            logger.info(f"Filtered outliers saved to: {filtered_path}")
    else:
        # No filtering - use all transformed records
        final_records = transformed_records

    # Opt-in dedup by uuid (itself = SHA-1(problem, final_generation)).
    num_dedup_dropped = 0
    dedup_dropped_records: list[dict[str, Any]] = []
    if args.deduplicate_by_uuid and final_records:
        logger.info("")
        logger.info("=" * 80)
        logger.info("DEDUPLICATION BY UUID")
        logger.info("=" * 80)
        seen_uuids: set[str] = set()
        deduped: list[dict[str, Any]] = []
        for record in final_records:
            uid = record.get("uuid")
            if uid and uid in seen_uuids:
                dedup_dropped_records.append(record)
                continue
            if uid:
                seen_uuids.add(uid)
            deduped.append(record)

        num_dedup_dropped = len(final_records) - len(deduped)
        pct_dropped = num_dedup_dropped / len(final_records) * 100
        logger.info(f"Input records:           {len(final_records):,}")
        logger.info(f"Unique uuids:            {len(deduped):,}")
        logger.info(f"Duplicates dropped:      {num_dedup_dropped:,} ({pct_dropped:.2f}%)")
        if pct_dropped > 10:
            logger.warning(
                "Duplicate rate %.2f%% exceeds 10%%; investigate upstream SDG.",
                pct_dropped,
            )
        final_records = deduped

        if dedup_dropped_records:
            dedup_path = output_path.parent / "duplicates.jsonl"
            with open(dedup_path, "w", encoding="utf-8") as dedup_file:
                for record in dedup_dropped_records:
                    dedup_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.info(f"Dropped duplicates saved to: {dedup_path}")

    # Write final records to chunks directory (always use chunking structure)
    num_chunks = args.num_chunks
    chunks_dir = output_path.parent / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    records_per_chunk = (len(final_records) + num_chunks - 1) // num_chunks

    logger.info(f"Writing {num_chunks} chunks (~{records_per_chunk:,} records each)")
    logger.info(f"Chunks directory: {chunks_dir}")

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * records_per_chunk
        end_idx = min(start_idx + records_per_chunk, len(final_records))
        chunk_records = final_records[start_idx:end_idx]

        if not chunk_records:
            continue

        chunk_file = chunks_dir / f"final_result_chunk{chunk_idx + 1}.jsonl"
        with open(chunk_file, "w", encoding="utf-8") as outfile:
            for record in chunk_records:
                outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(
            f"  → Chunk {chunk_idx + 1}: {len(chunk_records):,} records → {chunk_file.name}"
        )

    # Write error records if any
    if error_records:
        error_path = output_path.parent / "errors.jsonl"
        with open(error_path, "w", encoding="utf-8") as errfile:
            for error_record in error_records:
                errfile.write(json.dumps(error_record, ensure_ascii=False) + "\n")
        logger.info(f"Error records saved to: {error_path}")

    # Compute and display statistics
    logger.info("")
    logger.info("=" * 80)
    logger.info("TRANSFORMATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total records read:       {total_records}")
    logger.info(f"Successfully transformed: {len(transformed_records)}")
    logger.info(f"Errors encountered:       {len(error_records)}")
    if args.filter_outliers:
        logger.info(f"Outliers filtered:        {len(filtered_records)}")
    if args.deduplicate_by_uuid:
        logger.info(f"Duplicates (uuid) dropped: {num_dedup_dropped}")
    logger.info(f"Final records:            {len(final_records)}")

    # Statistics on final records (after filtering if enabled)
    if final_records:
        suffix = " (After Filtering)" if args.filter_outliers else ""

        logger.info("")
        logger.info("=" * 80)
        logger.info(f"LENGTH STATISTICS{suffix}")
        logger.info("=" * 80)

        stats = compute_length_statistics(final_records)
        for field in ["problem", "context", "reasoning_content", "generation"]:
            s = stats[field]
            logger.info(f"\n{field.upper()}:")
            logger.info(f"  Count:   {s['count']:,}")
            logger.info(f"  Missing: {s['missing']:,}")
            logger.info(f"  Min:     {s['min']:,} chars")
            logger.info(f"  Max:     {s['max']:,} chars")
            logger.info(f"  Mean:    {s['mean']:,.1f} chars")
            logger.info(f"  Median:  {s['median']:,} chars")
            logger.info(f"  P95:     {s['p95']:,} chars")
            logger.info(f"  P99:     {s['p99']:,} chars")

        # Question type distribution
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"QUESTION TYPE DISTRIBUTION{suffix}")
        logger.info("=" * 80)

        question_type_dist = compute_question_type_distribution(final_records)
        total = sum(question_type_dist.values())
        for qtype, count in sorted(question_type_dist.items(), key=lambda x: -x[1]):
            pct = (count / total * 100) if total > 0 else 0
            logger.info(f"  - {qtype:<20}: {count:>8} ({pct:>5.1f}%)")

    logger.info("")
    logger.info("=" * 80)
    logger.info("Transformation complete!")
    logger.info(f"Output: {output_path.parent}/chunks/ ({num_chunks} chunks)")
    logger.info("=" * 80)

    if error_records:
        error_rate = len(error_records) / total_records if total_records > 0 else 1.0
        if error_rate > 0.05:
            logger.error(
                f"Error rate {error_rate:.1%} ({len(error_records)}/{total_records}) "
                f"exceeds 5% threshold"
            )
            return 1
        logger.warning(
            f"{len(error_records)} error(s) ({error_rate:.2%} of {total_records}) "
            f"— below 5% threshold, treating as success"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
