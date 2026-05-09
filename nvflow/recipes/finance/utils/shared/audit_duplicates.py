#!/usr/bin/env python3
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
"""Audit a JSONL dataset for duplicates.

Reports three distinct notions of duplication, in increasing strictness:

1. ``uuid`` duplicates -- records sharing the deterministic uuid assigned
   by ``dataset_transformer.generate_uuid(problem, final_generation)``.
   Equivalent to same ``(problem, final_generation)``.

2. ``(problem, generation)`` duplicates -- the pair from which ``uuid``
   is derived; reported separately so this audit is useful on files
   that don't yet have a uuid field (e.g. SDG raw output, the
   prefiltered.jsonl inside validate_questions).

3. Same-``problem``-different-``generation`` clusters -- the same user
   question repeated with different candidate answers.  Typically
   indicates SDG over-generation or selection-index branching; not a
   "duplicate" that can be dropped silently but useful to quantify.

Read-only.  Does not modify input.
"""

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def _jsonl_rows(path: Path):
    """Yield rows from a JSONL file, skipping blank and malformed lines."""
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("  skipped malformed JSON at %s:%d: %s", path, line_num, exc)


def audit(path: Path, problem_field: str, generation_field: str) -> dict[str, Any]:
    """Compute duplicate statistics for *path* and log them.

    Returns the stats dict so callers can consume programmatically.
    """
    n_total = 0
    n_with_uuid = 0
    uuid_counts: Counter = Counter()
    pair_counts: Counter = Counter()
    problem_to_generations: dict[str, set[str]] = defaultdict(set)

    for row in _jsonl_rows(path):
        n_total += 1
        uid = row.get("uuid")
        if uid:
            n_with_uuid += 1
            uuid_counts[uid] += 1
        problem = row.get(problem_field, "")
        generation = row.get(generation_field, "")
        pair_counts[(problem, generation)] += 1
        if problem:
            problem_to_generations[problem].add(generation)

    uuid_dup_records = sum(c - 1 for c in uuid_counts.values() if c > 1)
    pair_dup_records = sum(c - 1 for c in pair_counts.values() if c > 1)
    problems_with_variants = {p: gs for p, gs in problem_to_generations.items() if len(gs) > 1}
    variant_record_count = sum(len(gs) for gs in problems_with_variants.values())

    # Cluster-size histograms (top 5 most duplicated keys)
    top_uuid_clusters = uuid_counts.most_common(5)
    top_pair_clusters = pair_counts.most_common(5)

    stats = {
        "path": str(path),
        "total_records": n_total,
        "records_with_uuid_field": n_with_uuid,
        # uuid duplicates
        "unique_uuids": len(uuid_counts),
        "uuid_duplicate_records": uuid_dup_records,
        "uuid_duplicate_pct": (uuid_dup_records / n_total * 100) if n_total else 0.0,
        # (problem, generation) pair duplicates
        "unique_problem_generation_pairs": len(pair_counts),
        "pair_duplicate_records": pair_dup_records,
        "pair_duplicate_pct": (pair_dup_records / n_total * 100) if n_total else 0.0,
        # same-problem-different-generation clusters
        "unique_problems": len(problem_to_generations),
        "problems_with_multiple_generations": len(problems_with_variants),
        "variant_record_count": variant_record_count,
        "variant_record_pct": (variant_record_count / n_total * 100 if n_total else 0.0),
        "top_uuid_clusters": top_uuid_clusters,
        "top_pair_clusters_count_only": [c for _, c in top_pair_clusters],
    }

    logger.info("")
    logger.info("=" * 80)
    logger.info("DUPLICATE AUDIT")
    logger.info("=" * 80)
    logger.info("Path:                           %s", path)
    logger.info("Total records:                  %d", n_total)
    logger.info("Records with uuid field:        %d", n_with_uuid)
    logger.info("")
    logger.info("-- uuid duplicates --")
    logger.info("Unique uuids:                   %d", len(uuid_counts))
    logger.info(
        "Duplicate records:              %d (%.2f%%)",
        uuid_dup_records,
        stats["uuid_duplicate_pct"],
    )
    if top_uuid_clusters:
        logger.info("Top uuid clusters (count x uuid):")
        for uid, count in top_uuid_clusters:
            if count > 1:
                logger.info("  %5d  %s", count, uid)
    logger.info("")
    logger.info("-- (problem, generation) duplicates --")
    logger.info(
        "Unique (problem, generation) pairs: %d",
        len(pair_counts),
    )
    logger.info(
        "Duplicate records:              %d (%.2f%%)",
        pair_dup_records,
        stats["pair_duplicate_pct"],
    )
    logger.info("")
    logger.info("-- same-problem-different-generation --")
    logger.info("Unique problems:                %d", len(problem_to_generations))
    logger.info(
        "Problems with >1 generation:    %d",
        len(problems_with_variants),
    )
    logger.info(
        "Records in variant clusters:    %d (%.2f%%)",
        variant_record_count,
        stats["variant_record_pct"],
    )

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Audit a JSONL file for duplicate records",
    )
    parser.add_argument("input_file", type=str, help="Path to input JSONL file")
    parser.add_argument(
        "--problem_field",
        default="problem",
        help='Field name for the "problem" content (default: problem)',
    )
    parser.add_argument(
        "--generation_field",
        default="generation",
        help='Field name for the "generation" content (default: generation; '
        'use "answer" for raw SDG that has not been through dataset_transformer)',
    )
    parser.add_argument(
        "--json_output",
        type=str,
        default=None,
        help="Optional path to write the stats dict as JSON (alongside stdout)",
    )
    args = parser.parse_args()

    path = Path(args.input_file)
    if not path.exists():
        raise SystemExit(f"Input file not found: {path}")

    stats = audit(path, args.problem_field, args.generation_field)

    if args.json_output:
        out_path = Path(args.json_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(stats, f, indent=2, default=str)
        logger.info("\nStats JSON -> %s", out_path)
