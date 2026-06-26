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
"""Split dataset into train and validation sets with stratification.

Memory-efficient streaming approach:
1. Pass 1: Count records per category, apply token length filter
2. Pass 2: Stream-write to train/val files
3. Shuffle or sort output files (curriculum ordering via --sort_by)
"""

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from nvflow.utils import setup_logger

logger = setup_logger(__name__)

_SFT_OUTPUT_FIELDS = ["input", "output", "uuid", "total_token_length"]


def filter_record(record: dict, output_fields: list[str] | None = None) -> dict:
    """Extract selected fields from a record.

    When *output_fields* is ``None`` all fields are kept as-is (GRPO mode).
    """
    if output_fields is None:
        return record
    return {k: record.get(k, 0 if k == "total_token_length" else "") for k in output_fields}


def _resolve_nested(record: dict, dotted_key: str):
    """Traverse a nested dict via dot-notation (e.g. 'difficulty_profile.avg_reward')."""
    obj = record
    for part in dotted_key.split("."):
        if isinstance(obj, dict):
            obj = obj.get(part)
        else:
            return None
    return obj


def shuffle_file(filepath: Path, seed: int) -> None:
    """Shuffle a JSONL file in place."""
    with open(filepath, encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    random.Random(seed).shuffle(lines)
    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(line if line.endswith("\n") else line + "\n" for line in lines)


def sort_file(filepath: Path, sort_by: str, descending: bool = True) -> None:
    """Sort a JSONL file in place by a (possibly nested) numeric field.

    Records missing the field are placed at the end regardless of sort order.
    """
    with open(filepath, encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]

    sentinel = float("-inf") if descending else float("inf")

    def sort_key(line: str):
        val = _resolve_nested(json.loads(line), sort_by)
        return val if isinstance(val, int | float) else sentinel

    lines.sort(key=sort_key, reverse=descending)

    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(line if line.endswith("\n") else line + "\n" for line in lines)

    n_missing = sum(
        1
        for line in lines
        if not isinstance(_resolve_nested(json.loads(line), sort_by), int | float)
    )
    logger.info(f"Sorted {len(lines):,} records by '{sort_by}' ({'desc' if descending else 'asc'})")
    if n_missing:
        logger.info(f"  {n_missing:,} records missing '{sort_by}' placed at end")


def perform_split(
    input_file: Path,
    output_dir: Path,
    stratify_field: str,
    val_ratio: float,
    seed: int,
    max_tokens: int | None = None,
    keep_all_fields: bool = False,
    sort_by: str | None = None,
    sort_order: str = "desc",
) -> tuple[int, int, int]:
    """Memory-efficient stratified split with optional token length filtering."""
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    def keep(rec: dict) -> bool:
        return max_tokens is None or rec.get("total_token_length", 0) <= max_tokens

    # Pass 1: Count per category
    logger.info("Pass 1: Counting records per category...")
    counts: dict[str, int] = defaultdict(int)
    filtered = 0

    with open(input_file, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if keep(record):
                counts[record.get(stratify_field, "<missing>")] += 1
            else:
                filtered += 1

    total = sum(counts.values())
    logger.info(f"Found {total:,} records across {len(counts)} categories")
    if filtered:
        logger.info(f"Filtered {filtered:,} records exceeding {max_tokens:,} tokens")

    if total == 0:
        raise ValueError(
            f"No records to split: {input_file} yielded 0 records after token-length "
            f"filtering (filtered={filtered}, max_tokens={max_tokens}). Cannot create a "
            f"train/validation split from an empty dataset — check the upstream stage output."
        )

    # Build validation indices per category
    logger.info(f"\nSplit by '{stratify_field}':")
    val_indices: dict[str, set[int]] = {}
    for cat, n in sorted(counts.items()):
        n_val = max(1, int(n * val_ratio)) if val_ratio > 0 else 0
        logger.info(f"  {cat:<20}: {n:>6} total | {n - n_val:>6} train | {n_val:>6} val")
        rng = random.Random(seed + int(hashlib.md5(cat.encode()).hexdigest(), 16))
        idx = list(range(n))
        rng.shuffle(idx)
        val_indices[cat] = set(idx[:n_val])

    # Pass 2: Write splits
    logger.info("\nPass 2: Writing split files...")
    cat_idx: dict[str, int] = defaultdict(int)
    train_count = val_count = 0
    output_fields = None if keep_all_fields else _SFT_OUTPUT_FIELDS

    with (
        open(input_file, encoding="utf-8") as f_in,
        open(train_path, "w", encoding="utf-8") as f_train,
        open(val_path, "w", encoding="utf-8") as f_val,
    ):
        for line in f_in:
            if not line.strip():
                continue
            record = json.loads(line)
            if not keep(record):
                continue

            cat = record.get(stratify_field, "<missing>")
            idx = cat_idx[cat]
            cat_idx[cat] += 1
            out = json.dumps(filter_record(record, output_fields), ensure_ascii=False) + "\n"

            if idx in val_indices[cat]:
                f_val.write(out)
                val_count += 1
            else:
                f_train.write(out)
                train_count += 1

    # Sort or shuffle output files
    if sort_by:
        logger.info(f"Sorting train file by '{sort_by}' ({sort_order})...")
        sort_file(train_path, sort_by, descending=(sort_order == "desc"))
    else:
        logger.info("Shuffling output files...")
        shuffle_file(train_path, seed)
    if val_count > 0:
        shuffle_file(val_path, seed + 1)

    return train_count, val_count, filtered


def main():
    parser = argparse.ArgumentParser(description="Split dataset with stratification")
    parser.add_argument("input_file", help="Input JSONL file")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio")
    parser.add_argument("--stratify_by", default="question_type", help="Field to stratify by")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_token_length", type=int, help="Filter records exceeding this")
    parser.add_argument(
        "--keep_all_fields",
        action="store_true",
        help="Keep all input fields (GRPO). Default: keep only SFT fields.",
    )
    parser.add_argument(
        "--sort_by",
        help="Sort train file by this field instead of shuffling. "
        "Supports dot-notation for nested fields (e.g. 'difficulty_profile.avg_reward').",
    )
    parser.add_argument(
        "--sort_order",
        default="desc",
        choices=["asc", "desc"],
        help="Sort order when --sort_by is set (default: desc for easy-to-hard curriculum).",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("TRAIN/VALIDATION SPLIT")
    logger.info("=" * 60)
    logger.info(f"Input:       {args.input_file}")
    logger.info(f"Output:      {args.output_dir}")
    logger.info(f"Val ratio:   {args.val_ratio:.1%}")
    logger.info(f"Stratify:    {args.stratify_by}")
    logger.info(f"Seed:        {args.random_seed}")
    if args.max_token_length:
        logger.info(f"Max tokens:  {args.max_token_length:,}")
    if args.sort_by:
        logger.info(f"Sort by:     {args.sort_by} ({args.sort_order})")
    logger.info("")

    train_n, val_n, filtered_n = perform_split(
        Path(args.input_file),
        Path(args.output_dir),
        args.stratify_by,
        args.val_ratio,
        args.random_seed,
        args.max_token_length,
        keep_all_fields=args.keep_all_fields,
        sort_by=args.sort_by,
        sort_order=args.sort_order,
    )

    total = train_n + val_n
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total:    {total:,}")
    logger.info(f"Train:    {train_n:,} ({train_n / total:.1%})")
    if val_n:
        logger.info(f"Val:      {val_n:,} ({val_n / total:.1%})")
    if filtered_n:
        logger.info(f"Filtered: {filtered_n:,}")
    logger.info(f"✅ Output: {args.output_dir}")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
