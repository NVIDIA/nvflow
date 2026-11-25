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
"""Split a JSONL dataset into train and validation sets, optionally stratified.

Memory-efficient streaming approach:
1. Pass 1: Count records per category, apply token length filter
2. Pass 2: Stream-write to train/val files
3. Shuffle or sort output files (curriculum ordering via --sort_by)

With --splits, --group_by, or several --stratify_by fields, whole groups are
assigned instead: each stratum's rows are shared out so every split's running
total stays nearest its ratio, and the stratum's groups, ordered by a seeded hash
of their --group_by or --id_field value, fill the splits in turn. --splits writes
NAME.jsonl per split plus manifest.jsonl.
"""

import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

from nvflow.stages.data.split_contract import (
    NAMED_SPLIT_MARKER_NAME,
    SPLIT_MANIFEST_NAME,
    SPLIT_MARKER_NAME,
    publish_named_split_marker,
    publish_split_marker,
)
from nvflow.utils import setup_logger

logger = setup_logger(__name__)


def filter_record(
    record: dict,
    fields: list[str] | None = None,
    token_length_field: str | None = None,
) -> dict:
    """Project a record onto *fields*, or keep it as-is when *fields* is ``None``.

    Missing fields become ``0`` for the token-length field and ``""`` otherwise.
    """
    if fields is None:
        return record
    return {k: record.get(k, 0 if k == token_length_field else "") for k in fields}


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
    stratify_field: str | None,
    val_ratio: float,
    seed: int,
    max_tokens: int | None = None,
    token_length_field: str | None = None,
    fields: list[str] | None = None,
    sort_by: str | None = None,
    sort_order: str = "desc",
) -> tuple[int, int, int]:
    """Memory-efficient stratified split with optional token length filtering."""
    if max_tokens is not None and not token_length_field:
        raise ValueError("max_tokens requires token_length_field")
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    def keep(rec: dict) -> bool:
        return max_tokens is None or rec.get(token_length_field, 0) <= max_tokens

    def category(rec: dict) -> str:
        return rec.get(stratify_field, "<missing>") if stratify_field else "<all>"

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
                counts[category(record)] += 1
            else:
                filtered += 1

    total = sum(counts.values())
    logger.info(f"Found {total:,} records across {len(counts)} categories")
    if filtered:
        logger.info(f"Filtered {filtered:,} records exceeding {max_tokens:,} tokens")

    # Build validation indices per category
    logger.info(f"\nSplit by '{stratify_field}':" if stratify_field else "\nUnstratified split:")
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

            cat = category(record)
            idx = cat_idx[cat]
            cat_idx[cat] += 1
            projected = filter_record(record, fields, token_length_field)
            out = json.dumps(projected, ensure_ascii=False) + "\n"

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


def _rows(input_file: Path):
    """Yield ``(row_number, record)`` for non-empty lines."""
    with open(input_file, encoding="utf-8") as f:
        index = 0
        for line in f:
            if line.strip():
                yield index, json.loads(line)
                index += 1


def _stratum(record: dict, fields: list[str]) -> str:
    if not fields:
        return "<all>"
    values = [_resolve_nested(record, field) for field in fields]
    if len(values) == 1:
        return "<missing>" if values[0] is None else str(values[0])
    return json.dumps(values, ensure_ascii=False, default=str)


def _group(record: dict, field: str | None, index: int) -> str:
    value = _resolve_nested(record, field) if field else None
    if value is None or value == "":
        return f"row:{index}"
    return "group:" + json.dumps(value, ensure_ascii=False, default=str)


def _order_key(record: dict, group: str, id_field: str | None) -> str:
    value = _resolve_nested(record, id_field) if id_field and group.startswith("row:") else None
    if value is None or value == "":
        return group
    return "id:" + json.dumps(value, ensure_ascii=False, default=str)


def _quotas(
    size: int,
    splits: dict[str, float],
    totals: dict[str, int],
    candidates: list[str],
    rng: random.Random,
) -> dict[str, int]:
    """Rows per split for a stratum of *size* rows, keeping each split's running total
    nearest its ratio; ties go to the splits in a seeded order."""
    target = sum(totals.values()) + size
    order = list(candidates)
    rng.shuffle(order)
    quotas = dict.fromkeys(candidates, 0)
    for _ in range(size):
        name = max(order, key=lambda n: splits[n] * target - totals[n] - quotas[n])
        quotas[name] += 1
    return quotas


def perform_group_split(
    input_file: Path,
    output_dir: Path,
    splits: dict[str, float],
    *,
    stratify_fields: list[str] | None = None,
    group_field: str | None = None,
    seed: int = 42,
    max_tokens: int | None = None,
    token_length_field: str | None = None,
    fields: list[str] | None = None,
    id_field: str | None = None,
    write_manifest: bool = False,
) -> tuple[dict[str, int], int]:
    """Assign whole groups to named splits, keeping each stratum's split shares.

    A group (records sharing ``group_field``; otherwise each record alone) takes
    the stratum of its first record. Within a stratum, groups fill the splits in
    turn, ordered by a seeded hash of their ``group_field`` or ``id_field`` value,
    or of their row number when neither is set. Returns rows per split and
    filtered rows.
    """
    if max_tokens is not None and not token_length_field:
        raise ValueError("max_tokens requires token_length_field")
    names = list(splits)
    output_dir.mkdir(parents=True, exist_ok=True)

    def keep(rec: dict) -> bool:
        return max_tokens is None or rec.get(token_length_field, 0) <= max_tokens

    sizes: dict[str, int] = defaultdict(int)
    strata: dict[str, list[str]] = defaultdict(list)
    ranks: dict[str, tuple[str, int]] = {}
    filtered = 0
    for index, record in _rows(input_file):
        if not keep(record):
            filtered += 1
            continue
        group = _group(record, group_field, index)
        if group not in sizes:
            strata[_stratum(record, stratify_fields or [])].append(group)
            key = _order_key(record, group, id_field)
            ranks[group] = (hashlib.sha256(f"{seed}\n{key}".encode()).hexdigest(), index)
        sizes[group] += 1

    logger.info(f"Assigning {len(sizes):,} groups across {len(strata)} strata:")
    candidates = [name for name in names if splits[name] > 0]
    assignment: dict[str, str] = {}
    totals = dict.fromkeys(names, 0)
    for stratum, groups in sorted(strata.items()):
        size = sum(sizes[group] for group in groups)
        quotas = _quotas(size, splits, totals, candidates, random.Random(f"{seed}\n{stratum}"))
        assigned = dict.fromkeys(names, 0)
        current = 0
        for group in sorted(groups, key=ranks.__getitem__):
            while (
                current < len(candidates) - 1
                and assigned[candidates[current]] >= quotas[candidates[current]]
            ):
                current += 1
            assignment[group] = candidates[current]
            assigned[candidates[current]] += sizes[group]
        for name in names:
            totals[name] += assigned[name]
        logger.info(f"  {stratum:<20}: " + " | ".join(f"{assigned[n]:>6} {n}" for n in names))
    for name in candidates:
        if not totals[name]:
            logger.warning(f"Split {name!r} has no rows")

    counts = dict.fromkeys(names, 0)
    handles = {name: open(output_dir / f"{name}.jsonl", "w", encoding="utf-8") for name in names}
    manifest = (
        open(output_dir / SPLIT_MANIFEST_NAME, "w", encoding="utf-8") if write_manifest else None
    )
    try:
        for index, record in _rows(input_file):
            if not keep(record):
                continue
            name = assignment[_group(record, group_field, index)]
            projected = filter_record(record, fields, token_length_field)
            handles[name].write(json.dumps(projected, ensure_ascii=False) + "\n")
            counts[name] += 1
            if manifest is not None:
                identifier = _resolve_nested(record, id_field) if id_field else index
                entry = {"row": index, "id": identifier, "split": name}
                manifest.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    finally:
        for handle in handles.values():
            handle.close()
        if manifest is not None:
            manifest.close()
    for position, name in enumerate(names):
        if counts[name]:
            shuffle_file(output_dir / f"{name}.jsonl", seed + position)
    return counts, filtered


def _parse_splits(values: list[str], parser: argparse.ArgumentParser) -> dict[str, float]:
    splits: dict[str, float] = {}
    for value in values:
        name, separator, ratio = value.partition("=")
        try:
            share = float(ratio)
        except ValueError:
            share = -1.0
        if not separator or not name or name in splits or not math.isfinite(share) or share < 0:
            parser.error(f"--splits expects distinct NAME=RATIO pairs, got {value!r}")
        splits[name] = share
    if len(splits) < 2 or abs(sum(splits.values()) - 1) > 1e-6:
        parser.error("--splits needs at least two splits whose ratios sum to 1")
    return splits


def main():
    parser = argparse.ArgumentParser(description="Split dataset with stratification")
    parser.add_argument("input_file", help="Input JSONL file")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio")
    parser.add_argument(
        "--stratify_by",
        nargs="+",
        help="Field(s) to stratify by; several fields stratify by their combination. "
        "Default: one stratum.",
    )
    parser.add_argument(
        "--group_by",
        help="Keep records with the same value of this field in one split.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        metavar="NAME=RATIO",
        help="Named splits instead of train/val: writes NAME.jsonl per split, "
        "manifest.jsonl, and splits.done.",
    )
    parser.add_argument(
        "--id_field",
        help="Record field written as the id in manifest.jsonl (default: row number).",
    )
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_token_length", type=int, help="Filter records exceeding this")
    parser.add_argument(
        "--token_length_field",
        help="Numeric record field compared with --max_token_length.",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        help="Fields to keep, in output order. Default: keep every field.",
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
    if args.max_token_length is not None and not args.token_length_field:
        parser.error("--max_token_length requires --token_length_field")
    splits = _parse_splits(args.splits, parser) if args.splits else None
    if splits is not None and args.sort_by:
        parser.error("--sort_by applies to train/val splits, not --splits")
    stratify = args.stratify_by or []

    logger.info("=" * 60)
    logger.info("TRAIN/VALIDATION SPLIT" if splits is None else "NAMED SPLIT")
    logger.info("=" * 60)
    logger.info(f"Input:       {args.input_file}")
    logger.info(f"Output:      {args.output_dir}")
    if splits is None:
        logger.info(f"Val ratio:   {args.val_ratio:.1%}")
    else:
        logger.info(f"Splits:      {splits}")
    logger.info(f"Stratify:    {', '.join(stratify) or None}")
    if args.group_by:
        logger.info(f"Group by:    {args.group_by}")
    logger.info(f"Seed:        {args.random_seed}")
    if args.max_token_length:
        logger.info(f"Max tokens:  {args.max_token_length:,}")
    if args.sort_by:
        logger.info(f"Sort by:     {args.sort_by} ({args.sort_order})")
    logger.info("")

    output_dir = Path(args.output_dir)
    group_options = {
        "stratify_fields": stratify,
        "group_field": args.group_by,
        "seed": args.random_seed,
        "max_tokens": args.max_token_length,
        "token_length_field": args.token_length_field,
        "fields": args.fields,
    }
    if splits is not None:
        (output_dir / NAMED_SPLIT_MARKER_NAME).unlink(missing_ok=True)
        counts, filtered_n = perform_group_split(
            Path(args.input_file),
            output_dir,
            splits,
            id_field=args.id_field,
            write_manifest=True,
            **group_options,
        )
        total = sum(counts.values())
        if total == 0:
            raise ValueError("named split produced zero usable rows")
        for name, count in counts.items():
            logger.info(f"{name + ':':<9} {count:,} ({count / total:.1%})")
        if filtered_n:
            logger.info(f"Filtered: {filtered_n:,}")
        marker = publish_named_split_marker(output_dir, list(splits))
        logger.info(f"✅ Completion: {marker}")
        return 0

    (output_dir / SPLIT_MARKER_NAME).unlink(missing_ok=True)
    if args.group_by or len(stratify) > 1:
        counts, filtered_n = perform_group_split(
            Path(args.input_file),
            output_dir,
            {"train": 1 - args.val_ratio, "val": args.val_ratio},
            **group_options,
        )
        train_n, val_n = counts["train"], counts["val"]
        if args.sort_by and train_n:
            sort_file(output_dir / "train.jsonl", args.sort_by, args.sort_order == "desc")
    else:
        train_n, val_n, filtered_n = perform_split(
            Path(args.input_file),
            output_dir,
            stratify[0] if stratify else None,
            args.val_ratio,
            args.random_seed,
            args.max_token_length,
            token_length_field=args.token_length_field,
            fields=args.fields,
            sort_by=args.sort_by,
            sort_order=args.sort_order,
        )

    total = train_n + val_n
    if total == 0:
        raise ValueError("train/validation split produced zero usable rows")
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
    marker = publish_split_marker(output_dir)
    logger.info(f"✅ Output: {args.output_dir}")
    logger.info(f"✅ Completion: {marker}")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
