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
"""Bucket training data by total sequence length (input + output).

If total_token_length is pre-computed (from prepare_for_sft), bucketing is fast.
Otherwise, computes token lengths using tokenizer.

Usage:
    python -m nvfinance.utils.sft.sequence_batcher \
        input.jsonl --output_dir /path/to/output --bucket_sizes 16000 32000 64000
"""

import argparse
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import orjson

from nvflow.utils import setup_logger

logger = setup_logger(__name__)


def process_line(line: str, tokenizer: Any = None) -> tuple[dict, int] | None:
    """Process a JSONL line. Uses pre-computed length if available."""
    line = line.strip()
    if not line:
        return None

    obj = orjson.loads(line)

    # Fast path: use pre-computed length
    if "total_token_length" in obj:
        return obj, obj["total_token_length"]

    # Slow path: compute with tokenizer
    if tokenizer is None:
        raise ValueError("Record missing total_token_length and no tokenizer provided")

    in_text, out_text = obj.get("input", ""), obj.get("output", "")
    in_len = len(tokenizer(in_text, add_special_tokens=False)["input_ids"]) if in_text else 0
    out_len = len(tokenizer(out_text, add_special_tokens=False)["input_ids"]) if out_text else 0
    obj["total_token_length"] = in_len + out_len

    return obj, obj["total_token_length"]


def bucket_for_length(length: int, sizes: list[int]) -> int | str:
    """Return bucket key for given length."""
    for size in sizes:
        if length <= size:
            return size
    return "overflow"


def process_file(
    input_path: Path,
    output_dir: Path,
    bucket_sizes: list[int],
    tokenizer: Any = None,
) -> dict[int | str, int]:
    """Process JSONL and write to bucket files."""
    # Setup buckets
    keys: list[int | str] = bucket_sizes + ["overflow"]
    paths = {k: output_dir / f"{input_path.stem}_bucket_{k}.jsonl" for k in keys}
    counts: dict[int | str, int] = dict.fromkeys(keys, 0)

    processed = failed = 0
    start = time.time()

    with open(input_path, encoding="utf-8") as f_in, ExitStack() as stack:
        handles = {k: stack.enter_context(open(p, "w", encoding="utf-8")) for k, p in paths.items()}

        for line in f_in:
            try:
                result = process_line(line, tokenizer)
                if result is None:
                    continue

                obj, length = result
                key = bucket_for_length(length, bucket_sizes)
                handles[key].write(orjson.dumps(obj).decode("utf-8") + "\n")
                counts[key] += 1
                processed += 1

                if processed % 50000 == 0:
                    rate = processed / (time.time() - start)
                    logger.info(f"Processed {processed:,} | {rate:.0f}/sec")

            except Exception as e:
                failed += 1
                if failed <= 5:
                    logger.error(f"Error: {e}")

    logger.info(f"Done: {processed:,} processed, {failed:,} failed")
    return counts


def main():
    parser = argparse.ArgumentParser(description="Bucket training data by sequence length")
    parser.add_argument("input_file", help="Input JSONL file")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--bucket_sizes", nargs="+", type=int, default=[16000, 32000, 64000])
    parser.add_argument(
        "--tokenizer_path", help="Optional tokenizer (if data lacks total_token_length)"
    )
    # Keep for backward compat but ignore
    parser.add_argument("--to_bucket", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()
    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer only if needed
    tokenizer = None
    if args.tokenizer_path:
        from transformers import AutoTokenizer

        logger.info(f"Loading tokenizer: {args.tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Buckets: {args.bucket_sizes}")

    counts = process_file(input_path, output_dir, args.bucket_sizes, tokenizer)

    # Log distribution
    logger.info("")
    logger.info("=" * 50)
    logger.info("BUCKET DISTRIBUTION")
    logger.info("=" * 50)
    total = sum(counts.values())
    for key in args.bucket_sizes + ["overflow"]:
        count = counts[key]
        pct = count / total * 100 if total else 0
        logger.info(f"  {str(key):<12}: {count:>8,} ({pct:>5.1f}%)")
    logger.info("=" * 50)
    logger.info(f"✅ Output: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
