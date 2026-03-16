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
"""Add token counts to prepared SFT data.

Reads JSONL file, computes total_token_length (input + output), and writes back.
Uses multiprocessing for efficient tokenization on multi-core machines.

Usage:
    python -m nvflow.recipes.finance.utils.sft.add_token_counts \
        input.jsonl \
        --output_file output.jsonl \
        --tokenizer_path /hf_models/Qwen/Qwen3-14B
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

from nvflow.utils import setup_logger

logger = setup_logger(__name__)

# Global tokenizer for worker processes (initialized in _init_worker)
_tokenizer: Any = None


def _init_worker(tokenizer_path: str) -> None:
    """Initialize tokenizer in worker process (called once per worker)."""
    global _tokenizer
    from transformers import AutoTokenizer

    _tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def _process_record(line: str) -> tuple[str, int] | None:
    """Process a single record. Returns (json_string, total_token_length) or None."""
    line = line.strip()
    if not line:
        return None

    record = json.loads(line)
    in_text: str = record.get("input", "")
    out_text: str = record.get("output", "")

    in_len = len(_tokenizer(in_text, add_special_tokens=False)["input_ids"]) if in_text else 0
    out_len = len(_tokenizer(out_text, add_special_tokens=False)["input_ids"]) if out_text else 0
    total = in_len + out_len

    record["input_token_length"] = in_len
    record["output_token_length"] = out_len
    record["total_token_length"] = total

    return json.dumps(record, ensure_ascii=False), total


def add_token_counts(
    input_file: str, output_file: str, tokenizer_path: str, num_workers: int | None = None
) -> dict:
    """Add token length fields to each record using multiprocessing."""
    input_path = Path(input_file).resolve()
    output_path = Path(output_file).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine worker count: min(available_cpus - 2, user_specified)
    max_workers = max(1, cpu_count() - 2)
    num_workers = min(num_workers or max_workers, max_workers)

    logger.info(f"Tokenizer: {tokenizer_path}")
    logger.info(f"Workers: {num_workers} (max: {max_workers})")
    logger.info(f"Input: {input_file}")

    # Handle in-place update: write to temp file first
    same_file = input_path == output_path
    if same_file:
        fd, temp_path = tempfile.mkstemp(suffix=".jsonl", dir=output_path.parent)
        os.close(fd)
        actual_output = Path(temp_path)
    else:
        actual_output = output_path

    lengths: list[int] = []
    batch_size = 10000

    try:
        with Pool(num_workers, initializer=_init_worker, initargs=(tokenizer_path,)) as pool:
            with (
                open(input_path, encoding="utf-8") as f_in,
                open(actual_output, "w", encoding="utf-8") as f_out,
            ):
                batch: list[str] = []
                processed = 0

                def process_batch():
                    nonlocal processed
                    for result in pool.map(_process_record, batch):
                        if result:
                            json_str, token_len = result
                            lengths.append(token_len)
                            f_out.write(json_str + "\n")
                    processed += len(batch)
                    logger.info(f"Processed {processed:,} records...")

                for line in f_in:
                    batch.append(line)
                    if len(batch) >= batch_size:
                        process_batch()
                        batch = []

                if batch:
                    process_batch()

        if same_file:
            shutil.move(str(actual_output), str(output_path))

    except Exception:
        if same_file and actual_output.exists():
            actual_output.unlink()
        raise

    if not lengths:
        return {}

    # Log statistics
    lengths.sort()
    n = len(lengths)

    def pct(p: int) -> int:
        return lengths[int(n * p / 100)]

    logger.info("")
    logger.info("=" * 60)
    logger.info("TOKEN LENGTH STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Total records:  {n:,}")
    logger.info(f"Min:            {lengths[0]:,}")
    logger.info(f"Max:            {lengths[-1]:,}")
    logger.info(f"Mean:           {sum(lengths) / n:,.0f}")
    logger.info(f"P50 (median):   {pct(50):,}")
    logger.info(f"P75:            {pct(75):,}")
    logger.info(f"P90:            {pct(90):,}")
    logger.info(f"P95:            {pct(95):,}")
    logger.info(f"P99:            {pct(99):,}")
    logger.info("=" * 60)

    return {"total_records": n, "min": lengths[0], "max": lengths[-1]}


def main():
    parser = argparse.ArgumentParser(description="Add token counts to SFT data")
    parser.add_argument("input_file", help="Input JSONL file")
    parser.add_argument("--output_file", required=True, help="Output JSONL file")
    parser.add_argument("--tokenizer_path", required=True, help="Path to tokenizer")
    parser.add_argument(
        "--num_workers", type=int, help="Max workers (default: auto, capped at cpus-2)"
    )

    args = parser.parse_args()
    stats = add_token_counts(
        args.input_file, args.output_file, args.tokenizer_path, args.num_workers
    )

    if stats:
        logger.info(f"✅ Output written to: {args.output_file}")
        return 0
    logger.error("No records processed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
