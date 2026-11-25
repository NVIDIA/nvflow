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
"""Filter answers based on answerability tags."""

import argparse

import orjson

from nvflow.utils import setup_logger

# Initialize logger
logger = setup_logger(__name__)

WRITE_BUFFER_SIZE = 1000  # Write every 1000 records


def apply_answer_filter(input_file, output_file, keep_tag="ANSWERABLE"):
    """Filter answers based on filter_tag field.

    Args:
        input_file: Path to input JSONL file with 'filter_tag' field
        output_file: Path to output JSONL file with filtered entries
        keep_tag: Tag value to keep (default: "ANSWERABLE")
    """
    log_file = output_file.replace(".jsonl", "_filter_log.txt")
    num_total_entries = 0
    num_kept = 0
    num_filtered = 0
    num_missing_tag = 0

    buffer = []

    with (
        open(input_file, "rb") as reader,
        open(output_file, "wb") as writer,
        open(log_file, "w") as log_writer,
    ):
        for line in reader:
            line = line.strip()
            if not line:
                continue

            num_total_entries += 1

            try:
                row = orjson.loads(line)
            except orjson.JSONDecodeError as e:
                msg = f"Failed to parse JSON at entry {num_total_entries}: {e}"
                log_writer.write(msg + "\n")
                num_missing_tag += 1
                continue

            filter_tag = row.get("filter_tag")
            problem = row["problem"]
            generation = row["generation"]
            if filter_tag is None:
                num_missing_tag += 1
                msg = f"Entry {num_total_entries} missing filter_tag field: {problem[:100]}"
                log_writer.write(msg + "\n")
                continue

            if filter_tag == keep_tag:
                # Keep this entry
                num_kept += 1

                # Add to buffer
                buffer.append(orjson.dumps(row))

                # Write buffer when it reaches threshold
                if len(buffer) >= WRITE_BUFFER_SIZE:
                    writer.write(b"\n".join(buffer) + b"\n")
                    buffer.clear()
            else:
                # Filter out this entry
                num_filtered += 1
                msg = f"Filtered entry {num_total_entries} (tag={filter_tag}):\n  Q: {problem}\n  A: {generation}\n"
                log_writer.write(msg + "\n")

        # Write remaining buffer
        if buffer:
            writer.write(b"\n".join(buffer) + b"\n")

        # Write summary to log
        log_writer.write(f"\n{'=' * 60}\n")
        log_writer.write("FILTER SUMMARY\n")
        log_writer.write(f"{'=' * 60}\n")
        log_writer.write(f"Total entries: {num_total_entries}\n")
        log_writer.write(f"Kept ({keep_tag}): {num_kept}\n")
        log_writer.write(f"Filtered out: {num_filtered}\n")
        log_writer.write(f"Missing tag: {num_missing_tag}\n")
        if num_total_entries > 0:
            log_writer.write(f"Keep rate: {num_kept / num_total_entries * 100:.2f}%\n")
            log_writer.write(f"Filter rate: {num_filtered / num_total_entries * 100:.2f}%\n")

        # Log summary
        logger.info(f"\n{'=' * 60}")
        logger.info("FILTER SUMMARY")
        logger.info(f"{'=' * 60}")
        logger.info(f"Total entries: {num_total_entries}")
        logger.info(f"Kept ({keep_tag}): {num_kept}")
        logger.info(f"Filtered out: {num_filtered}")
        logger.info(f"Missing tag: {num_missing_tag}")
        if num_total_entries > 0:
            logger.info(f"Keep rate: {num_kept / num_total_entries * 100:.2f}%")
            logger.info(f"Filter rate: {num_filtered / num_total_entries * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter answers based on answerability tags")
    parser.add_argument(
        "--input_file", required=True, help="Input JSONL file with filter_tag field"
    )
    parser.add_argument(
        "--output_file", required=True, help="Output JSONL file with filtered entries"
    )
    parser.add_argument(
        "--keep_tag", default="ANSWERABLE", help="Tag value to keep (default: ANSWERABLE)"
    )
    args = parser.parse_args()

    apply_answer_filter(args.input_file, args.output_file, args.keep_tag)
