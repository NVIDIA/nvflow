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
"""Merge sharded HopChain localization outputs."""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Merge sharded localization JSONL and summaries")
    parser.add_argument("--inputs", required=True, help="Glob for shard final_output JSONL files")
    parser.add_argument("--summaries", required=True, help="Glob for shard summary JSON files")
    parser.add_argument("--output", required=True, help="Merged final_output JSONL")
    parser.add_argument("--summary", required=True, help="Merged summary JSON")
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    input_paths = [Path(path) for path in sorted(glob.glob(args.inputs))]
    summary_paths = [Path(path) for path in sorted(glob.glob(args.summaries))]
    output_path = Path(args.output)
    summary_path = Path(args.summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as output_file:
        for input_path in input_paths:
            with input_path.open("r") as input_file:
                for line in input_file:
                    if line.strip():
                        output_file.write(line)

    instance_counts: Counter[str] = Counter()
    filtered_counts: Counter[str] = Counter()
    total_records = 0
    parse_errors = 0
    debug_annotated_images_saved = 0
    debug_annotated_image_errors = 0
    merged_summaries = []
    for shard_summary_path in summary_paths:
        summary = json.loads(shard_summary_path.read_text())
        merged_summaries.append(str(shard_summary_path))
        total_records += int(summary.get("total_records_processed", 0))
        parse_errors += int(summary.get("parse_errors", 0))
        debug_annotated_images_saved += int(summary.get("debug_annotated_images_saved", 0))
        debug_annotated_image_errors += int(summary.get("debug_annotated_image_errors", 0))
        instance_counts.update(summary.get("instance_category_counts", {}))
        filtered_counts.update(summary.get("filtered_instance_counts", {}))

    merged_summary = {
        "total_records_processed": total_records,
        "parse_errors": parse_errors,
        "successful": total_records - parse_errors,
        "instance_category_counts": dict(sorted(instance_counts.items())),
        "filtered_instance_count": sum(filtered_counts.values()),
        "filtered_instance_counts": dict(sorted(filtered_counts.items())),
        "debug_annotated_images_saved": debug_annotated_images_saved,
        "debug_annotated_image_errors": debug_annotated_image_errors,
        "merged_shard_outputs": [str(path) for path in input_paths],
        "merged_shard_summaries": merged_summaries,
        "output_file": str(output_path),
    }
    summary_path.write_text(json.dumps(merged_summary, indent=2))


if __name__ == "__main__":
    main()
