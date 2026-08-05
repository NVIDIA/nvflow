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
"""Normalize kept image-filter outputs into SDG inputs."""

from __future__ import annotations

import argparse
import json
import logging
import random
import uuid
from collections import Counter
from pathlib import Path

from nvflow.recipes.multimodal.utils.hopchain_sdg_common import (
    dedupe_preserve_order,
    normalize_category_name,
)
from nvflow.recipes.multimodal.utils.hopchain_sdg_models import FilteredImageInput
from nvflow.recipes.multimodal.utils.image_filter_models import ImageFilterRecord

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

HOPCHAIN_IMAGE_NAMESPACE = uuid.UUID("ef5375f1-d796-4d7d-9efe-8ef6c48784dd")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Prepare filtered images for HopChain SDG")
    parser.add_argument("--input", required=True, help="Path to kept_images.jsonl")
    parser.add_argument(
        "--output", required=True, help="Path to output filtered_image_inputs.jsonl"
    )
    parser.add_argument("--summary", required=True, help="Path to summary.json")
    parser.add_argument(
        "--sample-count",
        type=int,
        default=None,
        help="Optional number of records to randomly subsample before writing outputs",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed used when sample-count is provided",
    )
    parser.add_argument(
        "--sample-count-per-domain",
        type=int,
        default=None,
        help="Sample up to N images per image_directory (domain). Takes priority over --sample-count.",
    )
    return parser.parse_args()


def build_image_id(image_directory: str, image_file_name: str) -> str:
    """Create a stable deterministic image id."""
    key = f"{image_directory}/{image_file_name}"
    return str(uuid.uuid5(HOPCHAIN_IMAGE_NAMESPACE, key))


def main() -> None:
    """Entry point."""
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    prepared_count = 0
    directory_counts: Counter[str] = Counter()
    prepared_records: list[FilteredImageInput] = []

    with input_path.open("r") as input_file:
        for line_num, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record = ImageFilterRecord.model_validate(json.loads(line))
                generalized_name_hints = dedupe_preserve_order(
                    [
                        normalize_category_name(obj.generalized_name)
                        for obj in record.complex_objects
                        if obj.generalized_name.strip()
                    ]
                )
                prepared = FilteredImageInput(
                    image_id=build_image_id(record.image_directory, record.image_file_name),
                    image_file_name=record.image_file_name,
                    image_directory=record.image_directory,
                    image_path=str(Path(record.image_directory) / record.image_file_name),
                    filter_complexity_score=record.overall_complexity_score,
                    filter_quality_rating=record.overall_quality_rating,
                    filter_complex_objects=record.complex_objects,
                    filter_generalized_name_hints=generalized_name_hints,
                    filter_analysis=record.complexity_analysis,
                    filter_should_keep=record.should_keep,
                    metadata={
                        "source_image_index": record.source_image_index,
                        "catalog_entry_index": record.catalog_entry_index,
                        "shuffle_seed": None,
                        "generation_stats": record.generation_stats.model_dump(),
                    },
                )
                prepared_records.append(prepared)
            except Exception as exc:
                logger.exception("Failed to normalize line %s: %s", line_num, exc)

    total_prepared_records = len(prepared_records)
    if args.sample_count_per_domain is not None:
        rng = random.Random(args.sample_seed)
        by_domain: dict[str, list] = {}
        for r in prepared_records:
            by_domain.setdefault(r.image_directory, []).append(r)
        prepared_records = []
        for domain_records in by_domain.values():
            n = min(args.sample_count_per_domain, len(domain_records))
            prepared_records.extend(rng.sample(domain_records, n))
        logger.info(
            "Per-domain sampled %s records across %s domains",
            len(prepared_records),
            len(by_domain),
        )
    elif args.sample_count is not None and args.sample_count < total_prepared_records:
        rng = random.Random(args.sample_seed)
        prepared_records = rng.sample(prepared_records, args.sample_count)
        logger.info(
            "Randomly subsampled %s records out of %s using seed=%s",
            len(prepared_records),
            total_prepared_records,
            args.sample_seed,
        )

    with output_path.open("w") as output_file:
        for prepared in prepared_records:
            output_file.write(json.dumps(prepared.model_dump()) + "\n")
            prepared_count += 1
            directory_counts[prepared.image_directory] += 1

    summary = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "prepared_count": prepared_count,
        "sample_count": args.sample_count,
        "sample_count_per_domain": args.sample_count_per_domain,
        "sample_seed": args.sample_seed
        if (args.sample_count is not None or args.sample_count_per_domain is not None)
        else None,
        "image_directory_counts": dict(sorted(directory_counts.items())),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Prepared %s filtered SDG inputs", prepared_count)


if __name__ == "__main__":
    main()
