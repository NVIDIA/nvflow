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
"""Postprocess category-identification generations."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from nvflow.recipes.multimodal.utils.hopchain_sdg_common import (
    dedupe_preserve_order,
    extract_json_value,
    normalize_category_name,
)
from nvflow.recipes.multimodal.utils.hopchain_sdg_models import (
    CategoryIdentificationRecord,
    CategoryLocalizationTarget,
)
from nvflow.recipes.multimodal.utils.image_filter_models import GenerationStats

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


LOW_VALUE_CATEGORIES = {
    "character",
    "characters",
    "digit",
    "digits",
    "bar_code",
    "bar_codes",
    "barcode",
    "barcodes",
    "letter",
    "letters",
    "number",
    "numbers",
    "punctuation",
    "qr_code",
    "qr_codes",
    "text",
    "word",
    "words",
}
MAX_LOCALIZATION_PHRASES_PER_CATEGORY = 3


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Postprocess category-identification generations")
    parser.add_argument("--input", required=True, help="Raw nemo-skills output JSONL")
    parser.add_argument("--output", required=True, help="CategoryIdentificationRecord JSONL")
    parser.add_argument("--summary", required=True, help="Summary JSON path")
    return parser.parse_args()


def normalize_keep_category(category: str) -> str | None:
    """Normalize a category and drop low-value labels."""
    normalized = normalize_category_name(category)
    if not normalized or normalized in LOW_VALUE_CATEGORIES:
        return None
    return normalized


def parse_categories_and_targets(
    generation_text: str,
) -> tuple[list[str], list[CategoryLocalizationTarget]]:
    """Parse canonical categories and optional localization phrases."""
    value = extract_json_value(generation_text)
    categories: list[str] = []
    targets: list[CategoryLocalizationTarget] = []
    if isinstance(value, dict):
        raw_categories = value.get("categories", value.get("localization_targets", []))
    elif isinstance(value, list):
        raw_categories = value
    else:
        raw_categories = []
    for raw_category in raw_categories:
        if isinstance(raw_category, str) and raw_category.strip():
            normalized = normalize_keep_category(raw_category)
            if normalized is None:
                continue
            categories.append(normalized)
            targets.append(
                CategoryLocalizationTarget(
                    category=normalized, localization_phrases=[raw_category.strip()]
                )
            )
        elif isinstance(raw_category, dict):
            raw_name = raw_category.get("category") or raw_category.get("name")
            if not isinstance(raw_name, str) or not raw_name.strip():
                continue
            normalized = normalize_keep_category(raw_name)
            if normalized is None:
                continue
            phrases = [raw_name.strip()]
            raw_phrases = raw_category.get("localization_phrases", raw_category.get("phrases", []))
            if isinstance(raw_phrases, list):
                phrases.extend(
                    phrase.strip()
                    for phrase in raw_phrases
                    if isinstance(phrase, str) and phrase.strip()
                )
            phrases = dedupe_preserve_order(phrases)[:MAX_LOCALIZATION_PHRASES_PER_CATEGORY]
            categories.append(normalized)
            targets.append(
                CategoryLocalizationTarget(category=normalized, localization_phrases=phrases)
            )

    categories = dedupe_preserve_order(categories)
    target_by_category: dict[str, CategoryLocalizationTarget] = {}
    for target in targets:
        existing = target_by_category.get(target.category)
        if existing is None:
            target_by_category[target.category] = target
            continue
        existing.localization_phrases = dedupe_preserve_order(
            [*existing.localization_phrases, *target.localization_phrases]
        )[:MAX_LOCALIZATION_PHRASES_PER_CATEGORY]
    return categories, [
        target_by_category[category] for category in categories if category in target_by_category
    ]


def main() -> None:
    """Entry point."""
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    parse_errors = 0
    category_frequency: Counter[str] = Counter()
    with Path(args.input).open("r") as input_file, output_path.open("w") as output_file:
        for line_num, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            total += 1
            try:
                record = json.loads(line)
                metadata = record.get("_metadata", {})
                generation_text = record.get("generation", "").strip()
                categories, localization_targets = parse_categories_and_targets(generation_text)
                for category in categories:
                    category_frequency[category] += 1
                output = CategoryIdentificationRecord(
                    image_id=metadata["image_id"],
                    image_file_name=metadata["image_file_name"],
                    image_directory=metadata["image_directory"],
                    image_path=metadata["image_path"],
                    prior_generalized_name_hints=metadata.get("prior_generalized_name_hints", []),
                    identified_categories=categories,
                    localization_targets=localization_targets,
                    raw_generation=generation_text,
                    generation_stats=GenerationStats(
                        num_generated_tokens=record.get("num_generated_tokens"),
                        generation_time=record.get("generation_time"),
                    ),
                )
                output_file.write(json.dumps(output.model_dump()) + "\n")
            except Exception as exc:
                parse_errors += 1
                logger.exception("Failed to postprocess category line %s: %s", line_num, exc)

    summary = {
        "total_records_processed": total,
        "parse_errors": parse_errors,
        "successful": total - parse_errors,
        "category_frequency": dict(sorted(category_frequency.items())),
        "output_file": str(output_path),
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2))
    logger.info("Postprocessed %s category-identification records", total - parse_errors)


if __name__ == "__main__":
    main()
