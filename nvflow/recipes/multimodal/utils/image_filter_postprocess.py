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
"""Postprocess image filter generations into structured JSONL results."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from nvflow.recipes.multimodal.utils.image_filter_models import (
    GenerationStats,
    ImageFilterGeneration,
    ImageFilterRecord,
    RawInferenceOutput,
    SelectedImageRecord,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def extract_json_object(text: str) -> str:
    """Extract the first balanced JSON object from a model response."""
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model response")

    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text[start:], start=start):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    raise ValueError("Unbalanced JSON object in model response")


def parse_args() -> argparse.Namespace:
    """Parse CLI args for image filter postprocessing."""
    parser = argparse.ArgumentParser(description="Postprocess HopChain image filtering outputs")
    parser.add_argument("--input", required=True, help="Raw nemo-skills output JSONL")
    parser.add_argument("--output", required=True, help="Final structured JSONL output")
    parser.add_argument(
        "--min-complexity-score", type=int, default=4, help="Minimum score required to keep"
    )
    parser.add_argument(
        "--allowed-quality-rating",
        action="append",
        dest="allowed_quality_ratings",
        default=[],
        help="Quality ratings that are allowed to pass filtering",
    )
    return parser.parse_args()


def build_filter_reason(
    *,
    generation: ImageFilterGeneration,
    should_keep: bool,
    min_complexity_score: int,
    allowed_quality_ratings: set[str],
) -> str:
    """Create a concise explanation for the keep/drop decision."""
    if should_keep:
        return (
            f"quality={generation.overall_quality_rating} and "
            f"complexity={generation.overall_complexity_score} meet thresholds"
        )

    failures: list[str] = []
    if generation.overall_quality_rating not in allowed_quality_ratings:
        failures.append(
            f"quality={generation.overall_quality_rating} not in {sorted(allowed_quality_ratings)}"
        )
    if generation.overall_complexity_score < min_complexity_score:
        failures.append(
            f"complexity={generation.overall_complexity_score} below minimum={min_complexity_score}"
        )
    return "; ".join(failures) if failures else "did not satisfy filter criteria"


def main() -> None:
    """Parse model outputs into structured image filter records."""
    args = parse_args()
    allowed_quality_ratings = set(args.allowed_quality_ratings or ["High", "Medium"])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept_output_path = output_path.parent / "kept_images.jsonl"
    summary_path = output_path.parent / "summary.json"

    results: list[ImageFilterRecord] = []
    kept_count = 0
    parse_errors = 0
    quality_counts: Counter[str] = Counter()
    score_counts: Counter[int] = Counter()
    kept_score_counts: Counter[int] = Counter()
    image_directory_counts: Counter[str] = Counter()
    kept_image_directory_counts: Counter[str] = Counter()

    with Path(args.input).open("r") as input_file:
        for line_num, line in enumerate(input_file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
                finish_reason = record.get("finish_reason")
                if finish_reason and finish_reason != "stop":
                    logger.warning(
                        "Skipping line %s due to finish_reason=%s", line_num, finish_reason
                    )
                    parse_errors += 1
                    continue

                metadata = SelectedImageRecord.model_validate(record.get("_metadata", {}))
                generation_text = record.get("generation", "").strip()
                if not generation_text:
                    logger.warning("Skipping line %s because generation was empty", line_num)
                    parse_errors += 1
                    continue

                parsed_json = json.loads(extract_json_object(generation_text))
                generation = ImageFilterGeneration.model_validate(parsed_json)
                quality_counts[generation.overall_quality_rating] += 1
                score_counts[generation.overall_complexity_score] += 1
                image_directory_counts[metadata.image_directory] += 1

                should_keep = (
                    generation.overall_quality_rating in allowed_quality_ratings
                    and generation.overall_complexity_score >= args.min_complexity_score
                )
                if should_keep:
                    kept_count += 1
                    kept_score_counts[generation.overall_complexity_score] += 1
                    kept_image_directory_counts[metadata.image_directory] += 1

                result = ImageFilterRecord(
                    image_file_name=metadata.image_file_name,
                    image_directory=metadata.image_directory,
                    catalog_entry_index=metadata.catalog_entry_index,
                    source_image_index=metadata.source_image_index,
                    start_index=metadata.start_index,
                    end_index=metadata.end_index,
                    overall_complexity_score=generation.overall_complexity_score,
                    overall_quality_rating=generation.overall_quality_rating,
                    complexity_analysis=generation.complexity_analysis,
                    complex_objects=generation.complex_objects,
                    should_keep=should_keep,
                    filter_reason=build_filter_reason(
                        generation=generation,
                        should_keep=should_keep,
                        min_complexity_score=args.min_complexity_score,
                        allowed_quality_ratings=allowed_quality_ratings,
                    ),
                    generation_stats=GenerationStats(
                        num_generated_tokens=record.get("num_generated_tokens"),
                        generation_time=record.get("generation_time"),
                    ),
                    raw_inference=RawInferenceOutput(
                        generation=generation_text,
                        full_generation=record.get("_full_generation"),
                        finish_reason=finish_reason,
                        reasoning_content=record.get("reasoning_content")
                        or record.get("reasoning"),
                    ),
                )
                results.append(result)
            except Exception as exc:
                logger.exception("Error postprocessing line %s: %s", line_num, exc)
                parse_errors += 1

    with output_path.open("w") as output_file, kept_output_path.open("w") as kept_file:
        for result in results:
            serialized = json.dumps(result.model_dump())
            output_file.write(serialized + "\n")
            if result.should_keep:
                kept_file.write(serialized + "\n")

    summary = {
        "total_records_processed": len(results) + parse_errors,
        "successful": len(results),
        "kept": kept_count,
        "dropped": len(results) - kept_count,
        "parse_errors": parse_errors,
        "min_complexity_score": args.min_complexity_score,
        "allowed_quality_ratings": sorted(allowed_quality_ratings),
        "quality_counts": dict(quality_counts),
        "score_counts": {str(score): count for score, count in sorted(score_counts.items())},
        "kept_score_counts": {
            str(score): count for score, count in sorted(kept_score_counts.items())
        },
        "image_directory_counts": dict(sorted(image_directory_counts.items())),
        "kept_image_directory_counts": dict(sorted(kept_image_directory_counts.items())),
        "output_file": str(output_path),
        "kept_output_file": str(kept_output_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    logger.info("Postprocessed %s records", len(results))
    logger.info("Kept %s images after filtering", kept_count)
    logger.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
