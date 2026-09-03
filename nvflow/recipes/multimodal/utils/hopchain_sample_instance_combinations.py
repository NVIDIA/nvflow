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
"""Sample instance combinations for HopChain query generation."""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import random
import shutil
from collections import Counter, defaultdict
from collections.abc import Callable
from functools import partial
from pathlib import Path

from nvflow.recipes.multimodal.utils.hopchain_sdg_models import (
    InstanceCombinationRecord,
    LocalizedInstanceRecord,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Sample localized instance combinations")
    parser.add_argument("--input", required=True, help="LocalizedInstanceRecord JSONL")
    parser.add_argument("--output", required=True, help="InstanceCombinationRecord JSONL")
    parser.add_argument("--summary", required=True, help="Summary JSON path")
    parser.add_argument("--min-instances", type=int, default=3)
    parser.add_argument("--max-instances", type=int, default=6)
    parser.add_argument("--selection-strategy", default="balanced_by_category_then_size")
    parser.add_argument("--max-instances-considered-per-image", type=int, default=12)
    parser.add_argument("--max-instances-per-category", type=int, default=2)
    parser.add_argument("--max-combinations-per-image", type=int, default=1)
    parser.add_argument(
        "--combination-size-strategy",
        choices=(
            "largest_first",
            "stratified_random",
            "stratified_random_by_size",
            "weighted_random",
        ),
        default="largest_first",
        help="How to choose combination sizes before sampling object combinations.",
    )
    parser.add_argument(
        "--combination-size-weights",
        default=None,
        help='Optional JSON or comma list of size weights, e.g. \'{"5":1,"6":2}\' or "5:1,6:2".',
    )
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--debug-copy-selected-images", action="store_true")
    parser.add_argument(
        "--debug-selected-images-dir", help="Directory to copy selected source images into"
    )
    # area_confidence strategy params
    parser.add_argument(
        "--area-confidence-area-weight",
        type=float,
        default=0.5,
        help="Weight for normalized area in the area+confidence score (default: 0.5)",
    )
    parser.add_argument(
        "--area-confidence-confidence-weight",
        type=float,
        default=0.5,
        help="Weight for confidence in the area+confidence score (default: 0.5)",
    )
    parser.add_argument(
        "--min-confidence-threshold",
        type=float,
        default=None,
        help="Drop instances below this confidence before selection (default: no threshold)",
    )
    parser.add_argument(
        "--null-confidence-fallback",
        type=float,
        default=0.5,
        help="Confidence value assigned to instances with confidence=None (default: 0.5)",
    )
    parser.add_argument(
        "--iou-dedup-threshold",
        type=float,
        default=None,
        help="Drop lower-ranked selected instances whose bbox IOU with an earlier selected instance is at least this value.",
    )
    parser.add_argument(
        "--iou-dedup-candidate-pool-size",
        type=int,
        default=None,
        help="Rank this many instances before IOU dedup refill (default: final selection cap).",
    )
    return parser.parse_args()


def crop_area(instance) -> int:
    """Return the crop area in pixels for one localized instance."""
    width = instance.bbox_xyxy.x2 - instance.bbox_xyxy.x1
    height = instance.bbox_xyxy.y2 - instance.bbox_xyxy.y1
    return max(0, width) * max(0, height)


def _instance_area_px(instance) -> float:
    """Return the bounding-box area in pixels for one instance."""
    w = max(0, instance.bbox_xyxy.x2 - instance.bbox_xyxy.x1)
    h = max(0, instance.bbox_xyxy.y2 - instance.bbox_xyxy.y1)
    return float(w * h)


def _bbox_iou(instance_a, instance_b) -> float:
    """Return pixel-space bounding-box IOU for two localized instances."""
    bbox_a = instance_a.bbox_xyxy
    bbox_b = instance_b.bbox_xyxy

    inter_x1 = max(bbox_a.x1, bbox_b.x1)
    inter_y1 = max(bbox_a.y1, bbox_b.y1)
    inter_x2 = min(bbox_a.x2, bbox_b.x2)
    inter_y2 = min(bbox_a.y2, bbox_b.y2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = float(inter_w * inter_h)
    if inter_area <= 0:
        return 0.0

    area_a = _instance_area_px(instance_a)
    area_b = _instance_area_px(instance_b)
    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def deduplicate_instances_by_bbox_iou(
    instances: list,
    *,
    iou_threshold: float | None,
    max_instances: int | None = None,
) -> tuple[list, int]:
    """Drop duplicate boxes after ranking/capping but before random sampling."""
    if iou_threshold is None:
        return instances[:max_instances] if max_instances is not None else instances, 0
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_dedup_threshold must be in the range (0, 1]")
    if max_instances is not None and max_instances <= 0:
        return [], 0

    deduped: list = []
    dropped = 0
    for instance in instances:
        if any(_bbox_iou(instance, kept_instance) >= iou_threshold for kept_instance in deduped):
            dropped += 1
            continue
        deduped.append(instance)
        if max_instances is not None and len(deduped) >= max_instances:
            break
    return deduped, dropped


def _instance_area_confidence_score(
    instance,
    *,
    area_weight: float,
    confidence_weight: float,
    null_confidence_fallback: float,
    max_area_px: float,
) -> float:
    """Compute a combined area+confidence score for one instance.

    Area is normalized relative to the largest instance in the current
    candidate set (``max_area_px``), so ``norm_area`` is always in [0, 1]
    regardless of image resolution or object size distribution.

        norm_area  = instance_area_px / max_area_px
        score      = area_weight * norm_area + confidence_weight * conf

    This ensures area and confidence contribute on the same scale.
    """
    norm_area = min(1.0, _instance_area_px(instance) / max_area_px) if max_area_px > 0 else 0.0

    conf = instance.confidence
    if conf is None:
        conf = null_confidence_fallback

    return area_weight * norm_area + confidence_weight * conf


def select_instances_balanced_by_category_then_size(
    instances: list,
    *,
    max_instances_considered_per_image: int,
    max_instances_per_category: int,
    **kwargs,
) -> tuple[list, bool]:
    """Select instances with category coverage first, then size preference.

    1. Group by category and sort each group by crop area descending.
    2. Keep up to `max_instances_per_category` candidates from each category.
    3. First pass: take the best candidate from each category.
    4. Second pass: fill remaining slots from leftover candidates by crop area.
    """
    if len(instances) <= max_instances_considered_per_image:
        return instances, False

    per_category: dict[str, list] = defaultdict(list)
    for instance in instances:
        per_category[instance.category].append(instance)

    for category_instances in per_category.values():
        category_instances.sort(key=crop_area, reverse=True)

    first_pass: list = []
    leftovers: list = []
    for category in sorted(per_category):
        candidates = per_category[category][:max_instances_per_category]
        if not candidates:
            continue
        first_pass.append(candidates[0])
        leftovers.extend(candidates[1:])

    selected: list = first_pass[:max_instances_considered_per_image]
    if len(selected) < max_instances_considered_per_image:
        leftovers.sort(key=crop_area, reverse=True)
        remaining_slots = max_instances_considered_per_image - len(selected)
        selected.extend(leftovers[:remaining_slots])

    return selected, True


def select_instances_balanced_by_category_then_area_confidence(
    instances: list,
    *,
    max_instances_considered_per_image: int,
    max_instances_per_category: int | None,
    area_weight: float = 0.5,
    confidence_weight: float = 0.5,
    min_confidence_threshold: float | None = None,
    null_confidence_fallback: float = 0.5,
    **kwargs,
) -> tuple[list, bool]:
    """Select instances with category diversity first, then area+confidence score.

    Strategy C (hybrid): Preserves category diversity from the baseline strategy
    but ranks within-category candidates and leftover slots by area+confidence
    score instead of raw pixel area.

    1. Optionally filter out instances below ``min_confidence_threshold``.
    2. Group by category and sort each group by area+confidence score descending.
    3. Optionally keep up to ``max_instances_per_category`` candidates per category.
    4. First pass: take the top-scored candidate from each category.
    5. Second pass: fill remaining slots from leftovers by score.
    """
    if min_confidence_threshold is not None:
        instances = [
            inst
            for inst in instances
            if inst.confidence is None or inst.confidence >= min_confidence_threshold
        ]

    max_area_px = max((_instance_area_px(i) for i in instances), default=1.0) or 1.0
    score_fn = partial(
        _instance_area_confidence_score,
        area_weight=area_weight,
        confidence_weight=confidence_weight,
        null_confidence_fallback=null_confidence_fallback,
        max_area_px=max_area_px,
    )

    per_category: dict[str, list] = defaultdict(list)
    for instance in instances:
        per_category[instance.category].append(instance)

    for category_instances in per_category.values():
        category_instances.sort(key=score_fn, reverse=True)

    first_pass: list = []
    leftovers: list = []
    for category in sorted(per_category):
        category_instances = per_category[category]
        candidates = (
            category_instances
            if max_instances_per_category is None
            else category_instances[:max_instances_per_category]
        )
        if not candidates:
            continue
        first_pass.append(candidates[0])
        leftovers.extend(candidates[1:])

    selected: list = first_pass[:max_instances_considered_per_image]
    if len(selected) < max_instances_considered_per_image:
        leftovers.sort(key=score_fn, reverse=True)
        remaining_slots = max_instances_considered_per_image - len(selected)
        selected.extend(leftovers[:remaining_slots])

    return selected, len(selected) < len(instances)


def select_instances_by_area_confidence(
    instances: list,
    *,
    max_instances_considered_per_image: int,
    area_weight: float = 0.5,
    confidence_weight: float = 0.5,
    min_confidence_threshold: float | None = None,
    null_confidence_fallback: float = 0.5,
    **kwargs,
) -> tuple[list, bool]:
    """Select instances globally by area+confidence score, ignoring category.

    Strategy B (global): Ranks all instances by combined score and keeps the
    top ``max_instances_considered_per_image``. No category diversity guarantee.
    Useful as a comparison baseline against Strategy C.
    """
    if min_confidence_threshold is not None:
        instances = [
            inst
            for inst in instances
            if inst.confidence is None or inst.confidence >= min_confidence_threshold
        ]

    if len(instances) <= max_instances_considered_per_image:
        return instances, False

    max_area_px = max((_instance_area_px(i) for i in instances), default=1.0) or 1.0
    score_fn = partial(
        _instance_area_confidence_score,
        area_weight=area_weight,
        confidence_weight=confidence_weight,
        null_confidence_fallback=null_confidence_fallback,
        max_area_px=max_area_px,
    )

    selected = sorted(instances, key=score_fn, reverse=True)[:max_instances_considered_per_image]
    return selected, True


SelectionStrategy = Callable[..., tuple[list, bool]]

SELECTION_STRATEGIES: dict[str, SelectionStrategy] = {
    "balanced_by_category_then_size": select_instances_balanced_by_category_then_size,
    "balanced_by_category_then_area_confidence": select_instances_balanced_by_category_then_area_confidence,
    "area_confidence": select_instances_by_area_confidence,
}


def parse_combination_size_weights(raw_weights: str | None) -> dict[int, float]:
    """Parse optional combination-size weights from JSON or ``size:weight`` pairs."""
    if not raw_weights:
        return {}

    try:
        parsed = json.loads(raw_weights)
    except json.JSONDecodeError as exc:
        parsed = {}
        for item in raw_weights.split(","):
            if not item.strip():
                continue
            size_text, separator, weight_text = item.partition(":")
            if not separator:
                raise ValueError(
                    "Combination size weights must be JSON or comma-separated size:weight pairs"
                ) from exc
            parsed[int(size_text.strip())] = float(weight_text.strip())

    if not isinstance(parsed, dict):
        raise ValueError("Combination size weights must parse to a mapping")

    weights: dict[int, float] = {}
    for size, weight in parsed.items():
        size_int = int(size)
        weight_float = float(weight)
        if weight_float < 0:
            raise ValueError("Combination size weights must be non-negative")
        weights[size_int] = weight_float
    return weights


def _feasible_combination_sizes(
    *,
    num_instances: int,
    min_instances: int,
    max_instances: int,
) -> list[int]:
    """Return feasible combination sizes in ascending order."""
    return list(range(min_instances, min(max_instances, num_instances) + 1))


def _weight_for_size(size: int, weights: dict[int, float]) -> float:
    """Return the configured weight for one combination size."""
    if not weights:
        return 1.0
    return weights.get(size, 0.0)


def _sample_combinations_for_size(
    *,
    instances: list,
    size: int,
    count: int,
    per_image_random: random.Random,
) -> list[tuple[int, tuple]]:
    """Sample up to ``count`` unique combinations for one size."""
    total_possible = math.comb(len(instances), size)
    count = min(count, total_possible)
    if count <= 0:
        return []

    # Enumerating small candidate sets avoids duplicate retry churn.
    if total_possible <= count * 10:
        candidates = list(itertools.combinations(instances, size))
        per_image_random.shuffle(candidates)
        return [(idx, combo) for idx, combo in enumerate(candidates[:count])]

    sampled: list[tuple[int, tuple]] = []
    seen: set[tuple[str, ...]] = set()
    attempts = 0
    max_attempts = count * 100
    while len(sampled) < count and attempts < max_attempts:
        attempts += 1
        combo = tuple(per_image_random.sample(instances, size))
        signature = tuple(sorted(instance.instance_id for instance in combo))
        if signature in seen:
            continue
        seen.add(signature)
        sampled.append((attempts - 1, combo))

    if len(sampled) < count:
        candidates = list(itertools.combinations(instances, size))
        per_image_random.shuffle(candidates)
        for idx, combo in enumerate(candidates):
            signature = tuple(sorted(instance.instance_id for instance in combo))
            if signature in seen:
                continue
            sampled.append((idx, combo))
            if len(sampled) >= count:
                break

    return sampled


def sample_instance_combinations_for_image(
    *,
    instances: list,
    min_instances: int,
    max_instances: int,
    max_combinations_per_image: int,
    combination_size_strategy: str,
    combination_size_weights: dict[int, float],
    per_image_random: random.Random,
) -> list[tuple[int, int, tuple]]:
    """Sample instance combinations for one image.

    Returns ``(size, candidate_rank, combo)`` tuples. ``largest_first`` preserves
    the original behavior; ``stratified_random`` guarantees coverage across
    feasible sizes before weighted fill; ``weighted_random`` samples sizes only
    according to configured weights.
    """
    feasible_sizes = _feasible_combination_sizes(
        num_instances=len(instances),
        min_instances=min_instances,
        max_instances=max_instances,
    )
    if not feasible_sizes or max_combinations_per_image <= 0:
        return []

    if combination_size_strategy == "largest_first":
        sampled: list[tuple[int, int, tuple]] = []
        for size in reversed(feasible_sizes):
            candidates = list(itertools.combinations(instances, size))
            per_image_random.shuffle(candidates)
            for combo_idx, combo in enumerate(candidates):
                sampled.append((size, combo_idx, combo))
                if len(sampled) >= max_combinations_per_image:
                    return sampled
        return sampled

    if combination_size_strategy == "stratified_random_by_size":
        combination_size_strategy = "stratified_random"
    if combination_size_strategy not in {"stratified_random", "weighted_random"}:
        raise ValueError(f"Unknown combination size strategy: {combination_size_strategy}")

    remaining_by_size = {size: math.comb(len(instances), size) for size in feasible_sizes}
    weighted_sizes = [
        size
        for size in feasible_sizes
        if remaining_by_size[size] > 0 and _weight_for_size(size, combination_size_weights) > 0
    ]
    if not weighted_sizes:
        raise ValueError("No feasible combination sizes have positive sampling weight")

    target_count = min(max_combinations_per_image, sum(remaining_by_size.values()))
    size_slots: list[int] = []

    # Stratified mode guarantees coverage across feasible positive-weight sizes
    # when possible, then uses weights for the remaining slots. Weighted mode
    # skips the coverage pass so weights control the whole size distribution.
    if combination_size_strategy == "stratified_random" and target_count >= len(weighted_sizes):
        size_slots.extend(weighted_sizes)
        for size in weighted_sizes:
            remaining_by_size[size] -= 1

    while len(size_slots) < target_count:
        choices = [
            size
            for size in weighted_sizes
            if remaining_by_size[size] > 0 and _weight_for_size(size, combination_size_weights) > 0
        ]
        if not choices:
            break
        weights = [_weight_for_size(size, combination_size_weights) for size in choices]
        chosen_size = per_image_random.choices(choices, weights=weights, k=1)[0]
        size_slots.append(chosen_size)
        remaining_by_size[chosen_size] -= 1

    per_image_random.shuffle(size_slots)
    counts_by_size = Counter(size_slots)
    sampled = []
    for size, count in sorted(counts_by_size.items()):
        for candidate_rank, combo in _sample_combinations_for_size(
            instances=instances,
            size=size,
            count=count,
            per_image_random=per_image_random,
        ):
            sampled.append((size, candidate_rank, combo))
    per_image_random.shuffle(sampled)
    return sampled[:max_combinations_per_image]


def main() -> None:
    """Entry point."""
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    debug_selected_images_dir = None
    if args.debug_copy_selected_images:
        debug_selected_images_dir = (
            Path(args.debug_selected_images_dir)
            if args.debug_selected_images_dir
            else output_path.parent / "selected_images"
        )
        debug_selected_images_dir.mkdir(parents=True, exist_ok=True)

    total_images = 0
    total_combinations = 0
    skipped_images = 0
    capped_images = 0
    category_balanced_images = 0
    combination_size_weights = parse_combination_size_weights(args.combination_size_weights)
    debug_full_images_copied = 0
    debug_crops_copied = 0
    failed_debug_artifact_copies = 0
    iou_deduplicated_images = 0
    iou_deduplicated_instances = 0
    size_counts: Counter[int] = Counter()

    with Path(args.input).open("r") as input_file, output_path.open("w") as output_file:
        for line_num, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record = LocalizedInstanceRecord.model_validate(json.loads(line))
                total_images += 1
                selection_strategy = SELECTION_STRATEGIES.get(args.selection_strategy)
                if selection_strategy is None:
                    raise ValueError(
                        f"Unknown selection strategy: {args.selection_strategy}. "
                        f"Available: {sorted(SELECTION_STRATEGIES)}"
                    )
                selection_limit = args.max_instances_considered_per_image
                max_instances_per_category = args.max_instances_per_category
                if args.iou_dedup_threshold is not None:
                    selection_limit = args.iou_dedup_candidate_pool_size or selection_limit
                    if selection_limit < args.max_instances_considered_per_image:
                        raise ValueError(
                            "iou_dedup_candidate_pool_size must be >= max_instances_considered_per_image"
                        )
                    if args.selection_strategy == "balanced_by_category_then_area_confidence":
                        max_instances_per_category = None
                instances, was_capped = selection_strategy(
                    record.instances,
                    max_instances_considered_per_image=selection_limit,
                    max_instances_per_category=max_instances_per_category,
                    area_weight=args.area_confidence_area_weight,
                    confidence_weight=args.area_confidence_confidence_weight,
                    min_confidence_threshold=args.min_confidence_threshold,
                    null_confidence_fallback=args.null_confidence_fallback,
                )
                if was_capped:
                    capped_images += 1
                    if args.selection_strategy.startswith("balanced_by_category"):
                        category_balanced_images += 1
                instances, deduplicated_count = deduplicate_instances_by_bbox_iou(
                    instances,
                    iou_threshold=args.iou_dedup_threshold,
                    max_instances=args.max_instances_considered_per_image,
                )
                if deduplicated_count:
                    iou_deduplicated_images += 1
                    iou_deduplicated_instances += deduplicated_count
                if len(instances) < args.min_instances:
                    skipped_images += 1
                    continue

                per_image_random = random.Random(f"{args.sampling_seed}:{record.image_id}")
                sampled_combinations = sample_instance_combinations_for_image(
                    instances=instances,
                    min_instances=args.min_instances,
                    max_instances=args.max_instances,
                    max_combinations_per_image=args.max_combinations_per_image,
                    combination_size_strategy=args.combination_size_strategy,
                    combination_size_weights=combination_size_weights,
                    per_image_random=per_image_random,
                )
                for sampled_for_image, (size, combo_idx, combo) in enumerate(
                    sampled_combinations, start=1
                ):
                    combination_record = InstanceCombinationRecord(
                        image_id=record.image_id,
                        image_file_name=record.image_file_name,
                        image_directory=record.image_directory,
                        image_path=record.image_path,
                        combination_id=f"{record.image_id}_combo_{sampled_for_image}",
                        instance_ids=[instance.instance_id for instance in combo],
                        instances=list(combo),
                        combination_size=size,
                        sampling_metadata={
                            "sampling_seed": args.sampling_seed,
                            "candidate_rank": combo_idx,
                            "combination_size_strategy": args.combination_size_strategy,
                        },
                    )
                    output_file.write(json.dumps(combination_record.model_dump()) + "\n")
                    if debug_selected_images_dir is not None:
                        combination_debug_dir = (
                            debug_selected_images_dir / combination_record.combination_id
                        )
                        combination_debug_dir.mkdir(parents=True, exist_ok=True)

                        source_image_path = Path(record.image_path)
                        full_image_target_path = (
                            combination_debug_dir
                            / f"full_image__{record.image_id}{source_image_path.suffix}"
                        )
                        try:
                            shutil.copy2(source_image_path, full_image_target_path)
                            debug_full_images_copied += 1
                        except Exception as exc:
                            failed_debug_artifact_copies += 1
                            logger.exception(
                                "Failed to copy full image for %s from %s: %s",
                                combination_record.combination_id,
                                source_image_path,
                                exc,
                            )

                        for instance in combo:
                            source_crop_path = Path(instance.crop_path)
                            crop_target_path = (
                                combination_debug_dir
                                / f"crop__{instance.instance_id}__{instance.category}{source_crop_path.suffix}"
                            )
                            try:
                                shutil.copy2(source_crop_path, crop_target_path)
                                debug_crops_copied += 1
                            except Exception as exc:
                                failed_debug_artifact_copies += 1
                                logger.exception(
                                    "Failed to copy crop %s for %s from %s: %s",
                                    instance.instance_id,
                                    combination_record.combination_id,
                                    source_crop_path,
                                    exc,
                                )
                    total_combinations += 1
                    size_counts[size] += 1
            except Exception as exc:
                logger.exception("Failed to sample combinations on line %s: %s", line_num, exc)

    summary = {
        "total_images_processed": total_images,
        "skipped_images": skipped_images,
        "capped_images": capped_images,
        "category_balanced_images": category_balanced_images,
        "selection_strategy": args.selection_strategy,
        "max_instances_considered_per_image": args.max_instances_considered_per_image,
        "max_instances_per_category": args.max_instances_per_category,
        "iou_dedup_threshold": args.iou_dedup_threshold,
        "iou_dedup_candidate_pool_size": args.iou_dedup_candidate_pool_size,
        "iou_deduplicated_images": iou_deduplicated_images,
        "iou_deduplicated_instances": iou_deduplicated_instances,
        "combination_size_strategy": args.combination_size_strategy,
        "combination_size_weights": combination_size_weights,
        "total_combinations": total_combinations,
        "combination_size_counts": dict(sorted(size_counts.items())),
        "output_file": str(output_path),
        "debug_copy_selected_images": args.debug_copy_selected_images,
        "debug_selected_images_dir": str(debug_selected_images_dir)
        if debug_selected_images_dir
        else None,
        "debug_full_images_copied": debug_full_images_copied,
        "debug_crops_copied": debug_crops_copied,
        "failed_debug_artifact_copies": failed_debug_artifact_copies,
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2))
    logger.info("Sampled %s instance combinations", total_combinations)


if __name__ == "__main__":
    main()
