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
"""Run HopChain localization with native SAM 3.1."""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from importlib import import_module
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from nvflow.recipes.multimodal.utils.hopchain_localization_filters import (
    LocalizationFilterSpec,
    evaluate_localization_filters,
    parse_filter_list,
)
from nvflow.recipes.multimodal.utils.hopchain_sdg_common import (
    dedupe_preserve_order,
    load_prompt_template,
    normalize_category_name,
)
from nvflow.recipes.multimodal.utils.hopchain_sdg_models import (
    BoundingBoxNorm1000,
    BoundingBoxPixels,
    CategoryIdentificationRecord,
    CropContext,
    LocalizedInstance,
    LocalizedInstanceRecord,
)
from nvflow.recipes.multimodal.utils.image_filter_models import GenerationStats
from nvflow.recipes.multimodal.utils.image_utils import save_image_crop

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


NATIVE_SAM31_DTYPE = torch.bfloat16
BOX_COLORS = (
    "#e11d48",
    "#2563eb",
    "#059669",
    "#d97706",
    "#7c3aed",
    "#0891b2",
)


class Sam3Config(BaseModel):
    """Runtime configuration for the SAM3 localization script."""

    model_name_or_path: str
    prompt_template: str
    max_image_dimension: int | None = Field(default=None, ge=1)
    dataloader_num_workers: int = Field(default=4, ge=0)
    dataloader_prefetch_factor: int = Field(default=2, ge=1)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    filter_list: list[LocalizationFilterSpec] = Field(default_factory=list)
    debug_save_annotated_images: bool = False
    debug_annotated_images_dir: str | None = None
    max_localization_phrases_per_category: int = Field(default=3, ge=1)
    prompt_alias_iou_dedup_threshold: float = Field(default=0.9, gt=0.0, le=1.0)
    num_shards: int = Field(default=1, ge=1)
    shard_index: int = Field(default=0, ge=0)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run HopChain localization with native SAM 3.1")
    parser.add_argument("--input", required=True, help="CategoryIdentificationRecord JSONL")
    parser.add_argument("--output", required=True, help="LocalizedInstanceRecord JSONL")
    parser.add_argument("--summary", required=True, help="Summary JSON path")
    parser.add_argument(
        "--crop-dir", required=True, help="Directory where instance crops will be written"
    )
    parser.add_argument("--prompt", required=True, help="SAM3 prompt template")
    parser.add_argument("--model", required=True, help="SAM3 model path or HF repo id")
    parser.add_argument("--max-image-dimension", type=int, default=None)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--dataloader-prefetch-factor", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--filter-list", default=None, help="JSON-encoded list of localization filters"
    )
    parser.add_argument("--max-localization-phrases-per-category", type=int, default=3)
    parser.add_argument("--prompt-alias-iou-dedup-threshold", type=float, default=0.9)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--debug-save-annotated-images",
        action="store_true",
        help="Save original images annotated with kept instance bounding boxes",
    )
    parser.add_argument(
        "--debug-annotated-images-dir",
        default=None,
        help="Directory for debug annotated images; defaults to <crop-dir>/annotated_images",
    )
    return parser.parse_args()


def format_category_for_prompt(category: str, prompt_template: str) -> str:
    """Render a SAM3 prompt for one category."""
    category_text = category.replace("_", " ").strip()
    return prompt_template.format(category=category_text)


def build_localization_targets(
    record: CategoryIdentificationRecord, max_phrases_per_category: int
) -> list[dict[str, Any]]:
    """Build canonical categories with concrete SAM prompt phrases."""
    if record.localization_targets:
        targets: list[dict[str, Any]] = []
        seen_categories: set[str] = set()
        for target in record.localization_targets:
            category = normalize_category_name(target.category)
            if not category or category in seen_categories:
                continue
            phrases = dedupe_preserve_order(
                [phrase for phrase in target.localization_phrases if phrase.strip()]
                or [category.replace("_", " ")]
            )[:max_phrases_per_category]
            targets.append({"category": category, "phrases": phrases})
            seen_categories.add(category)
        if targets:
            return targets

    return [
        {"category": category, "phrases": [category.replace("_", " ")]}
        for category in dedupe_preserve_order(
            [
                normalize_category_name(category)
                for category in record.identified_categories
                if category.strip()
            ]
        )
    ]


def bbox_iou_xyxy(box_a: list[float], box_b: list[float]) -> float:
    """Return IOU for two XYXY boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def resize_image_for_inference(image: Image.Image, max_image_dimension: int | None) -> Image.Image:
    """Cap the largest image dimension before sending it to the model."""
    if max_image_dimension is None or max(image.size) <= max_image_dimension:
        return image

    original_size = image.size
    resized = image.copy()
    resized.thumbnail((max_image_dimension, max_image_dimension), Image.Resampling.LANCZOS)
    logger.info(
        "Resized image for inference: %sx%s -> %sx%s",
        original_size[0],
        original_size[1],
        resized.size[0],
        resized.size[1],
    )
    return resized


class Sam3Dataset(Dataset):
    """CPU-side dataset that preloads and resizes images for inference."""

    def __init__(
        self,
        records: list[tuple[int, CategoryIdentificationRecord]],
        config: Sam3Config,
    ) -> None:
        self.records = records
        self.config = config

    def __len__(self) -> int:
        """Return the number of records."""
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Prepare one inference item on a worker process."""
        line_num, record = self.records[index]
        record_data = record.model_dump(mode="json")

        try:
            localization_targets = build_localization_targets(
                record,
                max_phrases_per_category=self.config.max_localization_phrases_per_category,
            )
            if not localization_targets:
                return {"line_num": line_num, "record": record_data, "skip_inference": True}

            with Image.open(record.image_path) as raw_image:
                image = raw_image.convert("RGB")
            prepared_image = resize_image_for_inference(image, self.config.max_image_dimension)
            if prepared_image is image:
                prepared_image = image.copy()
                image.close()
            else:
                image.close()

            return {
                "line_num": line_num,
                "record": record_data,
                "skip_inference": False,
                "localization_targets": localization_targets,
                "image": prepared_image,
            }
        except Exception as exc:
            return {
                "line_num": line_num,
                "record": record_data,
                "error": f"{type(exc).__name__}: {exc}",
            }


def collate_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep DataLoader outputs as plain item lists."""
    return items


def iter_with_progress(dataloader: DataLoader, total: int) -> Any:
    """Wrap the dataloader with a progress bar."""
    return tqdm(
        dataloader,
        total=total,
        desc="SAM localization",
        unit="image",
        dynamic_ncols=True,
    )


def boxes_to_list(boxes: Any) -> list[list[float]]:
    """Normalize boxes into a plain Python list."""
    if boxes is None:
        return []
    if hasattr(boxes, "tolist"):
        return boxes.tolist()
    return list(boxes)


def scores_to_list(scores: Any, num_boxes: int) -> list[float | None]:
    """Normalize scores into a plain Python list."""
    if scores is None:
        return [None] * num_boxes
    if hasattr(scores, "tolist"):
        values = scores.tolist()
    else:
        values = list(scores)
    return values


def save_annotated_debug_image(
    *,
    image_path: str,
    instances: list[LocalizedInstance],
    output_path: Path,
) -> None:
    """Save the original image with kept instance boxes drawn on top."""
    if not instances:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as raw_image:
        image = raw_image.convert("RGB")

    try:
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        line_width = max(2, max(image.size) // 400)
        for idx, instance in enumerate(instances):
            color = BOX_COLORS[idx % len(BOX_COLORS)]
            bbox = instance.bbox_xyxy
            draw.rectangle((bbox.x1, bbox.y1, bbox.x2, bbox.y2), outline=color, width=line_width)
            label = f"{idx + 1}. {instance.instance_id} ({instance.category})"
            label_y = max(0, bbox.y1 - 14)
            draw.rectangle(
                (bbox.x1, label_y, bbox.x1 + max(80, len(label) * 6), label_y + 14), fill=color
            )
            draw.text((bbox.x1 + 2, label_y + 1), label, fill="white", font=font)
        image.save(output_path, quality=92)
    finally:
        image.close()


def load_native_sam31_runtime(config: Sam3Config) -> tuple[Any, Any]:
    """Load the native Meta SAM 3.1 model and processor."""
    image_processor_module = import_module("sam3.model.sam3_image_processor")
    model_builder = import_module("sam3.model_builder")

    checkpoint_path = config.model_name_or_path
    logger.info("Loading SAM 3.1 model from %s", checkpoint_path)
    model = (
        model_builder.build_sam3_image_model(
            checkpoint_path=checkpoint_path,
            load_from_HF=False,
        )
        .to("cuda")
        .eval()
    )
    processor = image_processor_module.Sam3Processor(model, confidence_threshold=config.threshold)
    return model, processor


def pixels_to_bbox_norm_1000(
    box_xyxy: list[float],
    image_width: int,
    image_height: int,
) -> BoundingBoxNorm1000 | None:
    """Convert absolute pixel XYXY box to normalized 0-1000 coordinates."""
    x1, y1, x2, y2 = [float(value) for value in box_xyxy]
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))

    norm = BoundingBoxNorm1000(
        x1=max(0, min(1000, round(x1 / image_width * 1000))),
        y1=max(0, min(1000, round(y1 / image_height * 1000))),
        x2=max(0, min(1000, round(x2 / image_width * 1000))),
        y2=max(0, min(1000, round(y2 / image_height * 1000))),
    )
    if norm.x2 <= norm.x1 or norm.y2 <= norm.y1:
        return None
    return norm


@torch.no_grad()
def run_native_sam31_for_category(
    *,
    model: Any,
    processor: Any,
    inference_image: Image.Image,
    category_prompt: str,
    config: Sam3Config,
) -> tuple[list[dict[str, Any]], float]:
    """Run native SAM 3.1 segmentation for one category prompt."""
    start_time = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=NATIVE_SAM31_DTYPE):
        state = processor.set_image(inference_image)
        output = processor.set_text_prompt(state=state, prompt=category_prompt)
    elapsed = time.perf_counter() - start_time

    boxes = boxes_to_list(output.get("boxes"))
    scores = scores_to_list(output.get("scores"), len(boxes))
    detections: list[dict[str, Any]] = []
    for idx, box in enumerate(boxes):
        detections.append(
            {
                "box_xyxy": box,
                "score": scores[idx] if idx < len(scores) else None,
            }
        )
    return detections, elapsed


def main() -> None:
    """Entry point."""
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("SAM3 localization requires CUDA")

    config = Sam3Config(
        model_name_or_path=args.model,
        prompt_template=load_prompt_template(args.prompt),
        max_image_dimension=args.max_image_dimension,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_prefetch_factor=args.dataloader_prefetch_factor,
        threshold=args.threshold,
        filter_list=parse_filter_list(args.filter_list),
        debug_save_annotated_images=args.debug_save_annotated_images,
        debug_annotated_images_dir=args.debug_annotated_images_dir,
        max_localization_phrases_per_category=args.max_localization_phrases_per_category,
        prompt_alias_iou_dedup_threshold=args.prompt_alias_iou_dedup_threshold,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    if config.shard_index >= config.num_shards:
        raise ValueError("--shard-index must be less than --num-shards")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop_dir = Path(args.crop_dir)
    crop_dir.mkdir(parents=True, exist_ok=True)
    debug_annotated_images_dir = (
        Path(config.debug_annotated_images_dir)
        if config.debug_annotated_images_dir
        else crop_dir / "annotated_images"
    )

    model, processor = load_native_sam31_runtime(config)

    total = 0
    parse_errors = 0
    instance_counts: Counter[str] = Counter()
    filtered_counts: Counter[str] = Counter()
    debug_annotated_images_saved = 0
    debug_annotated_image_errors = 0

    records: list[tuple[int, CategoryIdentificationRecord]] = []
    with Path(args.input).open("r") as input_file:
        record_idx = 0
        for line_num, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            if record_idx % config.num_shards == config.shard_index:
                records.append(
                    (line_num, CategoryIdentificationRecord.model_validate(json.loads(line)))
                )
            record_idx += 1

    dataset = Sam3Dataset(records=records, config=config)
    dataloader_kwargs: dict[str, Any] = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": config.dataloader_num_workers,
        "collate_fn": collate_items,
    }
    if config.dataloader_num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = config.dataloader_prefetch_factor

    logger.info(
        "Starting DataLoader with num_workers=%s prefetch_factor=%s",
        config.dataloader_num_workers,
        config.dataloader_prefetch_factor,
    )
    dataloader = DataLoader(dataset, **dataloader_kwargs)

    with output_path.open("w") as output_file:
        for batch in iter_with_progress(dataloader, total=len(dataset)):
            item = batch[0]
            line_num = int(item["line_num"])
            record = item["record"]
            total += 1
            try:
                if item.get("error"):
                    raise RuntimeError(str(item["error"]))

                if item.get("skip_inference"):
                    output = LocalizedInstanceRecord(
                        image_id=record["image_id"],
                        image_file_name=record["image_file_name"],
                        image_directory=record["image_directory"],
                        image_path=record["image_path"],
                        localizer_backend="sam3.1",
                        identified_categories=record["identified_categories"],
                        instances=[],
                        raw_generation="[]",
                        generation_stats=GenerationStats(),
                    )
                    output_file.write(json.dumps(output.model_dump()) + "\n")
                    continue

                inference_image = item["image"]
                try:
                    image_width, image_height = inference_image.size
                    localized_instances: list[LocalizedInstance] = []
                    raw_generation_payload: list[dict[str, Any]] = []
                    total_generation_time = 0.0
                    accepted_boxes_by_category: dict[str, list[list[float]]] = {}
                    detection_counters: Counter[str] = Counter()

                    for target in item["localization_targets"]:
                        instance_category = normalize_category_name(target["category"])
                        accepted_boxes = accepted_boxes_by_category.setdefault(
                            instance_category, []
                        )
                        for localization_phrase in target["phrases"]:
                            prompt_text = format_category_for_prompt(
                                localization_phrase, config.prompt_template
                            )
                            detections, elapsed = run_native_sam31_for_category(
                                model=model,
                                processor=processor,
                                inference_image=inference_image,
                                category_prompt=prompt_text,
                                config=config,
                            )
                            total_generation_time += elapsed
                            raw_generation_payload.append(
                                {
                                    "category": instance_category,
                                    "localization_phrase": localization_phrase,
                                    "prompt": prompt_text,
                                    "detections": detections,
                                }
                            )

                            for detection in detections:
                                box_xyxy = detection["box_xyxy"]
                                if any(
                                    bbox_iou_xyxy(box_xyxy, accepted_box)
                                    >= config.prompt_alias_iou_dedup_threshold
                                    for accepted_box in accepted_boxes
                                ):
                                    continue
                                accepted_boxes.append(box_xyxy)
                                detection_counters[instance_category] += 1
                                instance_id = (
                                    f"{instance_category}_{detection_counters[instance_category]}"
                                )
                                bbox_norm = pixels_to_bbox_norm_1000(
                                    box_xyxy,
                                    image_width=image_width,
                                    image_height=image_height,
                                )
                                if bbox_norm is None:
                                    continue

                                crop_path = crop_dir / record["image_id"] / f"{instance_id}.jpg"
                                crop_box, (original_image_width, original_image_height) = (
                                    save_image_crop(
                                        image_path=record["image_path"],
                                        bbox_norm_1000=(
                                            bbox_norm.x1,
                                            bbox_norm.y1,
                                            bbox_norm.x2,
                                            bbox_norm.y2,
                                        ),
                                        output_path=str(crop_path),
                                    )
                                )
                                crop_width = crop_box[2] - crop_box[0]
                                crop_height = crop_box[3] - crop_box[1]
                                filtered_by = evaluate_localization_filters(
                                    config.filter_list,
                                    width=crop_width,
                                    height=crop_height,
                                )
                                if filtered_by is not None:
                                    filtered_counts[filtered_by] += 1
                                    if config.debug_save_annotated_images:
                                        crop_path.unlink(missing_ok=True)
                                    else:
                                        filtered_dir = (
                                            crop_dir / record["image_id"] / "filtered_out"
                                        )
                                        filtered_dir.mkdir(parents=True, exist_ok=True)
                                        filtered_path = filtered_dir / f"{instance_id}.jpg"
                                        crop_path.rename(filtered_path)
                                    continue
                                localized_instances.append(
                                    LocalizedInstance(
                                        instance_id=instance_id,
                                        category=instance_category,
                                        object_name=instance_category.replace("_", " "),
                                        bbox_xyxy=BoundingBoxPixels(
                                            x1=crop_box[0],
                                            y1=crop_box[1],
                                            x2=crop_box[2],
                                            y2=crop_box[3],
                                        ),
                                        bbox_norm_1000=bbox_norm,
                                        crop_path=str(crop_path),
                                        crop_bbox_context=CropContext(
                                            padding_ratio=0.02,
                                            image_width=original_image_width,
                                            image_height=original_image_height,
                                        ),
                                        confidence=float(detection["score"])
                                        if detection["score"] is not None
                                        else None,
                                        metadata={
                                            "source_prompt": localization_phrase,
                                            "canonical_category": instance_category,
                                        },
                                    )
                                )
                                instance_counts[instance_category] += 1

                    if config.debug_save_annotated_images and localized_instances:
                        debug_image_path = debug_annotated_images_dir / f"{record['image_id']}.jpg"
                        try:
                            save_annotated_debug_image(
                                image_path=record["image_path"],
                                instances=localized_instances,
                                output_path=debug_image_path,
                            )
                            debug_annotated_images_saved += 1
                        except Exception:
                            debug_annotated_image_errors += 1
                            logger.exception(
                                "Failed to save debug annotated image for line %s", line_num
                            )

                    output = LocalizedInstanceRecord(
                        image_id=record["image_id"],
                        image_file_name=record["image_file_name"],
                        image_directory=record["image_directory"],
                        image_path=record["image_path"],
                        localizer_backend="sam3.1",
                        identified_categories=record["identified_categories"],
                        instances=localized_instances,
                        raw_generation=json.dumps(raw_generation_payload),
                        generation_stats=GenerationStats(generation_time=total_generation_time),
                    )
                    output_file.write(json.dumps(output.model_dump()) + "\n")
                finally:
                    inference_image.close()
            except Exception as exc:
                parse_errors += 1
                logger.exception("Failed to localize line %s: %s", line_num, exc)

    summary = {
        "total_records_processed": total,
        "parse_errors": parse_errors,
        "successful": total - parse_errors,
        "instance_category_counts": dict(sorted(instance_counts.items())),
        "filtered_instance_count": sum(filtered_counts.values()),
        "filtered_instance_counts": dict(sorted(filtered_counts.items())),
        "debug_save_annotated_images": config.debug_save_annotated_images,
        "debug_annotated_images_dir": str(debug_annotated_images_dir)
        if config.debug_save_annotated_images
        else None,
        "debug_annotated_images_saved": debug_annotated_images_saved,
        "debug_annotated_image_errors": debug_annotated_image_errors,
        "max_localization_phrases_per_category": config.max_localization_phrases_per_category,
        "prompt_alias_iou_dedup_threshold": config.prompt_alias_iou_dedup_threshold,
        "num_shards": config.num_shards,
        "shard_index": config.shard_index,
        "output_file": str(output_path),
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2))
    logger.info("Processed %s SAM 3.1 localization records", total - parse_errors)


if __name__ == "__main__":
    main()
