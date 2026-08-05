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
"""Shared helpers for HopChain SDG stages."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from nvflow.recipes.multimodal.utils.hopchain_sdg_models import LocalizedInstance
from nvflow.recipes.multimodal.utils.image_utils import get_image_url

logger = logging.getLogger(__name__)

FORBIDDEN_PUBLIC_TERMS = (
    "bounding box",
    "bounding boxes",
    "bbox",
    "patch image",
    "patch images",
    "patch",
    "patches",
    "mask",
    "masks",
    "crop image",
    "cropped patch",
    "0-1000 coordinate",
    "coordinates in the 0-1000",
)


def load_prompt_template(prompt_path: str) -> str:
    """Load a text prompt template from disk."""
    return Path(prompt_path).read_text()


def normalize_category_name(category: str) -> str:
    """Normalize a semantic category into a simple lowercase identifier."""
    normalized = re.sub(r"[^a-z0-9]+", "_", category.strip().lower()).strip("_")
    return normalized or "unknown"


def dedupe_preserve_order(values: list[str]) -> list[str]:
    """Dedupe a list while preserving first appearance order."""
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


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


def extract_json_value(text: str) -> object:
    """Extract the first JSON object or array from a model response."""
    object_start = text.find("{")
    array_start = text.find("[")
    starts = [pos for pos in (object_start, array_start) if pos != -1]
    if not starts:
        raise ValueError("No JSON value found in model response")

    start = min(starts)
    opening = text[start]
    closing = "}" if opening == "{" else "]"
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
        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return json.loads(text[start : index + 1])
    raise ValueError("Unbalanced JSON value in model response")


def build_multimodal_message(
    *,
    prompt_text: str,
    metadata: dict[str, Any],
    image_paths: list[str],
    use_base64: bool,
    max_image_dimension: int | None,
) -> dict[str, Any]:
    """Build an OpenAI-format multimodal message with one or more images."""
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
    for image_path in image_paths:
        image_url = get_image_url(
            image_path=image_path,
            use_base64=use_base64,
            max_dimension=max_image_dimension,
        )
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    return {
        "messages": [{"role": "user", "content": content}],
        "_metadata": metadata,
    }


def format_generalized_name_hints(hints: list[str]) -> str:
    """Render optional generalized-name hints for prompt templates."""
    if not hints:
        return "None"
    return ", ".join(dedupe_preserve_order(hints))


def format_object_list_for_prompt(instances: list[LocalizedInstance]) -> str:
    """Render localized instances for the paper-style query-generation prompt."""
    lines: list[str] = []
    for index, instance in enumerate(instances, start=2):
        bbox = instance.bbox_norm_1000
        lines.append(
            f"- {instance.instance_id}: category={instance.category}; "
            f"object_name={instance.object_name}; "
            f"patch_image=Image {index}; "
            f"bbox_norm_1000=[{bbox.x1}, {bbox.y1}, {bbox.x2}, {bbox.y2}]"
        )
    return "\n".join(lines)
