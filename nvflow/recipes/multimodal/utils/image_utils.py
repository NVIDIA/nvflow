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
"""Image discovery and encoding helpers for multimodal stages."""

from __future__ import annotations

import base64
import io
import logging
from functools import lru_cache
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".tif"}
DEFAULT_MAX_IMAGE_DIMENSION = 2048
IMAGE_CACHE_SIZE = 256


def discover_images(images_dir: str | Path, recursive: bool = True) -> list[str]:
    """Discover supported images in a directory, sorted by absolute path."""
    root = Path(images_dir)
    if not root.exists():
        raise FileNotFoundError(f"Images directory not found: {root}")

    image_paths: list[Path] = []
    patterns = [f"*{ext}" for ext in IMAGE_EXTENSIONS] + [
        f"*{ext.upper()}" for ext in IMAGE_EXTENSIONS
    ]
    for pattern in patterns:
        matches = root.rglob(pattern) if recursive else root.glob(pattern)
        image_paths.extend(matches)

    return sorted({str(path.resolve()) for path in image_paths})


def _load_and_convert_to_rgb(image_path: str) -> Image.Image:
    """Load an image and convert it to RGB mode for JPEG encoding."""
    image = Image.open(image_path)
    if image.mode in ("RGBA", "P", "LA"):
        background = Image.new("RGB", image.size, (255, 255, 255))
        if image.mode == "P":
            image = image.convert("RGBA")
        background.paste(image, mask=image.split()[-1] if image.mode in ("RGBA", "LA") else None)
        image = background
    elif image.mode != "RGB":
        image = image.convert("RGB")
    return image


def _resize_image(image_path: str, max_dimension: int) -> Image.Image:
    """Resize an image to fit within max_dimension while preserving aspect ratio."""
    image = _load_and_convert_to_rgb(image_path)
    original_size = image.size
    image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
    logger.info(
        "Resized image %s: %sx%s -> %sx%s",
        Path(image_path).name,
        original_size[0],
        original_size[1],
        image.size[0],
        image.size[1],
    )
    return image


@lru_cache(maxsize=IMAGE_CACHE_SIZE)
def encode_image_as_data_uri(image_path: str, max_dimension: int | None = None) -> str:
    """Encode an image as a base64 JPEG data URI."""
    image = (
        _resize_image(image_path, max_dimension)
        if max_dimension
        else _load_and_convert_to_rgb(image_path)
    )
    try:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{image_data}"
    finally:
        image.close()


def get_image_url(
    image_path: str, use_base64: bool = False, max_dimension: int | None = None
) -> str:
    """Return a `file://` or base64 image URL for OpenAI-compatible APIs."""
    if use_base64:
        return encode_image_as_data_uri(image_path, max_dimension=max_dimension)
    return f"file://{Path(image_path).resolve()}"


def bbox_norm_1000_to_pixels(
    image_width: int,
    image_height: int,
    bbox_norm_1000: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Convert a normalized 0-1000 bbox into absolute pixel coordinates."""
    x1, y1, x2, y2 = bbox_norm_1000
    px1 = max(0, min(image_width - 1, round(x1 / 1000 * image_width)))
    py1 = max(0, min(image_height - 1, round(y1 / 1000 * image_height)))
    px2 = max(px1 + 1, min(image_width, round(x2 / 1000 * image_width)))
    py2 = max(py1 + 1, min(image_height, round(y2 / 1000 * image_height)))
    return px1, py1, px2, py2


def save_image_crop(
    *,
    image_path: str,
    bbox_norm_1000: tuple[int, int, int, int],
    output_path: str,
    padding_ratio: float = 0.02,
) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
    """Crop an image using normalized coordinates and save as JPEG."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as image:
        if image.mode in ("RGBA", "P", "LA"):
            background = Image.new("RGB", image.size, (255, 255, 255))
            if image.mode == "P":
                image = image.convert("RGBA")
            background.paste(
                image, mask=image.split()[-1] if image.mode in ("RGBA", "LA") else None
            )
            image = background
        elif image.mode != "RGB":
            image = image.convert("RGB")

        width, height = image.size
        px1, py1, px2, py2 = bbox_norm_1000_to_pixels(width, height, bbox_norm_1000)

        pad_x = round((px2 - px1) * padding_ratio)
        pad_y = round((py2 - py1) * padding_ratio)
        crop_box = (
            max(0, px1 - pad_x),
            max(0, py1 - pad_y),
            min(width, px2 + pad_x),
            min(height, py2 + pad_y),
        )
        cropped = image.crop(crop_box)
        cropped.save(output, format="JPEG", quality=90)
        cropped.close()
        return crop_box, (width, height)
