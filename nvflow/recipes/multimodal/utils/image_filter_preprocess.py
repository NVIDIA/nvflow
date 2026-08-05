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
"""Preprocess image directory catalogs into OpenAI vision JSONL."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

from nvflow.recipes.multimodal.utils.image_filter_models import (
    ImageDirectoryCatalog,
    SelectedImageRecord,
)
from nvflow.recipes.multimodal.utils.image_utils import discover_images, get_image_url

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_prompt_template(prompt_path: str) -> str:
    """Load the image filtering prompt text from disk."""
    return Path(prompt_path).read_text()


def build_openai_vision_message(
    *,
    image_path: str,
    prompt_text: str,
    metadata: SelectedImageRecord,
    use_base64: bool,
    max_image_dimension: int | None,
) -> dict:
    """Build the message format expected by nemo-skills generate."""
    image_url = get_image_url(
        image_path=image_path,
        use_base64=use_base64,
        max_dimension=max_image_dimension,
    )
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ],
        "_metadata": metadata.model_dump(),
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI args for image filtering preprocess."""
    parser = argparse.ArgumentParser(
        description="Preprocess image catalogs for HopChain image filtering"
    )
    parser.add_argument(
        "--input-catalog", required=True, help="Input JSON file containing image directories"
    )
    parser.add_argument("--output", required=True, help="Output JSONL file in OpenAI vision format")
    parser.add_argument(
        "--catalog-output", required=True, help="Output JSONL file with resolved image catalog"
    )
    parser.add_argument("--prompt", required=True, help="Prompt text file")
    parser.add_argument(
        "--use-base64", action="store_true", help="Encode images as base64 data URIs"
    )
    parser.add_argument(
        "--max-image-dimension",
        type=int,
        default=None,
        help="Resize images before base64 encoding when provided",
    )
    return parser.parse_args()


def main() -> None:
    """Convert image directories into JSONL requests for model inference."""
    args = parse_args()

    prompt_template = load_prompt_template(args.prompt)
    catalog_data = json.loads(Path(args.input_catalog).read_text())
    catalog = ImageDirectoryCatalog.model_validate(catalog_data)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_output_path = Path(args.catalog_output)
    catalog_output_path.parent.mkdir(parents=True, exist_ok=True)

    total_selected = 0
    seen_paths: set[str] = set()

    with output_path.open("w") as message_file, catalog_output_path.open("w") as catalog_file:
        for spec_index, spec in enumerate(catalog.root):
            discovered = discover_images(spec.directory, recursive=spec.recursive)
            if spec.shuffle_seed is not None:
                random.Random(spec.shuffle_seed).shuffle(discovered)
                logger.info(
                    "Catalog entry %s: shuffled %s discovered images from %s with seed=%s",
                    spec_index,
                    len(discovered),
                    spec.directory,
                    spec.shuffle_seed,
                )
            slice_start = spec.start_index if spec.start_index is not None else 0
            slice_end = spec.end_index if spec.end_index is not None else len(discovered)
            selected = discovered[slice_start:slice_end]

            logger.info(
                "Catalog entry %s: %s images selected from %s [%s:%s]",
                spec_index,
                len(selected),
                spec.directory,
                slice_start,
                slice_end,
            )

            for local_offset, image_path in enumerate(selected):
                if image_path in seen_paths:
                    logger.warning("Skipping duplicate image path from catalog: %s", image_path)
                    continue
                seen_paths.add(image_path)

                metadata = SelectedImageRecord(
                    image_file_name=Path(image_path)
                    .relative_to(Path(spec.directory).resolve())
                    .as_posix(),
                    image_directory=spec.directory,
                    catalog_entry_index=spec_index,
                    source_image_index=slice_start + local_offset,
                    start_index=spec.start_index,
                    end_index=spec.end_index,
                    shuffle_seed=spec.shuffle_seed,
                )

                try:
                    message = build_openai_vision_message(
                        image_path=image_path,
                        prompt_text=prompt_template,
                        metadata=metadata,
                        use_base64=args.use_base64,
                        max_image_dimension=args.max_image_dimension,
                    )
                    message_file.write(json.dumps(message) + "\n")
                    catalog_file.write(json.dumps(metadata.model_dump()) + "\n")
                    total_selected += 1
                except Exception as exc:
                    logger.exception("Error preprocessing image %s: %s", image_path, exc)

    if total_selected == 0:
        raise ValueError("No images were selected from the provided catalog")

    logger.info("Created %s OpenAI vision requests at %s", total_selected, output_path)
    logger.info("Saved resolved image catalog at %s", catalog_output_path)


if __name__ == "__main__":
    main()
