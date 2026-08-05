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
"""Preprocess filtered images into category-identification prompts."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from nvflow.recipes.multimodal.utils.hopchain_sdg_common import (
    build_multimodal_message,
    format_generalized_name_hints,
    load_prompt_template,
)
from nvflow.recipes.multimodal.utils.hopchain_sdg_models import FilteredImageInput

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build category-identification prompts")
    parser.add_argument("--input", required=True, help="FilteredImageInput JSONL")
    parser.add_argument("--output", required=True, help="OpenAI-format JSONL output")
    parser.add_argument("--prompt", required=True, help="Category identification prompt template")
    parser.add_argument("--use-base64", action="store_true")
    parser.add_argument("--max-image-dimension", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    prompt_template = load_prompt_template(args.prompt)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with Path(args.input).open("r") as input_file, output_path.open("w") as output_file:
        for line_num, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record = FilteredImageInput.model_validate(json.loads(line))
                prompt_text = prompt_template.format(
                    image_file_name=record.image_file_name,
                    generalized_name_hints=format_generalized_name_hints(
                        record.filter_generalized_name_hints
                    ),
                )
                message = build_multimodal_message(
                    prompt_text=prompt_text,
                    metadata={
                        "image_id": record.image_id,
                        "image_file_name": record.image_file_name,
                        "image_directory": record.image_directory,
                        "image_path": record.image_path,
                        "prior_generalized_name_hints": record.filter_generalized_name_hints,
                    },
                    image_paths=[record.image_path],
                    use_base64=args.use_base64,
                    max_image_dimension=args.max_image_dimension,
                )
                output_file.write(json.dumps(message) + "\n")
                total += 1
            except Exception as exc:
                logger.exception(
                    "Failed to preprocess category record on line %s: %s", line_num, exc
                )

    logger.info("Created %s category-identification requests", total)


if __name__ == "__main__":
    main()
