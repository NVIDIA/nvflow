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
"""Preprocess instance combinations into query-generation prompts."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from nvflow.recipes.multimodal.utils.hopchain_sdg_common import (
    build_multimodal_message,
    format_object_list_for_prompt,
    load_prompt_template,
)
from nvflow.recipes.multimodal.utils.hopchain_sdg_models import InstanceCombinationRecord

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

NUM_QUERIES_WORD = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build HopChain query-generation prompts")
    parser.add_argument("--input", required=True, help="InstanceCombinationRecord JSONL")
    parser.add_argument("--output", required=True, help="OpenAI-format JSONL output")
    parser.add_argument("--prompt", required=True, help="Exact paper query-generation prompt")
    parser.add_argument("--num-queries", type=int, default=1)
    parser.add_argument("--target-hop-count-info", default="4-5 hops")
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
                record = InstanceCombinationRecord.model_validate(json.loads(line))
                object_list = format_object_list_for_prompt(record.instances)
                prompt_text = prompt_template.format(
                    num_queries=args.num_queries,
                    num_queries_word=NUM_QUERIES_WORD.get(args.num_queries, str(args.num_queries)),
                    object_list=object_list,
                    target_hop_count_info=args.target_hop_count_info,
                )
                image_paths = [record.image_path] + [
                    instance.crop_path for instance in record.instances
                ]
                message = build_multimodal_message(
                    prompt_text=prompt_text,
                    metadata={
                        "image_id": record.image_id,
                        "combination_id": record.combination_id,
                        "image_file_name": record.image_file_name,
                        "image_directory": record.image_directory,
                        "image_path": record.image_path,
                        "instance_ids": record.instance_ids,
                        "combination_size": record.combination_size,
                    },
                    image_paths=image_paths,
                    use_base64=args.use_base64,
                    max_image_dimension=args.max_image_dimension,
                )
                output_file.write(json.dumps(message) + "\n")
                total += 1
            except Exception as exc:
                logger.exception("Failed to preprocess query generation line %s: %s", line_num, exc)

    logger.info("Created %s query-generation requests", total)


if __name__ == "__main__":
    main()
