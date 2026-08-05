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
"""Expand kept HopChain RL rows into k SFT reasoning-trace requests."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from nvflow.recipes.multimodal.utils.hopchain_llm_judge_common import load_prompt_template
from nvflow.recipes.multimodal.utils.hopchain_sdg_common import build_multimodal_message
from nvflow.recipes.multimodal.utils.hopchain_sdg_models import ReconciledHopChainQuery

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build SFT reasoning-trace prompts")
    parser.add_argument("--input", required=True, help="Kept HopChain RL JSONL")
    parser.add_argument("--output", required=True, help="OpenAI-format JSONL output")
    parser.add_argument("--prompt", required=True, help="SFT trace generation prompt")
    parser.add_argument("--k", type=int, default=3, help="Samples per query")
    parser.add_argument("--use-base64", action="store_true")
    parser.add_argument("--max-image-dimension", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    prompt_template = load_prompt_template(args.prompt)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_records = 0
    total_requests = 0
    with Path(args.input).open("r") as input_file, output_path.open("w") as output_file:
        for line_num, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                raw_record = json.loads(line)
                record = ReconciledHopChainQuery.model_validate(raw_record)
                prompt_text = prompt_template.format(question=record.question)

                for sample_index in range(args.k):
                    message = build_multimodal_message(
                        prompt_text=prompt_text,
                        metadata={
                            "query_id": record.query_id,
                            "sample_index": sample_index,
                            "k": args.k,
                            "question": record.question,
                            "hypothetical_answer": record.hypothetical_answer,
                            "record": raw_record,
                        },
                        image_paths=[record.image_fullpath],
                        use_base64=args.use_base64,
                        max_image_dimension=args.max_image_dimension,
                    )
                    output_file.write(json.dumps(message) + "\n")
                    total_requests += 1
                total_records += 1
            except Exception as exc:
                logger.exception(
                    "Failed to preprocess SFT trace generation line %s: %s", line_num, exc
                )

    logger.info(
        "Preprocessed %d source records into %d SFT trace requests (k=%d)",
        total_records,
        total_requests,
        args.k,
    )


if __name__ == "__main__":
    main()
