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
"""Preprocess answer-matching SFT trace candidates into judge requests."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from nvflow.recipes.multimodal.utils.hopchain_llm_judge_common import load_prompt_template
from nvflow.recipes.multimodal.utils.hopchain_sdg_models import SFTReasoningTraceCandidate

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build SFT trace judge prompts")
    parser.add_argument("--input", required=True, help="SFTReasoningTraceCandidate JSONL")
    parser.add_argument("--output", required=True, help="OpenAI-format judge prompt JSONL")
    parser.add_argument("--prompt", required=True, help="SFT trace judge prompt")
    return parser.parse_args()


def format_synthetic_hops(source_record: dict[str, Any]) -> str:
    """Format synthetic reasoning hops for the judge prompt."""
    query_metadata = source_record.get("query_metadata") or {}
    hops = query_metadata.get("reasoning_hops") or []
    formatted: list[str] = []
    for hop in hops:
        if not isinstance(hop, dict):
            continue
        objects = ", ".join(str(item) for item in hop.get("objects_involved", []))
        parts = [
            f"Hop {hop.get('hop_number', '?')}",
            f"type: {hop.get('hop_type', '')}",
            f"description: {hop.get('description', '')}",
            f"output: {hop.get('output', '')}",
        ]
        if objects:
            parts.append(f"objects: {objects}")
        formatted.append(" | ".join(parts))
    return "\n".join(formatted)


def main() -> None:
    """Entry point."""
    args = parse_args()
    prompt_template = load_prompt_template(args.prompt)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_candidates = 0
    judged_candidates = 0
    skipped_answer_mismatch = 0
    with Path(args.input).open("r") as input_file, output_path.open("w") as output_file:
        for line_num, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                candidate = SFTReasoningTraceCandidate.model_validate(json.loads(line))
                total_candidates += 1
                if not candidate.answer_matches_hypothetical:
                    skipped_answer_mismatch += 1
                    continue

                prompt_text = prompt_template.format(
                    question=candidate.question,
                    hypothetical_answer=candidate.hypothetical_answer,
                    synthetic_hops=format_synthetic_hops(candidate.source_record),
                    reasoning_trace=candidate.reasoning_trace,
                    final_answer=candidate.final_answer,
                )
                message = {
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": prompt_text}],
                        }
                    ],
                    "_metadata": {
                        "query_id": candidate.query_id,
                        "sft_sample_id": candidate.sft_sample_id,
                        "sample_index": candidate.sample_index,
                    },
                }
                output_file.write(json.dumps(message) + "\n")
                judged_candidates += 1
            except Exception as exc:
                logger.exception("Failed to preprocess SFT trace judge line %s: %s", line_num, exc)

    logger.info(
        "Prepared %d trace judge requests from %d candidates (%d answer mismatches skipped)",
        judged_candidates,
        total_candidates,
        skipped_answer_mismatch,
    )


if __name__ == "__main__":
    main()
