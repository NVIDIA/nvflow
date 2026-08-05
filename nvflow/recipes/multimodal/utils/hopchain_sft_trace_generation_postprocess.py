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
"""Postprocess raw SFT reasoning-trace generations into typed candidates."""

from __future__ import annotations

import argparse
import json
import logging
import uuid
from pathlib import Path

from nvflow.recipes.multimodal.utils.hopchain_llm_judge_common import (
    normalize_answer_text,
    parse_judge_response_text,
)
from nvflow.recipes.multimodal.utils.hopchain_sdg_models import SFTReasoningTraceCandidate
from nvflow.recipes.multimodal.utils.image_filter_models import GenerationStats

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Postprocess SFT reasoning-trace generations")
    parser.add_argument("--input", required=True, help="Raw nemo-skills output JSONL")
    parser.add_argument("--output", required=True, help="Correct SFTReasoningTraceCandidate JSONL")
    parser.add_argument(
        "--incorrect-output", required=True, help="Incorrect-answer trace candidate JSONL"
    )
    parser.add_argument("--summary", required=True, help="Summary JSON path")
    parser.add_argument("--model", required=True, help="Model key used for trace generation")
    return parser.parse_args()


def _generation_text(record: dict) -> str:
    return (record.get("generation") or "").strip()


def parse_trace_response(raw_output: dict) -> tuple[str, str, str]:
    """Parse teacher output into reasoning and final answer.

    SFT generation uses the same XML answer-question prompt as the LLM judge and
    difficulty filter, so the answer must come from the ``<final_answer>`` tag.
    The SFT reasoning trace must be the model's captured ``reasoning_content``,
    not the visible ``<reasoning>`` field from the prompt response.
    """
    generation_text = _generation_text(raw_output)
    reasoning_trace = (raw_output.get("reasoning_content") or "").strip()
    parsed_response = parse_judge_response_text(generation_text)
    final_answer = parsed_response.final_answer
    if not reasoning_trace:
        raise ValueError("Missing server-side reasoning trace")
    return reasoning_trace.strip(), final_answer.strip(), generation_text


def main() -> None:
    """Entry point."""
    args = parse_args()
    output_path = Path(args.output)
    incorrect_output_path = Path(args.incorrect_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    incorrect_output_path.parent.mkdir(parents=True, exist_ok=True)

    total_records = 0
    parse_errors = 0
    answer_match_count = 0
    incorrect_answer_count = 0
    with (
        Path(args.input).open("r") as input_file,
        output_path.open("w") as output_file,
        incorrect_output_path.open("w") as incorrect_output_file,
    ):
        for line_num, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            total_records += 1
            try:
                raw_output = json.loads(line)
                metadata = raw_output["_metadata"]
                reasoning_trace, final_answer, generation_text = parse_trace_response(raw_output)

                normalized_hypothetical_answer = normalize_answer_text(
                    metadata["hypothetical_answer"]
                )
                normalized_final_answer = normalize_answer_text(final_answer)
                answer_matches = normalized_final_answer == normalized_hypothetical_answer
                if answer_matches:
                    answer_match_count += 1

                query_id = str(metadata["query_id"])
                sample_index = int(metadata["sample_index"])
                candidate = SFTReasoningTraceCandidate(
                    query_id=query_id,
                    sft_sample_id=str(
                        uuid.uuid5(uuid.NAMESPACE_URL, f"{query_id}:sft:{sample_index}")
                    ),
                    sample_index=sample_index,
                    k=int(metadata["k"]),
                    model=args.model,
                    question=str(metadata["question"]),
                    hypothetical_answer=str(metadata["hypothetical_answer"]),
                    final_answer=final_answer,
                    normalized_hypothetical_answer=normalized_hypothetical_answer,
                    normalized_final_answer=normalized_final_answer,
                    answer_matches_hypothetical=answer_matches,
                    reasoning_trace=reasoning_trace,
                    raw_generation=generation_text,
                    source_record=metadata["record"],
                    generation_stats=GenerationStats(
                        num_generated_tokens=raw_output.get("num_generated_tokens"),
                        generation_time=raw_output.get("generation_time"),
                    ),
                )
                if answer_matches:
                    output_file.write(json.dumps(candidate.model_dump()) + "\n")
                else:
                    incorrect_output_file.write(json.dumps(candidate.model_dump()) + "\n")
                    incorrect_answer_count += 1
            except Exception as exc:
                parse_errors += 1
                logger.exception(
                    "Failed to postprocess SFT trace generation line %s: %s", line_num, exc
                )

    summary = {
        "total_generation_records": total_records,
        "parse_errors": parse_errors,
        "answer_match_count": answer_match_count,
        "incorrect_answer_count": incorrect_answer_count,
        "output_file": str(output_path),
        "incorrect_output_file": str(incorrect_output_path),
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2))
    logger.info(
        "Postprocessed SFT traces (%d answer matches, %d incorrect answers, %d parse errors)",
        answer_match_count,
        incorrect_answer_count,
        parse_errors,
    )


if __name__ == "__main__":
    main()
