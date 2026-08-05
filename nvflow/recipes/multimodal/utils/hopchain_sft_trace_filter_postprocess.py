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
"""Postprocess SFT trace judge outputs and select one trace per query."""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, Field

from nvflow.recipes.multimodal.utils.hopchain_sdg_common import extract_json_value
from nvflow.recipes.multimodal.utils.hopchain_sdg_models import (
    SFTReasoningTraceCandidate,
    SFTTraceJudgeDecision,
)
from nvflow.recipes.multimodal.utils.image_filter_models import GenerationStats

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SFTTraceFilterSummary(BaseModel):
    """Summary for SFT trace filtering."""

    candidates_input_file: str
    judge_output_file: str
    output_file: str
    kept_output_file: str
    sft_output_file: str
    total_candidates: int = Field(ge=0)
    answer_match_candidates: int = Field(ge=0)
    answer_mismatch_candidates: int = Field(ge=0)
    judge_pass_candidates: int = Field(ge=0)
    judge_fail_candidates: int = Field(ge=0)
    judge_parse_errors: int = Field(ge=0)
    selected_sft_count: int = Field(ge=0)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Filter SFT reasoning traces")
    parser.add_argument("--candidates", required=True, help="SFTReasoningTraceCandidate JSONL")
    parser.add_argument("--judge-output", required=True, help="Raw SFT trace judge output JSONL")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write final_output.jsonl, kept_output.jsonl, and sft_output.jsonl",
    )
    parser.add_argument("--summary", required=True, help="Summary JSON path")
    return parser.parse_args()


def _response_text(record: dict) -> str:
    return (record.get("generation") or "").strip()


def parse_judge_decision(record: dict) -> SFTTraceJudgeDecision:
    """Parse one judge response."""
    response_text = _response_text(record)
    parsed = extract_json_value(response_text)
    if not isinstance(parsed, dict):
        raise ValueError("Judge response must be a JSON object")
    verdict = str(parsed.get("verdict", "")).strip().lower()
    if verdict not in {"pass", "fail"}:
        raise ValueError(f"Invalid or missing verdict: {verdict!r}")
    reasoning = str(parsed.get("reasoning", "")).strip()
    if not reasoning:
        raise ValueError("Missing reasoning in judge response")
    return SFTTraceJudgeDecision(
        verdict=verdict,
        reasoning=reasoning,
        raw_response=response_text,
        generation_stats=GenerationStats(
            num_generated_tokens=record.get("num_generated_tokens"),
            generation_time=record.get("generation_time"),
        ),
    )


def _sft_record(candidate: SFTReasoningTraceCandidate, judge_decision: dict) -> dict:
    source_record = candidate.source_record
    return {
        **candidate.model_dump(),
        "answer": candidate.hypothetical_answer,
        "teacher_model": candidate.model,
        "image_id": source_record.get("image_id"),
        "image_fullpath": source_record.get("image_fullpath"),
        "query_metadata": source_record.get("query_metadata"),
        "difficulty_pass_rate": source_record.get("difficulty_pass_rate"),
        "difficulty_filter_status": source_record.get("difficulty_filter_status"),
        "sft_filter_status": "selected",
        "sft_trace_judge": judge_decision,
    }


def main() -> None:
    """Entry point."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "final_output.jsonl"
    kept_output_path = output_dir / "kept_output.jsonl"
    sft_output_path = output_dir / "sft_output.jsonl"

    candidates: dict[str, SFTReasoningTraceCandidate] = {}
    answer_mismatch_candidates = 0
    with Path(args.candidates).open("r") as input_file:
        for line_num, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                candidate = SFTReasoningTraceCandidate.model_validate(json.loads(line))
                candidates[candidate.sft_sample_id] = candidate
                if not candidate.answer_matches_hypothetical:
                    answer_mismatch_candidates += 1
            except Exception as exc:
                logger.exception("Failed to load SFT trace candidate line %s: %s", line_num, exc)

    judge_decisions: dict[str, SFTTraceJudgeDecision] = {}
    judge_parse_errors = 0
    with Path(args.judge_output).open("r") as judge_file:
        for line_num, line in enumerate(judge_file, start=1):
            if not line.strip():
                continue
            try:
                raw_judge = json.loads(line)
                sft_sample_id = raw_judge["_metadata"]["sft_sample_id"]
                judge_decisions[sft_sample_id] = parse_judge_decision(raw_judge)
            except Exception as exc:
                judge_parse_errors += 1
                logger.exception("Failed to parse SFT trace judge line %s: %s", line_num, exc)

    passed_by_query: dict[str, list[tuple[SFTReasoningTraceCandidate, SFTTraceJudgeDecision]]] = (
        defaultdict(list)
    )
    judge_pass_candidates = 0
    judge_fail_candidates = 0
    answer_match_candidates = 0

    with output_path.open("w") as output_file, kept_output_path.open("w") as kept_file:
        for candidate in sorted(
            candidates.values(), key=lambda item: (item.query_id, item.sample_index)
        ):
            status = "rejected_answer_mismatch"
            judge_decision = None
            if candidate.answer_matches_hypothetical:
                answer_match_candidates += 1
                judge_decision = judge_decisions.get(candidate.sft_sample_id)
                if judge_decision is None:
                    status = "rejected_missing_judge"
                    judge_fail_candidates += 1
                elif judge_decision.verdict == "pass":
                    status = "kept"
                    judge_pass_candidates += 1
                    passed_by_query[candidate.query_id].append((candidate, judge_decision))
                else:
                    status = "rejected_trace_judge"
                    judge_fail_candidates += 1

            record = {
                **candidate.model_dump(),
                "sft_filter_status": status,
                "sft_trace_judge": judge_decision.model_dump() if judge_decision else None,
            }
            output_file.write(json.dumps(record) + "\n")
            if status == "kept":
                kept_file.write(json.dumps(record) + "\n")

    selected_sft_count = 0
    with sft_output_path.open("w") as sft_file:
        for query_id in sorted(passed_by_query):
            candidate, judge_decision = sorted(
                passed_by_query[query_id],
                key=lambda item: item[0].sample_index,
            )[0]
            sft_file.write(json.dumps(_sft_record(candidate, judge_decision.model_dump())) + "\n")
            selected_sft_count += 1

    summary = SFTTraceFilterSummary(
        candidates_input_file=args.candidates,
        judge_output_file=args.judge_output,
        output_file=str(output_path),
        kept_output_file=str(kept_output_path),
        sft_output_file=str(sft_output_path),
        total_candidates=len(candidates),
        answer_match_candidates=answer_match_candidates,
        answer_mismatch_candidates=answer_mismatch_candidates,
        judge_pass_candidates=judge_pass_candidates,
        judge_fail_candidates=judge_fail_candidates,
        judge_parse_errors=judge_parse_errors,
        selected_sft_count=selected_sft_count,
    )
    Path(args.summary).write_text(summary.model_dump_json(indent=2))
    logger.info(
        "Selected %d SFT traces from %d candidates (%d judge pass, %d judge fail)",
        selected_sft_count,
        len(candidates),
        judge_pass_candidates,
        judge_fail_candidates,
    )


if __name__ == "__main__":
    main()
