#!/usr/bin/env python
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
"""Reconcile multiple HopChain LLM judge outputs into a final accepted dataset."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field

from nvflow.recipes.multimodal.utils.hopchain_llm_judge_common import normalize_answer_text
from nvflow.recipes.multimodal.utils.hopchain_sdg_models import (
    LLMJudgeAnswerSummary,
    LLMJudgeEvaluationRecord,
    ReconciledHopChainQuery,
    VerifiedHopChainQuery,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ReconciliationSummary(BaseModel):
    """Summary metadata for judge reconciliation."""

    input_file: str
    judge_input_files: list[str] = Field(default_factory=list)
    reconciled_file: str
    accepted_file: str
    rejected_file: str
    total_records_processed: int = Field(ge=0)
    status_counts: dict[str, int] = Field(default_factory=dict)
    all_judges_returned_valid_answers_count: int = Field(default=0, ge=0)
    all_judges_agreed_count: int = Field(default=0, ge=0)
    all_judges_agreed_with_hypothetical_answer_count: int = Field(default=0, ge=0)
    all_judges_agreed_but_not_hypothetical_answer_count: int = Field(default=0, ge=0)
    all_judges_returned_valid_answers_but_disagreed_count: int = Field(default=0, ge=0)
    incomplete_or_missing_judge_answers_count: int = Field(default=0, ge=0)
    rejection_reason_counts: dict[str, int] = Field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Reconcile HopChain LLM judge outputs")
    parser.add_argument("--input", required=True, help="VerifiedHopChainQuery JSONL")
    parser.add_argument("--output", required=True, help="All reconciliation results JSONL")
    parser.add_argument(
        "--output-dir", required=True, help="Directory for accepted and rejected files"
    )
    parser.add_argument("--summary", required=True, help="Summary JSON path")
    parser.add_argument(
        "--judge-input",
        action="append",
        dest="judge_inputs",
        required=True,
        help="LLMJudgeEvaluationRecord JSONL. Pass once per judge.",
    )
    return parser.parse_args()


def load_candidates(input_path: Path) -> list[VerifiedHopChainQuery]:
    """Load candidate query records."""
    candidates: list[VerifiedHopChainQuery] = []
    with input_path.open("r") as input_file:
        for line_num, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                candidates.append(VerifiedHopChainQuery.model_validate(json.loads(line)))
            except Exception as exc:
                logger.exception("Failed to parse candidate line %s: %s", line_num, exc)
    return candidates


def load_judge_records(input_path: Path) -> dict[str, LLMJudgeEvaluationRecord]:
    """Load one judge output file keyed by query_id."""
    records: dict[str, LLMJudgeEvaluationRecord] = {}
    with input_path.open("r") as input_file:
        for line_num, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record = LLMJudgeEvaluationRecord.model_validate(json.loads(line))
                records[record.query_id] = record
            except Exception as exc:
                logger.exception(
                    "Failed to parse judge line %s from %s: %s", line_num, input_path, exc
                )
    return records


def reconcile_candidate_record(
    candidate: VerifiedHopChainQuery,
    judge_records: list[LLMJudgeEvaluationRecord | None],
) -> ReconciledHopChainQuery:
    """Reconcile one candidate across all judge outputs."""
    judge_summaries: list[LLMJudgeAnswerSummary] = []
    judge_names: list[str] = []
    normalized_answers: list[str] = []
    raw_answers: list[str] = []
    rejection_reasons: list[str] = []
    missing_judge_record = False
    invalid_present_judge_answer = False

    for judge_record in judge_records:
        if judge_record is None:
            missing_judge_record = True
            rejection_reasons.append("missing_judge_record")
            continue

        judge_names.append(judge_record.llm_judge.judge_name)
        judge_summaries.append(
            LLMJudgeAnswerSummary(
                judge_name=judge_record.llm_judge.judge_name,
                provider=judge_record.llm_judge.provider,
                model=judge_record.llm_judge.model,
                answer=judge_record.llm_judge.answer,
                normalized_answer=judge_record.llm_judge.normalized_answer,
            )
        )

        if judge_record.judge_status != "parsed":
            invalid_present_judge_answer = True
            rejection_reasons.append(
                f"judge_status_{judge_record.llm_judge.judge_name}_{judge_record.judge_status}"
            )
            continue

        normalized_answer = judge_record.llm_judge.normalized_answer
        if not normalized_answer:
            invalid_present_judge_answer = True
            rejection_reasons.append(f"empty_judge_answer_{judge_record.llm_judge.judge_name}")
            continue

        normalized_answers.append(normalized_answer)
        raw_answers.append(judge_record.llm_judge.answer)

    unique_normalized_answers = sorted(set(normalized_answers))
    if invalid_present_judge_answer and not missing_judge_record:
        rejection_reasons.append("incomplete_judge_answers")
    if len(unique_normalized_answers) > 1:
        rejection_reasons.append("judge_answer_disagreement")
    if not normalized_answers:
        rejection_reasons.append("no_valid_judge_answers")

    consensus_answer = raw_answers[0] if len(unique_normalized_answers) == 1 and raw_answers else ""
    consensus_normalized_answer = (
        unique_normalized_answers[0] if len(unique_normalized_answers) == 1 else ""
    )
    consensus_matches_candidate = bool(
        consensus_normalized_answer
    ) and consensus_normalized_answer == normalize_answer_text(candidate.hypothetical_answer)
    if consensus_normalized_answer and not consensus_matches_candidate:
        rejection_reasons.append("judge_consensus_mismatch_hypothetical_answer")

    deduped_rejection_reasons = list(dict.fromkeys(rejection_reasons))
    reconciliation_status = "accepted" if not deduped_rejection_reasons else "rejected"
    return ReconciledHopChainQuery(
        **candidate.model_dump(),
        llm_judge_names=judge_names,
        llm_judge_answers=judge_summaries,
        llm_judge_consensus_answer=consensus_answer,
        llm_judge_consensus_normalized_answer=consensus_normalized_answer,
        llm_judge_consensus_matches_hypothetical_answer=consensus_matches_candidate,
        llm_judge_reconciliation_status=reconciliation_status,
        llm_judge_rejection_reasons=deduped_rejection_reasons,
    )


def main() -> None:
    """Entry point."""
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    accepted_path = output_dir / "final_candidates.jsonl"
    rejected_path = output_dir / "rejected_candidates.jsonl"

    candidates = load_candidates(input_path)
    judge_maps = [load_judge_records(Path(judge_input)) for judge_input in args.judge_inputs]

    status_counts: Counter[str] = Counter()
    rejection_reason_counts: Counter[str] = Counter()
    all_judges_returned_valid_answers_count = 0
    all_judges_agreed_count = 0
    all_judges_agreed_with_hypothetical_answer_count = 0
    all_judges_agreed_but_not_hypothetical_answer_count = 0
    all_judges_returned_valid_answers_but_disagreed_count = 0
    incomplete_or_missing_judge_answers_count = 0
    with (
        output_path.open("w") as output_file,
        accepted_path.open("w") as accepted_file,
        rejected_path.open("w") as rejected_file,
    ):
        for candidate in candidates:
            judge_records = [judge_map.get(candidate.query_id) for judge_map in judge_maps]
            reconciled = reconcile_candidate_record(candidate, judge_records)
            all_judges_returned_valid_answers = all(
                judge_record is not None
                and judge_record.judge_status == "parsed"
                and bool(judge_record.llm_judge.normalized_answer)
                for judge_record in judge_records
            )
            judges_agreed = bool(reconciled.llm_judge_consensus_normalized_answer)

            if all_judges_returned_valid_answers:
                all_judges_returned_valid_answers_count += 1
                if judges_agreed:
                    all_judges_agreed_count += 1
                    if reconciled.llm_judge_consensus_matches_hypothetical_answer:
                        all_judges_agreed_with_hypothetical_answer_count += 1
                    else:
                        all_judges_agreed_but_not_hypothetical_answer_count += 1
                else:
                    all_judges_returned_valid_answers_but_disagreed_count += 1
            else:
                incomplete_or_missing_judge_answers_count += 1

            status_counts[reconciled.llm_judge_reconciliation_status] += 1
            rejection_reason_counts.update(reconciled.llm_judge_rejection_reasons)
            payload = reconciled.model_dump_json()
            output_file.write(payload + "\n")
            if reconciled.llm_judge_reconciliation_status == "accepted":
                accepted_file.write(payload + "\n")
            else:
                rejected_file.write(payload + "\n")

    summary = ReconciliationSummary(
        input_file=str(input_path),
        judge_input_files=args.judge_inputs,
        reconciled_file=str(output_path),
        accepted_file=str(accepted_path),
        rejected_file=str(rejected_path),
        total_records_processed=len(candidates),
        status_counts=dict(status_counts),
        all_judges_returned_valid_answers_count=all_judges_returned_valid_answers_count,
        all_judges_agreed_count=all_judges_agreed_count,
        all_judges_agreed_with_hypothetical_answer_count=all_judges_agreed_with_hypothetical_answer_count,
        all_judges_agreed_but_not_hypothetical_answer_count=all_judges_agreed_but_not_hypothetical_answer_count,
        all_judges_returned_valid_answers_but_disagreed_count=all_judges_returned_valid_answers_but_disagreed_count,
        incomplete_or_missing_judge_answers_count=incomplete_or_missing_judge_answers_count,
        rejection_reason_counts=dict(rejection_reason_counts),
    )
    Path(args.summary).write_text(summary.model_dump_json(indent=2))
    logger.info("Reconciled %s candidate queries", len(candidates))
    logger.info(
        (
            "Reconciliation stats: valid_answers=%s, judges_agreed=%s, "
            "agreed_with_hypothetical=%s, agreed_without_hypothetical=%s, "
            "valid_but_disagreed=%s, incomplete_or_missing=%s"
        ),
        all_judges_returned_valid_answers_count,
        all_judges_agreed_count,
        all_judges_agreed_with_hypothetical_answer_count,
        all_judges_agreed_but_not_hypothetical_answer_count,
        all_judges_returned_valid_answers_but_disagreed_count,
        incomplete_or_missing_judge_answers_count,
    )


if __name__ == "__main__":
    main()
