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
"""Machine-side verification for generated HopChain queries."""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path

from nvflow.recipes.multimodal.utils.hopchain_sdg_common import FORBIDDEN_PUBLIC_TERMS
from nvflow.recipes.multimodal.utils.hopchain_sdg_models import (
    GeneratedHopChainQuery,
    VerificationMetadata,
    VerifiedHopChainQuery,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

NUMERIC_PATTERN = re.compile(r"^-?\d+(\.\d+)?$")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Verify HopChain generated queries")
    parser.add_argument("--input", required=True, help="GeneratedHopChainQuery JSONL")
    parser.add_argument("--output", required=True, help="VerifiedHopChainQuery JSONL")
    parser.add_argument(
        "--output-dir", required=True, help="Directory for accepted/rejected dataset files"
    )
    parser.add_argument("--summary", required=True, help="Summary JSON path")
    parser.add_argument("--min-hop-count", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    accepted_path = output_dir / "final_candidates.jsonl"
    rejected_path = output_dir / "rejected_candidates.jsonl"

    status_counts: Counter[str] = Counter()
    total = 0
    with (
        Path(args.input).open("r") as input_file,
        output_path.open("w") as output_file,
        accepted_path.open("w") as accepted_file,
        rejected_path.open("w") as rejected_file,
    ):
        for line_num, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            total += 1
            try:
                record = GeneratedHopChainQuery.model_validate(json.loads(line))
                rejection_reasons: list[str] = []
                forbidden_hits = [
                    term
                    for term in FORBIDDEN_PUBLIC_TERMS
                    if term.lower() in record.question.lower()
                ]
                instance_id_hits = [
                    iid
                    for iid in record.involved_instance_ids
                    if re.search(r"\b" + re.escape(iid) + r"\b", record.question, re.IGNORECASE)
                ]
                if not record.question.strip():
                    rejection_reasons.append("empty_question")
                if not record.hypothetical_answer.strip():
                    rejection_reasons.append("missing_hypothetical_answer")
                elif not NUMERIC_PATTERN.match(record.hypothetical_answer.strip()):
                    rejection_reasons.append("hypothetical_answer_not_numeric")
                if record.hop_count < args.min_hop_count:
                    rejection_reasons.append("hop_count_below_minimum")
                if not set(record.involved_instance_ids):
                    rejection_reasons.append("no_involved_instances")
                if not record.query_metadata.instance_chain.strip():
                    rejection_reasons.append("missing_instance_chain")
                if forbidden_hits:
                    rejection_reasons.append("contains_forbidden_visual_aid_reference")
                if instance_id_hits:
                    rejection_reasons.append("contains_instance_identifiers")

                verified = VerifiedHopChainQuery(
                    **record.model_dump(),
                    verification_status="accepted" if not rejection_reasons else "rejected",
                    rejection_reasons=rejection_reasons,
                    verification_metadata=VerificationMetadata(
                        numeric_answer=NUMERIC_PATTERN.match(record.hypothetical_answer.strip())
                        is not None,
                        forbidden_reference_terms=forbidden_hits,
                        instance_identifier_terms=instance_id_hits,
                        references_all_instances=record.query_metadata.uses_all_instances,
                    ),
                )
                status_counts[verified.verification_status] += 1
                payload = json.dumps(verified.model_dump())
                output_file.write(payload + "\n")
                if verified.verification_status == "accepted":
                    accepted_file.write(payload + "\n")
                else:
                    rejected_file.write(payload + "\n")
            except Exception as exc:
                logger.exception("Failed to verify line %s: %s", line_num, exc)
                status_counts["parse_error"] += 1

    summary = {
        "total_records_processed": total,
        "verified_file": str(output_path),
        "accepted_file": str(accepted_path),
        "rejected_file": str(rejected_path),
        "status_counts": dict(status_counts),
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2))
    logger.info("Verified %s candidate queries", total)


if __name__ == "__main__":
    main()
