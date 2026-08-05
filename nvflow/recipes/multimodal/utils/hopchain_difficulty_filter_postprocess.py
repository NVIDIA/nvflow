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
"""Postprocess Omni difficulty filter outputs.

Groups inference results by query_id, computes pass_rate (fraction of k samples
whose normalized answer matches the consensus), and writes a single final_output.jsonl
with all records scored. Summary counts are written to summary.json.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, Field

from nvflow.recipes.multimodal.utils.hopchain_llm_judge_common import parse_judge_response_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DifficultyFilterSummary(BaseModel):
    """Summary metadata for one difficulty filter run."""

    input_file: str
    output_file: str
    kept_output_file: str
    total_input: int = Field(ge=0)
    kept_count: int = Field(ge=0)
    rejected_easy_count: int = Field(ge=0)
    rejected_impossible_count: int = Field(ge=0)
    error_count: int = Field(ge=0)
    k: int
    pass_rate_histogram: dict[str, int] = Field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score and filter Omni difficulty outputs")
    parser.add_argument("--input", required=True, help="nemo-skills output.jsonl")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write final_output.jsonl (all rows) and kept_output.jsonl (filtered rows)",
    )
    parser.add_argument("--summary", required=True, help="Summary JSON output path")
    parser.add_argument("--k", type=int, default=5, help="Expected samples per question")
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=0.0,
        help="Inclusive lower bound for kept rows; pass rates below this are rejected_impossible",
    )
    parser.add_argument(
        "--max-pass-rate",
        type=float,
        default=1.0,
        help="Inclusive upper bound for kept rows; pass rates above this are rejected_easy",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "final_output.jsonl"
    kept_output_path = output_dir / "kept_output.jsonl"

    # Group inference results by query_id, preserve sample_index order
    groups: dict[str, list[dict]] = defaultdict(list)
    with Path(args.input).open("r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                query_id = record["_metadata"]["query_id"]
                groups[query_id].append(record)
            except Exception as exc:
                logger.warning("Failed to parse inference output line: %s", exc)

    min_pass_rate = args.min_pass_rate
    max_pass_rate = args.max_pass_rate
    kept_count = rejected_easy_count = rejected_impossible_count = error_count = 0
    pass_rate_histogram: dict[str, int] = {}

    with output_path.open("w") as out, kept_output_path.open("w") as kept_out:
        for query_id, samples in groups.items():
            samples.sort(key=lambda x: x["_metadata"]["sample_index"])
            consensus = samples[0]["_metadata"]["consensus_normalized_answer"]
            original_record = samples[0]["_metadata"]["record"]
            k = len(samples)

            parsed_answers: list[str] = []
            matches = 0
            for sample in samples:
                raw = sample.get("generation", "") or sample.get("reasoning_content", "")
                try:
                    parsed = parse_judge_response_text(raw)
                    parsed_answers.append(parsed.normalized_answer)
                    if parsed.normalized_answer == consensus:
                        matches += 1
                except Exception as exc:
                    logger.warning("Failed to parse response for %s: %s", query_id, exc)
                    error_count += 1
                    parsed_answers.append("")

            pass_rate = matches / k if k > 0 else 0.0
            hist_key = f"{matches}/{k}"
            pass_rate_histogram[hist_key] = pass_rate_histogram.get(hist_key, 0) + 1

            if pass_rate < min_pass_rate:
                status = "rejected_impossible"
                rejected_impossible_count += 1
            elif pass_rate > max_pass_rate:
                status = "rejected_easy"
                rejected_easy_count += 1
            else:
                status = "kept"
                kept_count += 1

            output_record = {
                **original_record,
                "difficulty_pass_rate": pass_rate,
                "difficulty_filter_status": status,
                "difficulty_filter_raw_answers": parsed_answers,
            }
            out.write(json.dumps(output_record) + "\n")
            if status == "kept":
                kept_out.write(json.dumps(output_record) + "\n")

    total_input = kept_count + rejected_easy_count + rejected_impossible_count
    summary = DifficultyFilterSummary(
        input_file=args.input,
        output_file=str(output_path),
        kept_output_file=str(kept_output_path),
        total_input=total_input,
        kept_count=kept_count,
        rejected_easy_count=rejected_easy_count,
        rejected_impossible_count=rejected_impossible_count,
        error_count=error_count,
        k=args.k,
        pass_rate_histogram=pass_rate_histogram,
    )
    Path(args.summary).write_text(summary.model_dump_json(indent=2))
    logger.info(
        "Done. Kept: %d | Easy (rejected): %d | Impossible (rejected): %d | Parse errors: %d",
        kept_count,
        rejected_easy_count,
        rejected_impossible_count,
        error_count,
    )


if __name__ == "__main__":
    main()
