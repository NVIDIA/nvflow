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
"""Postprocess query-generation outputs into typed HopChain records."""

from __future__ import annotations

import argparse
import json
import logging
import uuid
from pathlib import Path

from nvflow.recipes.multimodal.utils.hopchain_sdg_common import extract_json_value
from nvflow.recipes.multimodal.utils.hopchain_sdg_models import (
    GeneratedHopChainQuery,
    GeneratedHopChainQueryMetadata,
    ReasoningHop,
)
from nvflow.recipes.multimodal.utils.image_filter_models import GenerationStats

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Postprocess HopChain query-generation output")
    parser.add_argument("--input", required=True, help="Raw nemo-skills output JSONL")
    parser.add_argument("--output", required=True, help="GeneratedHopChainQuery JSONL")
    parser.add_argument("--summary", required=True, help="Summary JSON path")
    parser.add_argument("--generator-prompt-version", default="paper_appendix_a")
    return parser.parse_args()


def parse_queries(generation_text: str) -> list[dict]:
    """Extract generated query objects from a model response.

    The HopChain prompt contract requires a top-level object with a
    `sub_queries` array, so we only accept that shape here.
    """
    value = extract_json_value(generation_text)
    if isinstance(value, dict):
        if "sub_queries" in value and isinstance(value["sub_queries"], list):
            return [item for item in value["sub_queries"] if isinstance(item, dict)]
        return []
    return []


def main() -> None:
    """Entry point."""
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_records = 0
    total_queries = 0
    parse_errors = 0
    with Path(args.input).open("r") as input_file, output_path.open("w") as output_file:
        for line_num, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            total_records += 1
            try:
                record = json.loads(line)
                metadata = record.get("_metadata", {})
                generation_text = record.get("generation", "").strip()
                parsed_queries = parse_queries(generation_text)
                if not parsed_queries:
                    parse_errors += 1
                    logger.warning(
                        "No query objects found in generation output on line %s; skipping record",
                        line_num,
                    )
                    continue
                for query_idx, query in enumerate(parsed_queries, start=1):
                    reasoning_hops = [
                        ReasoningHop.model_validate(hop)
                        for hop in query.get("reasoning_hops", [])
                        if isinstance(hop, dict)
                    ]
                    if not reasoning_hops:
                        parse_errors += 1
                        logger.warning(
                            "Query %s on line %s has no valid reasoning_hops; skipping query",
                            query_idx,
                            line_num,
                        )
                        continue
                    query_record = GeneratedHopChainQuery(
                        query_id=str(
                            uuid.uuid5(
                                uuid.NAMESPACE_URL, f"{metadata['combination_id']}:{query_idx}"
                            )
                        ),
                        image_id=metadata["image_id"],
                        combination_id=metadata["combination_id"],
                        image_file_name=metadata["image_file_name"],
                        image_fullpath=metadata["image_path"],
                        question=str(query.get("query", "")).strip(),
                        hypothetical_answer=str(query.get("hypothetical_answer", "")).strip(),
                        involved_instance_ids=[
                            str(item)
                            for item in query.get(
                                "involved_objects", metadata.get("instance_ids", [])
                            )
                        ],
                        hop_count=len(reasoning_hops),
                        query_metadata=GeneratedHopChainQueryMetadata(
                            primary_capability=str(query.get("primary_capability", "unknown")),
                            instance_chain=str(query.get("instance_chain", "")),
                            reasoning_hops=reasoning_hops,
                            design_rationale=str(query.get("design_rationale", "")),
                            uses_all_instances=set(metadata.get("instance_ids", []))
                            <= {str(item) for item in query.get("involved_objects", [])},
                            generator_prompt_version=args.generator_prompt_version,
                        ),
                        raw_generation=generation_text,
                        generation_stats=GenerationStats(
                            num_generated_tokens=record.get("num_generated_tokens"),
                            generation_time=record.get("generation_time"),
                        ),
                    )
                    output_file.write(json.dumps(query_record.model_dump()) + "\n")
                    total_queries += 1
            except Exception as exc:
                parse_errors += 1
                logger.exception(
                    "Failed to postprocess query-generation line %s: %s", line_num, exc
                )

    summary = {
        "total_prompt_records": total_records,
        "total_queries": total_queries,
        "parse_errors": parse_errors,
        "output_file": str(output_path),
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2))
    logger.info("Postprocessed %s generated HopChain queries", total_queries)


if __name__ == "__main__":
    main()
