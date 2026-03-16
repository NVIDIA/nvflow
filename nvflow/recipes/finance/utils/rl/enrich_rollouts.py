#!/usr/bin/env python3
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
"""Enrich rollout output with metadata from the original input.

NeMo-Gym environments may drop extra input fields (uuid, question,
template_metadata, etc.) because their Pydantic response models don't
use ``extra="allow"``.  This script restores those fields by matching
output rows to input rows using ``expected_answer`` as the join key.

For each match: ``enriched = input_row | output_row`` -- input fields
serve as defaults, output fields (response, reward, etc.) take
precedence and are never overwritten.

If any input row lacks a ``uuid``, a deterministic one is derived from
``expected_answer`` + ``problem`` so that all seeds produce the same UUID
for the same question.  The input file is never modified.

Standalone script that runs inside the Slurm container with python3.

Usage:
    python enrich_rollouts.py <input.jsonl> <rollouts.jsonl>
"""

import hashlib
import json
import sys
import uuid as uuid_mod
from collections import defaultdict

from nvflow.utils import setup_logger

logger = setup_logger(__name__)


def _extract_prompt(row: dict) -> str:
    """Extract the user prompt from responses_create_params.input."""
    rcp = row.get("responses_create_params", {})
    inputs = rcp.get("input", [])
    if inputs and isinstance(inputs, list):
        return inputs[0].get("content", "")
    return ""


def _deterministic_uuid(row: dict, index: int) -> str:
    """Derive a stable UUID from row content + position so all seeds agree."""
    key = (
        row.get("expected_answer", "")
        + "|"
        + row.get("problem", row.get("question", ""))
        + "|"
        + str(index)
    )
    return str(uuid_mod.UUID(hashlib.md5(key.encode()).hexdigest()))


def _ensure_uuids(inputs: list[dict]) -> int:
    """Add in-memory uuid to any input row missing one.  Returns count generated.

    UUIDs are deterministic (derived from expected_answer + problem + row index)
    so that concurrent merge jobs across seeds produce the same UUID for the
    same question.  The row index guarantees uniqueness even when content
    fields are duplicated.  The input file is NOT rewritten.
    """
    generated = 0
    for i, row in enumerate(inputs):
        if "uuid" not in row:
            row["uuid"] = _deterministic_uuid(row, i)
            generated += 1
    return generated


def enrich(input_file: str, rollouts_file: str) -> None:
    with open(input_file) as f:
        inputs = [json.loads(line) for line in f if line.strip()]

    with open(rollouts_file) as f:
        rollouts = [json.loads(line) for line in f if line.strip()]

    if not rollouts:
        logger.warning("No rollouts to enrich.")
        return

    # Ensure every input row has a uuid (in-memory only).
    num_generated = _ensure_uuids(inputs)
    if num_generated:
        logger.info("Generated in-memory UUIDs for %d/%d input rows", num_generated, len(inputs))

    # Fast path: if the environment already passes through all input fields
    # (e.g. extra="allow"), there's nothing to restore.  However, if we
    # generated deterministic UUIDs above, we must still enrich so that the
    # rollout output gets the correct (stable) UUID for cross-seed grouping.
    sample_input_keys = set(inputs[0].keys()) if inputs else set()
    sample_output_keys = set(rollouts[0].keys())
    missing_keys = sample_input_keys - sample_output_keys
    if not missing_keys and not num_generated:
        logger.info("All input fields already present in output -- nothing to enrich.")
        return

    # Build lookup: expected_answer -> list of input rows.
    # Most datasets have unique expected_answer per question, but we handle
    # duplicates by also matching on the prompt content.
    by_answer: dict[str, list[dict]] = defaultdict(list)
    for row in inputs:
        by_answer[row.get("expected_answer", "")].append(row)

    def _find_input(rollout: dict) -> dict | None:
        ea = rollout.get("expected_answer", "")
        candidates = by_answer.get(ea, [])
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            return None
        # Disambiguate by prompt content.
        prompt = _extract_prompt(rollout)
        for c in candidates:
            if c.get("question", "") and prompt and c["question"] in prompt:
                return c
        return candidates[0]

    matched = 0
    unmatched = 0
    with open(rollouts_file, "w") as f:
        for rollout in rollouts:
            match = _find_input(rollout)
            if match:
                merged = match | rollout
                if num_generated and "uuid" in match:
                    merged["uuid"] = match["uuid"]
                matched += 1
            else:
                merged = rollout
                unmatched += 1
            f.write(json.dumps(merged) + "\n")

    logger.info(
        "Enriched %d/%d rollouts (%d fields restored)", matched, len(rollouts), len(missing_keys)
    )
    if unmatched:
        logger.warning("%d rollouts could not be matched to input rows", unmatched)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        logger.error("Usage: python enrich_rollouts.py <input.jsonl> <rollouts.jsonl>")
        sys.exit(1)
    enrich(sys.argv[1], sys.argv[2])
