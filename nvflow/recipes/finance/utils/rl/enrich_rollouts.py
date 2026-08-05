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
output rows to input rows using ``expected_answer`` + prompt content
as the join key.

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
import os
import sys
import uuid as uuid_mod

from nvflow.utils import setup_logger

logger = setup_logger(__name__)


def _atomic_write_jsonl(path: str, rows: list[dict]) -> None:
    """Write *rows* to *path* atomically via ``tmp + os.replace``.

    A process killed mid-write leaves only ``{path}.tmp`` behind, never a
    truncated ``{path}``.  Critical when *path* is the SOURCE of truth on
    a re-run: a partial overwrite of the merged rollouts file (or the
    input file during deterministic UUID write-back) would lose data
    under SIGKILL / OOM / node-failure.

    Uses :func:`os.replace` which is atomic on POSIX provided tmp and
    target share a mount; emitting tmp in the same directory as target
    satisfies that requirement.
    """
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    os.replace(tmp, path)


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


def _ensure_uuids(inputs: list[dict], input_file: str | None = None) -> int:
    """Add uuid to any input row missing one.  Returns count generated.

    UUIDs are deterministic (derived from expected_answer + problem + row index)
    so that concurrent merge jobs across seeds produce the same UUID for the
    same question.  The row index guarantees uniqueness even when content
    fields are duplicated.

    When *input_file* is provided and UUIDs were generated, the input file
    is rewritten so that downstream steps (e.g. filter) can join on uuid.
    Because UUIDs are deterministic, concurrent seeds writing the same file
    produce identical content -- safe even under race conditions.
    """
    generated = 0
    for i, row in enumerate(inputs):
        if "uuid" not in row:
            row["uuid"] = _deterministic_uuid(row, i)
            generated += 1

    if generated and input_file:
        # Atomic write: a SIGKILL between truncate and full re-write would
        # otherwise leave the input file empty or partial, breaking the
        # next run's UUID indexing (deterministic_uuid depends on row index).
        _atomic_write_jsonl(input_file, inputs)

    return generated


def enrich(input_file: str, rollouts_file: str) -> None:
    with open(input_file) as f:
        inputs = [json.loads(line) for line in f if line.strip()]

    with open(rollouts_file) as f:
        rollouts = [json.loads(line) for line in f if line.strip()]

    if not rollouts:
        logger.warning("No rollouts to enrich.")
        return

    num_generated = _ensure_uuids(inputs, input_file=input_file)
    if num_generated:
        logger.info(
            "Generated UUIDs for %d/%d input rows (written back to %s)",
            num_generated,
            len(inputs),
            input_file,
        )

    sample_input_keys = set(inputs[0].keys()) if inputs else set()
    sample_output_keys = set(rollouts[0].keys())
    missing_keys = sample_input_keys - sample_output_keys

    def _match_key(row: dict) -> str:
        return row.get("expected_answer", "") + "|" + _extract_prompt(row)

    by_key: dict[str, dict] = {}
    key_collisions = 0
    for row in inputs:
        k = _match_key(row)
        if k in by_key:
            key_collisions += 1
        else:
            by_key[k] = row
    if key_collisions:
        logger.warning("%d input rows share the same (expected_answer, prompt) key", key_collisions)

    def _find_input(rollout: dict) -> dict | None:
        return by_key.get(_match_key(rollout))

    matched = 0
    unmatched = 0
    enriched: list[dict] = []
    for rollout in rollouts:
        match = _find_input(rollout)
        if match:
            merged = {**match, **rollout}
            if "uuid" in match:
                merged["uuid"] = match["uuid"]
            matched += 1
        else:
            merged = rollout
            unmatched += 1
        enriched.append(merged)

    # Atomic publish: write all enriched rows to a tmp file then rename.
    # If killed mid-write, the original rollouts_file (the chunk-merge
    # output) stays intact and the next merge run will re-enrich from
    # the same input.  A non-atomic write would silently corrupt the
    # merged data, since rollouts_file IS our source of rollouts.
    _atomic_write_jsonl(rollouts_file, enriched)

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
