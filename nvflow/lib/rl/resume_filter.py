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
"""Filter out already-completed rows from a rollout input file.

Supports fine-grained resume for ``ng_collect_rollouts``: given a
``-async`` output file (rows completed so far) and the original input
file, writes a ``remaining`` file containing only the rows whose
fingerprint does not appear in the partial output.

Fingerprint = md5(json(responses_create_params.input) + "|" + expected_answer).

Optional ``max_num_samples`` truncates the input before filtering,
so the original file is used directly without host-side copies.

Standalone script (stdlib only, no nvflow/Gym dependencies) that runs
inside the Slurm container with python3.

Usage:
    python -m nvflow.lib.rl.resume_filter <partial_file> <input_file> <remaining_file> [max_num_samples]

Exit codes:
    0 -- success (remaining_file written; may be empty if ALL_DONE)
    1 -- error
"""

import hashlib
import json
import os
import sys


def fingerprint(row: dict) -> str:
    """Content-based hash for deduplication across async output order."""
    rcp = row.get("responses_create_params", {})
    inp = rcp.get("input", [])
    ea = row.get("expected_answer", "")
    return hashlib.md5((json.dumps(inp, sort_keys=True) + "|" + str(ea)).encode()).hexdigest()


def load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    dropped = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                dropped += 1
    if dropped:
        print(f"WARNING: Dropped {dropped} malformed line(s) in {path}")
    return rows


def resume_filter(
    partial_file: str,
    input_file: str,
    remaining_file: str,
    max_num_samples: int = 0,
) -> None:
    inputs = load_jsonl(input_file)
    if 0 < max_num_samples < len(inputs):
        print(f"Truncating input from {len(inputs)} to {max_num_samples} rows (max_num_samples).")
        inputs = inputs[:max_num_samples]

    if not os.path.exists(partial_file) or os.path.getsize(partial_file) == 0:
        print(f"No partial output -- full run ({len(inputs)} rows).")
        with open(remaining_file, "w") as f:
            for r in inputs:
                f.write(json.dumps(r) + "\n")
        print(f"RESUME_STATUS: remaining={len(inputs)} completed=0 total={len(inputs)}")
        return

    completed = load_jsonl(partial_file)
    done_fps = {fingerprint(r) for r in completed}
    remaining = [r for r in inputs if fingerprint(r) not in done_fps]

    with open(remaining_file, "w") as f:
        for r in remaining:
            f.write(json.dumps(r) + "\n")

    print(
        f"RESUME_STATUS: remaining={len(remaining)} completed={len(completed)} total={len(inputs)}"
    )
    if not remaining:
        print("ALL_DONE")


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5):
        print(
            "Usage: python -m nvflow.lib.rl.resume_filter "
            "<partial_file> <input_file> <remaining_file> [max_num_samples]"
        )
        sys.exit(1)
    max_samples = int(sys.argv[4]) if len(sys.argv) == 5 else 0
    resume_filter(sys.argv[1], sys.argv[2], sys.argv[3], max_samples)
