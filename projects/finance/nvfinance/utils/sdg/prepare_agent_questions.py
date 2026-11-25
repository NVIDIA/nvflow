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

"""Merge, filter, deduplicate, and normalize questions for agent rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from nvflow.integrations.nemo_gym.native_contract import NativeGymOutputLayout
from nvflow.utils.jsonl import iter_jsonl, write_jsonl


def _question_key(record: dict[str, Any]) -> str:
    payload = f"{record.get('context', '')}||{record.get('problem', '')}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _verification_passed(record: dict[str, Any]) -> bool:
    generation = str(record.get("generation") or "")
    if not generation:
        serialized = record.get("serialized_output") or []
        if serialized and isinstance(serialized[0], dict):
            generation = str(serialized[0].get("content") or "")
    last_yes = generation.lower().rfind("yes")
    last_no = generation.lower().rfind("no")
    return last_yes >= 0 and last_yes > last_no


def _canonical_rollouts(directory: Path) -> Path:
    candidates = (
        Path(NativeGymOutputLayout.collection(directory).analyzed_artifacts.rollouts.as_posix()),
        directory / "output.jsonl",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No canonical verified-question rollouts found in {directory}")


def _iter_verified_questions(directory: Path, threshold: float) -> Iterator[dict[str, Any]]:
    """Yield one representative record for each question passing the vote threshold."""
    votes: dict[str, list[Any]] = {}
    for record in iter_jsonl(_canonical_rollouts(directory), on_error="skip"):
        key = _question_key(record)
        entry = votes.setdefault(key, [0, 0, None, record])
        entry[0] += 1
        if _verification_passed(record):
            entry[1] += 1
            if entry[2] is None:
                entry[2] = record

    for total, positive, first_positive, first_record in votes.values():
        if total and positive / total >= threshold:
            yield first_positive or first_record


def _iter_source(source: dict[str, Any], threshold: float) -> Iterator[dict[str, Any]]:
    source_format = source.get("format", "jsonl")
    path = Path(source["path"])
    if source_format == "jsonl":
        yield from iter_jsonl(path, on_error="skip")
    elif source_format == "verified_rollouts":
        yield from _iter_verified_questions(path, threshold)
    else:
        raise ValueError(f"Unsupported source format: {source_format!r}")


def prepare_agent_questions(
    sources: list[dict[str, Any]], output_file: Path, verification_threshold: float = 1.0
) -> int:
    """Merge and normalize question sources without supplying document context."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    count = 0
    with write_jsonl(output_file) as writer:
        for source in sources:
            for record in _iter_source(source, verification_threshold):
                problem = str(record.get("problem") or record.get("question") or "").strip()
                if not problem:
                    continue
                normalized_problem = " ".join(problem.lower().split())
                if normalized_problem in seen:
                    continue
                seen.add(normalized_problem)
                writer.write(
                    {
                        "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, normalized_problem)),
                        "problem": problem,
                        # The downstream agent prompt intentionally omits context. These
                        # non-empty sentinels satisfy the shared data normalizer while
                        # making the open-context contract explicit.
                        "context": "N/A",
                        "reasoning_content": "N/A",
                        "generation": "N/A",
                        "expected_answer": "N/A",
                        "question_type": record.get("question_type", "unknown"),
                    }
                )
                count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources_json", required=True)
    parser.add_argument("--output_file", required=True, type=Path)
    parser.add_argument("--verification_threshold", type=float, default=1.0)
    args = parser.parse_args()
    sources = json.loads(args.sources_json)
    if not isinstance(sources, list) or not sources:
        raise ValueError("sources_json must be a non-empty JSON list")
    count = prepare_agent_questions(sources, args.output_file, args.verification_threshold)
    print(f"Wrote {count} agent questions to {args.output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
