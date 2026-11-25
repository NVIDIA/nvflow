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

"""Deterministic parsing and validation for generated rubric artifacts."""

from __future__ import annotations

import json
from typing import Any

RUBRIC_SCHEMA_VERSION = 1
RUBRIC_TYPES = {"hard_rule", "fact", "calculation", "principle", "conclusion", "criterion"}


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object while tolerating a Markdown fence or surrounding prose."""

    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        value = "\n".join(lines).strip()

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = _trailing_json_object(value)
    if not isinstance(parsed, dict):
        raise ValueError("model output must contain one JSON object")
    return parsed


def _trailing_json_object(value: str) -> dict[str, Any]:
    """Return the last self-contained JSON object in a prose-wrapped response.

    Reasoning models emit their chain of thought before the answer, and that
    prose routinely contains braces and draft objects. Slicing from the first
    '{' to the last '}' therefore spans unrelated text and fails. Scanning
    candidate starts from the end instead lands on the object the model
    actually finished with; nested items are skipped because only the outer
    object carries the rubric payload.
    """

    decoder = json.JSONDecoder()
    fallback: dict[str, Any] | None = None
    for start in range(len(value) - 1, -1, -1):
        if value[start] != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(value[start:])
        except json.JSONDecodeError:
            continue
        if not isinstance(candidate, dict):
            continue
        if "rubric" in candidate:
            return candidate
        if fallback is None:
            fallback = candidate
    if fallback is not None:
        return fallback
    raise json.JSONDecodeError("no JSON object found in model output", value, 0)


def normalize_rubric(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a stable rubric schema or raise for structurally invalid output."""

    raw_rubric = payload.get("rubric")
    if not isinstance(raw_rubric, list) or not raw_rubric:
        raise ValueError("rubric must be a non-empty list")

    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_criteria: set[str] = set()
    for index, item in enumerate(raw_rubric, 1):
        if not isinstance(item, dict):
            raise ValueError(f"rubric item {index} must be an object")

        criterion = item.get("criterion")
        if not isinstance(criterion, str) or not criterion.strip():
            raise ValueError(f"rubric item {index} has no criterion")
        criterion = criterion.strip()
        fingerprint = " ".join(criterion.casefold().split())
        if fingerprint in seen_criteria:
            raise ValueError(f"rubric item {index} duplicates an earlier criterion")
        seen_criteria.add(fingerprint)

        rubric_id = str(item.get("id") or f"R{index}").strip()
        if not rubric_id or rubric_id in seen_ids:
            raise ValueError(f"rubric item {index} has an empty or duplicate id")
        seen_ids.add(rubric_id)

        severity = item.get("severity", 1)
        if not isinstance(severity, int) or isinstance(severity, bool) or severity not in {1, 2, 3}:
            raise ValueError(f"rubric item {index} severity must be 1, 2, or 3")
        must_pass = item.get("must_pass", False)
        if not isinstance(must_pass, bool):
            raise ValueError(f"rubric item {index} must_pass must be boolean")

        criterion_type = str(item.get("type") or "criterion").strip()
        if criterion_type not in RUBRIC_TYPES:
            raise ValueError(f"rubric item {index} type must be one of {sorted(RUBRIC_TYPES)}")
        # gold and lookalike are carried through without being required: a
        # missing one costs a criterion its offline discrimination check, which
        # is not worth failing an otherwise usable rubric over.
        normalized.append(
            {
                "id": rubric_id,
                "type": criterion_type,
                "criterion": criterion,
                "gold": str(item.get("gold") or "").strip(),
                "lookalike": str(item.get("lookalike") or "").strip(),
                "severity": severity,
                "must_pass": must_pass,
                "evidence_summary": str(item.get("evidence_summary") or "").strip(),
            }
        )

    if not any(item["must_pass"] for item in normalized):
        raise ValueError("rubric must contain at least one must-pass criterion")
    return normalized
