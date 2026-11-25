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

"""Adapters from portable API records into the canonical rubric contract."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

INPUT_FORMAT_ALIASES = {
    "canonical": "canonical",
    "canonical_jsonl": "canonical",
    "openai_response": "openai_responses",
    "openai_responses": "openai_responses",
}
INPUT_FORMATS = set(INPUT_FORMAT_ALIASES)


def normalize_input_format(value: str | None) -> str:
    """Return the canonical name for a supported input format."""

    name = (value or "canonical").strip().lower()
    try:
        return INPUT_FORMAT_ALIASES[name]
    except KeyError as exc:
        raise ValueError(f"input_format must be one of {sorted(INPUT_FORMATS)}") from exc


def _get_path(record: Mapping[str, Any], path: str) -> Any:
    value: Any = record
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _first(record: Mapping[str, Any], *paths: str) -> Any:
    for path in paths:
        value = _get_path(record, path)
        if value is not None:
            return value
    return None


def _content_text(value: Any) -> str:
    """Extract text from OpenAI message content without assuming a domain."""

    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [_content_text(item) for item in value]
        return "\n".join(part for part in parts if part).strip()
    if not isinstance(value, Mapping):
        return ""
    for key in ("text", "output_text", "input_text"):
        nested = value.get(key)
        if isinstance(nested, str) and nested.strip():
            return nested.strip()
    if "content" in value:
        return _content_text(value["content"])
    return ""


def _request_input(record: Mapping[str, Any]) -> Any:
    return _first(
        record,
        "responses_create_params.input",
        "request.input",
        "input",
    )


def _extract_question(record: Mapping[str, Any]) -> str:
    request_input = _request_input(record)
    if isinstance(request_input, str):
        return request_input.strip()
    if not isinstance(request_input, list):
        return ""

    # A Responses request may contain system/developer instructions, tool items,
    # and user messages. The last user message is the portable task text.
    for item in reversed(request_input):
        if isinstance(item, Mapping) and item.get("role") == "user":
            text = _content_text(item.get("content"))
            if text:
                return text
    for item in reversed(request_input):
        text = _content_text(item)
        if text:
            return text
    return ""


def _function_result(item: Mapping[str, Any]) -> str:
    arguments = item.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return ""
    if not isinstance(arguments, Mapping):
        return ""
    for key in ("final_result", "answer", "result", "output"):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def extract_response_answer(value: Any) -> str:
    """Extract the final answer from a serialized OpenAI Responses result."""

    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, Mapping):
        return ""
    if isinstance(value.get("response"), Mapping):
        value = value["response"]

    direct = _first(value, "output_text", "answer", "final_answer")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    output = value.get("output")
    if isinstance(output, str):
        return output.strip()
    if not isinstance(output, list):
        return ""

    # Prefer the last final message/tool submission over intermediate reasoning.
    for item in reversed(output):
        if not isinstance(item, Mapping):
            continue
        item_type = item.get("type")
        if item_type in {"function_call", "tool_call"}:
            result = _function_result(item)
            if result:
                return result
        if item.get("role") == "assistant" or item_type == "message":
            text = _content_text(item.get("content"))
            if text:
                return text
        if item_type in {"output_text", "text"}:
            text = _content_text(item)
            if text:
                return text
    return ""


def _candidate_answers(record: Mapping[str, Any]) -> list[str] | None:
    candidates = _first(record, "candidate_responses", "responses", "rollouts")
    if not isinstance(candidates, list):
        return None
    answers: list[str] = []
    for candidate in candidates:
        answers.append(extract_response_answer(candidate))
    return answers or None


def adapt_input_record(
    record: Mapping[str, Any],
    *,
    input_format: str = "canonical",
) -> Mapping[str, Any]:
    """Adapt one input record while preserving explicit user-supplied fields."""

    normalized_format = normalize_input_format(input_format)
    if normalized_format == "canonical":
        return record

    adapted = dict(record)
    sidecar_paths = {
        "id": ("metadata.rubric.id", "response.id"),
        "question": ("metadata.rubric.question",),
        "context": (
            "rubric_context",
            "metadata.rubric.context",
            "metadata.rubric_context",
            "metadata.context",
        ),
        "reference_answer": (
            "metadata.rubric.reference_answer",
            "metadata.reference_answer",
        ),
        "preferred_index": (
            "metadata.rubric.preferred_index",
            "metadata.preferred_index",
        ),
        "preferred_answer": ("metadata.rubric.preferred_answer",),
        "rejected_answer": ("metadata.rubric.rejected_answer",),
    }
    for field, paths in sidecar_paths.items():
        if adapted.get(field) is None:
            value = _first(record, *paths)
            if value is not None:
                adapted[field] = value

    if not adapted.get("question") and not adapted.get("problem"):
        adapted["question"] = _extract_question(record)

    if not any(
        adapted.get(key) is not None for key in ("reference_answer", "expected_answer", "answer")
    ):
        response = record.get("response")
        answer = extract_response_answer(response)
        if answer:
            adapted["reference_answer"] = answer

    if not any(
        adapted.get(key) is not None for key in ("candidate_answers", "solutions", "answers")
    ):
        candidates = _candidate_answers(record)
        if candidates is not None:
            adapted["candidate_answers"] = candidates

    return adapted
