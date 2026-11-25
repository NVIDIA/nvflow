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

"""Recover the exact full trajectory selected by GenSelect for tool-calling SFT."""

from __future__ import annotations

import argparse
import json
import re
import uuid
from pathlib import Path
from typing import Any

from nvflow.utils.jsonl import iter_jsonl, write_jsonl


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
    return str(content or "")


def extract_question(record: dict[str, Any]) -> str:
    inputs = (record.get("responses_create_params") or {}).get("input") or []
    content = _content_text(inputs[0].get("content", "")) if inputs else ""
    match = re.search(r"Question:\s*\n(.*?)(?:\n\n|\Z)", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return str(record.get("problem") or record.get("question") or "").strip()


def _system_and_question(record: dict[str, Any]) -> tuple[str, str]:
    inputs = (record.get("responses_create_params") or {}).get("input") or []
    content = _content_text(inputs[0].get("content", "")) if inputs else ""
    marker = content.find("Question:")
    system = content[:marker].strip() if marker > 0 else ""
    return system, extract_question(record)


def tool_arguments(value: Any) -> dict[str, Any]:
    """Return a tool call's arguments as a dict; anything unparsable becomes {}."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _final_answer(item: dict[str, Any]) -> str:
    arguments = tool_arguments(item.get("arguments"))
    return str(arguments.get("final_result") or "")


def _reasoning_text(item: dict[str, Any]) -> str:
    parts = item.get("summary") or item.get("content") or []
    texts = [str(part.get("text") or "") for part in parts if isinstance(part, dict)]
    return "\n\n".join(text for text in texts if text) or str(item.get("text") or "")


def _awaits_outputs(message: dict[str, Any]) -> bool:
    return message.get("role") == "assistant" and bool(message.get("tool_calls"))


def _unwrapped(content: str) -> str:
    opening, closing = "<think>\n", "\n</think>"
    if content.startswith(opening) and content.endswith(closing):
        return content[len(opening) : -len(closing)]
    return content


def extract_messages(record: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Convert one Responses API trajectory to structured OpenAI messages.

    Tool calls made together, with no text or tool output between them, share one
    assistant message.
    """
    output = (record.get("response") or {}).get("output") or []
    if not output:
        return None
    system, question = _system_and_question(record)
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": f"Question: {question}"})
    pending: list[str] = []

    for item in output:
        item_type = item.get("type")
        if item_type == "message":
            for content in item.get("content") or []:
                text = content if isinstance(content, str) else content.get("text", "")
                if text:
                    pending.append(str(text))
        elif item_type == "reasoning":
            text = _reasoning_text(item)
            if text:
                pending.append(text)
        elif item_type == "function_call":
            name = str(item.get("name") or "")
            thinking = "\n\n".join(pending).strip()
            pending.clear()
            if name == "submit_final_result" and _awaits_outputs(messages[-1]):
                # Calls made with the submission never get outputs; the final
                # message keeps their turn's thinking.
                unanswered = messages.pop()
                thinking = "\n\n".join(
                    text for text in (_unwrapped(unanswered["content"]), thinking) if text
                )
            prefix = f"<think>\n{thinking}\n</think>" if thinking else ""
            if name == "submit_final_result":
                answer = _final_answer(item)
                messages.append(
                    {
                        "role": "assistant",
                        "content": f"{prefix}\n\n{answer}".strip(),
                    }
                )
                break
            call = {
                "type": "function",
                "id": str(item.get("call_id") or ""),
                "function": {
                    "name": name,
                    "arguments": tool_arguments(item.get("arguments")),
                },
            }
            previous = messages[-1]
            if not thinking and previous.get("role") == "assistant" and previous.get("tool_calls"):
                previous["tool_calls"].append(call)
            else:
                messages.append({"role": "assistant", "content": prefix, "tool_calls": [call]})
        elif item_type == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "content": str(item.get("output") or ""),
                    "tool_call_id": str(item.get("call_id") or ""),
                }
            )

    if messages[-1].get("role") != "assistant":
        return None
    if not any(message.get("tool_calls") for message in messages):
        return None
    return messages


def _normalize_problem(value: str) -> str:
    return " ".join(value.lower().split())


def extract_selected_trajectories(
    selection_file: Path, rollout_files: list[Path], output_file: Path
) -> tuple[int, int]:
    """Join selections to source rollouts by exact model-file/repeat identity."""
    selections: dict[tuple[int, int, str], dict[str, Any]] = {}
    for selected in iter_jsonl(selection_file, on_error="skip"):
        reference = selected.get("selected_candidate_reference")
        if not isinstance(reference, dict):
            continue
        file_index = reference.get("source_file_index")
        rollout_index = reference.get("rollout_index")
        if not isinstance(file_index, int) or not isinstance(rollout_index, int):
            continue
        problem = _normalize_problem(str(selected.get("problem") or ""))
        selections[(file_index, rollout_index, problem)] = selected

    output_file.parent.mkdir(parents=True, exist_ok=True)
    matched: set[tuple[int, int, str]] = set()
    count = 0
    with write_jsonl(output_file) as writer:
        for file_index, rollout_file in enumerate(rollout_files):
            for rollout in iter_jsonl(rollout_file, on_error="skip"):
                rollout_index = rollout.get("_ng_rollout_index")
                if not isinstance(rollout_index, int):
                    continue
                problem = _normalize_problem(extract_question(rollout))
                key = (file_index, rollout_index, problem)
                selected = selections.get(key)
                if selected is None or key in matched:
                    continue
                messages = extract_messages(rollout)
                if messages is None:
                    continue
                writer.write(
                    {
                        "messages": messages,
                        "uuid": str(
                            uuid.uuid5(
                                uuid.NAMESPACE_URL,
                                f"{problem}||{selected.get('generation', '')}",
                            )
                        ),
                        "problem": selected.get("problem", ""),
                        "expected_answer": selected.get("generation", ""),
                        "question_type": selected.get("question_type", "unknown"),
                        "selected_original_seed_index": selected.get(
                            "selected_original_seed_index"
                        ),
                        "selected_candidate_reference": selected.get(
                            "selected_candidate_reference"
                        ),
                    }
                )
                matched.add(key)
                count += 1
    return count, len(selections) - len(matched)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection_file", required=True, type=Path)
    parser.add_argument("--rollout_files", nargs="+", required=True, type=Path)
    parser.add_argument("--output_file", required=True, type=Path)
    args = parser.parse_args()
    written, missing = extract_selected_trajectories(
        args.selection_file, args.rollout_files, args.output_file
    )
    print(f"Wrote {written} selected trajectories to {args.output_file}")
    if missing:
        raise RuntimeError(f"Could not recover {missing} GenSelect-selected trajectories")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
