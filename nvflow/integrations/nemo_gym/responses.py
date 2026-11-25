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
"""Responses API request rows, and the text of the rollouts that answer them."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def load_prompt_template(template_path: str | Path) -> tuple[str | None, str]:
    """Return the ``(system, user)`` templates of a YAML file with a required ``user`` key."""
    import yaml

    with open(template_path) as f:
        config = yaml.safe_load(f)
    user_template = config.get("user")
    if not user_template:
        raise ValueError(f"Prompt template {template_path} must have a 'user:' key")
    return config.get("system") or None, user_template


def render_prompt(record: dict[str, Any], template: str) -> str:
    """Fill ``{field}`` placeholders from *record*; missing fields become empty."""

    class _DefaultDict(dict):
        def __missing__(self, key: str) -> str:
            return ""

    return template.format_map(_DefaultDict(record))


def to_responses_api(
    record: dict[str, Any],
    rendered_prompt: str,
    rendered_system: str | None = None,
    system_role: str = "system",
) -> dict[str, Any]:
    """Return *record* with ``responses_create_params`` holding the rendered messages.

    Every chat template accepts the ``system`` role; use ``developer`` only for
    models whose chat format defines it.
    """
    messages: list[dict[str, str]] = []
    if rendered_system:
        messages.append({"role": system_role, "content": rendered_system})
    messages.append({"role": "user", "content": rendered_prompt})
    result = dict(record)
    result["responses_create_params"] = {"input": messages}
    return result


def canonical_input(responses_create_params: dict[str, Any]) -> str:
    """JSON of the request messages' ``role`` and ``content``.

    Gym re-emits each input message with extra keys, so these are the only
    fields that match between a request row and its rollout row.
    """
    messages = (responses_create_params or {}).get("input", [])
    normalized = [
        {"role": message.get("role"), "content": message.get("content")}
        if isinstance(message, dict)
        else message
        for message in messages
    ]
    return json.dumps(normalized, sort_keys=True, ensure_ascii=False)


def compute_join_id(record: dict[str, Any]) -> str:
    """Hash that links a rollout row to its request row.

    ``_ng_task_index`` is unique only within one collection, so it can repeat
    when rollouts from separate collections are merged.
    """
    payload = canonical_input(record.get("responses_create_params", {}))
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def assistant_text(response: dict[str, Any]) -> str:
    """The assistant's text in a Gym response, in either output shape."""
    texts: list[str] = []
    for item in response.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message" and item.get("role") == "assistant":
            content = item.get("content", "")
            if isinstance(content, list):
                texts.extend(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "output_text"
                )
            elif isinstance(content, str):
                texts.append(content)
        elif isinstance(item.get("content"), str):
            texts.append(item["content"])
        elif "text" in item and item.get("type") != "reasoning":
            texts.append(item["text"])
    return "\n".join(texts)


def reasoning_text(response: dict[str, Any]) -> str:
    """Reasoning text: Responses API ``reasoning`` items, else ``reasoning_content``."""
    texts: list[str] = []
    output = response.get("output", [])
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "reasoning":
                continue
            for key in ("summary", "content"):
                parts = item.get(key)
                if isinstance(parts, list):
                    texts.extend(
                        part["text"]
                        for part in parts
                        if isinstance(part, dict)
                        and isinstance(part.get("text"), str)
                        and part["text"]
                    )
                elif key == "content" and isinstance(parts, str) and parts:
                    texts.append(parts)
            if isinstance(item.get("text"), str) and item["text"]:
                texts.append(item["text"])
    if texts:
        return "\n".join(texts)
    for key in ("reasoning_content", "reasoning"):
        value = response.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


__all__ = [
    "assistant_text",
    "canonical_input",
    "compute_join_id",
    "load_prompt_template",
    "reasoning_text",
    "render_prompt",
    "to_responses_api",
]
