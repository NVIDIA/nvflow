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
"""Record text, normalized keys, and word shingles for dedup and decontamination."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

_WORD = re.compile(r"\w+")


def field_names(value: Any, option: str, *, required: bool = True) -> tuple[str, ...]:
    """Read a stage option that names one field or a list of fields."""
    names = [value] if isinstance(value, str) else value
    if (
        not isinstance(names, list | tuple)
        or not all(isinstance(name, str) and name for name in names)
        or (required and not names)
    ):
        raise ValueError(f"'{option}' must be a field name or a list of field names")
    return tuple(names)


def field_value(record: Mapping[str, Any], path: str) -> Any:
    """Return a dotted field such as ``a.b`` or ``messages.0.content``, or ``None``."""
    value: Any = record
    for part in path.split("."):
        if isinstance(value, Mapping):
            value = value.get(part)
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            return None
    return value


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return "\n".join(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def record_text(
    record: Mapping[str, Any],
    fields: Sequence[str],
    *,
    text_after: str | None = None,
) -> str:
    """Join the non-empty *fields* of *record* with newlines.

    With *text_after*, a regular expression, each field keeps only the text after
    its last match, such as the question at the end of a rendered prompt.
    """
    marker = re.compile(text_after) if text_after else None
    parts = []
    for path in fields:
        text = _as_text(field_value(record, path))
        if marker is not None:
            matches = list(marker.finditer(text))
            if matches:
                text = text[matches[-1].end() :]
        text = text.strip()
        if text:
            parts.append(text)
    return "\n".join(parts)


def words(text: str) -> list[str]:
    """Case-folded word tokens after Unicode NFKC normalization."""
    return _WORD.findall(unicodedata.normalize("NFKC", text).casefold())


def normalized_text(text: str) -> str:
    """Text that ignores case, punctuation, and whitespace differences."""
    return " ".join(words(text))


def folded_text(text: str) -> str:
    """Text that ignores only case and whitespace differences, so "-5%" differs from "5%"."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def word_shingles(text: str, size: int) -> set[str]:
    """Word n-grams of *text*; shorter text becomes one shingle of all its words."""
    tokens = words(text)
    if len(tokens) < size:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}


def exact_key(text: str, guard: Sequence[Any] = ()) -> str:
    """Digest of the normalized text plus guard values that must also match."""
    payload = json.dumps([normalized_text(text), list(guard)], ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "exact_key",
    "field_names",
    "field_value",
    "folded_text",
    "normalized_text",
    "record_text",
    "word_shingles",
    "words",
]
